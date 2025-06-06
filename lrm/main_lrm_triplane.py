# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import time

import misc.dist_helper as dist_helper
import misc.logging as logging
import misc.utils as utils
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from data_loader.dtc_dataset import DtcDataset
from geometry.triplane import Triplane
from geometry.triplane_sdf import TriplaneSdf
from loss.perceptual_loss import PerceptualLoss
from loss.triplane_loss_lowmem import triplane_loss_lowmem
from loss.triplane_loss_vanilla import triplane_loss_acc
from misc.env_utils import fix_random_seeds, init_distributed_mode
from misc.io_helper import mkdirs
from models.multiview_encoder import mvencoder_base
from models.triplane_decoder import TriplaneTransformer
from renderer.nerf_renderer import NerfRenderer
from renderer.sdf_renderer import SdfRenderer
from torch.utils.tensorboard import SummaryWriter

logger = logging.get_logger(__name__)


def create_output_dir(args):
    args.user = os.environ["USER"] or "default"
    args.output_dir = os.path.join(args.exp_root, args.exp_name)
    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    args.image_dir = os.path.join(args.output_dir, "images")
    args.logging_save_path = os.path.join(args.output_dir, "log_verbose_rank_0.txt")

    mkdirs(args.output_dir)
    mkdirs(args.checkpoint_dir)
    mkdirs(args.image_dir)

    # Setup logging format.
    logging.setup_logging(
        args.logging_save_path,
        mode="a" if args.auto_resume else "w",
        buffering=args.buffering,
    )
    return args


def get_args_parser():
    parser = argparse.ArgumentParser("LRM Triplane", add_help=False)

    parser.add_argument(
        "--exp_root",
        required=True,
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument(
        "--exp_name",
        default="default_exp_name",
        type=str,
        help="Path to save logs and checkpoints.",
    )

    # Model parameters
    parser.add_argument(
        "--transformer_depth",
        default=24,
        type=int,
        help="Depth of the triplane transformer",
    )
    parser.add_argument(
        "--patch_size",
        default=8,
        type=int,
        help="patch_size of the encoder",
    )
    parser.add_argument(
        "--mlp_depth", default=3, type=int, help="Depth of the triplane Mlp"
    )
    parser.add_argument(
        "--mlp_brdf_depth", default=2, type=int, help="Depth of the triplane Mlp"
    )
    parser.add_argument(
        "--mlp_geo_depth", default=2, type=int, help="Depth of the triplane Mlp"
    )
    parser.add_argument(
        "--mlp_dim", default=32, type=int, help="Number of channels of the triplane MLP"
    )
    parser.add_argument(
        "--embed_dim",
        default=1024,
        type=int,
        help="The hidden dimension of transformer",
    )
    parser.add_argument(
        "--triplane_dim",
        default=32,
        type=int,
        help="Number of channels of the triplane input",
    )
    parser.add_argument(
        "--triplane_token_size", default=64, type=int, help="Triplane token size"
    )
    parser.add_argument(
        "--triplane_out_token_size",
        default=512,
        type=int,
        help="Triplane output token size",
    )
    parser.add_argument(
        "--view_embed_size", default=32, type=int, help="View embed token size"
    )
    parser.add_argument(
        "--view_embed_out_size",
        default=128,
        type=int,
        help="View embed output size",
    )
    parser.add_argument(
        "--num_samples_per_ray",
        default=512,
        type=int,
        help="the number of samples per ray",
    )
    parser.add_argument(
        "--mvencoder_type",
        default="plucker",
        choices=["plucker"],
        help="Mvencoder type",
    )

    parser.add_argument(
        "--renderer_type",
        default="nerf",
        choices=["nerf", "sdf"],
        help="Choose the renderer type",
    )
    parser.add_argument(
        "--sep_geo_app_fea",
        action="store_true",
        help="whether to separate geometry and appearance feature",
    )
    parser.add_argument(
        "--sep_geo_app_dims",
        type=int,
        nargs=2,
        default=None,
        help="feature split for geometry and appearance feature",
    )
    parser.add_argument(
        "--input_image_res", default=512, type=int, help="input image resolution"
    )
    parser.add_argument(
        "--output_image_res", default=128, type=int, help="output image resolution"
    )
    parser.add_argument(
        "--output_res_range",
        default=[128, 384],
        nargs=2,
        type=int,
        help="output image resize range",
    )

    parser.add_argument(
        "--bbox_radius",
        default=0.5,
        type=float,
        help="the size of the bounding box; 1.05 for sdf",
    )
    parser.add_argument(
        "--fov",
        default=None,
        type=float,
        help="fov",
    )

    parser.add_argument(
        "--dataset_type",
        default="dtc_dataset",
        choices=["dtc_dataset"],
        type=str,
        help="The input dataset format",
    )
    parser.add_argument(
        "--dtc_white_envmap",
        action="store_true",
        help="whether to load data with white environment map",
    )
    parser.add_argument(
        "--use_adobe_view_selection",
        action="store_true",
        help="whether to use adobe view selection for training",
    )
    parser.add_argument(
        "--prediction_type",
        default="rgb",
        choices=["rgb", "brdf", "both"],
        type=str,
        help="The output of LRM, either rgb or brdf",
    )
    parser.add_argument(
        "--encode_background",
        type=str,
        default="none",
        choices=["none", "only", "augment"],
        help="whether to encode the background information",
    )
    parser.add_argument(
        "--view_dependent_type",
        default="none",
        choices=["none", "sh", "neural", "neural_mip"],
        help="whether to predict view dependent radiance field",
    )
    parser.add_argument(
        "--max_dataset_size",
        default=-1,
        type=int,
        help="maximum number of models to be used for training",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_number_scale",
        default=1,
        type=int,
        help="scale the size of dataset - useful for training small dataset with large batch size.",
    )
    parser.add_argument(
        "--relative_cam_pose",
        action="store_true",
        help="whether to use relative camera poses.",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=1.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=8,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument(
        "--image_num_per_batch_range",
        default=[4, 8],
        nargs=2,
        type=int,
        help="Total number of images loaded per batch.",
    )
    parser.add_argument(
        "--output_image_num", default=4, type=int, help="output image num"
    )
    parser.add_argument(
        "--epochs", default=10000, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--lr",
        default=4e-4,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_iters",
        default=3000,
        type=int,
        help="Number of iterations for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--start_sdf_inv_std",
        type=float,
        default=10,
        help="the sdf std when the training starts",
    )
    parser.add_argument(
        "--warmup_sdf_inv_std",
        type=float,
        default=20,
        help="the sdf std after the warmup steps.",
    )
    parser.add_argument(
        "--end_sdf_inv_std",
        type=float,
        default=200,
        help="the sdf std after training ends",
    )
    parser.add_argument(
        "--gradient_checkpointing_freq",
        default=1,
        type=int,
        help="Frequency of block for gradient checkpointing",
    )
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.0, help="stochastic depth rate"
    )
    parser.add_argument(
        "--restart_from_checkpoint",
        action="store_true",
        help="restart from checkpoints",
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="path to the checkpoint"
    )
    parser.add_argument(
        "--weights_dir",
        default="config",
        help="the folder that contains yaml files for weights",
    )
    parser.add_argument(
        "--ocgrid_acc",
        action="store_true",
        help="whether to use occupancy grid to accelerate rendering",
    )
    parser.add_argument(
        "--filter_normal",
        action="store_true",
        help="whether to filter normal when computing normal loss",
    )
    parser.add_argument(
        "--mesh_resolution",
        type=int,
        default=256,
        help="the resolution when output mesh",
    )

    # Misc
    parser.add_argument(
        "--data_path",
        type=str,
        help="Please specify path to the training data.",
    )
    parser.add_argument(
        "--sep_gt_dir",
        default=None,
        type=str,
        help="Please specify path to the ImageNet training data.",
    )
    parser.add_argument(
        "--saveckp_epoch_freq",
        default=5,
        type=int,
        help="Save checkpoint every x epochs.",
    )
    parser.add_argument(
        "--backup_ckp_epoch_freq",
        default=-1,
        type=int,
        help="backup checkpoint every x epochs.",
    )
    parser.add_argument(
        "--saveckp_iter_freq",
        default=500,
        type=int,
        help="Save checkpoint every x iters.",
    )
    parser.add_argument(
        "--saveimg_iter_freq",
        default=1000,
        type=int,
        help="Save images every x iterations.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="restart from checkpoints",
    )
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        help="only load weights",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="fp16",
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="save mesh",
    )
    parser.add_argument(
        "--freeze_transformer",
        action="store_true",
        help="whether to freeze transformer",
    )
    parser.add_argument(
        "--buffering",
        default=1024,
        type=int,
        help="logging buffer size",
    )
    parser.add_argument(
        "--centralized_cropping",
        action="store_true",
        help="crop the center of the image.",
    )
    parser.add_argument(
        "--perturb_color",
        action="store_true",
        help="whether to perturb the image color.",
    )
    parser.add_argument(
        "--loss_weights_file",
        default=None,
        type=str,
        help="the weights for losses.",
    )
    parser.add_argument(
        "--use_weight_norm",
        action="store_true",
        help="whether to use weight normalization or not",
    )
    parser.add_argument(
        "--attn_type",
        choices=["default"],
        default="default",
        help="whether to use rope encoding",
    )
    parser.add_argument(
        "--use_double_triplane",
        action="store_true",
        help="whether to use double triplane",
    )
    parser.add_argument(
        "--use_memory_efficient_rendering",
        action="store_true",
        help="whether to use complicated deffered rendering",
    )
    return parser


def train_lrm(args):
    init_distributed_mode(args)
    args = create_output_dir(args)
    fix_random_seeds(args.seed)

    logger.info("*" * 80)
    logger.info(
        f"GPU: {args.gpu}, Local rank: {args.global_rank}/{args.world_size} for training"
    )
    logger.info(args)
    logger.info("*" * 80)

    tb_writer = (
        SummaryWriter(log_dir=args.output_dir) if utils.is_main_process() else None
    )

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "args.txt"), "w") as fOut:
            fOut.write(
                "\n".join(
                    "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
                )
            )

    if args.loss_weights_file is None:
        loss_weights_file = os.path.join(
            os.path.dirname(__file__), "config", f"{args.renderer_type}.yaml"
        )
    else:
        loss_weights_file = os.path.join(
            os.path.dirname(__file__), "config", f"{args.loss_weights_file}.yaml"
        )

    if utils.is_main_process():
        shutil.copy2(
            loss_weights_file,
            os.path.join(args.output_dir, "weights.yaml"),
        )

    with open(loss_weights_file, "r") as fIn:
        loss_weights_dict = yaml.safe_load(fIn)

    normal_weight_max = max(
        loss_weights_dict["mse"]["normal"],
        loss_weights_dict["perceptual"]["normal"],
    )
    numerical_normal_weight_max = max(
        loss_weights_dict["mse"]["numerical_normal"],
        loss_weights_dict["perceptual"]["numerical_normal"],
    )
    depth_weight_max = max(
        loss_weights_dict["mse"]["depth"], loss_weights_dict["perceptual"]["depth"]
    )

    mvencoder = mvencoder_base(
        type=args.mvencoder_type,
        with_bg=(args.encode_background == "augment"),
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
    )

    if args.view_dependent_type == "neural_mip":
        args.view_embed_out_size *= 2
    tridecoder = TriplaneTransformer(
        triplane_size=args.triplane_token_size,
        triplane_out_size=args.triplane_out_token_size,
        embed_dim=args.embed_dim,
        depth=args.transformer_depth,
        output_dim=args.triplane_dim,
        drop_path_rate=args.drop_path_rate,
        cp_freq=args.gradient_checkpointing_freq,
        separate_feature=args.sep_geo_app_fea,
        separate_dims=args.sep_geo_app_dims,
        use_weight_norm=args.use_weight_norm,
        attn_type=args.attn_type,
        double_triplane=args.use_double_triplane,
        use_view_embed=(
            "neural" in args.view_dependent_type and args.prediction_type != "brdf"
        ),
        view_embed_size=args.view_embed_size,
        view_embed_out_size=args.view_embed_out_size,
    )

    if args.renderer_type == "nerf":
        triplane = Triplane(
            dim=args.mlp_dim,
            input_dim=args.triplane_dim * 3,
            rgb_depth=args.mlp_depth,
            density_depth=args.mlp_geo_depth,
            brdf_depth=args.mlp_brdf_depth,
            radius=(args.bbox_radius * 1.05),
            view_dependent_type=args.view_dependent_type,
            prediction_type=args.prediction_type,
            sep_geo_app=args.sep_geo_app_fea,
            sep_dims=args.sep_geo_app_dims,
            double_triplane=args.use_double_triplane,
            compute_normal=(normal_weight_max > 0),
        )
        renderer = NerfRenderer(
            pred_mode=args.prediction_type,
            radius=(args.bbox_radius * 1.05),
            num_samples_per_ray=args.num_samples_per_ray,
            occgrid_res=(256 if args.ocgrid_acc else 1),
            auto_cast_dtype=(torch.float16 if args.fp16 else torch.bfloat16),
        )
    elif args.renderer_type == "sdf":
        triplane = TriplaneSdf(
            dim=args.mlp_dim,
            input_dim=args.triplane_dim * 3,
            rgb_depth=args.mlp_depth,
            brdf_depth=args.mlp_brdf_depth,
            sdf_depth=args.mlp_geo_depth,
            radius=(args.bbox_radius * 1.05),
            view_dependent_type=args.view_dependent_type,
            prediction_type=args.prediction_type,
            sep_geo_app=args.sep_geo_app_fea,
            sep_dims=args.sep_geo_app_dims,
            double_triplane=args.use_double_triplane,
            compute_normal=(normal_weight_max > 0),
        )
        renderer = SdfRenderer(
            pred_mode=args.prediction_type,
            radius=(args.bbox_radius * 1.05),
            num_samples_per_ray=args.num_samples_per_ray,
            occgrid_res=(256 if args.ocgrid_acc else 1),
            auto_cast_dtype=(torch.float16 if args.fp16 else torch.bfloat16),
        )

    if args.use_memory_efficient_rendering:
        loss = triplane_loss_lowmem(
            h_chunk=4,
            filter_normal=args.filter_normal,
            filter_normal_threshold=args.bbox_radius / 20.0,
        )
    else:
        loss = triplane_loss_acc(
            filter_normal=args.filter_normal,
            filter_normal_threshold=args.bbox_radius / 20.0,
        )

    mse_loss_func = F.mse_loss
    perceptual_loss_func = PerceptualLoss()
    for param in perceptual_loss_func.parameters():
        param.requires_grad = False
    loss_func_dict = {}
    loss_func_dict["mse"] = mse_loss_func
    loss_func_dict["perceptual"] = perceptual_loss_func

    mvencoder = mvencoder.cuda()
    tridecoder = tridecoder.cuda()
    triplane = triplane.cuda()
    perceptual_loss_func = perceptual_loss_func.cuda()
    renderer = renderer.cuda()

    if args.dataset_type == "dtc_dataset":
        dataset = DtcDataset(
            root_dir=args.data_path,
            sep_gt_dir=args.sep_gt_dir,
            input_image_num=args.image_num_per_batch_range,
            input_image_res=args.input_image_res,
            output_image_num=args.output_image_num,
            output_image_res=args.output_image_res,
            output_res_range=args.output_res_range,
            fov=args.fov,
            radius=args.bbox_radius,
            load_normal=(max(normal_weight_max, numerical_normal_weight_max) > 0),
            load_depth=(depth_weight_max > 0),
            load_brdf=(
                args.prediction_type == "brdf" or args.prediction_type == "both"
            ),
            load_bg=(args.encode_background != "none"),
            relative_cam_pose=args.relative_cam_pose,
            centralized_cropping=args.centralized_cropping,
            use_adobe_view_selection=args.use_adobe_view_selection,
            perturb_color=args.perturb_color,
            white_env=args.dtc_white_envmap,
        )
    else:
        raise NotImplementedError(f"Unknown dataset type {args.dataset_type}")

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = utils.RepeatedDataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    logger.info(f"Data loaded: there are {len(dataset)} 3D models.")

    mvencoder = dist_helper.get_parallel_model(mvencoder, args.gpu)
    tridecoder = dist_helper.get_parallel_model(tridecoder, args.gpu)
    triplane = dist_helper.get_parallel_model(triplane, args.gpu)
    params_groups = utils.get_params_groups(
        args,
        mvencoder=mvencoder,
        tridecoder=tridecoder,
        triplane=triplane,
    )
    optimizer = torch.optim.AdamW(params_groups, betas=(0.9, 0.95), foreach=True)

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_iters=args.warmup_iters,
    )
    if args.renderer_type == "sdf":
        std_scheduler = utils.linear_scheduler(
            args.warmup_sdf_inv_std,
            args.end_sdf_inv_std,
            args.epochs,
            len(data_loader),
            warmup_iters=args.warmup_iters,
            start_warmup_value=args.start_sdf_inv_std,
        )
    else:
        std_scheduler = None

    logger.info("Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "start_it": -1}
    last_ckpt = os.path.join(args.checkpoint_dir, "last.pth")
    if args.auto_resume and os.path.isfile(last_ckpt):
        args.checkpoint = last_ckpt
        args.load_weights_only = False  # resume training
        utils.restart_from_checkpoint(
            args.checkpoint,
            run_variables=to_restore,
            mvencoder=mvencoder,
            tridecoder=tridecoder,
            triplane=triplane,
            optimizer=optimizer,
            load_weights_only=args.load_weights_only,
        )
    elif args.restart_from_checkpoint and args.checkpoint is not None:
        utils.restart_from_checkpoint(
            args.checkpoint,
            run_variables=to_restore,
            mvencoder=mvencoder,
            tridecoder=tridecoder,
            triplane=triplane,
            load_weights_only=args.load_weights_only,
        )
    start_epoch = to_restore["epoch"]
    start_it = to_restore["start_it"]

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler(growth_interval=200)
    else:
        scaler = None

    start_time = time.time()
    logger.info("Starting LRM training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            mvencoder,
            tridecoder,
            triplane,
            renderer,
            loss,
            loss_func_dict,
            loss_weights_dict,
            data_loader,
            optimizer,
            lr_schedule,
            std_scheduler,
            epoch,
            args,
            start_iter=start_it,
            scaler=scaler,
            tb_writer=tb_writer,
        )
        start_it = -1

        # ============ writing logs ... ============
        save_dict = {
            "mvencoder": mvencoder.state_dict(),
            "tridecoder": tridecoder.state_dict(),
            "triplane": triplane.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }

        if (epoch == args.epochs - 1) or (epoch + 1) % args.saveckp_epoch_freq == 0:
            backup_ckp_epoch = (
                epoch
                if args.backup_ckp_epoch_freq > 0
                and (epoch + 1) % args.backup_ckp_epoch_freq == 0
                else -1
            )
            utils.save_on_master(
                save_dict,
                os.path.join(args.checkpoint_dir, "last.pth"),
                backup_ckp_epoch=backup_ckp_epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))

    return


def train_one_epoch(
    mvencoder,
    tridecoder,
    triplane,
    renderer,
    loss,
    loss_func_dict,
    loss_weights_dict,
    data_loader,
    optimizer,
    lr_schedule,
    std_scheduler,
    epoch,
    args,
    start_iter=None,
    scaler=None,
    tb_writer=None,
):
    metric_logger = utils.MetricLogger(
        delimiter="  ",
        logger=logger,
        tb_writer=tb_writer,
        epoch=epoch,
    )
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 5, header)):
        # update weight decay and learning rate according to their schedule
        abs_it = it
        if start_iter is not None and abs_it < start_iter:
            continue
        it = len(data_loader) * epoch + it  # global training iteration
        for _, param_group in enumerate(optimizer.param_groups):
            curr_lr = (
                0
                if args.freeze_transformer and param_group["lr"] == 0
                else lr_schedule[it]
            )
            param_group["lr"] = curr_lr
        if args.renderer_type == "sdf":
            inv_std = std_scheduler[it]
            renderer.inv_std = inv_std

        auto_cast_dtype = torch.float16 if args.fp16 else torch.bfloat16
        preds, gts, planes, losses = train_one_iteration(
            batch,
            mvencoder,
            tridecoder,
            triplane,
            renderer,
            loss,
            loss_func_dict,
            loss_weights_dict,
            args,
            scaler=scaler,
            auto_cast_dtype=auto_cast_dtype,
            it=it,
        )
        if scaler is not None:
            scaler.unscale_(optimizer)

        if args.clip_grad:
            utils.clip_gradients(mvencoder, args.clip_grad)
            utils.clip_gradients(tridecoder, args.clip_grad)
            utils.clip_gradients(triplane, args.clip_grad)

        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        optimizer.zero_grad(set_to_none=True)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(**losses)
        metric_logger.update(lr=lr_schedule[it])
        if args.renderer_type == "sdf":
            metric_logger.update(inv_std=renderer.inv_std)

        if it % args.saveimg_iter_freq == 0 or it == 0 or abs_it == 0:
            if utils.is_main_process():
                input_image_list = os.path.join(args.image_dir, f"{it:06d}_inputs.txt")
                utils.save_image_list(batch["rgb_names_input"], input_image_list)

                input_image_out = os.path.join(args.image_dir, f"{it:06d}_inputs.png")
                inputs = batch["rgb_input"].detach().cpu()
                utils.save_image(inputs, input_image_out)
                if args.encode_background != "none":
                    bgs_image_out = os.path.join(args.image_dir, f"{it:06d}_bgs.png")
                    bgs = batch["bgs_input"].detach().cpu()
                    utils.save_image(bgs, bgs_image_out)

                if "depth" in gts:
                    depth_mask = batch["depth_masks_output"].detach().cpu()
                    depth_gt = gts["depth"]
                    depth_gt = depth_gt.reshape(-1)[depth_mask.reshape(-1) > 0]
                    if depth_gt.numel() != 0:
                        depth_min = depth_gt.min().item()
                        depth_max = depth_gt.max().item()
                    else:
                        depth_min = 0
                        depth_max = 0
                else:
                    depth_mask = None
                    depth_min = None
                    depth_max = None

                if "depth_input" in batch:
                    utils.save_depth(
                        batch["depth_input"].detach().cpu(),
                        batch["depth_masks_input"].detach().cpu(),
                        os.path.join(args.image_dir, f"{it:06d}_depth_inputs.png"),
                        depth_min,
                        depth_max,
                    )

                for key in gts.keys():
                    gt = gts[key].detach().cpu()
                    gts_image_out = os.path.join(
                        args.image_dir, f"{it:06d}_gts_{key}.png"
                    )
                    if key == "depth":
                        utils.save_depth(
                            gt, depth_mask, gts_image_out, depth_min, depth_max
                        )
                    else:
                        utils.save_image(gt, gts_image_out, is_gamma=False)

                for key in preds.keys():
                    if key == "gradient" or key == "points":
                        continue  # to be finished.
                    pred = preds[key]
                    if len(pred.shape) == 5:
                        pred = pred.permute(0, 1, 4, 2, 3).detach().cpu()
                    preds_image_out = os.path.join(
                        args.image_dir, f"{it:06d}_preds_{key}.png"
                    )
                    if key == "depth" and depth_mask is not None:
                        utils.save_depth(
                            pred, depth_mask, preds_image_out, depth_min, depth_max
                        )
                    elif key == "surface_depth":
                        utils.save_depth(
                            pred,
                            batch["depth_masks_input"].detach().cpu(),
                            preds_image_out,
                            depth_min,
                            depth_max,
                        )
                    else:
                        utils.save_image(pred, preds_image_out, is_gamma=False)

                if args.save_mesh:
                    batch_size = planes["plane_xy"].shape[0]
                    mesh_dir = os.path.join(args.image_dir, f"{it:06d}_mesh")
                    mkdirs(mesh_dir)
                    for n in range(0, batch_size):
                        with torch.no_grad():
                            mesh_out = renderer.sample_point(
                                planes["plane_xy"][n : n + 1, :],
                                planes["plane_xz"][n : n + 1, :],
                                planes["plane_yz"][n : n + 1, :],
                                triplane,
                                mode="grid",
                                N=args.mesh_resolution,
                            )
                        if args.renderer_type == "sdf":
                            utils.save_mesh(
                                path=os.path.join(mesh_dir, f"{n:03d}_mesh.ply"),
                                values=mesh_out["sdf"][0, :].float(),
                                N=args.mesh_resolution,
                                threshold=0,
                                radius=args.bbox_radius,
                            )
                        else:
                            utils.save_mesh(
                                path=os.path.join(mesh_dir, f"{n:03d}_mesh.ply"),
                                values=mesh_out["density"][0, :].float(),
                                N=args.mesh_resolution,
                                threshold=0.5,
                                radius=args.bbox_radius,
                            )

                cams = batch["cameras_output"].detach().cpu().numpy()
                with open(
                    os.path.join(args.image_dir, f"{it:06d}_cam.npy"), "wb"
                ) as fp:
                    np.save(fp, cams)

        if it % args.saveckp_iter_freq == 0 and it != 0 and abs_it != 0:
            save_dict = {
                "mvencoder": mvencoder.state_dict(),
                "tridecoder": tridecoder.state_dict(),
                "triplane": triplane.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "start_it": abs_it + 1,
                "args": args,
            }
            utils.save_on_master(
                save_dict,
                os.path.join(args.checkpoint_dir, "last.pth"),
                backup_ckp_epoch=-1,
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_iteration(
    batch,
    mvencoder,
    tridecoder,
    triplane,
    renderer,
    loss,
    loss_func_dict,
    loss_weights_dict,
    args,
    scaler=None,
    auto_cast_dtype=torch.bfloat16,
    it=0,
):
    if args.image_num_per_batch_range[0] < args.image_num_per_batch_range[1]:
        image_num = np.random.randint(
            args.image_num_per_batch_range[0],
            args.image_num_per_batch_range[1] + 1,
        )
        batch["rgb_input"] = batch["rgb_input"][:, :image_num, :]
        batch["mask_input"] = batch["mask_input"][:, :image_num, :]
        batch["rays_o_input"] = batch["rays_o_input"][:, :image_num, :]
        batch["rays_d_input"] = batch["rays_d_input"][:, :image_num, :]
        batch["rays_d_un_input"] = batch["rays_d_un_input"][:, :image_num, :]
        batch["uv_input"] = batch["uv_input"][:, :image_num, :]
        batch["cameras_input"] = batch["cameras_input"][:, :image_num, :]
        if "depth_input" in batch:
            batch["depth_input"] = batch["depth_input"][:, :image_num, :, :, :]
        if "normal_input" in batch:
            batch["normal_input"] = batch["normal_input"][:, :image_num, :, :, :]
        if "depth_masks_input" in batch:
            batch["depth_masks_input"] = batch["depth_masks_input"][
                :, :image_num, :, :, :
            ]
        if "bgs_input" in batch:
            batch["bgs_input"] = batch["bgs_input"][:, :image_num, :, :, :]

    rays_o = batch["rays_o_output"]
    rays_d = batch["rays_d_output"]
    cams_output = batch["cameras_output"]

    images = batch["rgb_input"]
    batch_size, image_num, _, height, width = images.shape
    images = images.reshape(batch_size, image_num, -1, height, width)
    images = images.to(device=args.gpu, dtype=auto_cast_dtype)

    if args.encode_background != "none":
        bg = batch["bgs_input"].reshape(batch_size, image_num, 3, height, width)
        bg = bg.to(device=args.gpu, dtype=auto_cast_dtype)
    else:
        bg = None

    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
        if "plucker" in args.mvencoder_type:
            rays_o_input = batch["rays_o_input"]
            rays_d_input = batch["rays_d_input"]

            rays_o_input = rays_o_input.permute(0, 1, 4, 2, 3)
            rays_d_input = rays_d_input.permute(0, 1, 4, 2, 3)

            plucker_rays = torch.cat(
                [rays_d_input, torch.cross(rays_o_input, rays_d_input, dim=2)], dim=2
            )
            plucker_rays = plucker_rays.to(device=args.gpu, dtype=auto_cast_dtype)

        if args.mvencoder_type == "plucker":
            if args.encode_background != "only":
                tokens = mvencoder(
                    images,
                    plucker_rays,
                    bg,
                )
            else:
                tokens = mvencoder(
                    bg,
                    plucker_rays,
                    None,
                )
        plane_xy, plane_xz, plane_yz, plane_view, y = tridecoder(tokens)

    rays_o = rays_o.to(device=args.gpu)
    rays_d = rays_d.to(device=args.gpu)
    cams_output = cams_output.to(device=args.gpu)

    gts = {}
    if args.prediction_type == "rgb":
        keys = ["rgb", "normal", "numerical_normal", "mask", "depth"]
    elif args.prediction_type == "brdf":
        keys = [
            "albedo",
            "roughness",
            "metallic",
            "normal",
            "numerical_normal",
            "mask",
            "depth",
        ]
    elif args.prediction_type == "both":
        keys = [
            "rgb",
            "albedo",
            "roughness",
            "metallic",
            "normal",
            "numerical_normal",
            "mask",
            "depth",
        ]

    for _, key in enumerate(keys):
        if key == "numerical_normal":
            if "normal_output" not in batch:
                continue
            gts[key] = batch["normal_output"].to(device=args.gpu)
        else:
            if f"{key}_output" not in batch:
                continue
            gts[key] = batch[f"{key}_output"].to(device=args.gpu)

    if "depth_masks_output" in batch:
        depth_mask = batch["depth_masks_output"].to(device=args.gpu)
    else:
        depth_mask = None

    if "normal_masks_output" in batch:
        normal_mask = batch["normal_masks_output"].to(device=args.gpu)
    else:
        normal_mask = None

    normal_weight_max = max(
        loss_weights_dict["mse"]["normal"], loss_weights_dict["perceptual"]["normal"]
    )
    numerical_normal_weight_max = max(
        loss_weights_dict["mse"]["numerical_normal"],
        loss_weights_dict["perceptual"]["numerical_normal"],
    )

    preds, losses = loss.forward_and_backward(
        triplane,
        renderer,
        plane_xy,
        plane_xz,
        plane_yz,
        plane_view,
        rays_o,
        rays_d,
        cams_output,
        gts,
        depth_mask,
        normal_mask,
        loss_weights_dict,
        loss_func_dict,
        args,
        use_occ_grid=args.ocgrid_acc,
        compute_normal=(normal_weight_max > 0),
        compute_numerical_normal=(numerical_normal_weight_max > 0),
        scaler=(scaler, None),
    )
    planes = {
        "plane_xy": plane_xy,
        "plane_xz": plane_xz,
        "plane_yz": plane_yz,
        "plane_view": plane_view,
    }

    return preds, gts, planes, losses


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    train_lrm(args)
