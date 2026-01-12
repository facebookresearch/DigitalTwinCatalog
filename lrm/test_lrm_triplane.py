# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import os
import shutil
import tempfile
import time

import cv2
import ffmpeg
import misc.utils as utils
import numpy as np
import torch
import yaml
from data_loader.dtc_dataset import DtcDataset
from geometry.triplane import Triplane
from geometry.triplane_sdf import TriplaneSdf
from loss.triplane_loss_vanilla import triplane_loss_acc
from misc.env_utils import fix_random_seeds, init_distributed_mode
from misc.io_helper import mkdirs
from models.multiview_encoder import mvencoder_base
from models.triplane_decoder import TriplaneTransformer
from renderer.nerf_renderer import NerfRenderer
from renderer.sdf_renderer import SdfRenderer


def create_output_dir(args):
    args.user = os.environ["USER"] or "default"
    args.output_dir = os.path.join(args.exp_root, args.exp_name)
    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    args.image_dir = os.path.join(args.output_dir, "images")

    mkdirs(args.output_dir)
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
        default=32,
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
        choices=[
            "plucker",
            "plucker_geometry",
            "nocam_noencoding",
            "nocam_svencoding",
            "nocam_uniencoding",
        ],
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
        "--output_image_res", default=512, type=int, help="output image resolution"
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
    parser.add_argument(
        "--relative_cam_pose",
        action="store_true",
        help="whether to use relative camera poses.",
    )

    # Dataset parameters
    parser.add_argument(
        "--batch_size_per_gpu",
        default=8,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument(
        "--image_num_per_batch",
        default=4,
        type=int,
        help="Total number of images loaded per batch.",
    )
    parser.add_argument(
        "--output_image_num", default=10, type=int, help="output image num"
    )
    parser.add_argument(
        "--sdf_inv_std",
        type=float,
        default=200,
        help="the sdf std after training ends",
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="path to the checkpoint"
    )
    parser.add_argument(
        "--weights_dir",
        default="config",
        help="the folder that contains yaml files for weights",
    )

    # Evaluate
    parser.add_argument(
        "--eva_input_views",
        nargs="*",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7],
        help="Input view ids",
    )
    parser.add_argument(
        "--eva_output_views",
        nargs="*",
        type=int,
        default=[8, 9, 10, 11, 12],
        help="Output view ids",
    )

    # Misc
    parser.add_argument(
        "--data_path",
        type=str,
        help="Please specify path to the training data.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
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
        "--save_video",
        action="store_true",
        help="save video",
    )
    parser.add_argument(
        "--save_interm_volume",
        action="store_true",
        help="save volume for marching cube.",
    )
    parser.add_argument(
        "--mesh_resolution",
        type=int,
        default=256,
        help="the density/sdf grid resolution when outputting mesh",
    )
    parser.add_argument(
        "--output_eval_json",
        action="store_true",
        help="whether to save the evaluation json file.",
    )
    parser.add_argument(
        "--centralized_cropping",
        action="store_true",
        help="crop the center of the image.",
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
        "--start_id",
        type=int,
        default=0,
        help="the starting model id of the data list.",
    )
    parser.add_argument(
        "--end_id", type=int, default=-1, help="the starting model id of the data list."
    )
    parser.add_argument(
        "--resave_gt_images",
        action="store_true",
        help="whether to resave gt images or not",
    )
    return parser


def test_lrm(args):
    init_distributed_mode(args)
    args = create_output_dir(args)
    fix_random_seeds(args.seed)

    print("*" * 80)
    print(
        f"GPU: {args.gpu}, Local rank: {args.global_rank}/{args.world_size} for training"
    )
    print(args)
    print("*" * 80)

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
        cp_freq=0,
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
            occgrid_res=1,
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
            occgrid_res=1,
            auto_cast_dtype=(torch.float16 if args.fp16 else torch.bfloat16),
        )
        renderer.inv_std = args.sdf_inv_std

    loss = triplane_loss_acc()

    args.output_image_num = min(len(args.eva_output_views), args.output_image_num)
    args.eva_output_views = args.eva_output_views[: args.output_image_num]
    if args.dataset_type == "dtc_dataset":
        dataset = DtcDataset(
            mode="TEST",
            root_dir=args.data_path,
            input_image_num=args.image_num_per_batch,
            input_image_res=args.input_image_res,
            output_image_num=args.output_image_num,
            output_image_res=args.output_image_res,
            eva_input_views=args.eva_input_views,
            eva_output_views=args.eva_output_views,
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
            start_id=args.start_id,
            end_id=args.end_id,
            white_env=args.dtc_white_envmap,
        )
    else:
        raise NotImplementedError(f"Unknown dataset type {args.dataset_type}")

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    print(f"Data loaded: there are {len(dataset)} 3D models.")

    mvencoder = mvencoder.cuda().eval()
    for para in mvencoder.parameters():
        para.requires_grad = False
    tridecoder = tridecoder.cuda().eval()
    for para in tridecoder.parameters():
        para.requires_grad = False
    triplane = triplane.cuda().eval()
    for para in triplane.parameters():
        para.requires_grad = False
    renderer = renderer.cuda()

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        args.checkpoint,
        run_variables=to_restore,
        mvencoder=mvencoder,
        tridecoder=tridecoder,
        triplane=triplane,
        load_weights_only=False,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting LRM inference !")
    data_loader.sampler.set_epoch(start_epoch)
    test_one_epoch(
        mvencoder,
        tridecoder,
        triplane,
        renderer,
        loss,
        loss_weights_dict,
        data_loader,
        args.eva_output_views,
        args,
    )
    log_stats = {"epoch": start_epoch}
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))

    return


def save_imgs_one_sample(preds, batch_id, im_ids, folder):
    eval_dict = [{"frame": n} for n in range(0, len(im_ids))]
    pred_keys = []
    for key in preds.keys():
        if key == "gradient":
            continue  # to be finished.
        pred = preds[key]
        if key == "rgb":
            name = os.path.join(folder, "%03d_rgb.png")
            utils.save_single_png(pred[batch_id, :].permute(0, 3, 1, 2), name, im_ids)

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_image_path"] = os.path.join(
                    folder, "%03d_rgb.png" % im_ids[n]
                )
            pred_keys.append("rgb")

        elif key == "albedo":
            name = os.path.join(folder, "%03d_albedo.png")
            utils.save_single_png(
                pred[batch_id, :].permute(0, 3, 1, 2),
                name,
                im_ids,
                is_gamma=False,
            )

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_albedo_path"] = os.path.join(
                    folder, "%03d_albedo.png" % im_ids[n]
                )
            pred_keys.append("albedo")

        elif key == "roughness":
            name = os.path.join(folder, "%03d_roughness.png")
            utils.save_single_png(
                pred[batch_id, :].permute(0, 3, 1, 2),
                name,
                im_ids,
                is_gamma=False,
            )

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_roughness_path"] = os.path.join(
                    folder, "%03d_roughness.png" % im_ids[n]
                )
            pred_keys.append("roughness")

        elif key == "normal":
            name = os.path.join(folder, "%03d_normal.exr")
            utils.save_single_exr(pred[batch_id, :].permute(0, 3, 1, 2), name, im_ids)

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_normal_path"] = os.path.join(
                    folder, "%03d_normal.exr" % im_ids[n]
                )
            pred_keys.append("normal")

        elif key == "metallic":
            name = os.path.join(folder, "%03d_metallic.png")
            utils.save_single_png(
                pred[batch_id, :].permute(0, 3, 1, 2),
                name,
                im_ids,
                is_gamma=False,
            )

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_metallic_path"] = os.path.join(
                    folder, "%03d_metallic.png" % im_ids[n]
                )
            pred_keys.append("metallic")

    return eval_dict, pred_keys


def test_one_epoch(
    mvencoder,
    tridecoder,
    triplane,
    renderer,
    loss,
    loss_weights_dict,
    data_loader,
    eva_output_views,
    args,
):
    triplane_path = os.path.join(args.output_dir, "triplane.pth")
    with open(triplane_path, "wb") as fp:
        torch.save(triplane.state_dict(), fp)

    if args.output_eval_json:
        eval_dict_array = []

    for _, batch in enumerate(data_loader):
        auto_cast_dtype = torch.float16 if args.fp16 else torch.bfloat16
        preds, planes = test_one_iteration(
            batch,
            mvencoder,
            tridecoder,
            triplane,
            renderer,
            loss,
            loss_weights_dict,
            args,
            auto_cast_dtype=auto_cast_dtype,
        )
        # logging
        torch.cuda.synchronize()

        model_ids = batch["name"]
        if args.save_video:
            if args.relative_cam_pose:
                init_cam_arr = batch["init_cam_rot"]
            else:
                init_cam_arr = None
            save_video(
                planes,
                triplane,
                renderer,
                loss,
                model_ids,
                args,
                fov=60,
                frame=120,
                init_cam_arr=init_cam_arr,
            )

        batch_size = len(model_ids)
        for b in range(0, batch_size):
            print("Model Id: %s" % model_ids[b])
            if "fov" in batch:
                args.fov = batch["fov"][b].item()
            if "eva_output_views" in batch:
                eva_output_views = batch["eva_output_views"][b, :]
                eva_output_views = eva_output_views.numpy().tolist()

            model_id = model_ids[b]
            model_dir = os.path.join(args.output_dir, model_id)
            mkdirs(model_dir)

            plane_path = os.path.join(model_dir, "plane.pth")
            plane_one_sample = {}
            plane_one_sample["plane_xy"] = (
                planes["plane_xy"][b : b + 1, :].detach().cpu()
            )
            plane_one_sample["plane_xz"] = (
                planes["plane_xz"][b : b + 1, :].detach().cpu()
            )
            plane_one_sample["plane_yz"] = (
                planes["plane_yz"][b : b + 1, :].detach().cpu()
            )
            if planes["plane_view"] is not None:
                plane_one_sample["plane_view"] = (
                    planes["plane_view"][b : b + 1, :].detach().cpu()
                )
            else:
                plane_one_sample["plane_view"] = None

            with open(plane_path, "wb") as fp:
                torch.save(plane_one_sample, fp)

            eval_dict, pred_keys = save_imgs_one_sample(
                preds, b, eva_output_views, model_dir
            )
            if args.output_eval_json:
                for key in pred_keys:
                    if key == "rgb":
                        if args.resave_gt_images and "rgb_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_rgb.png"
                            )
                            gt_rgbs = batch["rgb_output"][b, :]
                            utils.save_single_png(
                                gt_rgbs, name_template, eva_output_views
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_image_path"] = names[n]
                        elif "rgb_names_output" in batch:
                            names = batch["rgb_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_image_path"] = names[n][b]
                    elif key == "albedo":
                        if args.resave_gt_images and "albedo_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_albedo.png"
                            )
                            gt_albedos = batch["albedo_output"][b, :]
                            utils.save_single_png(
                                gt_albedos,
                                name_template,
                                eva_output_views,
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_albedo_path"] = names[n]
                        elif "albedo_names_output" in batch:
                            names = batch["albedo_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_albedo_path"] = names[n][b]
                    elif key == "roughness":
                        if args.resave_gt_images and "roughness_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_roughness.png"
                            )
                            gt_roughnesss = batch["roughness_output"][b, :]
                            utils.save_single_png(
                                gt_roughnesss,
                                name_template,
                                eva_output_views,
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_roughness_path"] = names[n]
                        elif "roughness_names_output" in batch:
                            names = batch["roughness_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_roughness_path"] = names[n][b]
                    elif key == "normal":
                        if args.resave_gt_images and "normal_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_normal.png"
                            )
                            gt_normals = batch["normal_output"][b, :]
                            utils.save_single_png(
                                gt_normals,
                                name_template,
                                eva_output_views,
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_normal_path"] = names[n]
                        elif "normal_names_output" in batch:
                            names = batch["normal_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_normal_path"] = names[n][b]
                    elif key == "metallic":
                        if args.resave_gt_images and "metallic_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_metallic.png"
                            )
                            gt_metallics = batch["metallic_output"][b, :]
                            utils.save_single_png(
                                gt_metallics,
                                name_template,
                                eva_output_views,
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_metallic_path"] = names[n]
                        if "metallic_names_output" in batch:
                            names = batch["metallic_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_metallic_path"] = names[n][b]
                eval_dict_array.append(eval_dict)

            if args.save_mesh:
                with torch.no_grad():
                    if args.renderer_type == "sdf":
                        with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
                            mesh_out = renderer.sample_point(
                                planes["plane_xy"][b : b + 1, :],
                                planes["plane_xz"][b : b + 1, :],
                                planes["plane_yz"][b : b + 1, :],
                                triplane,
                                mode="grid",
                                N=args.mesh_resolution,
                            )
                        utils.save_mesh(
                            path=os.path.join(model_dir, "mesh.obj"),
                            values=mesh_out["sdf"][0, :].float(),
                            N=args.mesh_resolution,
                            threshold=0,
                            radius=args.bbox_radius,
                            save_volume=args.save_interm_volume,
                        )
                    elif args.renderer_type == "nerf":
                        with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
                            mesh_out = renderer.sample_point(
                                planes["plane_xy"][b : b + 1, :],
                                planes["plane_xz"][b : b + 1, :],
                                planes["plane_yz"][b : b + 1, :],
                                triplane,
                                mode="grid",
                                N=args.mesh_resolution,
                            )

                        utils.save_mesh(
                            path=os.path.join(model_dir, "mesh.obj"),
                            values=mesh_out["density"][0, :].float(),
                            N=args.mesh_resolution,
                            threshold=10,
                            radius=args.bbox_radius,
                            save_volume=args.save_interm_volume,
                        )

        if args.output_eval_json:
            with open(os.path.join(args.output_dir, "eval.json"), "w") as fp:
                json.dump(eval_dict_array, fp, indent=4)

    # gather the stats from all processes
    return


def save_video(
    plane,
    triplane,
    renderer,
    loss,
    model_ids,
    args,
    fov=60,
    frame=120,
    init_cam_arr=None,
):
    batch_size = len(model_ids)
    for b in range(0, batch_size):
        model_id = model_ids[b]
        video_dir = os.path.join(args.output_dir, model_id, "video")
        mkdirs(video_dir)

        plane_xy = plane["plane_xy"][b : b + 1, :]
        plane_xz = plane["plane_xz"][b : b + 1, :]
        plane_yz = plane["plane_yz"][b : b + 1, :]
        if plane["plane_view"] is not None:
            plane_view = plane["plane_view"][b : b + 1, :]
        else:
            plane_view = None

        if init_cam_arr is not None:
            init_cam = init_cam_arr[b, :]
        else:
            init_cam = None
        cams_output, rays_o, rays_d, _ = utils.create_video_cameras(
            args.bbox_radius, frame, args.output_image_res, init_cam=init_cam
        )
        with open(os.path.join(video_dir, "extrinsic.npy"), "wb") as fOut:
            extr = cams_output[:, :, 0:16].reshape(-1, 4, 4)
            np.save(fOut, extr)

        with open(os.path.join(video_dir, "fov.txt"), "w") as fOut:
            fOut.write("%.3f\n" % fov)

        cams_output = torch.from_numpy(cams_output).to(device=args.gpu)
        rays_o = torch.from_numpy(rays_o).to(device=args.gpu)
        rays_d = torch.from_numpy(rays_d).to(device=args.gpu)

        preds = loss.forward(
            triplane,
            renderer,
            plane_xy,
            plane_xz,
            plane_yz,
            plane_view,
            rays_o,
            rays_d,
            cams_output,
            args,
            compute_normal=triplane.compute_normal,
        )
        im_ids = [n for n in range(0, frame)]
        name = os.path.join(video_dir, "%03d_rgb.png")
        utils.save_single_png(preds["rgb"][0, :].permute(0, 3, 1, 2), name, im_ids)

        with tempfile.TemporaryDirectory() as temp_dir:
            for n in range(0, frame):
                image = preds["rgb"][0, n, :, :, :]
                image = image.detach().to(torch.float32).cpu().numpy()
                image = np.clip(0.5 * (image + 1), 0, 1)
                image = (255 * image).astype(np.uint8)
                cv2.imwrite(os.path.join(temp_dir, "%03d.png" % n), image[:, :, ::-1])
            local_video_path = os.path.join(temp_dir, "video.mp4")
            video_path = os.path.join(video_dir, "video.mp4")
            ffmpeg.input(os.path.join(temp_dir, "%03d.png"), framerate=24).output(
                local_video_path, pix_fmt="yuv420p"
            ).run()
            shutil.copy2(local_video_path, video_path)

        return


def test_one_iteration(
    batch,
    mvencoder,
    tridecoder,
    triplane,
    renderer,
    loss,
    loss_weights_dict,
    args,
    auto_cast_dtype=torch.bfloat16,
):
    images = batch["rgb_input"]
    rays_o = batch["rays_o_output"]
    rays_d = batch["rays_d_output"]
    cams_output = batch["cameras_output"]

    batch_size, image_num, _, height, width = images.shape
    images = images.reshape(batch_size, image_num, -1, height, width)
    images = images.to(device=args.gpu, dtype=auto_cast_dtype)
    if args.encode_background != "none":
        bg = batch["bgs_input"].reshape(batch_size, image_num, -1, height, width)
        bg = bg.to(device=args.gpu, dtype=auto_cast_dtype)
    else:
        bg = None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
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

    end.record()
    torch.cuda.synchronize()
    print("Inference time: %.3fs" % (float(start.elapsed_time(end)) / 1000.0))

    device = plane_xy.device
    rays_o = rays_o.to(device=device)
    rays_d = rays_d.to(device=device)
    cams_output = cams_output.to(device=device)

    normal_weight_max = max(
        loss_weights_dict["mse"]["normal"], loss_weights_dict["perceptual"]["normal"]
    )
    numerical_normal_weight_max = max(
        loss_weights_dict["mse"]["numerical_normal"],
        loss_weights_dict["perceptual"]["numerical_normal"],
    )

    preds = loss.forward(
        triplane,
        renderer,
        plane_xy,
        plane_xz,
        plane_yz,
        plane_view,
        rays_o,
        rays_d,
        cams_output,
        args,
        compute_normal=(normal_weight_max > 0),
        compute_numerical_normal=(numerical_normal_weight_max > 0),
    )

    planes = {
        "plane_xy": plane_xy,
        "plane_xz": plane_xz,
        "plane_yz": plane_yz,
        "plane_view": plane_view,
    }

    return preds, planes


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    test_lrm(args)
