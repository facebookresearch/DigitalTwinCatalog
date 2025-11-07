#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import imageio

import mmcv
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
import neural_pbir_cuda_utils
from neural_surface_recon.lib import utils, utils_bbox, utils_ray, vol_repr
from neural_surface_recon.lib.camera_refiner import CamRefiner
from neural_surface_recon.lib.dataloader import load_data


def config_parser():
    """Define command line arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, help="config file path")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--reload_optimizer",
        action="store_true",
        help="do not reload optimizer state from saved ckpt",
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default="",
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="do not optimize, reload weights and only run testing mode",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=500,
        help="period of console printout and metric login",
    )
    parser.add_argument(
        "--i_weights",
        type=int,
        default=100000,
        help="period (iterations) of weight ckpt saving",
    )
    parser.add_argument(
        "--save_optimizer", action="store_true", help="save optimizer state as well"
    )
    parser.add_argument(
        "--run_vis_period", type=int, default=0, help="visualize intermediate results"
    )
    parser.add_argument(
        "--run_vis_factor",
        type=int,
        default=2,
        help="downsample factor for the visualization",
    )

    # video results options
    parser.add_argument(
        "--render_train", action="store_true", help="render train split"
    )
    parser.add_argument("--render_test", action="store_true", help="render test split")
    parser.add_argument(
        "--render_video", action="store_true", help="render virtual trajectory"
    )
    parser.add_argument(
        "--render_video_factor",
        type=float,
        default=0,
        help="downsampling factor to speed up rendering, eg, set 8 for fast preview",
    )
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--render_normal", action="store_true")
    parser.add_argument("--render_fg_only", action="store_true")
    parser.add_argument(
        "--dump_images",
        action="store_true",
        help="whether to dump each frames instead of just video",
    )
    parser.add_argument("--eval_ssim", action="store_true")
    parser.add_argument("--eval_lpips_alex", action="store_true")
    parser.add_argument("--eval_lpips_vgg", action="store_true")

    # mesh results options
    parser.add_argument("--export_surface", action="store_true")
    parser.add_argument("--mesh_iso", default=0, type=float)
    parser.add_argument("--mesh_no_cleanup", action="store_true")
    parser.add_argument("--mesh_dtu_clean", action="store_true")

    return parser


@torch.no_grad()
def render_viewpoints(
    model,
    render_poses,
    HW,
    Ks,
    ndc,
    render_kwargs,
    gt_imgs=None,
    savedir=None,
    dump_images=False,
    render_factor=0,
    eval_ssim=False,
    eval_lpips_alex=False,
    eval_lpips_vgg=False,
    silent=False,
):
    """Render images for the given viewpoints; run evaluation if gt given."""
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor != 0 and render_factor != 1:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW / render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    normals = []
    xyzs = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    if silent:
        iterator = enumerate(render_poses)
    else:
        iterator = enumerate(tqdm(render_poses))

    for i, c2w in iterator:
        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = utils_ray.get_rays_of_a_view(H, W, K, c2w, ndc)
        keys = ["rgb_marched", "depth", "alphainv_last", "normal", "xyz"]
        rays_o = rays_o.flatten(0, -2).contiguous()
        rays_d = rays_d.flatten(0, -2).contiguous()
        viewdirs = viewdirs.flatten(0, -2).contiguous()
        render_result = model.superfast_forward(
            rays_o, rays_d, viewdirs, **render_kwargs
        )
        render_result = {k: render_result[k].reshape(H, W, -1) for k in keys}
        rgb = render_result["rgb_marched"].cpu().numpy()
        depth = render_result["depth"].cpu().numpy()
        bgmap = render_result["alphainv_last"].cpu().numpy()
        normal = render_result["normal"].cpu().numpy()
        xyz = render_result["xyz"].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        normals.append(normal)
        xyzs.append(xyz)
        if i == 0 and not silent:
            print("Testing", rgb.shape)

    if gt_imgs is not None and render_factor == 0:
        for i, rgb in enumerate(tqdm(rgbs)):
            p = -10.0 * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(
                    utils.rgb_lpips(rgb, gt_imgs[i], net_name="alex", device=c2w.device)
                )
            if eval_lpips_vgg:
                lpips_vgg.append(
                    utils.rgb_lpips(rgb, gt_imgs[i], net_name="vgg", device=c2w.device)
                )
            if i == 0:
                print(
                    f"Evaluation ssim={eval_ssim} lpips_alex={eval_lpips_alex} lpips_vgg={eval_lpips_vgg}"
                )

    if len(psnrs):
        print("Testing psnr", np.mean(psnrs), "(avg)")
        if eval_ssim:
            print("Testing ssim", np.mean(ssims), "(avg)")
        if eval_lpips_vgg:
            print("Testing lpips (vgg)", np.mean(lpips_vgg), "(avg)")
        if eval_lpips_alex:
            print("Testing lpips (alex)", np.mean(lpips_alex), "(avg)")

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            if gt_imgs is not None and render_factor == 0:
                import cv2

                rgb8 = utils.to8b(np.concatenate([rgbs[i], gt_imgs[i]], 1))
                rgb8 = cv2.resize(
                    rgb8, (rgb8.shape[1] // 2, rgb8.shape[0] // 2), cv2.INTER_LINEAR
                )
            else:
                rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    normals = np.array(normals)
    xyzs = np.array(xyzs)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, normals, xyzs, bgmaps


def save_movie(
    savedir, rgbs=None, depths=None, normals=None, depthcmap="gray", fps=30, quality=8
):
    import matplotlib.pyplot as plt

    if rgbs is not None:
        imageio.mimwrite(
            os.path.join(savedir, "video.rgb.mp4"),
            utils.to8b(rgbs),
            fps=fps,
            quality=quality,
        )
    if depths is not None:
        dmin, dmax = np.percentile(depths, q=[0, 100])
        depth_vis = 1 - np.clip((depths - dmin) / (dmax - dmin), 0, 1)[..., 0]
        depth_vis = plt.get_cmap(depthcmap)(depth_vis)[..., :3]
        imageio.mimwrite(
            os.path.join(savedir, "video.depth.mp4"),
            utils.to8b(depth_vis),
            fps=fps,
            quality=quality,
        )
    if normals is not None:
        normnormals = normals / np.clip(
            np.sqrt(np.sum(normals**2, axis=-1, keepdims=True)), 1e-4, 1
        )
        normals = np.clip(normals * 0.5 + 0.5, 0, 1)
        imageio.mimwrite(
            os.path.join(savedir, "video.normal.mp4"),
            utils.to8b(normals),
            fps=fps,
            quality=quality,
        )


def cyan(s):
    return f"\033[96m{s}\033[0m"


def seed_everything(seed):
    """Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_everything(cfg):
    """Load images / poses / camera settings / data split."""
    data_dict = load_data(cfg.data)

    # construct data tensor
    data_dict["images"] = torch.FloatTensor(data_dict["images"], device="cpu")
    data_dict["poses"] = torch.Tensor(data_dict["poses"])
    if data_dict["depths"] is not None:
        data_dict["depths"] = torch.FloatTensor(data_dict["depths"], device="cpu")
    if data_dict["masks"] is not None:
        data_dict["masks"] = torch.FloatTensor(data_dict["masks"], device="cpu")
    return data_dict


def create_new_model(
    cfg, cfg_model, cfg_train, xyz_min, xyz_max, device, coarse_ckpt_path=None
):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop("num_voxels")
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (cfg_train.pg_scale_s ** len(cfg_train.pg_scale)))

    model = vol_repr.VolRepr(
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        num_voxels=num_voxels,
        mask_cache_path=coarse_ckpt_path,
        **model_kwargs,
    )
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, trainiter=0)
    return model, optimizer


def load_existed_model(args, cfg_train, device, reload_ckpt_path):
    model = utils.load_model(vol_repr.VolRepr, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, trainiter=0)
    model, optimizer, start = utils.load_checkpoint(
        model, optimizer, reload_ckpt_path, args.reload_optimizer
    )
    return model, optimizer, start


def scene_rep_reconstruction(
    args,
    cfg,
    cfg_model,
    cfg_train,
    xyz_min,
    xyz_max,
    data_dict,
    stage,
    cam_refiner=None,
    coarse_ckpt_path=None,
):
    # init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    # find whether there is existing checkpoint path
    last_ckpt_path = cfg.results_dir / f"{stage}_last.pth"
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f"scene_rep_reconstruction ({stage}): train from scratch")
        model, optimizer = create_new_model(
            cfg, cfg_model, cfg_train, xyz_min, xyz_max, device, coarse_ckpt_path
        )
        start = 0
    else:
        print(f"scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}")
        model, optimizer, start = load_existed_model(
            args, cfg_train, device, reload_ckpt_path
        )

    # init rendering setup
    render_kwargs = {
        "near": data_dict["near"],
        "far": data_dict["far"],
        "bg": cfg.data.bkgd,
        "rand_bkgd": cfg.data.rand_bkgd,
        "stepsize": cfg_model.stepsize,
    }

    # init batch rays sampler
    (
        rgb_tr,
        rays_o_tr,
        rays_d_tr,
        viewdirs_tr,
        depths_tr,
        masks_tr,
        image_id_tr,
        batch_index_sampler,
    ) = utils_ray.gather_training_rays(data_dict, cfg, cfg_train, device, model)

    # some more initialization
    huber_loss = utils.HuberLoss(c_min=cfg_train.huber_min_c).to(device)
    psnr_lst = []
    trainiter = -1
    if args.run_vis_period > 0:
        running_render_viewpoints_kwargs = copy.deepcopy(render_kwargs)
        running_render_viewpoints_kwargs["stepsize"] = 0.5
        running_render_viewpoints_kwargs["tick"] = 0
        running_render_viewpoints_kwargs["render_color"] = False
        running_render_viewpoints_kwargs["render_normal"] = True
        running_render_viewpoints_kwargs["render_depth"] = False
        running_render_viewpoints_kwargs["render_fg_only"] = True
        run_vis_dir = cfg.results_dir / "progression_normal"
        os.makedirs(run_vis_dir, exist_ok=True)

    # GOGO
    torch.cuda.empty_cache()
    for k, v in cfg_train.items():
        if k.startswith("weight_"):
            print("loss", k, v)
    time0 = time.time()
    n_rest_scales = len(cfg_train.pg_scale)
    for trainiter in trange(1 + start, 1 + cfg_train.N_iters):
        # update occupancy grid
        if (trainiter + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if trainiter in cfg_train.pg_scale:
            if model.bg_model is not None:
                bg_outview = optimizer.state[model.bg_model.density.grid]["step"] == 0
                model.bg_model.density.grid.data[bg_outview] -= 5

            n_rest_scales -= 1
            assert n_rest_scales >= 0
            cur_voxels = int(
                cfg_model.num_voxels / (cfg_train.pg_scale_s**n_rest_scales)
            )
            model.scale_volume_grid(cur_voxels)
            optimizer = utils.create_optimizer_or_freeze_model(
                model, cfg_train, trainiter=trainiter
            )
            torch.cuda.empty_cache()

        # random sample rays
        sel_i = batch_index_sampler()
        target = rgb_tr[sel_i].to(device)
        rays_o = rays_o_tr[sel_i].to(device)
        rays_d = rays_d_tr[sel_i].to(device)
        viewdirs = viewdirs_tr[sel_i].to(device)
        image_id = image_id_tr[sel_i].to(device)
        if depths_tr is not None:
            d_target = depths_tr[sel_i].to(device)
        if masks_tr is not None:
            target_fg_mask = masks_tr[sel_i].to(device)

        # (optional) get the refined rays
        if cam_refiner is not None:
            rays_o, rays_d, viewdirs = cam_refiner(rays_o, rays_d, viewdirs, image_id)

        # forward volume rendering
        render_result = model(
            rays_o,
            rays_d,
            viewdirs,
            trainiter=trainiter,
            is_train=True,
            **render_kwargs,
        )

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        mse_loss = F.mse_loss(render_result["rgb_marched"], target)
        psnr = utils.mse2psnr(mse_loss.detach())
        if cfg_train.main_loss == "mse":
            main_loss = mse_loss
        elif cfg_train.main_loss == "mae":
            main_loss = F.l1_loss(render_result["rgb_marched"], target)
        elif cfg_train.main_loss == "huber":
            fg_mask = render_result["bg_prob"].detach() < 0.5
            main_loss = huber_loss(
                render_result["rgb_marched"], target, update_c=True, fg_mask=fg_mask
            ).mean()
        else:
            raise NotImplementedError
        loss = cfg_train.weight_main * main_loss

        if cfg_train.weight_entropy_last > 0:
            bg_prob = render_result["bg_prob"].clamp(1e-4, 1 - 1e-4)
            entropy_last_loss = -(
                bg_prob * torch.log(bg_prob) + (1 - bg_prob) * torch.log(1 - bg_prob)
            ).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            q = render_result["q"]
            rgbper = (q.rgb - target[q.ray_id]).abs().sum(-1)
            rgbper_loss = (rgbper * q.weights).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss

        # supervision from external foreground mask
        if cfg_train.weight_mask > 0:
            bg_prob = render_result["bg_prob"].clamp(1e-6, 1 - 1e-6)
            target_bg_mask = 1 - target_fg_mask.float()
            mask_loss = F.binary_cross_entropy(bg_prob, target_bg_mask)
            loss += cfg_train.weight_mask * mask_loss

        loss.backward()
        if trainiter % 100 == 0 and model.density.type == "DenseGrid":
            gridgrad = model.density.grid.grad.cpu().numpy()
            mask = gridgrad != 0
            gridgrad = gridgrad[mask]
            print(
                f"geo grid grad stat: "
                f"%={mask.mean():.4f} "
                f"mu={gridgrad.mean():.2e} "
                f"mu={np.abs(gridgrad).mean():.2e} "
                f"std={gridgrad.std():.2e} "
                f"min={gridgrad.min():.2e} "
                f"max={gridgrad.max():.2e} "
            )

        # grid-level regularization
        if cfg_train.weight_laplace > 0:
            neural_pbir_cuda_utils.laplace_add_grad(
                model.density.grid.data,
                model.density.grid.grad,
                -1e9,
                1e9,
                cfg_train.weight_laplace,
                7,
                cfg_train.laplace_dense,
                1,
            )

        # optimizer step
        optimizer.step()
        if cam_refiner is not None:
            cam_refiner.step()
        psnr_lst.append(psnr.item())

        # decay learning rate
        decay_rate = cfg_train.lrate_decay_factor ** (1 / cfg_train.N_iters)
        geo_decay_rate = cfg_train.lrate_geo_decay_factor ** (1 / cfg_train.N_iters)
        for param_group in optimizer.param_groups:
            if param_group["k"] == "density" and param_group["is_fg"]:
                param_group["lr"] = param_group["lr"] * geo_decay_rate
            else:
                param_group["lr"] = param_group["lr"] * decay_rate

        # visualize training progress
        if args.run_vis_period > 0 and (
            trainiter % args.run_vis_period == 0 or trainiter == 1
        ):
            run_id = running_render_viewpoints_kwargs["tick"] % len(
                data_dict["render_poses"]
            )
            rgbs, depths, normals, xyzs, bgmaps = render_viewpoints(
                render_poses=data_dict["render_poses"][[run_id]],
                HW=data_dict["HW"][[0]],
                Ks=data_dict["Ks"][[0]],
                savedir=run_vis_dir,
                dump_images=False,
                silent=True,
                model=model,
                ndc=False,
                render_factor=args.run_vis_factor,
                render_kwargs=running_render_viewpoints_kwargs,
            )
            n0 = normals[0] * 0.5 + 0.5
            imageio.imwrite(
                os.path.join(run_vis_dir, f"n0_{trainiter}.png"),
                (n0 * 255).astype(np.uint8),
            )
            running_render_viewpoints_kwargs["tick"] += 1

        # check log & save
        eps_time = time.time() - time0
        if trainiter % args.i_print == 0:
            eps_time_str = (
                f"{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}"
            )
            tqdm.write(
                f"scene_rep_reconstruction ({stage}): iter {trainiter:6d} / "
                f"Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / "
                f"Eps: {eps_time_str}"
            )
            psnr_lst = []

        if trainiter % args.i_weights == 0:
            print(f"scene_rep_reconstruction ({stage}): saving checkpoints...")
            path = cfg.results_dir / f"{stage}_{trainiter:06d}.pth"
            utils.torch_save(
                {
                    "trainiter": trainiter,
                    "model_kwargs": model.get_kwargs(),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    if args.save_optimizer
                    else None,
                    "cam_refiner": cam_refiner.state_dict()
                    if cam_refiner is not None
                    else None,
                },
                path,
            )
            print(f"scene_rep_reconstruction ({stage}): saved checkpoints at", path)

    if trainiter != -1:
        print(f"scene_rep_reconstruction ({stage}): saving checkpoints...")
        utils.torch_save(
            {
                "trainiter": trainiter,
                "model_kwargs": model.get_kwargs(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                if args.save_optimizer
                else None,
                "cam_refiner": cam_refiner.state_dict()
                if cam_refiner is not None
                else None,
            },
            last_ckpt_path,
        )
        print(
            f"scene_rep_reconstruction ({stage}): saved checkpoints at", last_ckpt_path
        )


def train(args, cfg, data_dict):
    # init
    print("train: start")
    eps_time = time.time()
    cfg.results_dir.mkdir(exist_ok=True, parents=True)
    with open(cfg.results_dir / "args.txt", "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    cfg.results_dir = str(cfg.results_dir).replace("\\", "/")
    cfg.data.datadir = cfg.data.datadir.replace("\\", "/")
    cfg.dump(os.path.join(cfg.results_dir, "config.py"))
    cfg.results_dir = Path(cfg.results_dir)

    # train a coarse model to find a tight foreground bbox (optional)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = utils_bbox.compute_bbox(**data_dict, **cfg.data)
    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
            args=args,
            cfg=cfg,
            cfg_model=cfg.coarse_model_and_render,
            cfg_train=cfg.coarse_train,
            xyz_min=xyz_min_coarse,
            xyz_max=xyz_max_coarse,
            data_dict=data_dict,
            stage="coarse",
        )
        eps_coarse = time.time() - eps_coarse
        eps_time_str = (
            f"{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}"
        )
        print("train: coarse geometry searching in", eps_time_str)
        coarse_ckpt_path = cfg.results_dir / "coarse_last.pth"
    else:
        print("train: skip coarse geometry searching")
        coarse_ckpt_path = None

    # build camera refinement model
    cam_refiner = None
    if cfg.camera_refinement_model.trainable:
        cam_refiner = CamRefiner(
            data_dict["poses"][data_dict["i_train"]], cfg.camera_refinement_model
        )
        cam_refiner = cam_refiner.cuda()

    # main training stage
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = utils_bbox.compute_bbox_by_coarse_geo(
            model_class=vol_repr.VolRepr,
            model_path=coarse_ckpt_path,
            bbox_alpha_thres=cfg.fine_model_and_render.bbox_alpha_thres,
            bbox_largest_cc_only=cfg.fine_model_and_render.bbox_largest_cc_only,
        )
    scene_rep_reconstruction(
        args=args,
        cfg=cfg,
        cfg_model=cfg.fine_model_and_render,
        cfg_train=cfg.fine_train,
        xyz_min=xyz_min_fine,
        xyz_max=xyz_max_fine,
        data_dict=data_dict,
        stage="fine",
        cam_refiner=cam_refiner,
        coarse_ckpt_path=coarse_ckpt_path if cfg.fine_train.use_coarse_mask else None,
    )
    eps_fine = time.time() - eps_fine
    eps_time_str = f"{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}"
    print("train: fine detail reconstruction in", eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f"{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}"
    print("train: finish (eps time", eps_time_str, ")")


def dump_results(args, cfg, data_dict):
    """
    Below are simply dumping the results in the form of:
        interpolated videos, mesh, refined poses, or sparse 3d feature.
    """
    device = torch.device("cuda")

    # load model
    if (
        args.render_test
        or args.render_train
        or args.render_video
        or args.export_surface
    ):
        ckpt_coarse_path = cfg.results_dir / "coarse_last.pth"
        ckpt_fine_path = cfg.results_dir / "fine_last.pth"
        if args.ft_path:
            ckpt_path = args.ft_path
        elif ckpt_fine_path.is_file():
            ckpt_path = ckpt_fine_path
        else:
            ckpt_path = ckpt_coarse_path
        ckpt_name = ckpt_path.stem
        model = utils.load_model(vol_repr.VolRepr, ckpt_path).to(device)
        render_viewpoints_kwargs = {
            "model": model,
            "ndc": cfg.data.ndc,
            "render_kwargs": {
                "near": data_dict["near"],
                "far": data_dict["far"],
                "bg": cfg.data.bkgd,
                "stepsize": cfg.fine_model_and_render.stepsize,
                "render_depth": args.render_depth,
                "render_normal": args.render_normal,
                "render_fg_only": args.render_fg_only,
            },
        }
    else:
        print("Dump nothing.")
        return

    # render train split and eval
    if args.render_train:
        testsavedir = cfg.results_dir / f"render_train_{ckpt_name}"
        os.makedirs(testsavedir, exist_ok=True)
        print("All results are dumped into", testsavedir)
        rgbs, depths, normals, xyzs, bgmaps = render_viewpoints(
            render_poses=data_dict["poses"][data_dict["i_train"]],
            HW=data_dict["HW"][data_dict["i_train"]],
            Ks=data_dict["Ks"][data_dict["i_train"]],
            gt_imgs=[
                data_dict["images"][i].cpu().numpy() for i in data_dict["i_train"]
            ],
            savedir=testsavedir,
            dump_images=args.dump_images,
            eval_ssim=args.eval_ssim,
            eval_lpips_alex=args.eval_lpips_alex,
            eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs,
        )
        save_movie(testsavedir, rgbs=rgbs, depths=depths, normals=normals)
        imageio.mimwrite(
            os.path.join(testsavedir, "video.rgb.mp4"),
            utils.to8b(rgbs),
            fps=30,
            quality=8,
        )
        dmin, dmax = np.percentile(depths[bgmaps < 0.1], q=[5, 100])
        depth_vis = 1 - np.clip((depths - dmin) / (dmax - dmin), 0, 1)
        depth_vis = depth_vis * (1 - bgmaps)
        imageio.mimwrite(
            os.path.join(testsavedir, "video.depth.mp4"),
            utils.to8b(depth_vis),
            fps=30,
            quality=8,
        )
        imageio.mimwrite(
            os.path.join(testsavedir, "video.normal.mp4"),
            utils.to8b(normals * 0.5 + 0.5),
            fps=30,
            quality=8,
        )

    # render test split and eval
    if args.render_test:
        testsavedir = cfg.results_dir / f"render_test_{ckpt_name}"
        os.makedirs(testsavedir, exist_ok=True)
        print("All results are dumped into", testsavedir)
        rgbs, depths, normals, xyzs, bgmaps = render_viewpoints(
            render_poses=data_dict["poses"][data_dict["i_test"]],
            HW=data_dict["HW"][data_dict["i_test"]],
            Ks=data_dict["Ks"][data_dict["i_test"]],
            gt_imgs=[data_dict["images"][i].cpu().numpy() for i in data_dict["i_test"]],
            savedir=testsavedir,
            dump_images=args.dump_images,
            eval_ssim=args.eval_ssim,
            eval_lpips_alex=args.eval_lpips_alex,
            eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs,
        )
        save_movie(testsavedir, rgbs=rgbs, depths=depths, normals=normals)

    # render video
    if args.render_video:
        testsavedir = cfg.results_dir / f"render_video_{ckpt_name}"
        os.makedirs(testsavedir, exist_ok=True)
        print("All results are dumped into", testsavedir)
        rgbs, depths, normals, xyzs, bgmaps = render_viewpoints(
            render_poses=data_dict["render_poses"],
            HW=data_dict["HW"][data_dict["i_train"]][[0]].repeat(
                len(data_dict["render_poses"]), 0
            ),
            Ks=data_dict["Ks"][data_dict["i_train"]][[0]].repeat(
                len(data_dict["render_poses"]), 0
            ),
            render_factor=args.render_video_factor,
            savedir=testsavedir,
            dump_images=args.dump_images,
            **render_viewpoints_kwargs,
        )
        save_movie(testsavedir, rgbs=rgbs, depths=depths, normals=normals)

    # export mesh
    if args.export_surface:
        print("=" * 50)
        print("Extracting mesh")
        with torch.no_grad():
            sdf_grid, mask = model.get_sdf_grid()
        sdf_grid[~mask] = 100
        xyz_min = model.xyz_min.cpu().numpy()
        xyz_max = model.xyz_max.cpu().numpy()
        mesh = utils.extract_mesh(
            sdf_grid.cpu().numpy(),
            isovalue=args.mesh_iso,
            xyz_min=xyz_min,
            xyz_max=xyz_max,
            cleanup=not args.mesh_no_cleanup,
        )
        if args.mesh_dtu_clean:
            mesh = utils.DTU_clean_mesh_by_mask(mesh, cfg.data.datadir)
        mesh.remove_degenerate_faces()
        print("Dumping mesh")
        mesh.export(cfg.results_dir / "mesh_mc_raw.obj")

        import subprocess

        # remesh and uv-unwrap
        import trimesh

        # import pymeshfix
        from gpytoolbox import remesh_botsch

        v, f = mesh.vertices, mesh.faces
        # print("Fixing the mesh...")
        # v, f = pymeshfix.clean_from_arrays(v, f, remove_smallest_components=False)
        print("Remeshing...")
        edge_len = mesh.edges_unique_length.mean()
        v, f = remesh_botsch(
            v, f, i=5, h=min(edge_len, ((v.max(0) - v.min(0)) ** 2).sum() ** 0.5 / 512)
        )
        # print("Fixing the mesh the second time...")
        # v, f = pymeshfix.clean_from_arrays(v, f, remove_smallest_components=False)
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        export_mesh_path = cfg.results_dir / "mesh.obj"
        mesh.export(export_mesh_path)
        print("The mesh is dumped into", export_mesh_path)
        print("Using blender to uv unwrap for", export_mesh_path)
        cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(
            [
                "blender",
                "--background",
                "--python",
                cur_dir / "lib" / "uv_blender.py",
                "--",
                export_mesh_path,
                export_mesh_path,
            ]
        )
        mesh = trimesh.load(export_mesh_path)
        if model.on_known_board:
            print("Adding a bottom board...")
            export_plane_path = cfg.results_dir / "mesh_plane_raw.obj"
            plane_L = (model.xyz_max - model.xyz_min).cpu().numpy()
            plane_L[1] = plane_L[[0, 2]].mean() / 128
            plane_C = (model.xyz_max + model.xyz_min).cpu().numpy() * 0.5
            plane_C[1] = model.xyz_min[1].item() - plane_L[1] * 0.5
            mesh_plane = utils.create_mesh_two_plane_box(plane_C, plane_L, 128 * 128)
            mesh_plane.export(export_plane_path)
            subprocess.run(
                [
                    "blender",
                    "--background",
                    "--python",
                    cur_dir / "lib" / "uv_blender.py",
                    "--",
                    export_plane_path,
                    export_plane_path,
                ]
            )
            mesh_plane = trimesh.load(export_plane_path)

            mesh.visual.uv[:, 0] *= 0.5
            mesh_plane.visual.uv[:, 0] = 0.5 + mesh_plane.visual.uv[:, 0] * 0.5
            mesh = trimesh.util.concatenate([mesh, mesh_plane])
            mesh.export(export_mesh_path)
        print("The uv mesh is dumped into", cyan(export_mesh_path))
        print("=" * 50)

        # export bg360
        if model.bg_model is not None:
            with torch.no_grad():
                bg360 = model.render360().cpu().mul(255).numpy().astype(np.uint8)
                imageio.imwrite(cfg.results_dir / "bg360.png", bg360)
                print(
                    "The bg 360 image is dumped into",
                    cyan(cfg.results_dir / "bg360.png"),
                )

        # export camera poses in later pbir stage's format
        st = utils.torch_load(ckpt_path, map_location="cpu")
        cam_refiner = CamRefiner(
            data_dict["poses"][data_dict["i_train"]], cfg.camera_refinement_model
        )
        if st["cam_refiner"] is not None:
            cam_refiner.load_state_dict(st["cam_refiner"])
        with torch.no_grad():
            poses = cam_refiner.get_refined_poses().cpu().numpy()
        poses_homo = np.concatenate(
            [poses, np.array([[[0, 0, 0, 1]]]).repeat(len(poses), 0)], 1
        )
        poses_homo[:, :, :2] *= -1
        np.savetxt(cfg.results_dir / "to_worlds_after.txt", poses_homo.reshape(-1, 4))
        print(
            "The cam2world matrices are dumped into",
            cyan(cfg.results_dir / "to_worlds_after.txt"),
        )


def run(args, cfg):
    # init environment
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)

    # load images / poses / camera settings / data split
    data_dict = load_everything(cfg=cfg)

    # train
    if not args.test_only:
        train(args, cfg, data_dict)

    dump_results(args, cfg, data_dict)
    print("Done")


if __name__ == "__main__":
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    run(args, cfg)
