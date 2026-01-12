# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import glob
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import mmcv
import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
from tqdm import tqdm, trange

sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
from neural_surface_recon.lib import utils_ray, utils_sdf_rt, utils_sg, vol_repr
from neural_surface_recon.run import load_everything


def print_cyan(s):
    print(f"\033[96m{s}\033[0m")


def srgb2lin(s):
    return torch.where(
        s <= 0.04045, s / 12.92, ((s.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4
    )


def lin2srgb(lin):
    return torch.where(
        lin > 0.0031308,
        (lin.clamp_min(0.0031308) ** (1.0 / 2.4)) * 1.055 - 0.055,
        lin * 12.92,
    )


def compute_avg_bg(pre_model, cfg_path, H=128, W=256, split_sz=65535):
    # Load data
    cfg = mmcv.Config.fromfile(cfg_path)
    cfg.data.load2gpu_on_the_fly = True  # save gpu mem
    cfg.data.update_bg_bkgd = False  # dont mask out bg pixels
    cfg.fine_train.ray_sampler = "flatten"  # we need all rays to compute avg bg
    cfg.fine_train.ray_sampler_fg_rate = None  # we check if a ray hit fg ourself
    data_dict = load_everything(cfg)
    rgb, rays_o, rays_d, _, _, _, _, _ = utils_ray.gather_training_rays(
        data_dict, cfg, cfg.fine_train, "cuda", pre_model
    )

    render_kwargs = {
        "render_color": False,
        "render_depth": False,
        "render_normal": False,
        "bg": 0.5,
        "near": 0,
        "far": 1e9,
        "stepsize": 0.5,  # half voxel size
    }

    bg_avg = torch.zeros([H, W, 3]).cuda()
    bg_avg_alpha = torch.zeros([H, W, 3]).cuda()

    assert len(rgb.shape) == 2, "Rays should be flatten first"
    assert len(rgb) == len(rays_o)
    assert len(rgb) == len(rays_d)
    rgb = rgb.split(split_sz)
    rays_o = rays_o.split(split_sz)
    rays_d = rays_d.split(split_sz)
    for gt, ro, rd in zip(tqdm(rgb), rays_o, rays_d):
        # cast rays
        with torch.no_grad():
            ro = ro.contiguous().cuda()
            rd = rd.contiguous().cuda()
            viewdirs = torch.nn.functional.normalize(rd, dim=-1)
            render_result = pre_model.superfast_forward(
                ro, rd, viewdirs, **render_kwargs
            )
        bg_rgb = gt.cuda()[~render_result["is_hit"]]
        bg_viewdirs = viewdirs[~render_result["is_hit"]]
        bg_coord = utils_sg.uv2coord(utils_sg.vec2uv(bg_viewdirs)) * 2 - 1

        # compute rgba on 360
        im_dummy = torch.zeros([1, 3, H, W]).requires_grad_()
        out = torch.nn.functional.grid_sample(
            im_dummy, bg_coord[None, None], mode="bilinear", align_corners=True
        )[0, :, 0].T
        im_w = torch.autograd.grad(
            outputs=out,
            inputs=im_dummy,
            grad_outputs=torch.ones_like(out, requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        im_rgb = torch.autograd.grad(
            outputs=out,
            inputs=im_dummy,
            grad_outputs=bg_rgb,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # accumulate
        bg_avg += im_rgb.data[0].moveaxis(0, -1)
        bg_avg_alpha += im_w.data[0].moveaxis(0, -1)

    bg_avg = bg_avg / bg_avg_alpha
    valid = bg_avg_alpha > 0
    bg_avg = srgb2lin(bg_avg)
    bg_avg[~valid] = 0
    return bg_avg


@torch.no_grad()
def compute_normal(pre_model, xyz, stepsize=0.5):
    q = vol_repr.Query(ray_pts=xyz)
    pre_model._process_query_gradient(q)
    if pre_model.on_known_board:
        board_min = pre_model.xyz_min.clone()
        board_max = pre_model.xyz_max.clone()
        board_min[1] = pre_model.xyz_min[1] - pre_model.voxel_size * stepsize
        board_max[1] = pre_model.xyz_min[1] + pre_model.voxel_size * stepsize
        q.gradient[((board_min <= q.ray_pts) & (q.ray_pts <= board_max)).all(-1)] = (
            torch.Tensor([0, 1, 0])
        )
    return torch.nn.functional.normalize(q.gradient, dim=-1)


@torch.no_grad()
def cast2sdfzero(pre_model, ro, rd, stepsize=0.5, render_normal=True):
    # Note!!
    # All ro should be inside the scene bbox. All rd should be normalized.
    shape = ro.shape[:-1]
    ro = ro.view(-1, 3).contiguous()
    rd = rd.view(-1, 3).contiguous()
    xyz, is_hit = utils_sdf_rt.sdf_grid_trace_surface(
        sdfgrid=pre_model.density.grid.data,
        rs=ro,
        rd=rd,
        stepsize=stepsize,
        xyz_min=pre_model.density.xyz_min,
        xyz_max=pre_model.density.xyz_max,
        world_size=pre_model.density.world_size,
    )
    if pre_model.on_known_board:
        # check bottom board
        board_min = pre_model.xyz_min.clone()
        board_max = pre_model.xyz_max.clone()
        board_min[1] = pre_model.xyz_min[1] - pre_model.voxel_size * stepsize
        board_max[1] = pre_model.xyz_min[1] + pre_model.voxel_size * stepsize
        board_idx = torch.where(((board_min <= xyz) & (xyz <= board_max)).all(-1))[0]
        is_hit[board_idx] = 1
        xyz[board_idx, 1] = pre_model.xyz_min[1] + pre_model.voxel_size * 0.1
    # process normal
    normal = torch.zeros_like(xyz)
    if render_normal:
        hit_idx = torch.where(is_hit)[0]
        normal[hit_idx] = compute_normal(pre_model, xyz[hit_idx], stepsize=stepsize)
    return xyz.reshape(*shape, 3), is_hit.reshape(*shape), normal.reshape(*shape, 3)
