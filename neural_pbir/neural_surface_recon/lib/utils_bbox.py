# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compute scene bbox (xyz_min, xyz_max) to enclose the object of interest.
TODO: find a oriented bbox that better fit a coarse geometry.
"""

import os
import time

import numpy as np

import torch
import trimesh

from . import utils, utils_ray


def compute_bbox(HW, Ks, poses, i_train, aabb, use_only_train=True, **kwargs):
    """
    HW: a list of (H,W) tuple for each frame.
    Ks: a list of camera intrinsics for each frame.
    poses: a list of camera pose (camera to world matrix) for each frame.
    i_train: a list of training frame index.
    """
    print("compute_bbox: start")
    if use_only_train:
        HW = HW[i_train]
        Ks = Ks[i_train]
        poses = poses[i_train]

    if kwargs["fg_bbox_rule"] == "json":
        xyz_min = torch.Tensor(aabb[0])
        xyz_max = torch.Tensor(aabb[1])
    elif kwargs["fg_bbox_rule"] == "cam_centroid_cuboid":
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm_unbounded(
            HW=HW, Ks=Ks, poses=poses, **kwargs
        )
    elif kwargs["fg_bbox_rule"] == "cam_frustrum":
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm_bounded(
            HW=HW, Ks=Ks, poses=poses, **kwargs
        )
    else:
        raise NotImplementedError

    print("compute_bbox: xyz_min", xyz_min)
    print("compute_bbox: xyz_max", xyz_max)
    print("compute_bbox: finish")
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm_bounded(HW, Ks, poses, near, far, ndc, **kwargs):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW, Ks, poses):
        rays_o, rays_d, viewdirs = utils_ray.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc
        )
        if ndc:
            pts_nf = torch.stack([rays_o + rays_d * near, rays_o + rays_d * far])
        else:
            pts_nf = torch.stack([rays_o + viewdirs * near, rays_o + viewdirs * far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm_unbounded(HW, Ks, poses, near_clip, ndc, **kwargs):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW, Ks, poses):
        rays_o, rays_d, viewdirs = utils_ray.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc
        )
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0, 1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max()
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max


@torch.no_grad()
def compute_bbox_by_coarse_geo(
    model_class, model_path, bbox_alpha_thres, bbox_largest_cc_only
):
    print("compute_bbox_by_coarse_geo: start")
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    sdf_grid, mask = model.get_sdf_grid(alpha=bbox_alpha_thres)
    sdf_grid[~mask] = 100
    mesh = utils.extract_mesh(
        sdf_grid.cpu().numpy(),
        isovalue=0,
        xyz_min=model.xyz_min.cpu().numpy(),
        xyz_max=model.xyz_max.cpu().numpy(),
        cleanup=bbox_largest_cc_only,
    )
    xyz_min = torch.Tensor(mesh.vertices.min(0))
    xyz_max = torch.Tensor(mesh.vertices.max(0))
    print("compute_bbox_by_coarse_geo: xyz_min", xyz_min)
    print("compute_bbox_by_coarse_geo: xyz_max", xyz_max)
    eps_time = time.time() - eps_time
    print("compute_bbox_by_coarse_geo: finish (eps time:", eps_time, "secs)")
    return xyz_min, xyz_max
