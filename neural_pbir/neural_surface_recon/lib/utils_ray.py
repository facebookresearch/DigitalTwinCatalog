# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import os
import time

import neural_pbir_cuda_utils
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh


"""
Basic utils
"""


def get_rays(H, W, K, c2w, mode="center"):
    x, y = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device).float(),
        torch.linspace(0, H - 1, H, device=c2w.device).float(),
        indexing="xy",
    )
    if mode == "lefttop":
        pass
    elif mode == "center":
        x = x + 0.5
        y = y + 0.5
    elif mode == "random":
        x = x + torch.rand_like(x)
        y = y + torch.rand_like(y)
    else:
        raise NotImplementedError

    dirs = torch.stack(
        [(x - K[0][2]) / K[0][0], (y - K[1][2]) / K[1][1], torch.ones_like(x)], -1
    )
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, mode="center"):
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)
    return rays_o, rays_d, viewdirs


"""
Generate rays for all the frame in the dataset
"""


def batch_indices_generator(N, BS, fg_rate=None, fg_mask=None):
    # torch.randperm on cuda produce incorrect results on my machine
    if fg_rate is None:
        # randomly sample a batch of index
        while True:
            yield np.random.randint(0, N, [BS])
    else:
        # sample fg_rate percent of index from foreground
        # TODO: online sampler if we can iterate all the rays multiple times
        print("Adaptive sampling")
        fg_mask = fg_mask.cpu().numpy()
        fg_rate = max(fg_rate, fg_mask.mean())
        print("  original fg rate:", fg_mask.mean())
        print("  new      fg rate:", fg_rate)
        BS_fg = int(BS * fg_rate)
        BS_bg = BS - BS_fg
        idx_fg = np.where(fg_mask)[0]
        idx_bg = np.where(~fg_mask)[0]
        while True:
            yield np.concatenate(
                [
                    idx_fg[np.random.randint(0, len(idx_fg), [BS_fg])],
                    idx_bg[np.random.randint(0, len(idx_bg), [BS_bg])],
                ]
            )


def gather_training_rays(data_dict, cfg, cfg_train, device, model):
    HW, Ks, i_train, poses, render_poses, images, masks, depths = [
        data_dict[k]
        for k in [
            "HW",
            "Ks",
            "i_train",
            "poses",
            "render_poses",
            "images",
            "masks",
            "depths",
        ]
    ]

    rgb_tr_ori = images[i_train].to("cpu" if cfg.data.load2gpu_on_the_fly else device)
    depths_tr = None
    masks_tr = None

    __offline_samplers__ = dict(
        random=offline_sampler_flatten,
        flatten=offline_sampler_flatten,
        hitbbox=offline_sampler_hitbbox,
        in_fgmask=offline_sampler_in_fgmask,
        offline_sampler_in_bbox=offline_sampler_in_bbox,
        in_maskcache=offline_sampler_in_maskcache,
    )
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, depths_tr, masks_tr, imsz = (
        get_training_rays_offline_sampler(
            rgb_tr_ori=rgb_tr_ori,
            train_poses=poses[i_train],
            HW=HW[i_train],
            Ks=Ks[i_train],
            ndc=cfg.data.ndc,
            train_depths=depths,
            train_masks=masks,
            offline_sampler=__offline_samplers__[cfg_train.ray_sampler],
            model=model,
        )
    )

    image_id_tr = np.zeros(len(rgb_tr), dtype=np.int64)
    image_id_tr[np.cumsum(imsz[:-1])] = 1
    image_id_tr = np.cumsum(image_id_tr)
    assert image_id_tr[-1] == len(imsz) - 1
    image_id_tr = torch.LongTensor(image_id_tr).to(rgb_tr.device)

    fg_mask = None
    if cfg_train.ray_sampler_fg_rate is not None:
        fg_mask = torch.cat(
            [
                (
                    neural_pbir_cuda_utils.infer_t_minmax(
                        o.cuda(), d.cuda(), model.xyz_min, model.xyz_max, 0, 1e9
                    )[1]
                    > 0
                )
                for o, d in zip(rays_o_tr.split(10000000), rays_d_tr.split(10000000))
            ]
        )
    index_generator = batch_indices_generator(
        len(rgb_tr), cfg_train.N_rand, cfg_train.ray_sampler_fg_rate, fg_mask
    )
    batch_index_sampler = lambda: next(index_generator)

    return (
        rgb_tr,
        rays_o_tr,
        rays_d_tr,
        viewdirs_tr,
        depths_tr,
        masks_tr,
        image_id_tr,
        batch_index_sampler,
    )


def get_training_rays_offline_sampler(
    rgb_tr_ori,
    train_poses,
    HW,
    Ks,
    ndc,
    offline_sampler,
    model,
    train_depths=None,
    train_masks=None,
    **kwargs,
):
    print("get_training_rays_offline_sampler: start")
    assert (
        len(rgb_tr_ori) == len(train_poses)
        and len(rgb_tr_ori) == len(Ks)
        and len(rgb_tr_ori) == len(HW)
    )
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device

    use_depth = train_depths is not None
    use_mask = train_masks is not None

    if train_depths is None:
        train_depths = [None] * len(rgb_tr_ori)
    if train_masks is None:
        train_masks = [None] * len(rgb_tr_ori)

    # Allocate all tensors first (will truncate them later)
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, 3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    depths_tr = torch.zeros([N], device=DEVICE) if use_depth else None
    masks_tr = torch.zeros([N], device=DEVICE) if use_mask else None
    imsz = []
    top = 0

    # Gather all rays
    for i in range(len(rgb_tr_ori)):
        c2w = train_poses[i]
        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w, ndc=ndc)

        mask = offline_sampler(
            rays_o, rays_d, viewdirs, model, train_depths[i], train_masks[i], **kwargs
        )

        n = mask.sum()
        rgb_tr[top : top + n].copy_(rgb_tr_ori[i][mask.to(DEVICE)])
        rays_o_tr[top : top + n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top : top + n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top : top + n].copy_(viewdirs[mask].to(DEVICE))
        if use_depth:
            depths_tr[top : top + n].copy_(train_depths[i].to(DEVICE)[mask.to(DEVICE)])
        if use_mask:
            masks_tr[top : top + n].copy_(train_masks[i].to(DEVICE)[mask.to(DEVICE)])
        imsz.append(n.item())
        top += n

    # Truncate tensors
    print("get_training_rays_offline_sampler: ratio", top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    depths_tr = depths_tr[:top] if use_depth else None
    masks_tr = masks_tr[:top] if use_mask else None
    eps_time = time.time() - eps_time
    print("get_training_rays_offline_sampler: finish (eps time:", eps_time, "sec)")
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, depths_tr, masks_tr, imsz


def offline_sampler_flatten(
    rays_o, rays_d, viewdirs, model, gt_depth, gt_mask, **kwargs
):
    return torch.ones(rays_o.shape[:-1], dtype=torch.bool)


def offline_sampler_hitbbox(
    rays_o, rays_d, viewdirs, model, gt_depth, gt_mask, **kwargs
):
    t_min, t_max = neural_pbir_cuda_utils.infer_t_minmax(
        rays_o.reshape(-1, 3).contiguous(),
        rays_d.reshape(-1, 3).contiguous(),
        model.xyz_min,
        model.xyz_max,
        0,
        1e9,
    )
    mask = (t_min < t_max).reshape(rays_o.shape[:2])
    return mask


def offline_sampler_in_fgmask(
    rays_o, rays_d, viewdirs, model, gt_depth, gt_mask, **kwargs
):
    return gt_mask.clone()


def offline_sampler_in_bbox(
    rays_o, rays_d, viewdirs, model, gt_depth, gt_mask, **kwargs
):
    rays_pts = rays_o + rays_d * gt_depth[:, :, None].to(rays_o)
    mask = ((model.xyz_min <= rays_pts) & (rays_pts <= model.xyz_max)).all(-1)
    mask = cv2.dilate(mask.cpu().float().numpy(), np.ones([3, 3]), iterations=10)
    mask = (torch.tensor(mask) > 0).to(rays_pts.device)
    return mask


def offline_sampler_in_maskcache(
    rays_o, rays_d, viewdirs, model, gt_depth, gt_mask, **kwargs
):
    t_min, t_max = neural_pbir_cuda_utils.infer_t_minmax(
        rays_o.reshape(-1, 3).contiguous(),
        rays_d.reshape(-1, 3).contiguous(),
        model.xyz_min,
        model.xyz_max,
        0,
        1e9,
    )
    stepdist = model.voxel_size * 0.5
    max_n_steps = ((t_max - t_now) / stepdist).long() + 1
    rt_n_steps = model.mask_cache.ray_tracing(rays_o, viewdirs, stepdist, max_n_steps)
    mask = rt_n_steps < max_n_steps
    return mask
