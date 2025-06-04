# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn


def sh_decode_compute(coef, views):
    # hard coded 3 orders spherical harmonics
    sh_c0 = 0.28209479177387814
    sh_c1 = 0.4886025119029199
    sh_c2_0 = 1.0925484305920792
    sh_c2_1 = -1.0925484305920792
    sh_c2_2 = 0.31539156525252005
    sh_c2_3 = -1.0925484305920792
    sh_c2_4 = 0.5462742152960396

    with torch.no_grad():
        x = views[..., 0:1]
        y = views[..., 1:2]
        z = views[..., 2:3]
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z

    return (
        sh_c0 * coef[..., 0:1]
        + (-sh_c1) * y * coef[..., 1:2]
        + sh_c1 * z * coef[..., 2:3]
        + (-sh_c1) * x * coef[..., 3:4]
        + sh_c2_0 * xy * coef[..., 4:5]
        + sh_c2_1 * yz * coef[..., 5:6]
        + sh_c2_2 * (3 * zz - 1) * coef[..., 6:7]
        + sh_c2_3 * xz * coef[..., 7:8]
        + sh_c2_4 * (xx - yy) * coef[..., 8:9]
    )


def sh_decode(x, views):
    order0, order12 = torch.split(x, [3, 24], dim=-1)
    order0 = np.sqrt(np.pi * 4) * torch.sigmoid(order0)
    # The actual upper bound should be sqrt(3 pi) / 2.0, 2 / 3 * sqrt(15 / pi) and 2 * sqrt(15 pi) / 9.0
    # They are all close to 1.5, so we use 1.5 instead
    order12 = np.sqrt(3 * np.pi) / 2.0 * (2 * torch.sigmoid(order12) - 1)
    r_coef = torch.cat([order0[..., 0:1], order12[..., 0:8]], dim=-1)
    g_coef = torch.cat([order0[..., 1:2], order12[..., 8:16]], dim=-1)
    b_coef = torch.cat([order0[..., 2:3], order12[..., 16:24]], dim=-1)
    views = nn.functional.normalize(views, dim=-1)
    r = sh_decode_compute(r_coef, views)
    g = sh_decode_compute(g_coef, views)
    b = sh_decode_compute(b_coef, views)
    color = torch.cat([r, g, b], dim=-1)
    color = torch.clamp(color, 0, 1)
    return color


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def build_pytorch_mlp(input_dim, hidden_dim, output_dim, depth=10, bias=False):
    if depth == 0:
        return nn.Linear(input_dim, output_dim, bias=bias)
    mlp = []
    mlp.append(nn.Linear(input_dim, hidden_dim, bias=bias))
    mlp.append(nn.ReLU())
    for _ in range(depth - 1):
        mlp.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        mlp.append(nn.ReLU())
    mlp.append(nn.Linear(hidden_dim, output_dim, bias=bias))
    mlp = nn.Sequential(*mlp)
    return mlp


def grid_sample_3d(voxel, index):
    """
    Modified from 2d grid sample of tensoIR
    Aligned corner, repetitive padding
    image: Float[Tensor, B C VZ VY VX]
    index: Float[Tensor, B Z Y X 3]
    """
    N, C, VZ, VY, VX = voxel.shape
    _, Z, Y, X, _ = index.shape

    ix = index[..., 0]
    iy = index[..., 1]
    iz = index[..., 2]

    ix = (ix + 1) / 2 * (VX - 1)
    iy = (iy + 1) / 2 * (VY - 1)
    iz = (iz + 1) / 2 * (VZ - 1)
    with torch.no_grad():
        ix_d = torch.clamp(torch.floor(ix), 0, VX - 1).long()
        iy_d = torch.clamp(torch.floor(iy), 0, VY - 1).long()
        iz_d = torch.clamp(torch.floor(iz), 0, VZ - 1).long()

        ix_u = torch.clamp(ix_d + 1, 0, VX - 1).long()
        iy_u = torch.clamp(iy_d + 1, 0, VY - 1).long()
        iz_u = torch.clamp(iz_d + 1, 0, VZ - 1).long()

        index_ddd = (iz_d * VY * VX + iy_d * VX + ix_d).long().view(N, 1, Z * Y * X)
        index_ddu = (iz_u * VY * VX + iy_d * VX + ix_d).long().view(N, 1, Z * Y * X)
        index_dud = (iz_d * VY * VX + iy_u * VX + ix_d).long().view(N, 1, Z * Y * X)
        index_duu = (iz_u * VY * VX + iy_u * VX + ix_d).long().view(N, 1, Z * Y * X)

        index_udd = (iz_d * VY * VX + iy_d * VX + ix_u).long().view(N, 1, Z * Y * X)
        index_udu = (iz_u * VY * VX + iy_d * VX + ix_u).long().view(N, 1, Z * Y * X)
        index_uud = (iz_d * VY * VX + iy_u * VX + ix_u).long().view(N, 1, Z * Y * X)
        index_uuu = (iz_u * VY * VX + iy_u * VX + ix_u).long().view(N, 1, Z * Y * X)

    w_ddd = (ix - ix_d) * (iy - iy_d) * (iz - iz_d)
    w_ddu = (ix - ix_d) * (iy - iy_d) * (iz_u - iz)
    w_dud = (ix - ix_d) * (iy_u - iy) * (iz - iz_d)
    w_duu = (ix - ix_d) * (iy_u - iy) * (iz_u - iz)

    w_udd = (ix_u - ix) * (iy - iy_d) * (iz - iz_d)
    w_udu = (ix_u - ix) * (iy - iy_d) * (iz_u - iz)
    w_uud = (ix_u - ix) * (iy_u - iy) * (iz - iz_d)
    w_uuu = (ix_u - ix) * (iy_u - iy) * (iz_u - iz)

    voxel = voxel.reshape(N, C, VX * VY * VZ)

    v_ddd = torch.gather(voxel, 2, index_ddd.repeat(1, C, 1))
    v_ddu = torch.gather(voxel, 2, index_ddu.repeat(1, C, 1))
    v_dud = torch.gather(voxel, 2, index_dud.repeat(1, C, 1))
    v_duu = torch.gather(voxel, 2, index_duu.repeat(1, C, 1))

    v_udd = torch.gather(voxel, 2, index_udd.repeat(1, C, 1))
    v_udu = torch.gather(voxel, 2, index_udu.repeat(1, C, 1))
    v_uud = torch.gather(voxel, 2, index_uud.repeat(1, C, 1))
    v_uuu = torch.gather(voxel, 2, index_uuu.repeat(1, C, 1))

    out_val = (
        w_ddd.view(N, 1, Z, Y, X) * v_ddd.view(N, C, Z, Y, X)
        + w_ddu.view(N, 1, Z, Y, X) * v_ddu.view(N, C, Z, Y, X)
        + w_dud.view(N, 1, Z, Y, X) * v_dud.view(N, C, Z, Y, X)
        + w_duu.view(N, 1, Z, Y, X) * v_duu.view(N, C, Z, Y, X)
        + w_udd.view(N, 1, Z, Y, X) * v_udd.view(N, C, Z, Y, X)
        + w_udu.view(N, 1, Z, Y, X) * v_udu.view(N, C, Z, Y, X)
        + w_uud.view(N, 1, Z, Y, X) * v_uud.view(N, C, Z, Y, X)
        + w_uuu.view(N, 1, Z, Y, X) * v_uuu.view(N, C, Z, Y, X)
    )

    return out_val


def grid_sample_2d(image, index):
    """
    Mostly copy from tensoIR
    Aligned corner, repetitive padding
    image: Float[Tensor, B C IH IW]
    index: Float[Tensor, B H W 2]
    """

    N, C, IH, IW = image.shape
    _, H, W, _ = index.shape

    ix = index[..., 0]
    iy = index[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.contiguous().view(N, C, IH * IW)

    nw_val = torch.gather(
        image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)
    )

    out_val = (
        nw_val.view(N, C, H, W) * nw.view(N, 1, H, W)
        + ne_val.view(N, C, H, W) * ne.view(N, 1, H, W)
        + sw_val.view(N, C, H, W) * sw.view(N, 1, H, W)
        + se_val.view(N, C, H, W) * se.view(N, 1, H, W)
    )

    return out_val
