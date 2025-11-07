# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Code adapt from:
    https://github.com/Kai-46/PhySG
"""

import os

import imageio
import numpy as np
import torch
import torch.nn as nn

TINY_NUMBER = 1e-8


def fibonacci_sphere(samples):
    """
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    :param samples:
    :return:
    """
    assert samples > 1
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    points = np.array(points)
    return points


def equirec_sphere(H, W):
    v, u = np.meshgrid(
        -((np.arange(H) + 0.5) / H - 0.5) * np.pi,
        np.arange(W) / W * np.pi * 2 - np.pi / 2,
        indexing="ij",
    )
    points = np.stack(
        [np.cos(v) * np.cos(u), np.sin(v), np.cos(v) * np.sin(u)], -1
    ).reshape(-1, 3)
    return points


def spiral_sphere(N, round=4):
    v = np.linspace(-np.pi / 2, np.pi / 2, N)
    u = np.linspace(0, np.pi * 2 * round, N)
    points = np.stack(
        [np.cos(v) * np.cos(u), np.sin(v), np.cos(v) * np.sin(u)], -1
    ).reshape(-1, 3)
    return points


def vec2uv(vec):
    u = torch.atan2(vec[..., 2], vec[..., 0])
    v = torch.atan2(vec[..., 1], (vec[..., 2] ** 2 + vec[..., 0] ** 2).sqrt())
    return torch.stack([u, v], -1)


def uv2coord(uv):
    coorx = (uv[..., 0] / (2 * np.pi) + 0.5 - 0.25) % 1
    coory = (-uv[..., 1] / np.pi) + 0.5
    return torch.stack([coorx, coory], -1)


def SG_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4])
    lgtMu = torch.abs(lgtSGs[:, 4:])
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy


def SG2Envmap(
    lgtSGs,
    H=None,
    W=None,
    upper_hemi=False,
    viewdirs=None,
    fixed_lobe=False,
    fixed_sharpness=False,
):
    if viewdirs is None:
        assert H is not None and W is not None
        if upper_hemi:
            phi, theta = torch.meshgrid(
                [
                    torch.linspace(0.0, np.pi / 2.0, H),
                    torch.linspace(-0.5 * np.pi, 1.5 * np.pi, W),
                ]
            )
            viewdirs = torch.stack(
                [
                    torch.cos(theta) * torch.sin(phi),
                    torch.cos(phi),
                    torch.sin(theta) * torch.sin(phi),
                ],
                -1,
            )  # [H, W, 3]
        else:
            viewdirs = torch.FloatTensor(equirec_sphere(H, W))
        viewdirs = viewdirs.to(lgtSGs.device)

    lgtSGLobes, lgtSGLambdas, lgtSGMus = lgtSGs.split([3, 1, 3], dim=-1)
    if fixed_lobe:
        lgtSGLobes = lgtSGLobes.detach()
    if fixed_sharpness:
        lgtSGLambdas = lgtSGLambdas.detach()
    lgtSGLobes = torch.nn.functional.normalize(lgtSGLobes, dim=-1)
    lgtSGLambdas = lgtSGLambdas.abs()
    lgtSGMus = lgtSGMus.abs()
    lgt_w = torch.exp(lgtSGLambdas.squeeze(-1) * (torch.mm(viewdirs, lgtSGLobes.T) - 1))
    envmap = torch.mm(lgt_w, lgtSGMus)
    if H is not None and W is not None:
        envmap = envmap.reshape((H, W, 3))
    return envmap


def init_sg(numLgtSGs):
    # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
    lgtSGs = nn.Parameter(
        torch.randn(numLgtSGs, 7), requires_grad=True
    ).cuda()  # [M, 7]; lobe + lambda + mu
    lgtSGs.data[:, -2:] = lgtSGs.data[:, -3:-2].expand((-1, 2))

    # make sure lambda is not too close to zero
    lgtSGs.data[:, 3:4] = 10.0 + torch.abs(lgtSGs.data[:, 3:4] * 20.0)
    # init envmap energy
    energy = SG_energy(lgtSGs.data)
    lgtSGs.data[:, 4:] = (
        torch.abs(lgtSGs.data[:, 4:])
        / torch.sum(energy, dim=0, keepdim=True)
        * 2.0
        * np.pi
        * 0.8
    )
    energy = SG_energy(lgtSGs.data)

    # deterministicly initialize lobes
    lobes = fibonacci_sphere(numLgtSGs // 2).astype(np.float32)
    lgtSGs.data[: numLgtSGs // 2, :3] = torch.from_numpy(lobes) * 3
    lgtSGs.data[numLgtSGs // 2 :, :3] = torch.from_numpy(lobes) * 3
    return lgtSGs


def Envmap2SG(
    gt_envmap,
    numLgtSGs=128,
    N_iter=1000,
    fixed_lobe=False,
    init_sharpness=20,
    init_intensity=0.5,
    debug=False,
):
    H, W = gt_envmap.shape[:2]
    viewdirs = torch.FloatTensor(equirec_sphere(H, W)).cuda()
    lgtSGs = nn.Parameter(torch.empty(numLgtSGs, 7).cuda())  # lobe + lambda + mu
    lgtSGs.data[..., :3] = torch.FloatTensor(fibonacci_sphere(numLgtSGs)).cuda()
    lgtSGs.data[..., 3] = init_sharpness
    lgtSGs.data[..., 4:] = init_intensity
    lgtSGs_energy = SG_energy(lgtSGs)
    lgtSGs.data[:, 4:] = lgtSGs.data[:, 4:] * (
        2.0 * np.pi * 0.8 / lgtSGs_energy.sum(dim=0, keepdim=True)
    )
    optimizer = torch.optim.Adam(
        [
            lgtSGs,
        ],
        lr=1e-1,
    )

    for step in range(N_iter):
        optimizer.zero_grad()
        env_map = SG2Envmap(lgtSGs, H, W, viewdirs=viewdirs, fixed_lobe=fixed_lobe)
        loss = torch.nn.functional.mse_loss(env_map, gt_envmap)
        loss.backward()
        optimizer.step()

        if ((step + 1) % 100 == 0 or step == 0) and debug:
            print("Envmap2SG step: {}, loss: {}".format(step + 1, loss.item()))

        if step % 100 == 0 and debug:
            envmap_check = env_map.clone().detach().cpu().numpy()
            gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
            im = np.concatenate((gt_envmap_check, envmap_check), axis=0)
            im = np.power(im, 1.0 / 2.2)
            im = np.clip(im, 0.0, 1.0)
            # im = (im - im.min()) / (im.max() - im.min() + TINY_NUMBER)
            im = np.uint8(im * 255.0)
            imageio.imwrite(os.path.join("tmp", "log_im_{}.png".format(step)), im)

    return lgtSGs


#######################################################################################################
# visiblity util
#######################################################################################################
def compute_vis_ratio(lgtSGs, vis_16x32, vis_wl, vis_area):
    lgtSGLobes, lgtSGLambdas, lgtSGMus = lgtSGs.split([3, 1, 3], dim=-1)
    lgtSGLobes = torch.nn.functional.normalize(lgtSGLobes, dim=-1)
    lgtSGLambdas = lgtSGLambdas.abs()
    energy = (vis_wl @ lgtSGLobes.T - 1) * lgtSGLambdas.T
    G = torch.exp(energy) * vis_area
    vis_ratio = torch.cat(
        [
            (vis_gt.unsqueeze(-1) * G).sum(1) / G.sum(0)
            for vis_gt in vis_16x32.split(4096)
        ]
    )
    return vis_ratio


#######################################################################################################
# below is the SG renderer
#######################################################################################################
def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER

    # orig impl; might be numerically unstable
    # t = torch.sqrt(lambda_val) * (1.6988 * lambda_val * lambda_val + 10.8438 * lambda_val) / (lambda_val * lambda_val + 6.2201 * lambda_val + 10.2415)
    inv_lambda_val = 1.0 / lambda_val
    t = (
        torch.sqrt(lambda_val)
        * (1.6988 + 10.8438 * inv_lambda_val)
        / (1.0 + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)
    )

    # orig impl; might be numerically unstable
    # a = torch.exp(t)
    # b = torch.exp(t * cos_beta)
    # s = (a * b - 1.) / ((a - 1.) * (b + 1.))
    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.0))
    s1 = (1.0 - inv_a * inv_b) / (1.0 - inv_a + inv_b - inv_a * inv_b)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.0))
    s2 = (b - inv_a) / ((1.0 - inv_a) * (b + 1.0))
    s = mask * s1 + (1.0 - mask) * s2

    A_b = (
        2.0
        * np.pi
        / lambda_val
        * (torch.exp(-lambda_val) - torch.exp(-2.0 * lambda_val))
    )
    A_u = 2.0 * np.pi / lambda_val * (1.0 - torch.exp(-lambda_val))

    return A_b * (1.0 - s) + A_u * s


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1.0 + 2.0 * ratio * dot)
    tmp = torch.min(tmp, ratio + 1.0)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1.0 / tmp
    diff = lambda2 * (tmp - ratio - 1.0)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * torch.exp(diff)

    return final_lobes, final_lambdas, final_mus


def render_with_sg(normal, wo, lgtSGs, ks, roughness, kd, vis_ratio=None):
    """
    :param normal: [n_pts, 1, 3]; ----> camera; must have unit norm
    :param wo: [n_pts, 1, 3]; ----> camera; must have unit norm
    :param lgtSGs: [n_sg, 7]
    :param ks: float or [n_pts, 1, 1 or 3];
    :param roughness: [n_pts, 1, 1]; values must be positive
    :param kd: [n_pts, 1, 3]; values must lie in [0,1]
    """

    # M = lgtSGs.shape[1]
    # n_pts = list(normal.shape[:-1])

    ########################################
    # light
    ########################################
    lgtSGLobes, lgtSGLambdas, lgtSGMus = lgtSGs.split([3, 1, 3], dim=-1)
    lgtSGLobes = torch.nn.functional.normalize(lgtSGLobes, dim=-1)
    lgtSGLambdas = lgtSGLambdas.abs()
    lgtSGMus = lgtSGMus.abs()
    if vis_ratio is not None:
        lgtSGMus = lgtSGMus * vis_ratio.unsqueeze(-1)

    ########################################
    # specular color
    ########################################
    # NDF
    brdfSGLobes = normal  # use normal as the brdf SG lobes
    brdfSGLambdas = 2.0 / roughness.pow(4)  # [n_pts, 1, 1]
    brdfSGMus = brdfSGLambdas / np.pi

    # perform spherical warping
    v_dot_lobe = (brdfSGLobes * wo).sum(dim=-1, keepdim=True).clamp_min(0)
    warpBrdfSGLobes = torch.nn.functional.normalize(
        2 * v_dot_lobe * brdfSGLobes - wo, dim=-1
    )  # [n_pts, 1, 3]
    warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)
    warpBrdfSGMus = brdfSGMus  # [n_pts, 1, 1]

    new_half = torch.nn.functional.normalize(warpBrdfSGLobes + wo, dim=-1)
    v_dot_h = (wo * new_half).sum(dim=-1, keepdim=True).clamp_min(0)

    F = ks + (1 - ks) * torch.pow(
        2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h
    )  # [n_pts, 1, 1 or 3]

    dot1 = (
        (warpBrdfSGLobes * normal).sum(dim=-1, keepdim=True).clamp_min(0)
    )  # equals <o, n>
    dot2 = (wo * normal).sum(dim=-1, keepdim=True).clamp_min(0)  # equals <o, n>
    k = (roughness + 1.0).pow(2) / 8.0  # [n_pts, 1, 1]
    G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
    G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    G = G1 * G2

    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = warpBrdfSGMus * Moi

    # multiply with light sg
    print("lgtSGLobes", lgtSGLobes.shape)
    print("warpBrdfSGLobes", warpBrdfSGLobes.shape)
    final_lobes, final_lambdas, final_mus = lambda_trick(
        lgtSGLobes,
        lgtSGLambdas,
        lgtSGMus,
        warpBrdfSGLobes,
        warpBrdfSGLambdas,
        warpBrdfSGMus,
    )
    print("final_lobes", final_lobes.shape)

    # now multiply with clamped cosine, and perform hemisphere integral
    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003
    print("normal", normal.shape)
    lobe_prime, lambda_prime, mu_prime = lambda_trick(
        normal, lambda_cos, mu_cos, final_lobes, final_lambdas, final_mus
    )
    print("lobe_prime", lobe_prime.shape)

    dot1 = (lobe_prime * normal).sum(dim=-1, keepdim=True)
    dot2 = (final_lobes * normal).sum(dim=-1, keepdim=True)
    specular_rgb = mu_prime * hemisphere_int(
        lambda_prime, dot1
    ) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    specular_rgb = specular_rgb.sum(dim=-2).clamp_min(0)

    ########################################
    # per-point hemisphere integral of envmap
    ########################################
    # diffuse visibility
    diffuse = kd / np.pi  # [n_pts, 1, 3]
    # multiply with light sg
    final_lobes = lgtSGLobes
    final_lambdas = lgtSGLambdas
    final_mus = lgtSGMus * diffuse

    # now multiply with clamped cosine, and perform hemisphere integral
    lobe_prime, lambda_prime, mu_prime = lambda_trick(
        normal, lambda_cos, mu_cos, final_lobes, final_lambdas, final_mus
    )
    print(normal.shape, final_lobes.shape, lobe_prime.shape)

    dot1 = (lobe_prime * normal).sum(dim=-1, keepdim=True)
    dot2 = (final_lobes * normal).sum(dim=-1, keepdim=True)
    diffuse_rgb = mu_prime * hemisphere_int(
        lambda_prime, dot1
    ) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    diffuse_rgb = diffuse_rgb.sum(dim=-2).clamp_min(0)

    # for debugging
    if torch.isnan(specular_rgb).sum() > 0:
        print("NaN !?")
        import pdb

        pdb.set_trace()
    if torch.isnan(diffuse_rgb).sum() > 0:
        print("NaN !?")
        import pdb

        pdb.set_trace()

    # combine diffuse and specular rgb
    rgb = specular_rgb + diffuse_rgb
    return rgb
