#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

from tqdm import tqdm, trange

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import imageio
import imageio.v3 as iio
import numpy as np
import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch
import trimesh

sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
from distill_util import (
    cast2sdfzero,
    compute_avg_bg,
    compute_normal,
    lin2srgb,
    print_cyan,
)
from neural_surface_recon.lib import utils_sg, vol_repr


def write_exr(path, exr):
    imageio.plugins.freeimage.download()
    iio.imwrite(path, exr, flags=imageio.plugins.freeimage.IO_FLAGS.EXR_NONE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("ckptroot", help="path to ckptroot", type=Path)

    # model related
    parser.add_argument(
        "--n_lobe", type=int, default=256, help="Number of sg lobes in envmap."
    )

    # rendering related
    parser.add_argument(
        "--n_samp", type=int, default=256, help="Number of reflect ray samples."
    )
    parser.add_argument("--F0", type=float, default=0.04)
    parser.add_argument("--use_metal", action="store_true")

    # optimization related
    parser.add_argument("--n_iter", type=int, default=1000, help="# of iters")
    parser.add_argument(
        "--step_every", type=int, default=1, help="# of iters to accmulate gradient"
    )
    parser.add_argument("--split_size", type=int, default=20000)
    parser.add_argument("--lr_warmup_iters", type=int, default=100)
    parser.add_argument("--lr_albedo", type=float, default=0.01)
    parser.add_argument("--lr_rough", type=float, default=0.01)
    parser.add_argument("--lr_metal", type=float, default=0.01)
    parser.add_argument("--lr_lgt", type=float, default=0.01)
    parser.add_argument("--w_envobs", type=float, default=0.1)

    # misc
    parser.add_argument("--rng_seed", type=int, default=777)
    args = parser.parse_args()

    """ init environment """
    outdir = args.ckptroot / "neural_distillation"
    outdir.mkdir(exist_ok=True)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.backends.cudnn.benchmark = True
    random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    """ load model """
    print_cyan("==> Load checkpoint")
    ckpt = torch.load(
        args.ckptroot / "neural_surface_recon" / "fine_last.pth", map_location="cpu"
    )
    teacher = vol_repr.VolRepr(**ckpt["model_kwargs"])
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.cuda()
    for param in teacher.parameters():
        param.requires_grad = False

    """ compute average background observation """
    print_cyan(f"==> Compute avg bg observation")
    avg_bg = compute_avg_bg(
        pre_model=teacher,
        cfg_path=args.ckptroot / "neural_surface_recon" / "config.py",
        H=128,
        W=256,
    ).cuda()
    avg_bg_alpha = (avg_bg > 0).all(-1, keepdim=True).float().cuda()
    avg_bg_viewdirs = torch.Tensor(utils_sg.equirec_sphere(128, 256)).cuda()
    torch.save(avg_bg, outdir / "avg_bg.pth")
    cv2.imwrite(
        str(outdir / "avg_bg.exr"),
        torch.cat([avg_bg.flip(dims=(-1,)), avg_bg_alpha], -1).cpu().numpy(),
    )

    """ load unwrapped mesh """
    print_cyan("==> Load mesh")
    mesh_path = outdir / "mesh.obj"
    shutil.copy(args.ckptroot / "neural_surface_recon" / "mesh.obj", mesh_path)
    mesh = trimesh.load(mesh_path)
    uv = torch.Tensor(mesh.visual.uv)
    vert = torch.Tensor(mesh.vertices)
    vert[:, 1].clamp_min_(teacher.xyz_min[1] + teacher.voxel_size * 0.1)
    face = torch.LongTensor(mesh.faces).cuda()
    print_cyan(f"====> # verts: {len(vert):10d}")
    print_cyan(f"====> # faces: {len(face):10d}")

    # compute xyz map
    vert_uv = torch.cat([uv * 2 - 1, torch.ones_like(uv[:, [0]])], -1)
    vert_uv[..., 0] *= -2
    mesh_torch = pytorch3d.structures.Meshes(verts=vert_uv[None], faces=face[None])
    pix_to_face, zbuf, bary_coords, dists = pytorch3d.renderer.mesh.rasterize_meshes(
        mesh_torch, image_size=(1024, 2048), faces_per_pixel=1, perspective_correct=True
    )
    frag = pytorch3d.renderer.mesh.rasterizer.Fragments(
        pix_to_face, zbuf, bary_coords, dists
    )
    mesh_torch.textures = pytorch3d.renderer.TexturesVertex(verts_features=vert[None])
    tex_xyz = mesh_torch.sample_textures(frag).squeeze()
    tex_mask = (pix_to_face != -1).squeeze()
    tex_xyz[~tex_mask] = tex_xyz[tex_mask].amin(0) - 1
    n_texels = tex_mask.sum().item()
    print_cyan(f"====> # texel to optimize: {n_texels:10d}")

    """ create material/light model and optimizer """
    print_cyan("==> Create material field / envmap sg / optimizer")
    xyz = tex_xyz[tex_mask]
    normal = compute_normal(teacher, xyz)
    wo = torch.Tensor(utils_sg.fibonacci_sphere(128)).cuda()
    albedo = []
    for xyz_i, normal_i in zip(tqdm(xyz.split(8192)), normal.split(8192)):
        colors = teacher.query_radiance(
            xyz_i.repeat_interleave(len(wo), dim=0), wo.repeat(len(xyz_i), 1)
        )
        masks = (normal_i @ wo.T).unsqueeze(-1) > 0
        albedo.append(
            (colors.view(len(xyz_i), len(wo), 3) * masks).quantile(0.75, dim=1)
        )
    albedo_logit = [
        v.requires_grad_()
        for v in torch.cat(albedo)
        .clamp(1e-2, 1 - 1e-2)
        .logit()
        .unsqueeze(1)
        .split(args.split_size)
    ]
    rough_logit = [
        v.requires_grad_()
        for v in torch.full([n_texels, 1, 1], -1.0).split(args.split_size)
    ]
    metal_logit = [
        v.requires_grad_()
        for v in torch.full([n_texels, 1, 1], -4.0).split(args.split_size)
    ]

    # spherical gaussian lighting
    lgtSGs = utils_sg.Envmap2SG(
        avg_bg,
        numLgtSGs=args.n_lobe,
        N_iter=1000,
        fixed_lobe=True,
        init_sharpness=20,
        init_intensity=0.5,
    )
    lgtSGs = lgtSGs.detach().clone().requires_grad_()

    # optimizer
    optim = torch.optim.Adam(
        [
            {"params": albedo_logit, "lr": args.lr_albedo},
            {"params": rough_logit, "lr": args.lr_rough},
            {"params": metal_logit, "lr": args.lr_metal},
            {"params": lgtSGs, "lr": args.lr_lgt},
        ]
    )
    for param_group in optim.param_groups:
        param_group["base_lr"] = param_group["lr"]

    """ Pre-compute visibility and indirect illumination """
    print_cyan("==> Pre-compute visibility and indirect illumination")
    wl = torch.Tensor(utils_sg.fibonacci_sphere(args.n_samp))
    vis = torch.zeros([len(xyz), len(wl)], dtype=torch.bool)
    gi = torch.zeros([len(xyz), len(wl), 3])
    for i in trange(len(wl)):
        NoL = (normal * wl[i]).sum(-1)
        wl_i = wl[None, i].repeat_interleave(len(xyz), dim=0)
        xyz_2nd, is_hit_2nd, _ = cast2sdfzero(
            teacher,
            ro=xyz + wl_i * teacher.voxel_size,
            rd=wl_i,
            stepsize=0.25,
            render_normal=False,
        )
        vis[:, i] = (NoL > 0) & ~is_hit_2nd
        ind_idx = torch.where((NoL > 0) & is_hit_2nd)[0]
        gi[ind_idx, i] = teacher.query_radiance(xyz_2nd[ind_idx], -wl_i[ind_idx])
    NoL = (normal @ wl.T).unsqueeze(-1).clip(0, 1)  # [n_vert, n_samp, 1]
    area = 4 * np.pi / args.n_samp
    p_intergral = NoL * area  # [n_vert, n_samp, 1]

    n_chunk = len(albedo_logit)
    if n_chunk > 1:
        print_cyan(f"==> Split texels into {n_chunk} chunks to save gpu mem")

    normal = normal.unsqueeze(1).split(
        args.split_size
    )  # [[split_sz, n_sample, 3], ...]
    xyz = xyz.unsqueeze(1).split(args.split_size)  # [[split_sz, n_sample, 1], ...]
    p_intergral = p_intergral.split(args.split_size)  # [[split_sz, n_sample, 1], ...]
    vis = vis.unsqueeze(-1).split(args.split_size)  # [[split_sz, n_sample, 1], ...]
    NoL = NoL.split(args.split_size)  # [[split_sz, n_sample, 1], ...]
    gi = gi.split(args.split_size)  # [[split_sz, n_sample, 3], ...]

    """ start training """
    print_cyan("==> Start training")
    optim.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    psnr_hist = []
    for ith_iter in trange(args.n_iter):
        # lr warmup
        for param_group in optim.param_groups:
            rate = min(1, ith_iter / args.lr_warmup_iters)
            param_group["lr"] = param_group["base_lr"] * rate

        # inner loop to iterate vertices chunk
        sum_square_err = 0
        for i_chunk in range(n_chunk):
            albedo = albedo_logit[i_chunk].sigmoid()
            rough = rough_logit[i_chunk].sigmoid()
            metal = metal_logit[i_chunk].sigmoid()
            envmap_lgt = utils_sg.SG2Envmap(lgtSGs, viewdirs=wl, fixed_lobe=True)

            N_i = normal[i_chunk]
            xyz_i = xyz[i_chunk]
            p_intergral_i = p_intergral[i_chunk]
            vis_i = vis[i_chunk]
            NoL_i = NoL[i_chunk]
            gi_i = gi[i_chunk]

            # sample wo
            wo = torch.nn.functional.normalize(
                torch.randn(len(albedo), 1, 3), dim=-1
            )  # [n_vert, 1, 3]
            wo *= (wo * N_i).sum(-1, keepdim=True).sign()
            gt = teacher.query_radiance(xyz_i.view(-1, 3), wo.view(-1, 3))

            # some vectors and dots
            wh = torch.nn.functional.normalize(wl + wo, dim=-1)  # [n_vert, n_samp, 3]
            NoV = (N_i * wo).sum(-1, keepdim=True).clip(0, 1)  # [n_vert, 1, 1]
            NoH = (N_i * wh).sum(-1, keepdim=True).clip(0, 1)  # [n_vert, n_samp, 1]
            NoL = (N_i * wl).sum(-1, keepdim=True).clip(0, 1)  # [n_vert, n_samp, 1]
            VoH = (wo * wh).sum(-1, keepdim=True).clip(0, 1)  # [n_vert, n_samp, 1]

            # shading
            rough_4 = rough.pow(4)
            k = (rough + 1).pow(2) / 8
            if args.use_metal:
                ks = albedo * metal + args.F0 * (1 - metal)
                kd = albedo * (1 - metal)
            else:
                ks = args.F0
                kd = albedo
            F = ks + (1 - ks) * (
                2 ** (-5.55473 * VoH.pow(2) - 6.98316 * VoH)
            )  # [n_vert, n_samp, 1]
            D = rough_4 / (
                np.pi * (NoH.pow(2) * (rough_4 - 1) + 1).pow(2)
            )  # [n_vert, n_samp, 1]
            # G = NoV * NoL / ((NoV * (1-k) + k) * (NoL * (1-k) + k))
            # f_spec = F * D * G / (4 * NoL * NoV) * ((NoL>0) & (NoV>0) & (NoH>0))
            f_spec = (
                F * D / (4 * (NoV * (1 - k) + k) * (NoL * (1 - k) + k))
            )  # [n_vert, n_samp, 1 or 3]
            # f_spec = f_spec * ((NoL>0) & (NoV>0) & (NoH>0)) # sanity check
            f_diff = kd / np.pi
            f = f_spec + f_diff

            L_NoL_vis = p_intergral_i * (
                vis_i * envmap_lgt + gi_i
            )  # [n_vert, n_samp, 3]
            shaded_color = (L_NoL_vis * f).sum(-2).clamp_max(1)

            # photometric loss
            loss_render = (shaded_color - gt).abs().sum() / (n_texels * 3)
            loss_render.backward()
            # log
            with torch.no_grad():
                sum_square_err += (shaded_color - gt).pow(2).sum()

        # envmap regularization
        envmap_lgt = utils_sg.SG2Envmap(
            lgtSGs, H=128, W=256, viewdirs=avg_bg_viewdirs, fixed_lobe=True
        )
        loss_envobs = (
            (envmap_lgt - avg_bg).abs()
            * avg_bg_alpha
            * (1 - avg_bg.amax(-1, keepdim=True)).pow(2)
        ).mean()
        (args.w_envobs * loss_envobs).backward()

        # optimizer step
        if (ith_iter + 1) % args.step_every == 0:
            optim.step()
            optim.zero_grad(set_to_none=True)

        # log
        with torch.no_grad():
            mse = sum_square_err / (n_texels * 3)
            psnr = -10 * torch.log10(mse)
            psnr_hist.append(psnr.item())

        if ith_iter % 100 == 0 or ith_iter == args.n_iter - 1:
            print_cyan(f"==> iter={ith_iter:5d}: psnr={np.mean(psnr_hist):4.1f}")
            psnr_hist = []

    """ Export results """
    torch.cuda.empty_cache()
    print_cyan(f"==> Post-processing")

    # compute material map
    with torch.no_grad():
        tex_albedo = torch.zeros(1024, 2048, 3)
        tex_rough = torch.zeros(1024, 2048)
        tex_metal = torch.zeros(1024, 2048)
        tex_albedo[tex_mask] = torch.cat(albedo_logit).sigmoid().squeeze()
        tex_rough[tex_mask] = torch.cat(rough_logit).sigmoid().squeeze()
        tex_metal[tex_mask] = torch.cat(metal_logit).sigmoid().squeeze()

    # dilate texture map to mitigate the seam problem
    m = tex_mask[None, None].float().clone()
    tex_albedo = tex_albedo.moveaxis(-1, 0)[None]
    tex_rough = tex_rough[None, None]
    tex_metal = tex_metal[None, None]
    tex_xyz = tex_xyz.moveaxis(-1, 0)[None]
    for _ in range(10):
        tex_albedo = m * tex_albedo + (1 - m) * torch.nn.functional.max_pool2d(
            tex_albedo, kernel_size=3, stride=1, padding=1
        )
        tex_rough = m * tex_rough + (1 - m) * torch.nn.functional.max_pool2d(
            tex_rough, kernel_size=3, stride=1, padding=1
        )
        tex_metal = m * tex_metal + (1 - m) * torch.nn.functional.max_pool2d(
            tex_metal, kernel_size=3, stride=1, padding=1
        )
        tex_xyz = m * tex_xyz + (1 - m) * torch.nn.functional.max_pool2d(
            tex_xyz, kernel_size=3, stride=1, padding=1
        )
        m = torch.nn.functional.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    tex_albedo = tex_albedo.squeeze(0).moveaxis(0, -1)
    tex_rough = tex_rough.squeeze()
    tex_metal = tex_metal.squeeze()
    tex_xyz = tex_xyz.squeeze(0).moveaxis(0, -1)
    tex_mask = m.squeeze() > 0

    # compute env map
    with torch.no_grad():
        envmap_lgt = utils_sg.SG2Envmap(lgtSGs, H=128, W=256, viewdirs=avg_bg_viewdirs)

    # some more visualization
    viz_tex_xyz = (tex_xyz - vert.amin(0)) / (vert.amax(0) - vert.amin(0))
    viz_tex_xyz[~tex_mask] = 0
    viz_vis = torch.zeros(1024, 2048)
    with torch.no_grad():
        normal_flat = compute_normal(teacher, tex_xyz[tex_mask])
        viz_tex_normal = torch.full([1024, 2048, 3], 0.5)
        viz_tex_normal[tex_mask] = (normal_flat * 0.5 + 0.5).clip(0, 1)
        viz_tex_teacher = torch.zeros([1024, 2048, 3])
        viz_tex_teacher[tex_mask] = lin2srgb(
            teacher.query_radiance(tex_xyz[tex_mask], normal_flat)
        )
        viz_vis[(pix_to_face != -1).squeeze()] = (
            torch.cat(vis).float().mean(1).squeeze()
        )

    # export results
    print_cyan(f"==> Save to {outdir}/")
    iio.imwrite(outdir / "mask.png", tex_mask.cpu().numpy())
    write_exr(outdir / "albedo.exr", tex_albedo.cpu().numpy())
    write_exr(outdir / "roughness.exr", tex_rough.cpu().numpy())
    iio.imwrite(
        outdir / "metallic.png", tex_metal.mul(255).cpu().numpy().astype(np.uint8)
    )
    write_exr(outdir / "envmap.exr", envmap_lgt.cpu().numpy())
    iio.imwrite(
        outdir / "viz_xyz.png", viz_tex_xyz.mul(255).cpu().numpy().astype(np.uint8)
    )
    iio.imwrite(
        outdir / "viz_normal.png",
        viz_tex_normal.mul(255).cpu().numpy().astype(np.uint8),
    )
    iio.imwrite(
        outdir / "viz_teacher.png",
        viz_tex_teacher.mul(255).cpu().numpy().astype(np.uint8),
    )
    iio.imwrite(outdir / "viz_vis.png", viz_vis.mul(255).cpu().numpy().astype(np.uint8))
