# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Some utility functions.
"""

import zipfile

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from .masked_adam import MaskedAdam


""" Misc
"""
mse2psnr = lambda x: -10.0 * torch.log10(x)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


class HuberLoss(torch.nn.Module):
    def __init__(self, c_init=0.5, momentum=0.99, c_min=1 / 255):
        super(HuberLoss, self).__init__()
        self.c = c_init
        self.c_min = c_min
        self.momentum = momentum
        self.tick = 0

    def forward(self, x, y, update_c=False, fg_mask=None):
        diff = (x - y).abs().mean(-1)
        l2_term = (x - y).pow(2).mean(-1)
        l1_term = 2 * self.c * diff - self.c**2
        loss = torch.where(diff < self.c, l2_term, l1_term)
        if update_c:
            assert fg_mask is not None
            if fg_mask.sum() > 100:
                diff = diff.detach()[fg_mask]
                new_c = diff.median()
                self.c = self.c * self.momentum + new_c * (1 - self.momentum)
                self.c = max(self.c, self.c_min)
        return loss


""" Optimizer utils
"""


def create_optimizer_or_freeze_model(model, cfg_train, trainiter):
    decay_factor = cfg_train.lrate_decay_factor ** (trainiter / cfg_train.N_iters)
    geo_decay_factor = cfg_train.lrate_geo_decay_factor ** (
        trainiter / cfg_train.N_iters
    )

    param_group = []
    for ith_bg in range(model.num_bg_scale + 1):
        for k in cfg_train.keys():
            if not k.startswith("lrate_"):
                continue
            k = k[len("lrate_") :]

            root = model
            name = k
            for _ in range(ith_bg):
                root = getattr(root, "bg_model")
                name = f"bg_model.{name}"

            if not hasattr(root, k):
                continue

            param = getattr(root, k)
            if param is None:
                print(f"create_optimizer_or_freeze_model: param {name} not exist")
                continue

            lr = getattr(cfg_train, f"lrate_{k}") * decay_factor
            if ith_bg > 0:
                if hasattr(cfg_train, f"bg_lrate_{k}"):
                    lr = getattr(cfg_train, f"bg_lrate_{k}") * decay_factor
            if ith_bg == 0 and k == "density":
                lr = getattr(cfg_train, f"lrate_{k}") * geo_decay_factor

            if lr > 0:
                print(f"create_optimizer_or_freeze_model: param {name} lr {lr}")
                if (k == "density" or k == "k0") and ith_bg == 0:
                    param_grid = []
                    param_mlp = []
                    for n, p in param.named_parameters():
                        if "mlp" in n:
                            param_mlp.append(p)
                        else:
                            param_grid.append(p)
                    param_group.append(
                        {
                            "params": param_grid,
                            "lr": lr,
                            "skip_zero_grad": (k in cfg_train.skip_zero_grad_fields),
                            "root": root,
                            "k": k,
                            "is_fg": True,
                        }
                    )
                    if len(param_mlp) > 0:
                        print(
                            f"create_optimizer_or_freeze_model: param {name}-mlp lr {cfg_train.lrate_rgbnet}"
                        )
                        param_group.append(
                            {
                                "params": param_mlp,
                                "lr": cfg_train.lrate_rgbnet,
                                "skip_zero_grad": (
                                    k in cfg_train.skip_zero_grad_fields
                                ),
                                "root": root,
                                "k": k,
                                "is_fg": True,
                            }
                        )
                else:
                    if isinstance(param, nn.Module):
                        param = param.parameters()
                    param_group.append(
                        {
                            "params": param,
                            "lr": lr,
                            "skip_zero_grad": (k in cfg_train.skip_zero_grad_fields),
                            "root": root,
                            "k": k,
                            "is_fg": (ith_bg == 0),
                        }
                    )
            else:
                print(f"create_optimizer_or_freeze_model: param {name} freeze")
                param.requires_grad = False
    return MaskedAdam(
        params=param_group,
        betas=cfg_train.optim_betas,
        eps=cfg_train.optim_eps,
        warmup_iter=cfg_train.lrate_warmup_iter,
    )


""" Checkpoint utils
"""


def load_checkpoint(model, optimizer, ckpt_path, reload_optimizer):
    ckpt = torch_load(ckpt_path)
    start = ckpt["trainiter"]
    model.load_state_dict(ckpt["model_state_dict"])
    if reload_optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, start


def load_model(model_class, ckpt_path):
    ckpt = torch_load(ckpt_path)
    model = model_class(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def torch_save(state_dict, path, *args, **kwargs):
    if str(path).endswith("zip"):
        with zipfile.ZipFile(
            path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as zip_f:
            with zip_f.open("ckpt.pt", "w") as f:
                torch.save(state_dict, f, *args, **kwargs)
    else:
        torch.save(state_dict, path, *args, **kwargs)


def torch_load(path, *args, **kwargs):
    if str(path).endswith("zip"):
        with zipfile.ZipFile(
            path, "r", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as zip_f:
            with zip_f.open("ckpt.pt", "r") as f:
                ckpt = torch.load(f, *args, **kwargs)
    else:
        ckpt = torch.load(path, *args, **kwargs)
    return ckpt


""" Evaluation metrics (ssim, lpips)
"""


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    filt_fn = lambda z: np.stack(
        [
            convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
            for i in range(z.shape[-1])
        ],
        -1,
    )
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    import lpips

    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


""" Mesh extraction
"""


def mesh_keep_largest_cc(mesh):
    mesh_cc = mesh.split(only_watertight=False)
    max_id = np.argmax([len(m.vertices) for m in mesh_cc])
    mesh = mesh_cc[max_id]
    print(f"Removed {len(mesh_cc) - 1} isolated connected components.")
    return mesh


def extract_mesh(sdf_grid, isovalue=0, xyz_min=None, xyz_max=None, cleanup=True):
    import mcubes
    import trimesh

    vertices, triangles = mcubes.marching_cubes(-sdf_grid, -isovalue)
    if xyz_min is None:
        vmin = vertices.min(0)
        vmax = vertices.max(0)
        R = (vmax - vmin).max() * 0.5
        c = (vmax + vmin) * 0.5
        vertices = (vertices - c) / R
    else:
        # normalized to [0, 1]
        vertices /= np.array(sdf_grid.shape) - 1
        # transform to the original coordiante
        vertices = vertices * (xyz_max - xyz_min) + xyz_min

    mesh = trimesh.Trimesh(vertices, triangles)

    if cleanup:
        mesh = mesh_keep_largest_cc(mesh)
    return mesh


def DTU_clean_mesh_by_mask(mesh, datadir):
    print("Running DTU_clean_mesh_by_mask...")

    import glob

    import cv2
    import pytorch3d
    import pytorch3d.renderer
    import pytorch3d.structures
    import trimesh
    from tqdm import trange

    verts = np.copy(mesh.vertices[:])
    faces = np.copy(mesh.faces[:])
    cameras = np.load(f"{datadir}/cameras_sphere.npz")
    mask_lis = sorted(glob.glob(f"{datadir}/mask/*.png"))

    n_images = len(mask_lis)
    mask = np.ones(len(verts), dtype=bool)
    for i in trange(n_images):
        P = cameras[f"world_mat_{i}"]
        pts_image = (
            np.matmul(P[None, :3, :3], verts[:, :, None]).squeeze() + P[None, :3, 3]
        )
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1
        mask_image = cv2.imread(mask_lis[i])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
        mask_image = cv2.dilate(mask_image, kernel, iterations=1)
        mask_image = mask_image[:, :, 0] > 128
        mask_image = np.concatenate(
            [np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0
        )
        mask_image = np.concatenate(
            [np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1
        )
        curr_mask = mask_image[
            (pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))
        ]
        mask &= curr_mask.astype(bool)

    print("Valid vertices ratio:", mask.mean())

    indexes = np.full(len(verts), -1, dtype=np.int64)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
    new_faces = faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = verts[np.where(mask)]

    mesh = trimesh.Trimesh(new_vertices, new_faces)
    mesh = mesh_keep_largest_cc(mesh)
    return mesh


def create_mesh_two_plane_box(plane_C, plane_L, N_pt=128 * 128):
    import trimesh

    H, W = plane_L[[2, 0]]
    l = (H * W / N_pt) ** 0.5
    H = int(H / l)
    W = int(W / l)

    x, y = np.meshgrid(
        np.linspace(-0.5, 0.5, W), np.linspace(-0.5, 0.5, H), indexing="xy"
    )
    i, j = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    idx = i * W + j
    off = H * W

    v = plane_C + plane_L * np.concatenate(
        [
            np.stack([x.flatten(), np.full([H * W], 0.5), y.flatten()], -1),  # top
            np.stack([x.flatten(), np.full([H * W], -0.5), y.flatten()], -1),  # btn
        ]
    )
    f_top = np.concatenate(
        [
            np.stack([idx[:-1, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3),
            np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3),
        ]
    )
    f_btn = off + np.flip(f_top, axis=1)
    f_T = np.concatenate(
        [
            np.stack([idx[0, :-1], idx[0, 1:], off + idx[0, 1:]], -1).reshape(-1, 3),
            np.stack([idx[0, :-1], off + idx[0, 1:], off + idx[0, :-1]], -1).reshape(
                -1, 3
            ),
        ]
    )
    f_R = np.concatenate(
        [
            np.stack([idx[:-1, -1], idx[1:, -1], off + idx[1:, -1]], -1).reshape(-1, 3),
            np.stack([idx[:-1, -1], off + idx[1:, -1], off + idx[:-1, -1]], -1).reshape(
                -1, 3
            ),
        ]
    )
    f_B = np.concatenate(
        [
            np.stack([idx[-1, :-1], off + idx[-1, 1:], idx[-1, 1:]], -1).reshape(-1, 3),
            np.stack([idx[-1, :-1], off + idx[-1, :-1], off + idx[-1, 1:]], -1).reshape(
                -1, 3
            ),
        ]
    )
    f_L = np.concatenate(
        [
            np.stack([idx[:-1, 0], off + idx[1:, 0], idx[1:, 0]], -1).reshape(-1, 3),
            np.stack([idx[:-1, 0], off + idx[:-1, 0], off + idx[1:, 0]], -1).reshape(
                -1, 3
            ),
        ]
    )
    f = np.concatenate([f_top, f_btn, f_T, f_R, f_B, f_L])

    return trimesh.Trimesh(v, f)
