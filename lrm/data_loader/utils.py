# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np


def linear_to_srgb(l):
    s = np.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055 * (l[~m] ** (1.0 / 2.4)) - 0.055
    return s


def srgb_to_linear(s):
    l = np.zeros_like(s)
    m = s <= 0.0404482362771082
    l[m] = s[m] / 12.92
    l[~m] = ((s[~m] + 0.055) / 1.055) ** 2.4
    return l


def compute_rays(fov, matrix, res):
    if isinstance(res, int):
        res = (res, res)

    matrix = np.array(matrix)
    rays_o = np.zeros((res[0], res[1], 3), dtype=np.float32) + matrix[0:3, 3].reshape(
        1, 1, 3
    )

    x_axis = np.linspace(0.5, res[1] - 0.5, res[1]) / res[1]
    x_axis = 2 * x_axis - 1
    y_axis = np.linspace(0.5, res[0] - 0.5, res[0]) / res[0]
    y_axis = 2 * y_axis - 1

    x, y = np.meshgrid(x_axis, y_axis)
    x = x * np.tan(fov / 2.0)
    y = -y * np.tan(fov / 2.0) / res[1] * res[0]
    uv = np.stack([x, y], axis=-1)

    z = -np.ones((res[0], res[1]))
    rays_d_un = np.stack([x, y, z], axis=-1)
    rays_d = rays_d_un / np.linalg.norm(rays_d_un, axis=-1)[:, :, None]
    rot = matrix[0:3, 0:3][None, None, :, :]
    rays_d_un = np.sum(rot * rays_d_un[:, :, None, :], axis=-1)
    rays_d = np.sum(rot * rays_d[:, :, None, :], axis=-1)

    return rays_o, rays_d, rays_d_un, uv


def load_one_frame(
    fov, im=None, c2w=None, image_res=512, hdr_to_ldr=False, normalize=True
):
    im, mask = load_one_image(
        im=im, image_res=image_res, normalize=normalize, hdr_to_ldr=hdr_to_ldr
    )
    rays_o, rays_d, rays_d_un, uv = compute_rays(fov, c2w, image_res)

    camera_ext = np.array(c2w).reshape(16)
    camera_int = np.array([fov, fov, 0.5, 0.5])
    camera = np.concatenate([camera_ext, camera_int])

    return (
        im.astype(np.float32),
        rays_o.astype(np.float32),
        rays_d.astype(np.float32),
        camera.astype(np.float32),
        mask.astype(np.float32) if mask is not None else None,
        rays_d_un.astype(np.float32),
        uv.astype(np.float32),
    )


def load_one_image(
    im, image_res=512, normalize=False, hdr_to_ldr=False, ldr_to_hdr=False
):
    if isinstance(im, str):
        with open(im, "rb") as fIn:
            buffer = fIn.read()
            buffer = np.asarray(bytearray(buffer), dtype=np.uint8)
            im = cv2.imdecode(buffer, -1)
            if len(im.shape) == 3:
                if im.shape[2] == 3:
                    im = np.ascontiguousarray(im[:, :, ::-1])
                else:
                    tmp_im = im[:, :, :3]
                    tmp_im = np.ascontiguousarray(im[:, :, ::-1])
                    tmp_mask = im[:, :, 3:4]
                    im = np.concatenate([tmp_im, tmp_mask], axis=-1)
            else:
                im = np.stack(
                    [im, im, im, im], axis=-1
                )  # Important to load mask with 1 channel

    if im.dtype != np.float32:
        if im.dtype == np.uint16:
            im = im.astype(np.float32) / 65535
        else:
            im = im.astype(np.float32) / 255.0

    if hdr_to_ldr:
        im = linear_to_srgb(im)
    if ldr_to_hdr:
        im = srgb_to_linear(im)

    if image_res is None:
        image_res = im.shape[0:2]
    elif isinstance(image_res, int):
        image_res = (image_res, image_res)

    if len(im.shape) == 2:
        im = np.stack([im, im, im, im], axis=-1)

    if im.shape[2] == 4:
        mask = im[:, :, 3:4]
        im = im[:, :, 0:3] + (1.0 - mask)
        im = np.clip(im, 0, 1)
        mask = cv2.resize(
            mask[:, :, 0], (image_res[1], image_res[0]), interpolation=cv2.INTER_AREA
        )
        mask = mask[None, :, :]
    else:
        mask = None

    if normalize:
        im = 2 * im - 1
    im = cv2.resize(im, (image_res[1], image_res[0]), interpolation=cv2.INTER_AREA)
    im = im.transpose(2, 0, 1)

    return im, mask


def load_specular(im, image_res):
    im, _ = load_one_image(im, image_res, normalize=True, ldr_to_hdr=False)
    metallic = im[0:1, :]
    roughness = im[1:2, :]
    return roughness, metallic


def load_depth(depth, image_res):
    depth = cv2.resize(depth, (image_res, image_res), interpolation=cv2.INTER_AREA)
    if len(depth.shape) == 3:
        depth = depth.transpose(2, 0, 1)
        depth = depth[0:1, :, :]
    elif len(depth.shape) == 2:
        depth = depth[None, :, :]
    return depth


def load_envmap(env, width, height):
    orig_height, orig_width = env.shape[0:2]
    env = np.ascontiguousarray(env[:, :, ::-1])

    env_mask = np.ones((orig_height, orig_width), dtype=np.float32)
    check_height = int(
        orig_height * 0.85
    )  # Hard coded a threshold as part of env is missing
    env_mask[check_height:, :] = np.max(env[check_height:, :, :], axis=2) > 0.05
    env_mask = env_mask[:, :, None]

    if orig_height != height and orig_width != width:
        theta = np.linspace(0, orig_height - 1, orig_height) + 0.5
        weight = np.sin(theta / orig_height * np.pi)[:, None, None]
        weight = np.tile(weight, [1, orig_width, 1])

        env = cv2.resize(env * weight, (width, height), interpolation=cv2.INTER_AREA)
        env_mask = cv2.resize(
            env_mask * weight, (width, height), interpolation=cv2.INTER_AREA
        )
        weight = cv2.resize(weight, (width, height), interpolation=cv2.INTER_AREA)

        env_mask = env_mask.reshape(height, width, 1)
        weight = weight.reshape(height, width, 1)

        env = env / np.maximum(weight, 1e-6)
        env_mask = env_mask / np.maximum(weight, 1e-6)

    env = env.transpose(2, 0, 1)
    env_mask = env_mask.transpose(2, 0, 1)
    return env, env_mask


def transform_cams(cam, cam_rot):
    cam_rot = cam_rot.transpose(1, 0)
    ext_orig = cam[:16].reshape(4, 4)
    inv = np.eye(4).astype(cam.dtype)
    inv[0:3, 0:3] = cam_rot
    ext = np.matmul(inv, ext_orig)
    cam[:16] = ext.reshape(16)
    return cam


def transform_rays(rays, cam_rot):
    rays = rays[:, :, :, None]
    cam_rot = cam_rot.reshape(1, 1, 3, 3)
    rays = np.sum(cam_rot * rays, axis=-2)
    return rays


def transform_normal(normal, mask, cam_rot):
    normal = normal[:, None, :, :]
    cam_rot = cam_rot[:, :, None, None]
    normal = np.sum(normal * cam_rot, axis=0)
    normal = normal * mask + (1 - mask)
    normal = np.clip(normal, -1, 1)
    return normal


def importance_selection(mask, crop_size, offset=0.001):
    mask = mask.squeeze() + offset
    height, width = mask.shape
    m_row = np.sum(mask, axis=1)
    m_col = np.sum(mask, axis=0)
    m_row = np.concatenate([np.zeros(1, dtype=m_row.dtype), m_row])
    m_col = np.concatenate([np.zeros(1, dtype=m_row.dtype), m_col])

    m_row_acc = np.cumsum(m_row)
    m_col_acc = np.cumsum(m_col)
    m_row_int = m_row_acc[crop_size:] - m_row_acc[:-crop_size]
    m_col_int = m_col_acc[crop_size:] - m_col_acc[:-crop_size]

    m_row_cdf = np.cumsum(m_row_int)
    m_col_cdf = np.cumsum(m_col_int)

    m_row_cdf = m_row_cdf / np.maximum(np.max(m_row_cdf), 1e-6)
    m_col_cdf = m_col_cdf / np.maximum(np.max(m_col_cdf), 1e-6)

    row_rand = np.random.random()
    col_rand = np.random.random()

    hs = np.searchsorted(m_row_cdf, row_rand, side="right")
    he = hs + crop_size

    ws = np.searchsorted(m_col_cdf, col_rand, side="right")
    we = ws + crop_size

    return hs, he, ws, we


def crop_and_resize(im, hs, he, ws, we, height, width):
    im = im[hs:he, ws:we]
    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
    return im


def compute_cropping_from_mask(mask):
    height, width = mask.shape

    if np.sum(mask) == 0:
        return 0, height, 0, width

    mask_row = np.sum(mask, axis=0)
    mask_col = np.sum(mask, axis=1)

    m_row_nonzero = np.nonzero(mask_row)[0]
    m_row_s = m_row_nonzero.min()
    m_row_e = min(m_row_nonzero.max() + 1, width)
    m_row_len = m_row_e - m_row_s
    m_row_center = (m_row_s + m_row_e) // 2

    m_col_nonzero = np.nonzero(mask_col)[0]
    m_col_s = m_col_nonzero.min()
    m_col_e = min(m_col_nonzero.max() + 1, height)
    m_col_len = m_col_e - m_col_s
    m_col_center = (m_col_s + m_col_e) // 2

    if m_col_len > m_row_len:
        hs = m_col_s
        he = m_col_e
        sq_size = m_col_len

        ws = m_row_center - sq_size // 2
        we = ws + sq_size

        if ws < 0:
            ws = 0
            we = sq_size
        elif we > width:
            we = width
            ws = we - sq_size
    else:
        ws = m_row_s
        we = m_row_e
        sq_size = m_row_len

        hs = m_col_center - sq_size // 2
        he = hs + sq_size

        if hs < 0:
            hs = 0
            he = sq_size
        elif he > height:
            he = height
            hs = height - sq_size
    return hs, he, ws, we
