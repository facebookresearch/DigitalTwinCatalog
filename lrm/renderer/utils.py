# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def validate_empty_rays(ray_indices, t_start, t_end):
    if ray_indices.nelement() == 0:
        ray_indices = torch.LongTensor([0]).to(ray_indices)
        t_start = torch.Tensor([0]).to(ray_indices)
        t_end = torch.Tensor([0]).to(ray_indices)
    return ray_indices, t_start, t_end


def render_flash(
    origin,
    points,
    views,
    normals,
    basecolor,
    roughness,
    metallic,
    opacity,
    intensity=10,
):
    # origin:       [B, I, H, W, 3]
    # points:       [B, I, H, W, 3]
    # views:        [B, I, H, W, 3]
    # basecolor:    [B, I, H, W, 3]
    # roughness:    [B, I, H, W, 1]
    # metallic:     [B, I, H, W, 1]
    # Use spherical Gaussian approximation, for flashlight F = sc as vh=1

    basecolor = 0.5 * (basecolor + 1)
    roughness = 0.5 * (roughness + 1)
    metallic = 0.5 * (metallic + 1)
    roughness = torch.clamp(roughness, min=0.05)

    dc = basecolor * (1 - metallic)
    sc = basecolor * metallic

    k = (roughness + 1) * (roughness + 1) / 8.0
    alpha = roughness * roughness
    ndv = torch.sum(-views * normals, dim=-1, keepdim=True)
    ndh = ndv

    sc_term = alpha / 2 / (ndh * ndh * (alpha * alpha - 1) + 1) / (ndv * (1 - k) + k)
    brdf = 1 / np.pi * (dc + sc * sc_term * sc_term)

    dist2 = torch.sum((origin - points) * (origin - points), dim=-1, keepdim=True)
    intensity = brdf * intensity / torch.clamp(dist2, min=1e-2)
    intensity = torch.clamp(intensity * opacity + (1 - opacity), 0, 1)
    intensity = 2 * intensity - 1

    return intensity
