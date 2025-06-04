# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn

from .utils import PatchEmbed, PatchEmbedPlucker, zero_module


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


class MultiviewTransformerPlucker(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        norm_layer=nn.LayerNorm,
        with_bg=False,
        with_geometry=False,
        feature_dim=None,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedPlucker(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        if with_bg:
            self.patch_embed_bg = PatchEmbed(
                patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
            )
            zero_module(self.patch_embed_bg)
        if with_geometry:
            self.patch_embed_normal = PatchEmbed(
                patch_size=patch_size, in_chans=3, embed_dim=embed_dim
            )
            zero_module(self.patch_embed_normal)

            self.patch_embed_depth = PatchEmbed(
                patch_size=patch_size, in_chans=1, embed_dim=embed_dim
            )
            zero_module(self.patch_embed_depth)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if feature_dim is not None:
            self.input_feature_proj = nn.Linear(
                feature_dim + embed_dim, embed_dim, bias=False
            )

        self.norm = norm_layer(embed_dim)

    def prepare_tokens(self, x, plucker_rays, x_bg=None, x_normal=None, x_depth=None):
        B, I, _, w, h = x.shape
        x = x.reshape(B * I, -1, w, h)
        plucker_rays = plucker_rays.reshape(B * I, -1, w, h)
        x = self.patch_embed(x, plucker_rays)  # patch linear embedding
        if x_bg is not None:
            x_bg = x_bg.reshape(B * I, -1, w, h)
            x_bg = self.patch_embed_bg(x_bg)
            x = x + x_bg
        if x_normal is not None:
            x_normal = x_normal.reshape(B * I, -1, w, h)
            x_normal = self.patch_embed_normal(x_normal)
            x = x + x_normal
        if x_depth is not None:
            x_depth = x_depth.reshape(B * I, -1, w, h)
            x_depth = self.patch_embed_depth(x_depth)
            x = x + x_depth
        x = x.reshape(B, -1, self.embed_dim)
        return x

    def forward(
        self, x, plucker_rays, x_bg=None, x_normal=None, x_depth=None, feature=None
    ):
        x = self.prepare_tokens(x, plucker_rays, x_bg, x_normal, x_depth)
        if feature is not None:
            x = self.input_feature_proj(torch.cat([feature, x], dim=-1))
        x = self.norm(x)
        return x


def mvencoder_base(
    type,
    patch_size=16,
    with_bg=False,
    embed_dim=768,
    feature_dim=None,
):
    if "plucker" in type:
        model = MultiviewTransformerPlucker(
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            with_bg=with_bg,
            with_geometry=(type == "plucker_geometry"),
            feature_dim=feature_dim,
        )
    return model
