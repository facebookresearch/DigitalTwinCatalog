# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .utils import Block, trunc_normal_


class TriplaneMlpUpsampler(nn.Module):
    def __init__(
        self,
        repeat_factor,
        embed_dim=1024,
        output_dim=32,
        separate_feature=False,
        separate_dims=None,
        use_weight_norm=False,
        use_view_embed=False,
        view_output_dim=32,
        view_repeat_factor=4,
    ):
        super().__init__()
        # deconv for original 64x64 resolution
        if separate_dims is None:
            separate_dims = []
            separate_dims.append(output_dim // 2)
            separate_dims.append(output_dim // 2)

        if separate_feature:
            if separate_dims[0] + separate_dims[1] != output_dim:
                raise ValueError(
                    "The sum of two separate dims should be the same as output dim"
                )

        self.separate_feature = separate_feature
        self.separate_dims = separate_dims
        self.output_dim = output_dim
        self.repeat_factor = repeat_factor

        if separate_feature:
            self.dconv2d_geo = nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=separate_dims[0],
                kernel_size=repeat_factor,
                stride=repeat_factor,
            )
            self.dconv2d_app = nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=separate_dims[1],
                kernel_size=repeat_factor,
                stride=repeat_factor,
            )
        else:
            # deconv for original 64x64 resolution
            self.dconv2d = nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=output_dim,
                kernel_size=repeat_factor,
                stride=repeat_factor,
            )

        if use_weight_norm:
            if separate_feature:
                self.dconv2d_geo = nn.utils.parametrizations.weight_norm(
                    self.dconv2d_geo
                )
                self.dconv2d_app = nn.utils.parametrizations.weight_norm(
                    self.dconv2d_app
                )
            else:
                self.dconv2d = nn.utils.parametrizations.weight_norm(self.dconv2d)

        self.use_view_embed = use_view_embed
        if use_view_embed:
            self.view_output_dim = view_output_dim
            self.view_repeat_factor = view_repeat_factor
            self.dconv2d_view = nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=view_output_dim,
                kernel_size=view_repeat_factor,
                stride=view_repeat_factor,
            )
            if use_weight_norm:
                self.dconv2d_view = nn.utils.parametrizations.weight_norm(
                    self.dconv2d_view
                )

    def upsample_one_plane(self, plane, dconv2d):
        out = dconv2d(plane)
        return out

    def upsample_plane(self, plane_xy, plane_xz, plane_yz, dconv2d):
        out_xy = self.upsample_one_plane(plane_xy, dconv2d)
        out_xz = self.upsample_one_plane(plane_xz, dconv2d)
        out_yz = self.upsample_one_plane(plane_yz, dconv2d)
        return out_xy, out_xz, out_yz

    def forward(self, plane_xy, plane_xz, plane_yz, plane_view=None):
        if self.separate_feature:
            plane_xy_geo, plane_xz_geo, plane_yz_geo = self.upsample_plane(
                plane_xy,
                plane_xz,
                plane_yz,
                self.dconv2d_geo,
            )

            plane_xy_app, plane_xz_app, plane_yz_app = self.upsample_plane(
                plane_xy,
                plane_xz,
                plane_yz,
                self.dconv2d_app,
            )
            plane_xy = torch.cat([plane_xy_geo, plane_xy_app], dim=1)
            plane_xz = torch.cat([plane_xz_geo, plane_xz_app], dim=1)
            plane_yz = torch.cat([plane_yz_geo, plane_yz_app], dim=1)
        else:
            plane_xy, plane_xz, plane_yz = self.upsample_plane(
                plane_xy, plane_xz, plane_yz, self.dconv2d
            )

        if self.use_view_embed:
            plane_view = self.upsample_one_plane(
                plane_view,
                self.dconv2d_view,
            )
            return plane_xy, plane_xz, plane_yz, plane_view
        else:
            return plane_xy, plane_xz, plane_yz


class TriplaneTransformer(nn.Module):
    def __init__(
        self,
        triplane_size=32,
        triplane_out_size=64,
        embed_dim=1024,
        output_dim=32,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        cp_freq=1,
        separate_feature=False,
        separate_dims=None,
        use_weight_norm=False,
        double_triplane=False,
        use_view_embed=False,
        view_embed_size=32,
        view_embed_out_size=128,
        view_output_dim=32,
        no_upsampler=False,
        attn_type="default",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.triplane_size = triplane_size
        self.triplane_out_size = triplane_out_size
        self.double_triplane = double_triplane
        if self.double_triplane:
            self.num_token = triplane_size * triplane_size * 3 * 2
            self.plane_size = triplane_size * triplane_size * 2
        else:
            self.num_token = triplane_size * triplane_size * 3
            self.plane_size = triplane_size * triplane_size

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_token, embed_dim)
            * 1
            / math.sqrt(float(self.plane_size))
        )

        self.use_view_embed = use_view_embed
        if use_view_embed:
            self.view_embed_size = view_embed_size
            self.view_embed_out_size = view_embed_out_size
            self.num_view_token = view_embed_size * view_embed_size
            self.view_embed = nn.Parameter(
                torch.randn(1, self.num_view_token, embed_dim)
                * 1
                / math.sqrt(float(self.plane_size))  # Keep the same magnitude?
            )

        self.attn_type = attn_type
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        block_list = []
        for i in range(depth):
            if attn_type == "default":
                block_list.append(
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        norm_layer=norm_layer,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        use_weight_norm=use_weight_norm,
                        norm_bias=True,
                    )
                )

        self.cross_blocks = nn.ModuleList(block_list)
        self.norm = norm_layer(embed_dim)

        repeat_factor = int(triplane_out_size // triplane_size)
        view_repeat_factor = int(view_embed_out_size // view_embed_size)

        self.no_upsampler = no_upsampler
        if not no_upsampler:
            self.upsampler = TriplaneMlpUpsampler(
                repeat_factor=repeat_factor,
                embed_dim=embed_dim,
                output_dim=output_dim,
                separate_feature=separate_feature,
                separate_dims=separate_dims,
                use_weight_norm=use_weight_norm,
                use_view_embed=use_view_embed,
                view_output_dim=view_output_dim,
                view_repeat_factor=view_repeat_factor,
            )

        self.apply(self._init_weights)
        self.cp_freq = int(cp_freq)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def decode_plane(self, plane, plane_size, double_plane):
        plane = plane.permute(0, 2, 1)
        if double_plane:
            plane = plane.reshape(-1, self.embed_dim, plane_size, plane_size * 2)
        else:
            plane = plane.reshape(-1, self.embed_dim, plane_size, plane_size)
        return plane

    def forward(self, y):
        batch_size = y.shape[0]
        x = self.pos_embed.repeat(batch_size, 1, 1)
        if self.use_view_embed:
            x_view = self.view_embed.repeat(batch_size, 1, 1)
            x = torch.cat([x, x_view], dim=1)

        for idx in range(0, len(self.cross_blocks)):
            blk = self.cross_blocks[idx]
            if self.cp_freq > 0 and idx % self.cp_freq == 0:
                x_token_num = x.shape[1]
                y_token_num = y.shape[1]
                x = torch.cat([x, y], dim=1)
                x = cp.checkpoint(blk, x, use_reentrant=False)
                x, y = torch.split(x, [x_token_num, y_token_num], dim=1)
            else:
                x_token_num = x.shape[1]
                y_token_num = y.shape[1]
                x = torch.cat([x, y], dim=1)
                x = blk(x)
                x, y = torch.split(x, [x_token_num, y_token_num], dim=1)
        x = self.norm(x)
        y = self.norm(y)

        # triplane at 64x64 resolution
        if self.use_view_embed:
            x_xy, x_xz, x_yz, x_view = torch.split(
                x,
                [
                    self.plane_size,
                    self.plane_size,
                    self.plane_size,
                    self.num_view_token,
                ],
                dim=1,
            )
            plane_xy = self.decode_plane(x_xy, self.triplane_size, self.double_triplane)
            plane_xz = self.decode_plane(x_xz, self.triplane_size, self.double_triplane)
            plane_yz = self.decode_plane(x_yz, self.triplane_size, self.double_triplane)
            plane_view = self.decode_plane(x_view, self.view_embed_size, False)

            if not self.no_upsampler:
                plane_xy, plane_xz, plane_yz, plane_view = self.upsampler(
                    plane_xy, plane_xz, plane_yz, plane_view
                )
            return plane_xy, plane_xz, plane_yz, plane_view, y
        else:
            x_xy, x_xz, x_yz = torch.split(
                x, [self.plane_size, self.plane_size, self.plane_size], dim=1
            )
            plane_xy = self.decode_plane(x_xy, self.triplane_size, self.double_triplane)
            plane_xz = self.decode_plane(x_xz, self.triplane_size, self.double_triplane)
            plane_yz = self.decode_plane(x_yz, self.triplane_size, self.double_triplane)
            if not self.no_upsampler:
                plane_xy, plane_xz, plane_yz = self.upsampler(
                    plane_xy, plane_xz, plane_yz
                )
            return plane_xy, plane_xz, plane_yz, None, y
