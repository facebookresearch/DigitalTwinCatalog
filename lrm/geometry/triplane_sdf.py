# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from .utils import build_pytorch_mlp, grid_sample_2d, grid_sample_3d, sh_decode


class TriplaneSdf(nn.Module):
    def __init__(
        self,
        dim=32,
        input_dim=96,
        input_view_dim=32,
        rgb_depth=3,
        sh_depth=2,
        sdf_depth=2,
        normal_depth=3,
        brdf_depth=3,
        rho_depth=2,
        radius=0.5,
        grid_sample_mode="original",
        view_dependent_type="none",
        prediction_type="rgb",
        sdf_bias=True,
        normal_bias=True,
        sep_geo_app=False,
        sep_dims=None,
        chunk_size=1048576,
        double_triplane=False,
        compute_normal=False,
    ):
        super().__init__()
        # Define the network
        self.dim = dim
        self.input_dim = input_dim
        self.input_view_dim = input_view_dim
        self.sdf_depth = sdf_depth
        self.rgb_depth = rgb_depth
        self.brdf_depth = brdf_depth
        self.rho_depth = rho_depth
        self.radius = radius
        self.prediction_type = prediction_type
        self.sdf_bias = sdf_bias
        self.normal_bias = normal_bias
        self.chunk_size = chunk_size
        self.sep_geo_app = sep_geo_app
        self.double_triplane = double_triplane
        self.compute_normal = compute_normal
        self.view_dependent_type = view_dependent_type
        if self.view_dependent_type == "neural":
            if not compute_normal:
                raise ValueError("Normal is required for neural view encoding!")

        if sep_geo_app:
            if sep_dims is None:
                sep_dims = [input_dim // 6, input_dim // 6]
        self.sep_dims = sep_dims

        self.mlp_sdf = build_pytorch_mlp(
            sep_dims[0] * 3 if sep_geo_app else input_dim,
            dim,
            1,
            depth=sdf_depth,
            bias=False,
        )

        if self.prediction_type == "rgb" or self.prediction_type == "both":
            if self.view_dependent_type == "none":
                self.mlp_rgb = build_pytorch_mlp(
                    sep_dims[1] * 3 if sep_geo_app else input_dim,
                    dim,
                    3,
                    depth=rgb_depth,
                    bias=False,
                )
            elif self.view_dependent_type == "sh":
                self.mlp_rgb = build_pytorch_mlp(
                    sep_dims[1] * 3 if sep_geo_app else input_dim,
                    dim,
                    3,
                    depth=rgb_depth,
                    bias=False,
                )
                self.mlp_order12 = build_pytorch_mlp(
                    sep_dims[1] * 3 if sep_geo_app else input_dim,
                    dim,
                    24,
                    depth=sh_depth,
                    bias=False,
                )
            elif self.view_dependent_type == "neural":
                self.mlp_rgb = build_pytorch_mlp(
                    (sep_dims[1] * 3 if sep_geo_app else input_dim) + input_view_dim,
                    dim,
                    3,
                    depth=rgb_depth,
                    bias=False,
                )
                self.mlp_normal = build_pytorch_mlp(
                    sep_dims[1] * 3 if sep_geo_app else input_dim,
                    dim,
                    3,
                    depth=normal_depth,
                    bias=False,
                )
            elif self.view_dependent_type == "neural_mip":
                self.mlp_rgb = build_pytorch_mlp(
                    (sep_dims[1] * 3 if sep_geo_app else input_dim) + input_view_dim,
                    dim,
                    3,
                    depth=rgb_depth,
                    bias=False,
                )
                self.mlp_rho = build_pytorch_mlp(
                    sep_dims[1] * 3 if sep_geo_app else input_dim,
                    dim,
                    1,
                    depth=rho_depth,
                    bias=False,
                )
                self.mlp_normal = build_pytorch_mlp(
                    sep_dims[1] * 3 if sep_geo_app else input_dim,
                    dim,
                    3,
                    depth=normal_depth,
                    bias=False,
                )

        if self.prediction_type == "brdf" or self.prediction_type == "both":
            self.mlp_basecolor = build_pytorch_mlp(
                sep_dims[1] * 3 if sep_geo_app else input_dim,
                dim,
                3,
                depth=brdf_depth,
                bias=False,
            )
            self.mlp_specular = build_pytorch_mlp(
                sep_dims[1] * 3 if sep_geo_app else input_dim,
                dim,
                2,
                depth=brdf_depth,
                bias=False,
            )

        if self.compute_normal and "neural" not in self.view_dependent_type:
            self.mlp_normal = build_pytorch_mlp(
                sep_dims[1] * 3 if sep_geo_app else input_dim,
                dim,
                3,
                depth=normal_depth,
                bias=False,
            )

        self.grid_sample_mode = grid_sample_mode

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.25)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def sampling(self, plane, index_1, index_2, index_3):
        dim = plane.shape[1]
        index = torch.stack([index_3, index_2, index_1], dim=-1)  # pytorch convention
        index = index.reshape(1, 1, 1, -1, 3)
        if self.grid_sample_mode == "customized":
            feature = grid_sample_3d(plane, index)
        elif self.grid_sample_mode == "original":
            dtype = plane.dtype
            feature = nn.functional.grid_sample(
                plane.to(torch.float32),
                index,
                align_corners=True,
                mode="bilinear",
                padding_mode="border",
            )
            feature = feature.to(dtype=dtype)
        else:
            raise ValueError(
                "Unrecognizable grid sampling mode %s." % self.grid_sampling
            )

        feature = feature.reshape(dim, -1)
        feature = feature.permute(1, 0)
        return feature

    def sampling_points(self, plane, index_1, index_2):
        batch_size, dim = plane.shape[0:2]
        index = torch.stack([index_2, index_1], dim=-1)  # pytorch convention
        index = index.reshape(batch_size, -1, 1, 2)
        if self.grid_sample_mode == "customized":
            feature = grid_sample_2d(plane, index)
        elif self.grid_sample_mode == "original":
            dtype = plane.dtype
            feature = nn.functional.grid_sample(
                plane.to(torch.float32),
                index,
                align_corners=True,
                mode="bilinear",
                padding_mode="border",
            )
            feature = feature.to(dtype=dtype)
        else:
            raise ValueError(
                "Unrecognizable grid sampling mode %s." % self.grid_sampling
            )
        feature = feature.reshape(batch_size, dim, -1)
        feature = feature.permute(0, 2, 1).reshape(-1, dim)
        return feature

    def get_sdf_bias(self, positions):
        with torch.no_grad():
            positions = positions.reshape(-1, 3)
            dist = torch.sqrt(torch.sum(positions * positions, dim=-1, keepdim=True))
            sdf = dist - 0.1 * self.radius
            return sdf

    def get_normal_bias(self, positions):
        with torch.no_grad():
            positions = positions.reshape(-1, 3)
            normal = nn.functional.normalize(positions, dim=-1, eps=1e-4)
            return normal

    def split_plane(self, plane):
        h, w = plane.shape[2:]
        hh, ww = h // 2, w // 2
        plane_0 = plane[:, :, :hh, :ww]
        plane_1 = plane[:, :, :hh, ww:]
        plane_2 = plane[:, :, hh:, :ww]
        plane_3 = plane[:, :, hh:, ww:]
        plane = torch.stack([plane_0, plane_1, plane_2, plane_3], dim=2)
        return plane

    def forward_one_set(
        self,
        ray_indices,
        positions,
        im_num,
        height,
        width,
        plane_xy,
        plane_xz,
        plane_yz,
        plane_view,
        mode="all",
        views=None,
    ):
        if self.view_dependent_type != "none":
            if views is not None:
                views = nn.functional.normalize(views, dim=-1)

        batch_size = plane_xy.shape[0]
        if batch_size != 1:
            raise ValueError(
                "Rendering of multiple instances of triplanes have been removed."
            )
        p0 = positions[..., 0] / self.radius
        p1 = positions[..., 1] / self.radius
        p2 = positions[..., 2] / self.radius

        if self.double_triplane:
            feature_xy = self.sampling_points(
                plane_xy,
                p0,
                (p1 / 2.0 - 0.5) + (p2 > 0).float(),
            )
            feature_xz = self.sampling_points(
                plane_xz,
                p0,
                (p2 / 2.0 - 0.5) + (p1 > 0).float(),
            )
            feature_yz = self.sampling_points(
                plane_yz,
                p1,
                (p2 / 2.0 - 0.5) + (p0 > 0).float(),
            )
        else:
            feature_xy = self.sampling_points(
                plane_xy,
                p0,
                p1,
            )
            feature_xz = self.sampling_points(
                plane_xz,
                p0,
                p2,
            )
            feature_yz = self.sampling_points(
                plane_yz,
                p1,
                p2,
            )

        if self.sep_geo_app:
            feature_xy_geo, feature_xy_app = torch.split(
                feature_xy, self.sep_dims, dim=-1
            )
            feature_xz_geo, feature_xz_app = torch.split(
                feature_xz, self.sep_dims, dim=-1
            )
            feature_yz_geo, feature_yz_app = torch.split(
                feature_yz, self.sep_dims, dim=-1
            )
            feature_geo = torch.cat(
                [feature_xy_geo, feature_xz_geo, feature_yz_geo], dim=-1
            )
            feature_app = torch.cat(
                [feature_xy_app, feature_xz_app, feature_yz_app], dim=-1
            )
        else:
            feature_geo = torch.cat([feature_xy, feature_xz, feature_yz], dim=-1)
            feature_app = feature_geo

        if mode == "sdf":
            sdf = self.mlp_sdf(feature_geo)
            sdf = torch.tanh(sdf) * self.radius * 1.732 * 1.5
            if self.sdf_bias:
                sdf = sdf + self.get_sdf_bias(positions)
            return sdf
        elif mode == "image":
            if self.prediction_type == "rgb" or self.prediction_type == "both":
                if self.view_dependent_type == "none":
                    x = torch.sigmoid(self.mlp_rgb(feature_app))
                elif self.view_dependent_type == "sh":
                    order0 = self.mlp_rgb(feature_app)
                    order12 = self.mlp_order12(feature_app)
                    x = torch.cat([order0, order12], dim=-1)
                    x = sh_decode(x, views)
                elif self.view_dependent_type == "neural":
                    normal = self.mlp_normal(feature_geo)
                    if self.normal_bias:
                        normal = normal + self.get_normal_bias(positions)
                    normal = nn.functional.normalize(normal, dim=-1)

                    n = normal.detach()

                    if views is not None:
                        r = -2 * torch.sum(views * n, dim=-1, keepdim=True) * n + views
                        r = nn.functional.normalize(r, dim=-1)
                    else:
                        r = normal.detach()
                    x, y, z = r[..., 0], r[..., 1], r[..., 2]
                    p0 = torch.atan2(x, y) / np.pi
                    p1 = z
                    feature_view = self.sampling_points(plane_view, p0, p1)
                    x = torch.sigmoid(
                        self.mlp_rgb(torch.cat([feature_app, feature_view], dim=-1))
                    )
                elif self.view_dependent_type == "neural_mip":
                    normal = self.mlp_normal(feature_geo)
                    if self.normal_bias:
                        normal = normal + self.get_normal_bias(positions)
                    normal = nn.functional.normalize(normal, dim=-1)

                    n = normal.detach()

                    if views is not None:
                        r = -2 * torch.sum(views * n, dim=-1, keepdim=True) * n + views
                        r = nn.functional.normalize(r, dim=-1)
                    else:
                        r = normal.detach()
                    x, y, z = r[..., 0], r[..., 1], r[..., 2]
                    p0 = 2 * torch.sigmoid(self.mlp_rho(feature_app)) - 1
                    p0 = p0.squeeze(-1)
                    p1 = torch.atan2(x, y) / np.pi
                    p2 = z

                    plane_view = self.split_plane(plane_view)
                    feature_view = self.sampling(plane_view, p0, p1, p2)
                    x = torch.sigmoid(
                        self.mlp_rgb(torch.cat([feature_app, feature_view], dim=-1))
                    )

            if self.prediction_type == "brdf" or self.prediction_type == "both":
                basecolor = torch.sigmoid(
                    self.mlp_basecolor(
                        feature_geo if self.prediction_type == "both" else feature_app
                    )
                )
                specular = torch.sigmoid(
                    self.mlp_specular(
                        feature_geo if self.prediction_type == "both" else feature_app
                    )
                )
                if self.prediction_type == "both":
                    x = torch.cat([x, basecolor, specular], dim=-1)
                else:
                    x = torch.cat([basecolor, specular], dim=-1)

            if self.compute_normal and "neural" not in self.view_dependent_type:
                normal = self.mlp_normal(feature_geo)
                if self.normal_bias:
                    normal = normal + self.get_normal_bias(positions)
                normal = nn.functional.normalize(normal, dim=-1)
                x = torch.cat([x, normal], dim=-1)
            elif "neural" in self.view_dependent_type:
                x = torch.cat([x, normal], dim=-1)

            return x
        elif mode == "all":
            sdf = self.mlp_sdf(feature_geo)
            sdf = torch.tanh(sdf) * self.radius * 1.732 * 1.5
            if self.sdf_bias:
                sdf = sdf + self.get_sdf_bias(positions)

            if self.prediction_type == "rgb" or self.prediction_type == "both":
                if self.view_dependent_type == "none":
                    x = torch.sigmoid(self.mlp_rgb(feature_app))
                elif self.view_dependent_type == "sh":
                    order0 = self.mlp_rgb(feature_app)
                    order12 = self.mlp_order12(feature_app)
                    x = torch.cat([order0, order12], dim=-1)
                    x = sh_decode(x, views)
                elif self.view_dependent_type == "neural":
                    normal = self.mlp_normal(feature_geo)
                    if self.normal_bias:
                        normal = normal + self.get_normal_bias(positions)
                    normal = nn.functional.normalize(normal, dim=-1)

                    n = normal.detach()

                    if views is not None:
                        r = -2 * torch.sum(views * n, dim=-1, keepdim=True) * n + views
                        r = nn.functional.normalize(r, dim=-1)
                    else:
                        r = normal.detach()
                    x, y, z = r[..., 0], r[..., 1], r[..., 2]
                    p0 = torch.atan2(x, y) / np.pi
                    p1 = z
                    feature_view = self.sampling_points(plane_view, p0, p1)
                    x = torch.sigmoid(
                        self.mlp_rgb(torch.cat([feature_app, feature_view], dim=-1))
                    )
                elif self.view_dependent_type == "neural_mip":
                    normal = self.mlp_normal(feature_geo)
                    if self.normal_bias:
                        normal = normal + self.get_normal_bias(positions)
                    normal = nn.functional.normalize(normal, dim=-1)

                    n = normal.detach()

                    if views is not None:
                        r = -2 * torch.sum(views * n, dim=-1, keepdim=True) * n + views
                        r = nn.functional.normalize(r, dim=-1)
                    else:
                        r = normal.detach()
                    x, y, z = r[..., 0], r[..., 1], r[..., 2]
                    p0 = 2 * torch.sigmoid(self.mlp_rho(feature_app)) - 1
                    p0 = p0.squeeze(-1)
                    p1 = torch.atan2(x, y) / np.pi
                    p2 = z
                    plane_view = self.split_plane(plane_view)
                    feature_view = self.sampling(plane_view, p0, p1, p2)
                    x = torch.sigmoid(
                        self.mlp_rgb(torch.cat([feature_app, feature_view], dim=-1))
                    )

            if self.prediction_type == "brdf" or self.prediction_type == "both":
                basecolor = torch.sigmoid(
                    self.mlp_basecolor(
                        feature_geo if self.prediction_type == "both" else feature_app
                    )
                )
                specular = torch.sigmoid(
                    self.mlp_specular(
                        feature_geo if self.prediction_type == "both" else feature_app
                    )
                )
                if self.prediction_type == "both":
                    x = torch.cat([x, basecolor, specular], dim=-1)
                else:
                    x = torch.cat([basecolor, specular], dim=-1)

            if self.compute_normal and "neural" not in self.view_dependent_type:
                normal = self.mlp_normal(feature_geo)
                if self.normal_bias:
                    normal = normal + self.get_normal_bias(positions)
                normal = nn.functional.normalize(normal, dim=-1)
                x = torch.cat([x, normal], dim=-1)
            elif "neural" in self.view_dependent_type:
                x = torch.cat([x, normal], dim=-1)

            return torch.cat([sdf, x], dim=-1)
        else:
            raise ValueError(f"Unrecognizable mode {mode}.")

    def forward(
        self,
        ray_indices,
        positions,
        im_num,
        height,
        width,
        plane_xy,
        plane_xz,
        plane_yz,
        plane_view,
        mode="all",
        views=None,
    ):
        if torch.is_grad_enabled():
            return self.forward_one_set(
                ray_indices,
                positions,
                im_num,
                height,
                width,
                plane_xy,
                plane_xz,
                plane_yz,
                plane_view,
                mode,
                views,
            )
        else:
            point_num = positions.shape[0]
            chunk_num = int(np.ceil(float(point_num) / self.chunk_size))
            pred_arr = []
            for n in range(0, chunk_num):
                xs = n * self.chunk_size
                xe = min(xs + self.chunk_size, point_num)
                pred = self.forward_one_set(
                    ray_indices if ray_indices is None else ray_indices[xs:xe],
                    positions[xs:xe, :],
                    im_num,
                    height,
                    width,
                    plane_xy,
                    plane_xz,
                    plane_yz,
                    plane_view,
                    mode,
                    views if views is None else views[xs:xe, :],
                )
                pred_arr.append(pred)
            pred_arr = torch.cat(pred_arr, dim=0)
            return pred_arr
