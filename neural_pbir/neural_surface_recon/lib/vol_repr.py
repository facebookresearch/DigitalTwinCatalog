# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Implement the main scene representation and rendering.
"""

import neural_pbir_cuda_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import grid, utils_sdf_rt


"""Model"""


class VolRepr(torch.nn.Module):
    def __init__(
        self,
        xyz_min,
        xyz_max,
        num_voxels,
        # limit the maximum resolution
        num_voxels_k0_max=None,
        num_voxels_bg_max=None,
        # use sdf grid or density grid
        sdf_mode=True,
        # sdf sharpness term (inverse variance) setup
        sdf_scale_init=1,
        sdf_scale_max=None,
        sdf_anneal_step=-1,  # allow backward faces at the early training
        sdf_scale_step=0,  # increment sdf_scale every train iter
        # use gradient descent if zero
        detach_sharpness=True,  # disable sdf_scale weight the sdf value in backprop
        # init activated alpha in density mode
        alpha_init=None,
        # init sdf grid to a constant + small random
        constant_init=False,
        constant_init_val=0,
        # init sdf grid to a sphere
        sphere_init=False,
        sphere_init_scale=1,
        sphere_init_shift=-0.5,
        # if the bottom of the bbox is a solid plane
        on_known_board=False,
        # binary mask to skip free-space
        mask_cache_path=None,
        mask_cache_init_thres=1e-3,  # thres for free-space
        mask_cache_upd_thres=1e-6,
        mask_cache_world_size=None,
        # weight threshold to skip query points on a ray
        fast_color_thres=0,
        # grids setups (see grid.py for detail)
        density_type="DenseGrid",
        k0_type="DenseGrid",
        density_config=None,
        k0_config=None,
        # mlp setups
        rgbnet_dim=0,
        rgbnet_depth=3,
        rgbnet_width=128,
        rgbnet_tcnn=True,
        rgbnet_last_act="sigmoid",
        posbase_pe=-1,
        viewbase_pe=4,
        # background model setups
        num_bg_scale=0,
        bg_scale=None,
        i_am_fg=True,
        **kwargs,
    ):
        super(VolRepr, self).__init__()
        if density_config is None:
            density_config = {}
        if k0_config is None:
            k0_config = {}
        self.register_buffer("xyz_min", torch.Tensor(xyz_min))
        self.register_buffer("xyz_max", torch.Tensor(xyz_max))
        self.i_am_fg = i_am_fg
        self.on_known_board = on_known_board
        self._name = "VolRepr-" + ("fg" if self.i_am_fg else "bg")

        # some geometry related scalar parameters
        self.sdf_mode = sdf_mode
        self.sdf_scale_max = sdf_scale_max
        self.sdf_scale_step = sdf_scale_step
        self.detach_sharpness = detach_sharpness
        self.alpha_init = alpha_init
        self.fast_color_thres = fast_color_thres
        if self.sdf_mode:
            self.sdf_anneal_step = sdf_anneal_step
            if sdf_scale_step == 0:
                self.sdf_scale = nn.Parameter(torch.FloatTensor([sdf_scale_init]))
            else:
                self.register_buffer("sdf_scale", torch.FloatTensor([sdf_scale_init]))
            self.stepdist_s = 1
        else:
            # determine the density bias shift (only for volume density not SDF)
            self.register_buffer(
                "act_shift", torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)])
            )
            print(f"{self._name}: set density bias shift to", self.act_shift)

        # determine init grid resolution
        self.num_voxels_k0_max = num_voxels_k0_max
        self.num_voxels_bg_max = num_voxels_bg_max
        self._set_grid_resolution(num_voxels)

        # geometry grid initialization
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
            density_type,
            channels=1,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
            config=self.density_config,
        )

        geo_grid_dirty = False

        def update_geo_grid(init_val):
            nonlocal geo_grid_dirty
            if geo_grid_dirty:
                self.density.grid.data.copy_(
                    torch.minimum(init_val, self.density.grid.data)
                )
            else:
                self.density.grid.data.copy_(init_val)
            geo_grid_dirty = True

        with torch.no_grad():
            if constant_init:
                init_val = torch.full_like(self.density.grid.data, constant_init_val)
                init_val += torch.randn_like(init_val) * 0.01
                update_geo_grid(init_val)
            if sphere_init:
                C = (self.xyz_max + self.xyz_min) * 0.5
                R = (self.xyz_max - self.xyz_min) * 0.5
                xyz = (self.get_coord_grid().moveaxis(0, -1) - C) / R
                xyz_norm = xyz.norm(dim=-1)
                init_val = xyz_norm * sphere_init_scale + sphere_init_shift
                update_geo_grid(init_val[None, None])
        print(f"{self._name}: geometry grid", self.density)

        # color representation initialization
        self.rgbnet_kwargs = {
            "rgbnet_dim": rgbnet_dim,
            "rgbnet_depth": rgbnet_depth,
            "rgbnet_width": rgbnet_width,
            "rgbnet_tcnn": rgbnet_tcnn,
            "rgbnet_last_act": rgbnet_last_act,
            "posbase_pe": posbase_pe,
            "viewbase_pe": viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.use_xyz_encoding = posbase_pe >= 0
        self.use_view_encoding = viewbase_pe >= 0
        if rgbnet_dim <= 0:
            # color grid
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type,
                channels=self.k0_dim,
                world_size=self.world_size_k0,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
                config=self.k0_config,
            )
            self.rgbnet = None
        else:
            # feature grid + shallow MLP
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                k0_type,
                channels=self.k0_dim,
                world_size=self.world_size_k0,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
                config=self.k0_config,
            )
            self.register_buffer(
                "viewfreq", torch.FloatTensor([(2**i) for i in range(viewbase_pe)])
            )

            if self.use_view_encoding:
                dim_i = self.k0_dim + (3 + 3 * viewbase_pe * 2)
            else:
                dim_i = self.k0_dim

            if self.use_xyz_encoding:
                import tinycudann as tcnn

                self.xyz_encoding = tcnn.Encoding(
                    3, {"otype": "Frequency", "n_frequencies": posbase_pe}
                )
                dim_i += 3 + self.xyz_encoding.n_output_dims

            MLP = MLP_tcnn if rgbnet_tcnn else MLP_torch
            self.rgbnet = MLP(dim_i, 3, rgbnet_width, rgbnet_depth)
            self.rgbnet_last_act = rgbnet_last_act
            print(f"{self._name}: feature voxel grid", self.k0)
            print(f"{self._name}: mlp", self.rgbnet)

        # using the coarse geometry if provided (used to determine known free space and unknown space)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_init_thres = mask_cache_init_thres
        self.mask_cache_upd_thres = mask_cache_upd_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                path=mask_cache_path, mask_cache_init_thres=mask_cache_init_thres
            ).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]
                    ),
                    torch.linspace(
                        self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]
                    ),
                    torch.linspace(
                        self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]
                    ),
                    indexing="ij",
                ),
                -1,
            )
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
            path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max
        )

        # build background models recursively
        self.num_bg_scale = num_bg_scale
        self.bg_scale = bg_scale
        self.bg_model = None
        if num_bg_scale > 0:
            C = (xyz_max + xyz_min) * 0.5
            R = (xyz_max - xyz_min).max() * 0.5
            self.bg_model = VolRepr(
                xyz_min=C - R * bg_scale,
                xyz_max=C + R * bg_scale,
                num_voxels=self.num_voxels_bg,
                alpha_init=alpha_init,
                mask_cache_upd_thres=mask_cache_upd_thres,
                fast_color_thres=fast_color_thres,
                density_type="DenseGrid",
                k0_type="DenseGrid",
                sdf_mode=False,
                rgbnet_dim=8,
                rgbnet_depth=3,
                rgbnet_width=64,
                num_bg_scale=num_bg_scale - 1,
                bg_scale=bg_scale,
                i_am_fg=False,
            )

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / (160**3)).pow(
            1 / 3
        )
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base

        # num voxel for k0 and bg
        if self.num_voxels_k0_max is None:
            self.num_voxels_k0 = num_voxels
        else:
            self.num_voxels_k0 = min(num_voxels, self.num_voxels_k0_max)

        if self.num_voxels_bg_max is None:
            self.num_voxels_bg = num_voxels
        else:
            self.num_voxels_bg = min(num_voxels, self.num_voxels_bg_max)

        # world size for k0 and bg
        self.voxel_size_k0 = (
            (self.xyz_max - self.xyz_min).prod() / self.num_voxels_k0
        ).pow(1 / 3)
        self.world_size_k0 = ((self.xyz_max - self.xyz_min) / self.voxel_size_k0).long()
        self.voxel_size_bg = (
            (self.xyz_max - self.xyz_min).prod() / self.num_voxels_bg
        ).pow(1 / 3)
        self.world_size_bg = ((self.xyz_max - self.xyz_min) / self.voxel_size_bg).long()

        # log
        print(f"{self._name}: voxel_size      ", self.voxel_size)
        print(f"{self._name}: world_size      ", self.world_size)
        print(f"{self._name}: world_size_k0   ", self.world_size_k0)
        print(f"{self._name}: voxel_size_base ", self.voxel_size_base)
        print(f"{self._name}: voxel_size_ratio", self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            "xyz_min": self.xyz_min.cpu().numpy(),
            "xyz_max": self.xyz_max.cpu().numpy(),
            "num_voxels": self.num_voxels,
            "num_voxels_k0_max": self.num_voxels_k0_max,
            "num_voxels_bg_max": self.num_voxels_bg_max,
            "sdf_mode": self.sdf_mode,
            "sdf_scale_max": self.sdf_scale_max,
            "sdf_scale_step": self.sdf_scale_step,
            "detach_sharpness": self.detach_sharpness,
            "alpha_init": self.alpha_init,
            "on_known_board": self.on_known_board,
            "mask_cache_path": self.mask_cache_path,
            "mask_cache_init_thres": self.mask_cache_init_thres,
            "mask_cache_upd_thres": self.mask_cache_upd_thres,
            "mask_cache_world_size": list(self.mask_cache.mask.shape),
            "fast_color_thres": self.fast_color_thres,
            "density_type": self.density_type,
            "k0_type": self.k0_type,
            "density_config": self.density_config,
            "k0_config": self.k0_config,
            **self.rgbnet_kwargs,
            "num_bg_scale": self.num_bg_scale,
            "i_am_fg": self.i_am_fg,
            "bg_scale": self.bg_scale,
        }

    def get_coord_grid(self, num_voxels=None):
        # Return grid point coordinate in shape [3, NX, NY, NZ]
        if num_voxels is None:
            world_size = self.world_size
        else:
            voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
            world_size = ((self.xyz_max - self.xyz_min) / voxel_size).long()
        return torch.stack(
            torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], world_size[2]),
                indexing="ij",
            ),
            0,
        )

    def get_sdf_grid(self, num_voxels=None, scale=1, alpha=0.5):
        coord_grid = self.get_coord_grid(num_voxels)[None]
        if num_voxels is not None:
            dense_grid = torch.cat(
                [self.density(i) for i in coord_grid.moveaxis(1, -1).split(8, dim=1)],
                dim=1,
            ).unsqueeze(1)
        else:
            dense_grid = self.density.get_dense_grid()
        if scale != 1:
            interp_kwargs = dict(
                scale_factor=scale, mode="trilinear", align_corners=True
            )
            dense_grid = F.interpolate(dense_grid, **interp_kwargs)
            coord_grid = F.interpolate(coord_grid, **interp_kwargs)
        dense_grid = dense_grid.squeeze().contiguous()
        coord_grid = coord_grid.squeeze().moveaxis(0, -1).contiguous()
        if self.sdf_mode:
            sdf_grid = dense_grid * self.sdf_scale
        else:
            raw_iso = alpha2raw(
                alpha, self.act_shift.item(), self.voxel_size_ratio.item()
            )
            sdf_grid = -(dense_grid - raw_iso)
        mask = self.mask_cache(coord_grid)
        return sdf_grid, mask

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print(f"{self._name}: scale_volume_grid start")
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print(
            f"{self._name}: scale_volume_grid scale world_size from",
            ori_world_size.tolist(),
            "to",
            self.world_size.tolist(),
        )

        self.density.scale_volume_grid(self.world_size)
        if (self.world_size_k0 != self.k0.world_size).any():
            self.k0.scale_volume_grid(self.world_size_k0)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        self.xyz_min[0], self.xyz_max[0], self.world_size[0]
                    ),
                    torch.linspace(
                        self.xyz_min[1], self.xyz_max[1], self.world_size[1]
                    ),
                    torch.linspace(
                        self.xyz_min[2], self.xyz_max[2], self.world_size[2]
                    ),
                    indexing="ij",
                ),
                -1,
            )
            if (self.world_size != self.density.world_size).any():
                densegrid = self.density.get_dense_grid(self.world_size)
            else:
                densegrid = self.density.get_dense_grid()
            self_alpha = F.max_pool3d(
                self.activate_density(grid=densegrid),
                kernel_size=3,
                padding=1,
                stride=1,
            )[0, 0]
            thres = (
                self.fast_color_thres
                if self.mask_cache_upd_thres is None
                else self.mask_cache_upd_thres
            )
            self.mask_cache = grid.MaskGrid(
                path=None,
                mask=self.mask_cache(self_grid_xyz) & (self_alpha > thres),
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
            )
        print(f"{self._name}: scale_volume_grid finish")

        if self.bg_model is not None and (
            self.num_voxels_bg != self.bg_model.num_voxels
        ):
            self.bg_model.scale_volume_grid(self.num_voxels_bg)

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]
                ),
                torch.linspace(
                    self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]
                ),
                torch.linspace(
                    self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]
                ),
                indexing="ij",
            ),
            -1,
        )
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(grid=cache_grid_density)
        cache_grid_alpha = F.max_pool3d(
            cache_grid_alpha, kernel_size=3, padding=1, stride=1
        )[0, 0]
        ori_p = self.mask_cache.mask.float().mean()
        thres = (
            self.fast_color_thres
            if self.mask_cache_upd_thres is None
            else self.mask_cache_upd_thres
        )
        self.mask_cache.mask &= cache_grid_alpha > thres
        new_p = self.mask_cache.mask.float().mean()
        if self.sdf_mode:
            print(f"{self._name}: sharpness {self.sdf_scale.data.item():.2f}")
        print(
            f"{self._name}: occupancy mask activated ratio {ori_p:.4f} => {new_p:.4f}"
        )
        if self.bg_model is not None:
            self.bg_model.update_occupancy_cache()

    def activate_density(self, grid=None, q=None, trainiter=None):
        density = q.density if grid is None else grid
        shape = density.shape
        density = density.flatten()
        if len(density) == 0:
            return torch.zeros_like(density)

        if self.sdf_mode:
            if q is None:
                sdf = density + density.detach() * (self.sdf_scale - 1)
                T = torch.sigmoid(sdf)
                alpha = 1 - T
            else:
                true_cos = (F.normalize(q.rays_d, dim=-1)[q.ray_id] * q.gradient).sum(
                    dim=-1
                )
                if trainiter is None or trainiter > self.sdf_anneal_step:
                    iter_cos = true_cos
                else:
                    anneal_p = 1 - trainiter / self.sdf_anneal_step
                    fake_cos = true_cos * 0.5 - 0.5 * q.gradient.detach().norm(dim=-1)
                    iter_cos = torch.lerp(
                        true_cos.clamp_max(0), fake_cos.clamp_max(0), anneal_p
                    )

                iter_cos_halfstep = q.stepdist * self.stepdist_s * 0.5 * iter_cos
                sdf = density - iter_cos_halfstep
                sdf_next = density + iter_cos_halfstep
                T = torch.sigmoid(scale_fw(sdf, self.sdf_scale, self.detach_sharpness))
                T_next = torch.sigmoid(
                    scale_fw(sdf_next, self.sdf_scale, self.detach_sharpness)
                )
                alpha = ((T - T_next) / T.clamp_min(1e-5)).clip(0, 1)
        else:
            interval = q.interval if q is not None else self.voxel_size_ratio
            alpha = Raw2Alpha.apply(density, self.act_shift, interval)
        return alpha.reshape(shape)

    def _process_query_alpha(self, q, trainiter):
        # query for alpha
        if self.sdf_mode:
            q.density, q.gradient = self.density(q.ray_pts, with_grad=True)
        else:
            q.density = self.density(q.ray_pts)
        q.alpha = self.activate_density(q=q, trainiter=trainiter)

    def _process_query_gradient(self, q):
        # analytical sdf grid gradient
        q.gradient = self.density.compute_grad(q.ray_pts)

    def _process_query_color(self, q, trainiter=None):
        k0 = self.k0(q.ray_pts)

        if len(q.ray_pts) == 0:
            q.rgb = torch.zeros([0, 3])
        elif self.rgbnet is None:
            # no view-depend effect
            q.rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            embs = []
            embs.append(k0)

            if self.use_view_encoding:
                viewdirs_emb = (q.viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
                viewdirs_emb = torch.cat(
                    [q.viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1
                )
                viewdirs_emb = viewdirs_emb.flatten(0, -2)[q.ray_id]
                embs.append(viewdirs_emb)

            if self.use_xyz_encoding:
                xyz = (q.ray_pts.detach() - self.xyz_min) / (
                    self.xyz_max - self.xyz_min
                )
                xyz_emb = torch.cat([xyz, self.xyz_encoding(xyz)], -1)
                embs.append(xyz_emb)

            embs = torch.cat(embs, -1)
            rgb_logit = self.rgbnet(embs)
            if self.rgbnet_last_act == "sigmoid":
                q.rgb = torch.sigmoid(rgb_logit)
            elif self.rgbnet_last_act == "softplus":
                q.rgb = F.softplus(rgb_logit)
            else:
                raise NotImplementedError

    def query_radiance(self, xyz, wo):
        q = Query(
            ray_pts=xyz, viewdirs=-wo, ray_id=torch.arange(len(xyz)).to(xyz.device)
        )
        self._process_query_color(q)
        # rgb = q.rgb.pow(2.2)
        rgb = torch.where(
            q.rgb <= 0.04045,
            q.rgb / 12.92,
            ((q.rgb.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4,
        )
        return rgb

    def process_query(
        self, q, N, render_color=True, render_normal=False, trainiter=None
    ):
        # compute the geometry the first time in non-differentiable mode
        with torch.no_grad():
            q.masking(((q.ray_pts > self.xyz_min) & (q.ray_pts < self.xyz_max)).all(-1))

            # skip known free space outside the occupancy mask
            q.masking(self.mask_cache(q.ray_pts))

            # skip low alpha
            self._process_query_alpha(q, trainiter)
            q.masking((q.alpha > self.fast_color_thres))

            # skip low weight
            # Alphas2Weights will stop when transmittance is lower than 1e-4;
            # the weights rest for the rest of the points would be 0
            q.weights, alphainv_last = Alphas2Weights.apply(q.alpha, q.ray_id, N)
            q.masking((q.weights > self.fast_color_thres))

            # compute the depth (z-distance) of each point
            q.t = (
                q.t_min[q.ray_id]
                + (q.step_id * q.stepdist) / q.rays_d.detach().norm(dim=-1)[q.ray_id]
            )
            # ray_pts = q.rays_o[q.ray_id] + q.rays_d[q.ray_id] * q.t.unsqueeze(-1)
            # assert ((ray_pts - q.ray_pts) / (self.xyz_max - self.xyz_min)).abs().max() < 1e-5

        # compute the geometry the second time if we need gradient (training mode)
        if trainiter is not None or render_normal:
            if self.i_am_fg:
                if q.rays_o is not None and q.rays_o.requires_grad:
                    q.ray_pts = q.rays_o[q.ray_id] + q.rays_d[q.ray_id] * q.t.unsqueeze(
                        -1
                    )
            self._process_query_alpha(q, trainiter)
            q.weights, alphainv_last = Alphas2Weights.apply(q.alpha, q.ray_id, N)

        # compute sdf/density gradient
        if self.i_am_fg and render_normal and q.gradient is None:
            self._process_query_gradient(q)

        # compute normal
        if q.gradient is not None:
            q.normal = F.normalize(q.gradient, dim=-1)
            if not self.sdf_mode:
                q.normal = -q.normal

        # compute colors
        if render_color:
            self._process_query_color(q, trainiter)
        else:
            q.rgb = torch.zeros([len(q.ray_pts), 3])

        return q, alphainv_last

    def _process_known_board(
        self, q, rgb_marched, alphainv_last, depth, normal, trainiter=None
    ):
        rd = q.rays_d
        pt_tmax = q.rays_o + rd * q.t_max.unsqueeze(-1)
        boardmask = pt_tmax[:, 1] < self.xyz_min[1] + self.voxel_size * 0.01
        boardidx = torch.where(boardmask & (alphainv_last > self.fast_color_thres))[0]
        pt_board = pt_tmax[boardidx]
        pt_board += (-pt_board[:, 1] / rd[boardidx, 1]).unsqueeze(-1) * rd[boardidx]
        if len(boardidx) == 0:
            return rgb_marched, alphainv_last, depth, normal

        lb = self.xyz_min - self.voxel_size * 1e-5
        rb = self.xyz_max + self.voxel_size * 1e-5
        insideidx = torch.where(((pt_board >= lb) & (pt_board <= rb)).all(-1))[0]
        boardidx = boardidx[insideidx]
        pt_board = pt_board[insideidx]
        if len(boardidx) == 0:
            return rgb_marched, alphainv_last, depth, normal

        q_board = Query(
            viewdirs=q.viewdirs[boardidx],
            ray_pts=pt_board,
            ray_id=torch.arange(len(boardidx)),
            density=torch.Tensor([0]).repeat(len(pt_board)),
            normal=torch.Tensor([[0, 1, 0]]).repeat(len(pt_board), 1),
        )
        self._process_query_color(q_board)
        if trainiter is not None:
            anneal = min(1, 0.1 + trainiter / 500)
            rgb_marched[boardidx] = rgb_marched[boardidx] + (
                alphainv_last[boardidx].unsqueeze(-1) * anneal * q_board.rgb
            )
        else:
            rgb_marched[boardidx] = rgb_marched[boardidx] + (
                alphainv_last[boardidx].unsqueeze(-1) * q_board.rgb
            )
        if depth is not None:
            depth[boardidx] = depth[boardidx] + (
                alphainv_last[boardidx] * q.t_max[boardidx]
            )
        if normal is not None:
            normal[boardidx] = normal[boardidx] + (
                alphainv_last[boardidx].unsqueeze(-1) * torch.Tensor([0, 1, 0])
            )
        alphainv_last = alphainv_last.clone()
        alphainv_last[boardidx] = alphainv_last[boardidx] * 0
        return rgb_marched, alphainv_last, depth, normal

    def sample_ray(self, rays_o, rays_d, near, far, stepsize):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, ray_id, step_id, N_steps, t_min, t_max = (
            neural_pbir_cuda_utils.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist
            )
        )
        # Note: At this point, ray_pts is non_differentiable wrt rays_o, rays_d.
        #       We will make them differentiable when we skip more sample points.
        return ray_pts, ray_id, step_id, t_min, t_max, stepdist, N_steps

    def forward(
        self, rays_o, rays_d, viewdirs, stepsize, trainiter=None, **render_kwargs
    ):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, (
            "Only suuport point queries in [N, 3] format"
        )

        ret_dict = {}
        N = len(rays_o)
        render_normal = render_kwargs.get("render_normal", False)
        if trainiter is not None and self.sdf_mode and self.sdf_scale_step > 0:
            self.sdf_scale += self.sdf_scale_step
        if trainiter is not None and self.sdf_mode and self.sdf_scale_max is not None:
            self.sdf_scale.data.clamp_max_(self.sdf_scale_max)

        # sample points on rays
        q = Query(
            rays_o=rays_o,
            rays_d=rays_d,
            viewdirs=viewdirs,
            stepsize=stepsize,
            interval=stepsize * self.voxel_size_ratio,
        )
        q.ray_pts, q.ray_id, q.step_id, q.t_min, q.t_max, q.stepdist, q.N_steps = (
            self.sample_ray(
                rays_o=rays_o,
                rays_d=rays_d,
                near=render_kwargs["near"],
                far=render_kwargs["far"],
                stepsize=stepsize,
            )
        )

        # processing the point queries
        q, alphainv_last = self.process_query(
            q=q, N=N, render_normal=render_normal, trainiter=trainiter
        )
        bg_prob = alphainv_last.clone()

        # accumulated queired points into pixels
        rgb_marched = torch.zeros([N, 3]).index_add_(
            dim=0, index=q.ray_id, source=(q.weights.unsqueeze(-1) * q.rgb)
        )

        depth = None
        if render_kwargs.get("render_depth", False):
            depth = torch.zeros([N]).index_add_(
                dim=0, index=q.ray_id, source=(q.weights * q.t)
            )

        normal = None
        if self.i_am_fg and render_normal:
            normal = torch.zeros([N, 3]).index_add_(
                dim=0, index=q.ray_id, source=(q.weights.unsqueeze(-1) * q.normal)
            )

        # compute board colors
        if self.on_known_board:
            rgb_marched, alphainv_last, depth, normal = self._process_known_board(
                q, rgb_marched, alphainv_last, depth, normal, trainiter=trainiter
            )

        # process background
        bg_q = None
        bg_color = None
        bg_maskidx = None
        if self.bg_model is not None and not render_kwargs.get("render_fg_only", False):
            maskidx = torch.where(alphainv_last > self.fast_color_thres)[0]
            if len(maskidx):
                # shift ray to bg model starting point
                ro = rays_o.detach()[maskidx]
                rd = rays_d.detach()[maskidx]
                rd = rd / rd.norm(dim=-1, keepdim=True)
                _, t_max = neural_pbir_cuda_utils.infer_t_minmax(
                    ro, rd, self.xyz_min, self.xyz_max, 0, 1e9
                )
                ro = ro + rd * t_max.unsqueeze(-1)
                bg_dict = self.bg_model(
                    ro, rd, rd, stepsize=stepsize, trainiter=trainiter, **render_kwargs
                )

                # combine colors
                bg_color = bg_dict["rgb_marched"]
                bg_maskidx = maskidx
                rgb_marched[maskidx] = rgb_marched[maskidx] + (
                    alphainv_last[maskidx].unsqueeze(-1) * bg_color
                )

                # combine depth
                if render_kwargs.get("render_depth", False):
                    bg_depth = bg_dict["depth"] / rays_d[maskidx].detach().norm(dim=-1)
                    ret_dict["depth"][maskidx] = (
                        ret_dict["depth"][maskidx] + alphainv_last[maskidx] * bg_depth
                    )

                # combine transmittance
                alphainv_last = alphainv_last.clone()
                alphainv_last[maskidx] = (
                    alphainv_last[maskidx] * bg_dict["alphainv_last"]
                )

        elif render_kwargs["rand_bkgd"]:
            rgb_marched += alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched)
        else:
            rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs["bg"]

        ret_dict.update(
            {
                "alphainv_last": alphainv_last,  # it's the total bg_prob
                "bg_prob": bg_prob,  # it's the foreground bg_prob
                "rgb_marched": rgb_marched,
                "depth": depth,
                "normal": normal,
                "q": q,
                "bg_maskidx": bg_maskidx,
                "bg_color": bg_color,
            }
        )

        return ret_dict

    def fast_forward(self, rays_o, rays_d, viewdirs, stepsize, near, **render_kwargs):
        """Volume rendering for test-time"""
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, (
            "Only suuport point queries in [N, 3] format"
        )
        MAX_N_Q = 800 * 800 * 4

        N = len(rays_o)
        rgb_marched = torch.zeros([N, 3])
        depth = torch.zeros([N])
        xyz = torch.zeros([N, 3])
        alphainv_last = torch.ones([N])
        normal = torch.zeros([N, 3])
        render_color = render_kwargs.get("render_color", True)
        render_depth = render_kwargs.get("render_depth", False)
        render_normal = render_kwargs.get("render_normal", False)
        render_xyz = render_kwargs.get("render_xyz", False)
        stepdist = stepsize * self.voxel_size
        interval = stepsize * self.voxel_size_ratio

        # shift points to starting section
        t_now, t_max = neural_pbir_cuda_utils.infer_t_minmax(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, 1e9
        )
        ori_t_max = t_max.clone()
        rs, rd = neural_pbir_cuda_utils.infer_ray_start_dir(rays_o, rays_d, t_now)

        # init ray rendering
        alive_idx = torch.where(t_now <= t_max)[0]
        rs = rs[alive_idx]
        rd = rd[alive_idx]
        vd = viewdirs[alive_idx]
        step_norm = rays_d.norm(dim=-1)[alive_idx]
        t_now = t_now[alive_idx]
        t_max = t_max[alive_idx]

        # maskcache ray tracing
        rt_n_steps = self.mask_cache.ray_tracing(
            rs, rd, stepdist, ((t_max - t_now) / stepdist).long() + 1
        )
        iter_stepsize = stepdist * rt_n_steps
        t_now += iter_stepsize / step_norm
        rs += rd * iter_stepsize.unsqueeze(-1)
        maskidx = torch.where(t_now <= t_max)[0]
        alive_idx = alive_idx[maskidx]
        rs = rs[maskidx]
        rd = rd[maskidx]
        vd = vd[maskidx]
        step_norm = step_norm[maskidx]
        t_now = t_now[maskidx]
        t_max = t_max[maskidx]

        # fast volume rendering
        T = alphainv_last[alive_idx]
        while len(alive_idx):
            n = len(alive_idx)
            m = min(MAX_N_Q // n, 256)
            t = torch.arange(m)[None, :, None] * stepdist
            q = Query(
                rays_o=rs,
                rays_d=rd,
                viewdirs=vd,
                t_min=torch.zeros_like(t_now),
                ray_pts=(rs.unsqueeze(-2) + rd.unsqueeze(-2) * t).reshape(-1, 3),
                ray_id=torch.arange(n).view(-1, 1).repeat(1, m).flatten(),
                step_id=torch.arange(m).view(1, -1).repeat(n, 1).flatten(),
                stepdist=stepdist,
                stepsize=stepsize,
                interval=interval,
            )
            q, local_alphainv_last = self.process_query(
                q=q,
                N=n,
                render_color=render_color,
                render_normal=render_normal,
                trainiter=None,
            )

            local_rgb_marched = torch.zeros([n, 3]).index_add_(
                dim=0, index=q.ray_id, source=(q.weights.unsqueeze(-1) * q.rgb)
            )

            local_depth = 0
            if render_depth:
                local_depth = torch.zeros([n]).index_add_(
                    dim=0, index=q.ray_id, source=(q.weights * q.t)
                )

            local_normal = 0
            if self.i_am_fg and render_normal:
                local_normal = torch.zeros([n, 3]).index_add_(
                    dim=0, index=q.ray_id, source=(q.weights.unsqueeze(-1) * q.normal)
                )

            local_xyz = 0
            if render_xyz:
                local_xyz = torch.zeros([n, 3]).index_add_(
                    dim=0, index=q.ray_id, source=(q.weights.unsqueeze(-1) * q.ray_pts)
                )

            rgb_marched[alive_idx] += T.unsqueeze(-1) * local_rgb_marched
            depth[alive_idx] += T * local_depth
            xyz[alive_idx] += T.unsqueeze(-1) * local_xyz
            normal[alive_idx] += T.unsqueeze(-1) * local_normal
            T *= local_alphainv_last
            alphainv_last[alive_idx] = T

            iter_stepsize = stepdist * m
            t_now += iter_stepsize / step_norm
            rs += rd * iter_stepsize

            maskidx = torch.where((t_now <= t_max) & (T > self.fast_color_thres))[0]
            alive_idx = alive_idx[maskidx]
            rs = rs[maskidx]
            rd = rd[maskidx]
            vd = vd[maskidx]
            step_norm = step_norm[maskidx]
            t_now = t_now[maskidx]
            t_max = t_max[maskidx]
            T = T[maskidx]

        # compute board colors
        if self.on_known_board:
            q_board = Query(
                rays_o=rays_o, rays_d=rays_d, viewdirs=viewdirs, t_max=ori_t_max
            )
            rgb_marched, alphainv_last, depth, normal = self._process_known_board(
                q_board, rgb_marched, alphainv_last, depth, normal
            )

        # add bg color
        if self.bg_model is not None and not render_kwargs.get("render_fg_only", False):
            maskidx = torch.where(alphainv_last > self.fast_color_thres)[0]
            ro = rays_o[maskidx]
            rd = rays_d[maskidx]
            rd = rd / rd.norm(dim=-1, keepdim=True)
            _, t_max = neural_pbir_cuda_utils.infer_t_minmax(
                ro, rd, self.xyz_min, self.xyz_max, 0, 1e9
            )
            ro = ro + rd * t_max.unsqueeze(-1)
            bg_dict = self.bg_model.fast_forward(
                rays_o=ro,
                rays_d=rd,
                viewdirs=rd,
                stepsize=stepsize,
                near=0,
                **render_kwargs,
            )
            rgb_marched[maskidx] += (
                alphainv_last[maskidx].unsqueeze(-1) * bg_dict["rgb_marched"]
            )
            depth[maskidx] += alphainv_last[maskidx] * (t_max + bg_dict["depth"])
            xyz[maskidx] += alphainv_last[maskidx].unsqueeze(-1) * bg_dict["xyz"]
            alphainv_last[maskidx] *= bg_dict["alphainv_last"]
        else:
            rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs["bg"]

        return dict(
            rgb_marched=rgb_marched,
            depth=depth,
            alphainv_last=alphainv_last,
            normal=normal,
            xyz=xyz,
        )

    def superfast_forward(
        self, rays_o, rays_d, viewdirs, stepsize, near, **render_kwargs
    ):
        """Volume rendering for test-time"""
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, (
            "Only suuport point queries in [N, 3] format"
        )
        assert self.density.type == "DenseGrid"
        assert self.sdf_mode

        N = len(rays_o)
        rgb_marched = torch.full([N, 3], render_kwargs["bg"]).float()
        depth = torch.zeros([N])
        xyz = torch.zeros([N, 3])
        alphainv_last = torch.ones([N])
        normal = torch.zeros([N, 3])
        render_color = render_kwargs.get("render_color", True)
        render_depth = render_kwargs.get("render_depth", False)
        render_normal = render_kwargs.get("render_normal", False)
        stepdist = stepsize * self.voxel_size
        interval = stepsize * self.voxel_size_ratio

        # shift points to starting section
        t_now, t_max = neural_pbir_cuda_utils.infer_t_minmax(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, 1e9
        )
        ori_t_max = t_max.clone()
        rs, _ = neural_pbir_cuda_utils.infer_ray_start_dir(rays_o, rays_d, t_now)

        # init ray rendering
        alive_idx = torch.where(t_now <= t_max)[0]
        rs = rs[alive_idx]
        rd = viewdirs[alive_idx]
        step_norm = rays_d.norm(dim=-1)[alive_idx]
        t_now = t_now[alive_idx]
        t_max = t_max[alive_idx]

        # trace surfact
        hitxyz, is_hit = utils_sdf_rt.sdf_grid_trace_surface(
            self.density.grid.data,
            rs,
            rd,
            stepsize,
            self.xyz_min,
            self.xyz_max,
            self.world_size,
        )
        if self.on_known_board:
            board_min = self.xyz_min.clone()
            board_max = self.xyz_max.clone()
            board_max[1] = self.xyz_min[1] + self.voxel_size * 0.1
            is_hit[((board_min < hitxyz) & (hitxyz < board_max)).all(-1)] = 1
        xyz[alive_idx] = hitxyz
        alphainv_last[alive_idx] = 1 - is_hit.float()
        depth[alive_idx] = (hitxyz - rs).norm(dim=-1)

        # query surface
        q = Query(
            ray_pts=hitxyz[is_hit],
            viewdirs=viewdirs[is_hit],
            ray_id=torch.arange(is_hit.sum()),
        )
        if render_normal:
            self._process_query_gradient(q)
            if self.on_known_board:
                q.gradient[
                    ((board_min < q.ray_pts) & (q.ray_pts < board_max)).all(-1)
                ] = torch.Tensor([0, 1, 0])
            q.normal = F.normalize(q.gradient, dim=-1)
            normal[alive_idx[is_hit]] = q.normal
        if render_color:
            self._process_query_color(q)
            rgb_marched[alive_idx[is_hit]] = q.rgb

        return dict(
            rgb_marched=rgb_marched,
            depth=depth,
            alphainv_last=alphainv_last,
            normal=normal,
            xyz=xyz,
            is_hit=is_hit,
        )

    def render360(self, return_alpha=False):
        """Render a 360 background image"""
        H, W = 512, 1024
        C = (self.xyz_min + self.xyz_max) * 0.5
        R = (self.xyz_max - self.xyz_min) * 0.5
        v, u = torch.meshgrid(
            -((torch.arange(H) + 0.5) / H - 0.5) * np.pi,
            torch.arange(W) / W * np.pi * 2 - np.pi / 2,
            indexing="ij",
        )
        rays_z = torch.sin(v)
        rays_y = torch.cos(v) * torch.sin(u)
        rays_x = torch.cos(v) * torch.cos(u)
        rays_d = torch.stack([rays_x, rays_z, rays_y], -1).reshape(-1, 3) * (
            R.min() * 0.1
        )
        rays_o = C[None].repeat(len(rays_d), 1)
        _, t_max = neural_pbir_cuda_utils.infer_t_minmax(
            rays_o, rays_d, self.xyz_min, self.xyz_max, 0, 1e9
        )
        rays_o = rays_o + rays_d * t_max.unsqueeze(-1)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        view360 = self.bg_model.fast_forward(
            rays_o=rays_o, rays_d=rays_d, viewdirs=rays_d, stepsize=0.5, near=0, bg=0
        )
        bg360 = view360["rgb_marched"].reshape(H, W, 3)
        if return_alpha:
            bg360_a = 1 - view360["alphainv_last"].reshape(H, W)
            if self.on_known_board:
                boardmask = rays_o[:, 1] < self.xyz_min[1] + self.voxel_size * 0.1
                bg360_a *= 1 - boardmask.reshape(bg360_a.shape).float()
            return bg360, bg360_a
        return bg360


""" Query bundle
"""


class Query(object):
    def __init__(
        self,
        rays_o=None,
        rays_d=None,
        viewdirs=None,
        t_min=None,
        t_max=None,
        N_steps=None,
        stepdist=None,
        stepsize=None,
        interval=None,
        ray_pts=None,
        ray_id=None,
        step_id=None,
        t=None,
        density=None,
        alpha=None,
        weights=None,
        gradient=None,
        normal=None,
        rgb=None,
    ):
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.viewdirs = viewdirs
        self.t_min = t_min
        self.t_max = t_max
        self.stepdist = stepdist
        self.stepsize = stepsize
        self.interval = interval
        self.N_steps = N_steps
        self.ray_pts = ray_pts
        self.ray_id = ray_id
        self.step_id = step_id
        self.t = t
        self.density = density
        self.alpha = alpha
        self.weights = weights
        self.gradient = gradient
        self.normal = normal
        self.rgb = rgb
        self._properties = [
            "ray_pts",
            "ray_id",
            "step_id",
            "t",
            "density",
            "alpha",
            "weights",
            "gradient",
            "normal",
            "rgb",
        ]

    def masking(self, mask):
        # Compute index once and for all.
        # Using masking is costly especially in differentiable mode.
        idx = torch.where(mask)[0]
        for name in self._properties:
            prop = getattr(self, name)
            if prop is not None:
                setattr(self, name, prop[idx])

    def concat(self, other):
        # Concatenate two Query instance
        a_rng = torch.arange(len(self.ray_id))
        b_rng = torch.arange(len(other.ray_id))
        a_bin = self.ray_id.bincount(minlength=len(self.rays_o) + 1)
        b_bin = other.ray_id.bincount(minlength=len(self.rays_o) + 1)
        t_bin = a_bin + b_bin
        a_cum = a_bin.roll(1).cumsum_(0)
        b_cum = b_bin.roll(1).cumsum_(0)
        t_cum = t_bin.roll(1).cumsum_(0)
        a_idx = a_rng - a_cum[self.ray_id]
        b_idx = b_rng - b_cum[other.ray_id]
        order = torch.cat(
            [
                a_idx + t_cum[self.ray_id],
                b_idx + t_cum[other.ray_id] + a_bin[other.ray_id],
            ]
        )
        reorder = torch.empty([len(a_rng) + len(b_rng)], dtype=torch.long)
        reorder[order] = torch.arange(len(a_rng) + len(b_rng))
        for name in self._properties:
            a_prop = getattr(self, name)
            b_prop = getattr(other, name)
            if a_prop is not None:
                setattr(self, name, torch.cat([a_prop, b_prop])[reorder])


""" Computation for alpha values and weights
"""


class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        """
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        """
        exp, alpha = neural_pbir_cuda_utils.raw2alpha(density, shift, interval)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        """
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp'(density + shift)
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        """
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return (
            neural_pbir_cuda_utils.raw2alpha_backward(
                exp, grad_back.contiguous(), interval
            ),
            None,
            None,
        )


class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = neural_pbir_cuda_utils.alpha2weight(
            alpha, ray_id, N
        )
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = neural_pbir_cuda_utils.alpha2weight_backward(
            alpha,
            weights,
            T,
            alphainv_last,
            i_start,
            i_end,
            ctx.n_rays,
            grad_weights,
            grad_last,
        )
        return grad, None, None


def alpha2raw(alpha, bias, interval):
    raw = (1 - alpha) ** (-1 / interval) - 1
    if torch.is_tensor(alpha):
        raw = torch.log(raw) - bias
    else:
        raw = np.log(raw) - bias
    return raw


def scale_fw(x, s, detach_sharpness):
    if detach_sharpness:
        return x + x.detach() * (s - 1)
    return x * s


""" MLP wrapper
"""


class MLP_torch(torch.nn.Module):
    def __init__(self, dim0, dimO, width, depth):
        super(MLP_torch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim0, width),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, dimO),
        )
        nn.init.constant_(self.mlp[-1].bias, 0)
        self.channels = dimO

    def forward(self, x):
        return self.mlp(x)


class MLP_tcnn(torch.nn.Module):
    def __init__(self, dim0, dimO, width, depth):
        super(MLP_tcnn, self).__init__()
        import tinycudann as tcnn

        self.mlp = tcnn.Network(
            n_input_dims=dim0,
            n_output_dims=dimO,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": width,
                "n_hidden_layers": depth - 1,
            },
        )
        with torch.no_grad():
            bd = 1 / np.sqrt(width)
            bd0 = 1 / np.sqrt(dim0)
            self.mlp.params.data[:] = (
                torch.rand(len(self.mlp.params.data)) * bd * 2 - bd
            )
            self.mlp.params.data[: dim0 * width] = (
                torch.rand(dim0 * width) * bd0 * 2 - bd0
            )
        self.bias = nn.Parameter(torch.zeros(dimO))
        self.channels = dimO

    def forward(self, x):
        return self.mlp(x).to(x) + self.bias
