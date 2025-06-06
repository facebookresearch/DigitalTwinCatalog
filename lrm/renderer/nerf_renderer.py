# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import nerfacc
import numpy as np
import torch
import torch.nn as nn

from .utils import validate_empty_rays


class NerfRenderer(nn.Module):
    def __init__(
        self,
        radius=0.5,
        num_samples_per_ray=128,
        pred_mode="rgb",
        opacity_threshold=0.98,
        occgrid_res=128,
        occgrid_thr=1e-4,
        auto_cast_dtype=torch.bfloat16,
    ):
        super().__init__()
        # Define the bounding box
        self.radius = radius
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-radius, -radius, -radius],
                    [radius, radius, radius],
                ],
                dtype=torch.float32,
            ),
        )
        self.num_samples_per_ray = num_samples_per_ray
        self.render_step_size = 1.732 * 2 * radius / num_samples_per_ray
        self.eps = 8 * radius / num_samples_per_ray
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=self.bbox.view(-1), resolution=occgrid_res, levels=1
        )
        self.estimator.occs.fill_(True)
        self.estimator.binaries.fill_(True)
        self.occgrid_res = occgrid_res
        self.occgrid_thr = occgrid_thr

        if pred_mode != "rgb" and pred_mode != "brdf" and pred_mode != "both":
            raise ValueError(f"Undefined prediction mode {pred_mode}")
        self.mode = pred_mode
        self.opacity_threshold = opacity_threshold
        self.auto_cast_dtype = auto_cast_dtype

    def update_occupancy_grid(self, plane_xy, plane_xz, plane_yz, triplane):
        def occ_eval_fn(x):
            chunk_size = 2097152
            point_num = x.shape[0]
            chunk_num = int(np.ceil(float(point_num) / chunk_size))
            density_arr = []
            for n in range(0, chunk_num):
                xs = n * chunk_size
                xe = min(xs + chunk_size, point_num)

                with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype):
                    density = triplane(
                        ray_indices=None,
                        positions=x[xs:xe, :],
                        im_num=None,
                        height=None,
                        width=None,
                        plane_xy=plane_xy,
                        plane_xz=plane_xz,
                        plane_yz=plane_yz,
                        plane_view=None,
                        mode="density",
                    )
                    density = density.to(dtype=torch.float32)
                density_arr.append(density)
            density = torch.cat(density_arr, dim=0)

            return density * self.render_step_size

        self.estimator._update(
            step=0,
            occ_eval_fn=occ_eval_fn,
            occ_thre=self.occgrid_thr,
            ema_decay=0.0,
        )

    def reset_occupancy_grid(self):
        self.estimator.occs.fill_(True)
        self.estimator.binaries.fill_(True)

    def compose_output(
        self,
        weights,
        values,
        ray_indices,
        n_rays,
        opacity,
        batch_size,
        im_num,
        height,
        width,
        is_scale=True,
    ):
        comp_fg = nerfacc.accumulate_along_rays(
            weights, values=values, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_fg = comp_fg.view(batch_size, im_num, height, width, -1)
        if opacity is not None:
            comp = comp_fg + (1 - opacity)
        else:
            comp = comp_fg
        if is_scale:
            comp = 2 * comp - 1
        return comp

    def grid_sample(self, N):
        device = self.bbox.device
        dtype = self.bbox.dtype
        x = torch.linspace(-self.radius, self.radius, N, device=device)
        y = torch.linspace(-self.radius, self.radius, N, device=device)
        z = torch.linspace(-self.radius, self.radius, N, device=device)
        x, y, z = torch.meshgrid(x, y, z)
        points = torch.stack([x, y, z], dim=-1)
        points = points.reshape(-1, 3).to(device=device, dtype=dtype)
        return points

    def uniform_sample(self, N):
        device = self.bbox.device
        dtype = self.bbox.dtype
        points = torch.rand(N, 3, device=device) * 2 * self.radius - self.radius
        points = points.to(device=device, dtype=dtype)
        return points

    def evaluate_points_density(
        self, points, plane_xy, plane_xz, plane_yz, triplane, chunk=1048576
    ):
        batch_size, point_num = points.shape[0:2]
        p_interval = int(np.ceil(float(point_num) / chunk))

        density_arr = []
        for b in range(0, batch_size):
            for n in range(0, p_interval):
                rs = n * chunk
                re = min(rs + chunk, point_num)
                with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype):
                    density = triplane(
                        None,
                        points[b, rs:re, :],
                        None,
                        None,
                        None,
                        plane_xy[b : b + 1, :],
                        plane_xz[b : b + 1, :],
                        plane_yz[b : b + 1, :],
                        None,
                        mode="density",
                    )
                    density = density.to(dtype=torch.float32)
                density_arr.append(density.reshape(re - rs, 1))
        density = torch.cat(density_arr, dim=0).reshape(batch_size, point_num, 1)
        return density

    def sample_point(
        self, plane_xy, plane_xz, plane_yz, triplane, mode, N, chunk=1024 * 1024
    ):
        """
        plane_xy: Float[Tensor, "B C H W"],
        plane_xz: Float[Tensor, "B C H W"],
        plane_yz: Float[Tensor, "B C H W"],
        triplane: triplane network
        mode: how to sample points, either grid or uniform
        N: control the number of points sampled. In grid mode
        we will sample N**3 points while in uniform mode, we
        will sample exactly N points
        """
        batch_size = plane_xy.shape[0]
        if mode == "grid":
            points = self.grid_sample(N)
        elif mode == "uniform":
            points = self.uniform_sample(N)
        else:
            raise ValueError("Unrecognizable point sampling mode")
        points = points[None, :, :]
        points = points.repeat(batch_size, 1, 1)

        out = {}
        out["density"] = self.evaluate_points_density(
            points, plane_xy, plane_xz, plane_yz, triplane
        )
        out["points"] = points

        return out

    def compute_numerical_normal(
        self,
        density,
        triplane,
        ray_indices,
        points,
        im_num,
        height,
        width,
        plane_xy,
        plane_xz,
        plane_yz,
        reverse=False,
    ):
        device = plane_xy.device
        point_num = points.shape[0]

        eps = [[self.eps, 0, 0], [0, self.eps, 0], [0, 0, self.eps]]
        eps = torch.Tensor(eps).to(device=device)
        eps = eps.reshape(3, 1, 3)
        eps = -eps if reverse else eps
        points = points[None, :, :]
        points_eps = (points + eps).reshape(-1, 3)

        with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype):
            density_epsilon = triplane(
                ray_indices.repeat(3) if ray_indices is not None else None,
                points_eps,
                im_num,
                height,
                width,
                plane_xy,
                plane_xz,
                plane_yz,
                None,
                mode="density",
            )
            density_epsilon = density_epsilon.to(dtype=torch.float32)
        density_x, density_y, density_z = torch.split(density_epsilon, point_num, dim=0)
        grad_x = -(density_x - density) / self.eps
        grad_y = -(density_y - density) / self.eps
        grad_z = -(density_z - density) / self.eps

        gradient = torch.cat([grad_x, grad_y, grad_z], dim=-1)
        gradient = -gradient if reverse else gradient
        normal = nn.functional.normalize(gradient, dim=-1, eps=1e-4)

        return normal, gradient

    def forward(
        self,
        plane_xy,
        plane_xz,
        plane_yz,
        plane_view,
        triplane,
        rays_o,
        rays_d,
        cams,
        compute_normal=False,
        compute_numerical_normal=False,
        is_camera_space_normal=False,
        return_intermediate=False,
        sampled_points_per_ray=None,
    ):
        batch_size, im_num, height, width = rays_o.shape[:4]
        rays_o_flatten = rays_o.reshape(-1, 3)
        rays_d_flatten = rays_d.reshape(-1, 3)
        n_rays = rays_o_flatten.shape[0]

        if sampled_points_per_ray is None:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=None,
                    near_plane=0,
                    far_plane=1e10,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.0,
                    stratified=False,
                    cone_angle=0.0,
                    early_stop_eps=0,
                )
                ray_indices, t_starts_, t_ends_ = validate_empty_rays(
                    ray_indices, t_starts_, t_ends_
                )
                ray_indices = ray_indices.long()
                t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        else:
            ray_indices = sampled_points_per_ray["ray_indices"]
            t_starts = sampled_points_per_ray["t_starts"]
            t_ends = sampled_points_per_ray["t_ends"]

        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions

        with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype):
            pred = triplane(
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
                views=t_dirs,
            )
            pred = pred.to(dtype=torch.float32)
            density, pred = pred[:, :1], pred[:, 1:]

        weights, transmittance, alpha = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            density[..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        if compute_numerical_normal:
            numerical_normal, gradient = self.compute_numerical_normal(
                density,
                triplane,
                ray_indices,
                positions,
                im_num,
                height,
                width,
                plane_xy,
                plane_xz,
                plane_yz,
                reverse=False,
            )

        opacity = nerfacc.accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        points = nerfacc.accumulate_along_rays(
            weights, values=positions, ray_indices=ray_indices, n_rays=n_rays
        )
        points = points / (torch.clamp(opacity, min=self.opacity_threshold)).detach()

        opacity = opacity.view(batch_size, im_num, height, width, 1)
        points = points.view(batch_size, im_num, height, width, 3)
        valid_mask = (opacity > self.opacity_threshold).to(dtype=points.dtype).detach()

        comp_pred = self.compose_output(
            weights,
            pred,
            ray_indices,
            n_rays,
            opacity,
            batch_size,
            im_num,
            height,
            width,
        )
        out = {}
        if self.mode == "rgb":
            if compute_normal:
                comp_rgb, comp_normal = torch.split(comp_pred, [3, 3], dim=-1)
                comp_normal = 0.5 * (comp_normal + 1)
                out["rgb"] = comp_rgb
                out["normal"] = comp_normal
            else:
                out["rgb"] = comp_pred
        elif self.mode == "brdf":
            if compute_normal:
                comp_albedo, comp_roughness, comp_metallic, comp_normal = torch.split(
                    comp_pred, [3, 1, 1, 3], dim=-1
                )
                comp_normal = 0.5 * (comp_normal + 1)
                out["albedo"] = comp_albedo
                out["roughness"] = comp_roughness
                out["metallic"] = comp_metallic
                out["normal"] = comp_normal
            else:
                comp_albedo, comp_roughness, comp_metallic = torch.split(
                    comp_pred, [3, 1, 1], dim=-1
                )
                out["albedo"] = comp_albedo
                out["roughness"] = comp_roughness
                out["metallic"] = comp_metallic
        elif self.mode == "both":
            if compute_normal:
                comp_rgb, comp_albedo, comp_roughness, comp_metallic, comp_normal = (
                    torch.split(comp_pred, [3, 3, 1, 1, 3], dim=-1)
                )
                comp_normal = 0.5 * (comp_normal + 1)
                out["rgb"] = comp_rgb
                out["albedo"] = comp_albedo
                out["roughness"] = comp_roughness
                out["metallic"] = comp_metallic
                out["normal"] = comp_normal
            else:
                comp_rgb, comp_albedo, comp_roughness, comp_metallic = torch.split(
                    comp_pred, [3, 3, 1, 1], dim=-1
                )
                out["rgb"] = comp_rgb
                out["albedo"] = comp_albedo
                out["roughness"] = comp_roughness
                out["metallic"] = comp_metallic

        if compute_numerical_normal:
            comp_numerical_normal = self.compose_output(
                weights,
                numerical_normal,
                ray_indices,
                n_rays,
                None,
                batch_size,
                im_num,
                height,
                width,
                is_scale=False,
            )
            comp_numerical_normal = comp_numerical_normal.view(
                batch_size, im_num, height, width, 3
            )
            comp_numerical_normal = nn.functional.normalize(
                comp_numerical_normal, dim=-1, eps=1e-4
            )
            comp_numerical_normal = comp_numerical_normal * valid_mask + (
                1 - valid_mask
            )

        cams = cams[:, :, 0:16].reshape(batch_size, im_num, 4, 4)
        if is_camera_space_normal:
            # Rotate normal into camera coordinate
            rots = cams[:, :, :3, :3].unsqueeze(2).unsqueeze(2)

            if compute_normal:
                comp_normal = comp_normal.unsqueeze(-1)
                comp_normal = torch.sum(comp_normal * rots, dim=-2)

            if compute_numerical_normal:
                comp_numerical_normal = comp_numerical_normal.unsqueeze(-1)
                comp_numerical_normal = torch.sum(comp_numerical_normal * rots, dim=-2)

        z_axis = cams[:, :, :3, 2].unsqueeze(-2).unsqueeze(-2)
        depth = torch.sum((points - rays_o) * -z_axis, dim=-1, keepdim=True)
        depth = depth * valid_mask

        out["mask"] = opacity
        out["valid_mask"] = valid_mask
        out["depth"] = depth
        out["points"] = points
        if compute_normal:
            out["normal"] = comp_normal
        if compute_numerical_normal:
            out["numerical_normal"] = comp_numerical_normal
            out["gradient"] = gradient

        if return_intermediate:
            intermediate = {
                "ray_indices": ray_indices,
                "t_starts": t_starts,
                "t_ends": t_ends,
                "alpha": alpha,
                "transmittance": transmittance,
            }
            return out, intermediate
        else:
            return out
