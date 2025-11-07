# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Various explicit (grid) and hybrid (grid+MLP) implementation.

To add your new one, you should implement
- forward
- get_dense_grid
- scale_volume_grid (optional. just in case you want to progressively scale your grid)
See DenseGrid for an concrete example.
Finally, remember to add your new representation into the create_grid function.
"""

import neural_pbir_cuda_utils
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import torch_load


def create_grid(type, **kwargs):
    if type == "DenseGrid":
        return DenseGrid(**kwargs)
    elif type == "MSDenseGrid":
        return MSDenseGrid(**kwargs)
    elif type == "DenseGrid_MLP":
        return DenseGrid_MLP(**kwargs)
    elif type == "TensoRFGrid":
        return TensoRFGrid(**kwargs)
    elif type == "HashGrid":
        return HashGrid(**kwargs)
    else:
        raise NotImplementedError


""" Dense 3D grid
"""


class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.type = "DenseGrid"
        self.channels = channels
        self.world_size = world_size
        self.spatial_numel = np.prod(self.world_size.cpu().numpy())
        self.register_buffer("xyz_min", torch.Tensor(xyz_min))
        self.register_buffer("xyz_max", torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz, with_grad=False):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1,)
        ) * 2 - 1

        out = grid_sample_3d(self.grid, ind_norm)

        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)

        if with_grad:
            grid_grad = GeoGridGradientSample.apply(self.grid, ind_norm).reshape(
                *shape, 3
            )
            grid_grad = 2 / (self.xyz_max - self.xyz_min) * grid_grad.flip((-1,))
            return out, grid_grad
        return out

    def compute_grad(self, xyz):
        assert self.channels == 1
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1,)
        ) * 2 - 1
        grid_grad = GeoGridGradientSample.apply(self.grid, ind_norm).reshape(*shape, 3)
        grid_grad = 2 / (self.xyz_max - self.xyz_min) * grid_grad.flip((-1,))
        return grid_grad

    def scale_volume_grid(self, new_world_size):
        self.world_size = new_world_size
        self.spatial_numel = np.prod(self.world_size.cpu().numpy())
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(
                    self.grid.data,
                    size=tuple(new_world_size),
                    mode="trilinear",
                    align_corners=True,
                )
            )

    def get_dense_grid(self, sz=None):
        assert sz is None
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f"channels={self.channels}, world_size={self.world_size.tolist()}"


class MSDenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, config, **kwargs):
        super(MSDenseGrid, self).__init__()
        self.type = "MSDenseGrid"
        self.channels = channels
        self.register_buffer("xyz_min", torch.Tensor(xyz_min))
        self.register_buffer("xyz_max", torch.Tensor(xyz_max))

        grids = []
        for scale in config.upscales:
            grids.append(
                nn.Parameter(
                    torch.zeros([1, channels, *(world_size.float() * scale).long()])
                )
            )
        self.grids = nn.ParameterList(grids)
        self.register_buffer("activated_scale", torch.LongTensor([1]).squeeze())
        self.register_buffer("world_size", torch.LongTensor(tuple(grids[0].shape[2:])))
        self.spatial_numel = np.prod(self.world_size.cpu().numpy())

    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1,)
        ) * 2 - 1

        out = 0
        for ith_scale, grid in enumerate(self.grids):
            if ith_scale >= self.activated_scale:
                break
            out += grid_sample_3d(grid, ind_norm)

        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        self.activated_scale += 1
        self.world_size[0] = self.grids[self.activated_scale - 1].shape[0]
        self.world_size[1] = self.grids[self.activated_scale - 1].shape[1]
        self.world_size[2] = self.grids[self.activated_scale - 1].shape[2]
        self.spatial_numel = np.prod(self.world_size.cpu().numpy())

    def get_dense_grid(self, sz=None):
        if sz is None:
            sz = self.world_size
        sum_grid = 0
        for ith_scale, grid in enumerate(self.grids):
            if ith_scale >= self.activated_scale:
                break
            sum_grid += F.interpolate(
                grid, size=tuple(sz), mode="trilinear", align_corners=True
            )
        return sum_grid

    def extra_repr(self):
        return f"channels={self.channels}, world_size={self.world_size.tolist()}"


class DenseGrid_MLP(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, config, **kwargs):
        super(DenseGrid_MLP, self).__init__()
        self.type = "DenseGrid_MLP"
        self.channels = channels
        self.grid = DenseGrid(config.mid_channels, world_size, xyz_min, xyz_max)
        import tinycudann as tcnn

        self.mlp = tcnn.Network(
            n_input_dims=config.mid_channels,
            n_output_dims=channels,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": config.n_neurons,
                "n_hidden_layers": config.n_hidden_layers,
            },
        )
        with torch.no_grad():
            bd = 1 / np.sqrt(config.n_neurons)
            bd0 = 1 / np.sqrt(config.n_neurons)
            self.mlp.params.data[:] = (
                torch.rand(len(self.mlp.params.data)) * bd * 2 - bd
            )
            self.mlp.params.data[: config.n_neurons * config.n_neurons] = (
                torch.rand(config.n_neurons * config.n_neurons) * bd0 * 2 - bd0
            )

    def forward(self, xyz):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        feat = self.grid(xyz)
        out = self.mlp(feat).reshape(*shape, self.channels).float()
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        self.grid.scale_volume_grid(new_world_size)

    def get_dense_grid(self, sz=None):
        assert sz is None
        feat = self.grid.get_dense_grid()[0].moveaxis(0, -1)
        shape = feat.shape[:3]
        out = self.mlp(feat.flatten(0, 2)).float()
        return out.moveaxis(-1, 0).unsqueeze(0).unflatten(-1, shape)

    def extra_repr(self):
        return f"channels={self.channels} {str(self.grid)}"


""" Vector-Matrix decomposited grid
See TensoRF: Tensorial Radiance Fields
    (https://arxiv.org/abs/2203.09517)
"""


class TensoRFGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, config):
        super(TensoRFGrid, self).__init__()
        self.type = "TensoRFGrid"
        self.channels = channels
        self.world_size = world_size
        self.spatial_numel = np.prod(self.world_size.cpu().numpy())
        self.config = config
        self.register_buffer("xyz_min", torch.Tensor(xyz_min))
        self.register_buffer("xyz_max", torch.Tensor(xyz_max))
        X, Y, Z = world_size
        R = config["n_comp"]
        Rxy = config.get("n_comp_xy", R)
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y]) * 0.01)
        self.xz_plane = nn.Parameter(torch.randn([1, R, X, Z]) * 0.01)
        self.yz_plane = nn.Parameter(torch.randn([1, R, Y, Z]) * 0.01)
        self.x_vec = nn.Parameter(torch.randn([1, R, X, 1]) * 0.01)
        self.y_vec = nn.Parameter(torch.randn([1, R, Y, 1]) * 0.01)
        self.z_vec = nn.Parameter(torch.randn([1, Rxy, Z, 1]) * 0.01)
        if self.channels > 1:
            self.f_vec = nn.Parameter(torch.ones([R + R + Rxy, channels]))
            nn.init.kaiming_uniform_(self.f_vec, a=np.sqrt(5))

    def forward(self, xyz):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, -1, 3)
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        ind_norm = torch.cat([ind_norm, torch.zeros_like(ind_norm[..., [0]])], dim=-1)
        if self.channels > 1:
            out = compute_tensorf_feat(
                self.xy_plane,
                self.xz_plane,
                self.yz_plane,
                self.x_vec,
                self.y_vec,
                self.z_vec,
                self.f_vec,
                ind_norm,
            )
            out = out.reshape(*shape, self.channels)
        else:
            out = compute_tensorf_val(
                self.xy_plane,
                self.xz_plane,
                self.yz_plane,
                self.x_vec,
                self.y_vec,
                self.z_vec,
                ind_norm,
            )
            out = out.reshape(*shape)
        return out

    def scale_volume_grid(self, new_world_size):
        self.world_size = new_world_size
        self.spatial_numel = np.prod(self.world_size.cpu().numpy())
        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        self.xy_plane = nn.Parameter(
            F.interpolate(
                self.xy_plane.data, size=[X, Y], mode="bilinear", align_corners=True
            )
        )
        self.xz_plane = nn.Parameter(
            F.interpolate(
                self.xz_plane.data, size=[X, Z], mode="bilinear", align_corners=True
            )
        )
        self.yz_plane = nn.Parameter(
            F.interpolate(
                self.yz_plane.data, size=[Y, Z], mode="bilinear", align_corners=True
            )
        )
        self.x_vec = nn.Parameter(
            F.interpolate(
                self.x_vec.data, size=[X, 1], mode="bilinear", align_corners=True
            )
        )
        self.y_vec = nn.Parameter(
            F.interpolate(
                self.y_vec.data, size=[Y, 1], mode="bilinear", align_corners=True
            )
        )
        self.z_vec = nn.Parameter(
            F.interpolate(
                self.z_vec.data, size=[Z, 1], mode="bilinear", align_corners=True
            )
        )

    def get_dense_grid(self, sz=None):
        assert sz is None
        if self.channels > 1:
            feat = torch.cat(
                [
                    torch.einsum(
                        "rxy,rz->rxyz", self.xy_plane[0], self.z_vec[0, :, :, 0]
                    ),
                    torch.einsum(
                        "rxz,ry->rxyz", self.xz_plane[0], self.y_vec[0, :, :, 0]
                    ),
                    torch.einsum(
                        "ryz,rx->rxyz", self.yz_plane[0], self.x_vec[0, :, :, 0]
                    ),
                ]
            )
            grid = torch.einsum("rxyz,rc->cxyz", feat, self.f_vec)[None]
        else:
            grid = (
                torch.einsum("rxy,rz->xyz", self.xy_plane[0], self.z_vec[0, :, :, 0])
                + torch.einsum("rxz,ry->xyz", self.xz_plane[0], self.y_vec[0, :, :, 0])
                + torch.einsum("ryz,rx->xyz", self.yz_plane[0], self.x_vec[0, :, :, 0])
            )
            grid = grid[None, None]
        return grid

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}, n_comp={self.config["n_comp"]}'


def compute_tensorf_feat(
    xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, f_vec, ind_norm
):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = grid_sample_2d(xy_plane, ind_norm[:, :, :, [1, 0]]).flatten(0, 2).T
    xz_feat = grid_sample_2d(xz_plane, ind_norm[:, :, :, [2, 0]]).flatten(0, 2).T
    yz_feat = grid_sample_2d(yz_plane, ind_norm[:, :, :, [2, 1]]).flatten(0, 2).T
    x_feat = grid_sample_2d(x_vec, ind_norm[:, :, :, [3, 0]]).flatten(0, 2).T
    y_feat = grid_sample_2d(y_vec, ind_norm[:, :, :, [3, 1]]).flatten(0, 2).T
    z_feat = grid_sample_2d(z_vec, ind_norm[:, :, :, [3, 2]]).flatten(0, 2).T
    # Aggregate components
    feat = torch.cat(
        [
            xy_feat * z_feat,
            xz_feat * y_feat,
            yz_feat * x_feat,
        ],
        dim=-1,
    )
    feat = torch.mm(feat, f_vec)
    return feat


def compute_tensorf_val(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = grid_sample_2d(xy_plane, ind_norm[:, :, :, [1, 0]]).flatten(0, 2)
    xz_feat = grid_sample_2d(xz_plane, ind_norm[:, :, :, [2, 0]]).flatten(0, 2)
    yz_feat = grid_sample_2d(yz_plane, ind_norm[:, :, :, [2, 1]]).flatten(0, 2)
    x_feat = grid_sample_2d(x_vec, ind_norm[:, :, :, [3, 0]]).flatten(0, 2)
    y_feat = grid_sample_2d(y_vec, ind_norm[:, :, :, [3, 1]]).flatten(0, 2)
    z_feat = grid_sample_2d(z_vec, ind_norm[:, :, :, [3, 2]]).flatten(0, 2)
    # Aggregate components
    feat = (
        (xy_feat * z_feat).sum(0)
        + (xz_feat * y_feat).sum(0)
        + (yz_feat * x_feat).sum(0)
    )
    return feat


""" Mask grid
It supports query for the known free space and unknown space.
"""


class MaskGrid(nn.Module):
    def __init__(
        self,
        path=None,
        mask_cache_init_thres=None,
        mask=None,
        xyz_min=None,
        xyz_max=None,
    ):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch_load(path)
            self.mask_cache_init_thres = mask_cache_init_thres
            density = F.max_pool3d(
                st["model_state_dict"]["density.grid"],
                kernel_size=3,
                padding=1,
                stride=1,
            )
            alpha = 1 - torch.exp(
                -F.softplus(density + st["model_state_dict"]["act_shift"])
                * st["model_kwargs"]["voxel_size_ratio"]
            )
            mask = (alpha >= self.mask_cache_init_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st["model_kwargs"]["xyz_min"])
            xyz_max = torch.Tensor(st["model_kwargs"]["xyz_max"])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer("mask", mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer(
            "xyz2ijk_scale", (torch.Tensor(list(mask.shape)) - 1) / xyz_len
        )
        self.register_buffer("xyz2ijk_shift", -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        """Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = neural_pbir_cuda_utils.maskcache_lookup(
            self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift
        )
        mask = mask.reshape(shape)
        return mask

    @torch.no_grad()
    def ray_tracing(self, rs, rd, stepdist, max_n_steps):
        """Ray tracing on the occupancy grid
        @rs:          [B, 3] the starting position of the rays.
        @rd:          [B, 3] the ray direction (assume unit length).
        @stepdist:           step size.
        @max_n_steps: [B]    (t_max - t_now) / stepdist
        """
        return neural_pbir_cuda_utils.maskcache_ray_tracing(
            self.mask,
            rs,
            rd,
            stepdist,
            max_n_steps,
            self.xyz2ijk_scale,
            self.xyz2ijk_shift,
        )

    def extra_repr(self):
        return f"mask.shape={list(self.mask.shape)}"


""" Grid sampling to allow second derivative
"""


def grid_sample_2d(voxel, coord):
    return F.grid_sample(
        voxel, coord, padding_mode="border", mode="bilinear", align_corners=True
    )


def grid_sample_3d(voxel, coord):
    return F.grid_sample(
        voxel, coord, padding_mode="border", mode="bilinear", align_corners=True
    )


class GeoGridGradientSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxel, coord):
        assert voxel.ndim == coord.ndim
        assert voxel.ndim == 5
        assert voxel.shape[0] == 1
        assert voxel.shape[1] == 1
        coord = coord.contiguous()
        ctx.save_for_backward(voxel, coord)
        op = torch._C._jit_get_operation(f"aten::grid_sampler_3d_backward")
        grad_output = (
            torch.ones(coord.shape[:-1], requires_grad=False).unsqueeze(1).to(coord)
        )
        if isinstance(op, tuple):
            op = op[0]
            output_mask = (False, True)
            grad_coord = op(grad_output, voxel, coord, 0, 1, True, output_mask)[1]
        else:
            grad_coord = op(grad_output, voxel, coord, 0, 1, True)[1]
        ctx.save_for_backward(coord)
        ctx.grid_shape = voxel.shape
        return grad_coord

    @staticmethod
    def backward(ctx, grad_grad_coord):
        (coord,) = ctx.saved_tensors
        grad_voxel = None
        if ctx.needs_input_grad[0]:
            NB, NC, NZ, NY, NX = ctx.grid_shape
            grad_voxel = (
                neural_pbir_cuda_utils.grid_sample_3d_second_derivative_to_voxel(
                    coord.reshape(-1, 3),
                    grad_grad_coord.reshape(-1, 3),
                    NB,
                    NC,
                    NZ,
                    NY,
                    NX,
                )
            )
        if ctx.needs_input_grad[1]:
            # TODO
            pass
        return grad_voxel, None


""" Mesh & SDF utils
"""


def compute_sdf_from_mesh(coord_grid, mesh):
    from pytorch3d.ops import knn_points

    shape = coord_grid.shape[:-1]
    coord_grid = coord_grid.flatten(0, -2)
    mesh_vert = torch.Tensor(mesh.vertices)
    mesh_norm = torch.Tensor(np.copy(mesh.vertex_normals))
    nn_idx = knn_points(coord_grid[None], mesh_vert[None], K=1).idx.squeeze()
    mask_inside = ((mesh_vert[nn_idx] - coord_grid) * mesh_norm[nn_idx]).sum(-1) > 0
    sdf = (mesh_vert[nn_idx] - coord_grid).norm(dim=-1)
    sdf[mask_inside] = -sdf[mask_inside]
    return sdf.reshape(shape)


##########################
# Below is untested zone #
##########################
""" Hash grid
See Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
    (https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)
"""


class HashGrid(nn.Module):
    def __init__(
        self, channels, n_rest_scales, xyz_min, xyz_max, reproject=True, **kwargs
    ):
        super(HashGrid, self).__init__()
        self.type = "HashGrid"
        import tinycudann as tcnn

        config = kwargs["config"]
        self.L = config.get("L", 14)
        self.F = config.get("F", 2)
        self.log2_T = config.get("log2_T", 19)
        self.N_min = config.get("N_min", 32)
        self.N_max = config.get("N_max", 2048)
        self.per_level_scale = (self.N_max / self.N_min) ** (1 / (self.L - 1))
        self.grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.L,
                "n_features_per_level": self.F,
                "log2_hashmap_size": self.log2_T,
                "base_resolution": self.N_min,
                "per_level_scale": self.per_level_scale,
            },
        )
        if reproject:
            self.width = config.get("width", 64)
            self.depth = config.get("depth", 1)
            self.mlp = nn.Sequential(
                nn.Linear(self.L * self.F, self.width),
                nn.ReLU(inplace=True),
                *[
                    nn.Sequential(
                        nn.Linear(self.width, self.width),
                        nn.ReLU(inplace=True),
                    )
                    for _ in range(self.depth - 1)
                ],
                nn.Linear(self.width, channels),
            )
            # tcnn.Network(
            #     n_input_dims=self.L * self.F,
            #     n_output_dims=channels,
            #     network_config={
            #         "otype": "FullyFusedMLP",
            #         "activation": "ReLU",
            #         "output_activation": "None",
            #         "n_neurons": config.get('width', 64),
            #         "n_hidden_layers": config.get('depth', 1),
            #     }
            # )
            self.channels = channels
        else:
            self.mlp = None
            self.channels = self.L * self.F
        self.register_buffer("xyz_min", torch.Tensor(xyz_min))
        self.register_buffer("xyz_max", torch.Tensor(xyz_max))
        self.register_buffer("center", (self.xyz_max + self.xyz_min) * 0.5)
        self.register_buffer("radius", (self.xyz_max - self.xyz_min).max())

        self.n_rest_scales = n_rest_scales
        self.register_buffer("masking", torch.ones(self.L * self.F))
        if self.n_rest_scales > 0:
            self.masking[-self.n_rest_scales * self.F :] = 0
        assert self.n_rest_scales < self.L, "Too much progressive checkpoint"

        # for computing finite difference
        self.register_buffer(
            "finite_step",
            torch.FloatTensor(
                [
                    [-1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, 1],
                ]
            ),
        )

    @property
    def _world_size(self):
        scale = self.per_level_scale ** (self.L - 1 - self.n_rest_scales)
        return round(self.N_min * scale)

    @property
    def world_size(self):
        sz = self._world_size
        return torch.LongTensor([sz, sz, sz])

    @property
    def world_size_effective(self):
        sz = self._world_size
        lens = self.xyz_max - self.xyz_min
        return torch.Tensor(sz * lens / lens.max()).round().long()

    def world_size_effective_clip(self, maxlen):
        sz = self.world_size_effective
        if sz.max() > maxlen:
            sz = sz.float()
            sz = (maxlen * sz / sz.max()).round().long()
        return sz

    @property
    def voxel_size(self):
        sz = self._world_size
        return (self.xyz_max - self.xyz_min).max() / sz

    def compute_grad(self, xyz):
        shape = xyz.shape[:-1]
        voxel_size = self.voxel_size
        step = voxel_size * self.finite_step
        out_step = self.forward(
            (xyz.reshape(-1, 1, 3) + step).reshape(-1, 3), with_grad=False
        )
        if self.channels == 1:
            out_step = out_step.reshape(*shape, 3, 2)
        else:
            out_step = out_step.reshape(*shape, 3, 2, self.channels)
        grad = out_step.diff(dim=-1).squeeze(-1) / (2 * voxel_size)
        return out_step, grad

    def forward(self, xyz, with_grad=False):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        ind_norm = (xyz.reshape(-1, 3) - self.center) / self.radius + 0.5
        out = self.grid(ind_norm).float() * self.masking
        if self.mlp is not None:
            out = self.mlp(out).float()

        if self.channels == 1:
            out = out.reshape(*shape)
        else:
            out = out.reshape(*shape, self.channels)

        if with_grad:
            out_step, grad = self.compute_grad(xyz)
            return out, out_step, grad
        return out

    def scale_volume_grid(self):
        assert self.n_rest_scales > 0, "Already at the highest scale"
        self.n_rest_scales -= 1
        if self.n_rest_scales == 0:
            self.masking[:] = 1
        else:
            self.masking[: -self.n_rest_scales * self.F] = 1

    def get_dense_grid(self):
        world_size_effective = self.world_size_effective
        grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    self.xyz_min[0], self.xyz_max[0], world_size_effective[0]
                ),
                torch.linspace(
                    self.xyz_min[1], self.xyz_max[1], world_size_effective[1]
                ),
                torch.linspace(
                    self.xyz_min[2], self.xyz_max[2], world_size_effective[2]
                ),
                indexing="ij",
            ),
            -1,
        )
        dg = self(grid_xyz)
        if len(dg.shape) == 3:
            return dg[None, None]
        return dg.moveaxis(-1, 0)[None]
