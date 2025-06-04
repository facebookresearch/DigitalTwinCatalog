# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from irtk.model import Model
from irtk.io import *
from irtk.config import *

from pathlib import Path

import gin
import torch


@gin.configurable
class MicrofacetBasis(Model):
    def __init__(
        self,
        scene,
        mat_id,
        N=50,
        init_iter=300,
        d_max=1,
        d_min=0,
        s_max=1,
        s_min=0,
        r_max=1,
        r_min=0,
        weight_map_lr=1e-1,
        d_lr=1e-1,
        s_lr=1e-1,
        r_lr=1e-2,
        t_res=None,
        optimizer_kwargs={},
    ):
        super().__init__(scene)

        self.brdf = self.scene[mat_id]

        if isinstance(t_res, int):
            self.brdf["d"] = (
                torch.ones((t_res, t_res, 3), dtype=ftype, device=device) * 0.5
            )
            self.brdf["s"] = (
                torch.ones((t_res, t_res, 3), dtype=ftype, device=device) * 0.5
            )
            self.brdf["r"] = (
                torch.ones((t_res, t_res, 1), dtype=ftype, device=device) * 0.5
            )

        d_range = d_max - d_min
        s_range = s_max - s_min
        r_range = r_max - r_min

        self.t_range = torch.tensor(
            [d_range, d_range, d_range, s_range, s_range, s_range, r_range],
            dtype=ftype,
            device=device,
        )
        self.t_min = torch.tensor(
            [d_min, d_min, d_min, s_min, s_min, s_min, r_min],
            dtype=ftype,
            device=device,
        )

        self.t_shape = list(self.brdf["d"].shape)
        self.t_shape[2] = 7

        self.weight_map = torch.normal(
            0, 1e-4, (self.t_shape[0] * self.t_shape[1], N), dtype=ftype, device=device
        )
        self.weight_map.requires_grad = True

        brdf_points = torch.rand((N, 7), dtype=ftype, device=device) * 2 - 1
        self._d = brdf_points[:, 0:3]
        self._s = brdf_points[:, 3:6]
        self._r = brdf_points[:, 6:7]
        self._d.requires_grad = True
        self._s.requires_grad = True
        self._r.requires_grad = True

        print("Initializing using NMF...")
        textures = torch.cat([self.brdf["d"], self.brdf["s"], self.brdf["r"]], dim=2)
        init_opt = torch.optim.Adam(
            [self.weight_map, self._d, self._s, self._r], lr=1e-1
        )
        for _ in range(init_iter):
            init_opt.zero_grad()
            t = self.get_textures()
            diff = textures - t
            loss = diff.pow(2).sum()
            loss.backward()
            init_opt.step()
        print("Done.")

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.weight_map, "lr": weight_map_lr},
                {"params": self._d, "lr": d_lr},
                {"params": self._s, "lr": s_lr},
                {"params": self._r, "lr": r_lr},
            ],
            **optimizer_kwargs,
        )

    def get_textures(self):
        bp = torch.concat([self._d, self._s, self._r], dim=1)
        bp = torch.sigmoid(bp)
        w = torch.softmax(self.weight_map, dim=1)
        t = (w @ bp).reshape(self.t_shape)
        t = t * self.t_range + self.t_min
        return t

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_data(self):
        t = self.get_textures()
        self.brdf["d"] = t[:, :, 0:3]
        self.brdf["s"] = t[:, :, 3:6]
        self.brdf["r"] = t[:, :, 6:7]

    def step(self):
        self.optimizer.step()

    def get_results(self):
        t = self.get_textures().detach()
        results = {
            "d": t[:, :, 0:3],
            "s": t[:, :, 3:6],
            "r": t[:, :, 6:7],
        }
        return results

    def write_results(self, result_path):
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        write_image(result_path / f"diffuse.exr", results["d"])
        write_image(result_path / f"specular.exr", results["s"])
        write_image(result_path / f"roughness.exr", results["r"])
