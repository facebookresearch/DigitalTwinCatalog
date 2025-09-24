# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import gin
import numpy as np
import torch
from irtk.config import configs
from irtk.io import write_image
from irtk.model import Model

ftype = configs["ftype"]
device = configs["device"]

from .utils_largestep import connect_strip, LargeStepsOptimizer


@gin.configurable
class EnvmapLS(Model):
    def __init__(
        self,
        scene,
        emitter_id,
        t_res_h=None,
        optimizer_kwargs={},
    ):
        super().__init__(scene)

        self.envmap = scene[emitter_id]

        if isinstance(t_res_h, int):
            self.envmap["radiance"] = (
                torch.ones((t_res_h, t_res_h * 2, 3), dtype=ftype, device=device) * 0.5
            )

        self.envmap_radiance = self.envmap["radiance"].clone()

        # Compute connectivity
        self.h, self.w = self.envmap_radiance.shape[0:2]
        idx = np.arange(self.h * self.w).reshape(self.h, self.w)
        F = []
        for i in range(self.h - 1):
            F.append(connect_strip(idx[i], idx[i + 1], connect_ends=True))
        F = torch.from_numpy(np.concatenate(F))

        self.envmap_radiance = self.envmap_radiance.reshape(-1, 3)
        self.envmap_radiance.requires_grad = True

        self.optimizer = LargeStepsOptimizer(
            self.envmap_radiance, F, **optimizer_kwargs
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_data(self):
        self.envmap["radiance"] = self.envmap_radiance.reshape(self.h, self.w, 3)

    def step(self):
        self.optimizer.step()
        with torch.no_grad():
            self.envmap_radiance.copy_(torch.clamp_min(self.envmap_radiance, 0))

    def get_results(self):
        results = {
            "envmap": self.envmap["radiance"].detach(),
        }
        return results

    def write_results(self, result_path):
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        write_image(result_path / f"envmap.exr", results["envmap"])
