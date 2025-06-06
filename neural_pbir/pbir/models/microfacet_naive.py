# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from irtk.model import Model
from irtk.io import *
from irtk.config import configs

ftype = configs["ftype"]
device = configs["device"]

from pathlib import Path

import gin
import torch


@gin.configurable
class MicrofacetNaive(Model):
    def __init__(
        self,
        scene,
        mat_id,
        d_min=0.0,
        d_max=1.0,
        s_min=0.0,
        s_max=1.0,
        r_min=0.0,
        r_max=1.0,
        d_lr=1e-2,
        s_lr=1e-2,
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

        self._d_v_min = d_min
        self._s_v_min = s_min
        self._r_v_min = r_min

        self._d_v_span = d_max - d_min
        self._s_v_span = s_max - d_min
        self._r_v_span = r_max - d_min

        self._d_data = torch.logit(
            torch.clip(self.brdf["d"], 1e-5, 1 - 1e-5)
        ).requires_grad_()
        self._s_data = torch.logit(
            torch.clip(self.brdf["s"], 1e-5, 1 - 1e-5)
        ).requires_grad_()
        self._r_data = torch.logit(
            torch.clip(self.brdf["r"], 1e-5, 1 - 1e-5)
        ).requires_grad_()

        self.optimizer = torch.optim.Adam(
            [self._d_data, self._s_data, self._r_data], **optimizer_kwargs
        )

        self.optimizer = torch.optim.Adam(
            [
                {"params": self._d_data, "lr": d_lr},
                {"params": self._s_data, "lr": s_lr},
                {"params": self._r_data, "lr": r_lr},
            ],
            **optimizer_kwargs,
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_data(self):
        self.brdf["d"] = torch.sigmoid(self._d_data) * self._d_v_span + self._d_v_min
        self.brdf["s"] = torch.sigmoid(self._s_data) * self._s_v_span + self._s_v_min
        self.brdf["r"] = torch.sigmoid(self._r_data) * self._r_v_span + self._r_v_min

    def step(self):
        self.optimizer.step()

    def get_results(self):
        results = {
            "d": self.brdf["d"].detach(),
            "s": self.brdf["s"].detach(),
            "r": self.brdf["r"].detach(),
        }
        return results

    def write_results(self, result_path):
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        write_image(result_path / f"diffuse.exr", results["d"])
        write_image(result_path / f"specular.exr", results["s"])
        write_image(result_path / f"roughness.exr", results["r"])
