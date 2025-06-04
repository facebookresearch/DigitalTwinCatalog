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

from .utils_sg import Envmap2SG, SG2Envmap


@gin.configurable
class EnvmapSG(Model):
    def __init__(
        self,
        scene,
        emitter_id,
        numLgtSGs=128,
        num_init_iter=1000,
        t_res_h=None,
        optimizer_kwargs={},
    ):
        super().__init__(scene)

        self.envmap = scene[emitter_id]

        if isinstance(t_res_h, int):
            self.envmap["radiance"] = (
                torch.ones((t_res_h, t_res_h * 2, 3), dtype=ftype, device=device) * 0.5
            )

        self.h = self.envmap["radiance"].shape[0]
        self.w = self.envmap["radiance"].shape[1]

        self.lgtSGs = Envmap2SG(
            self.envmap["radiance"].clone(),
            numLgtSGs=numLgtSGs,
            N_iter=num_init_iter,
            fixed_lobe=True,
        )
        self.lgtSGs = self.lgtSGs.to(device).requires_grad_()

        self.optimizer = torch.optim.Adam([self.lgtSGs], **optimizer_kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_data(self):
        envmap = SG2Envmap(self.lgtSGs, H=self.h, W=self.w)
        envmap = torch.clip(envmap, min=0, max=None)
        self.envmap["radiance"] = envmap

    def step(self):
        self.optimizer.step()

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
