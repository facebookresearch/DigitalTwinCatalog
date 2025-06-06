# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from copy import deepcopy
from pathlib import Path

import gin
import numpy as np

import torch
from irtk.io import read_image, to_torch_f
from irtk.renderer import Renderer
from irtk.sampling import sample_sphere
from irtk.scene import (
    EnvironmentLight,
    HDRFilm,
    Integrator,
    Mesh,
    MicrofacetBRDF,
    PerspectiveCameraFull,
    Scene,
)
from torch.utils.data import Dataset


@gin.configurable
class NeuralPBIRDataset(Dataset):
    def __init__(
        self,
        dataroot,
        ckptroot,
        resultroot=None,
        savemem=True,
        F0=0.04,
        integrator_type="path",
        integrator_config={"max_depth": 3, "hide_emitters": False},
    ):
        self.data_root = Path(dataroot)
        self.ckpt_root = Path(ckptroot)
        self.result_root = self.ckpt_root / "pbir" if resultroot is None else resultroot
        self.save_mem = savemem
        self.F0 = F0
        self.integrator_type = integrator_type
        self.integrator_config = integrator_config

        with open(self.data_root / "cameras.json", "r") as f:
            self.cameras = json.load(f)

        self.num_cameras = len(self.cameras["frames"])
        self.image_paths = []
        for i in range(self.num_cameras):
            self.image_paths.append(self.data_root / self.cameras["frames"][i]["path"])

        if not self.save_mem:
            self.images = []
            for image_path in self.image_paths:
                image = to_torch_f(read_image(image_path))
                self.images.append(image)

    def __len__(self):
        return self.num_cameras

    def __getitem__(self, idx):
        if self.save_mem:
            return to_torch_f(read_image(self.image_paths[idx]))
        else:
            return self.images[idx]

    def get_scene(self):
        scene = Scene()

        h, w, _ = self[0].shape

        fx = self.cameras["fx"] / w
        fy = self.cameras["fy"] / h
        cx = self.cameras["cx"] / w
        cy = self.cameras["cy"] / h

        for i in range(len(self.cameras["frames"])):
            to_world = np.array(self.cameras["frames"][i]["to_world"])
            scene.set(f"sensor {i}", PerspectiveCameraFull(fx, fy, cx, cy, to_world))

        stage_1_path = self.ckpt_root / "neural_surface_recon"
        stage_2_path = self.ckpt_root / "neural_distillation"

        mesh_path = stage_1_path / "mesh.obj"
        scene.set("mesh", Mesh.from_file(mesh_path, "mat", use_face_normal=False))

        diffuse = read_image(stage_2_path / "albedo.exr")
        specular = np.ones_like(diffuse) * self.F0
        roughness = read_image(stage_2_path / "roughness.exr")
        scene.set("mat", MicrofacetBRDF(diffuse, specular, roughness))

        envmap_path = stage_2_path / "envmap.exr"
        scene.set("envmap", EnvironmentLight.from_file(envmap_path))

        scene.set("film", HDRFilm(w, h))

        scene.set(
            "integrator",
            Integrator(
                self.integrator_type,
                self.integrator_config,
            ),
        )

        return scene
