# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
from argparse import ArgumentParser
from pathlib import Path

import gin
import torch
from mmcv import Config
from opt import optimize


@gin.configurable
def pipeline(configroot, ckptroot, dataset_class):
    cfg = Config.fromfile(ckptroot / "neural_surface_recon" / "config.py")

    contains_board = cfg.fine_model_and_render.on_known_board

    dataset = dataset_class(dataroot=cfg.data.datadir, ckptroot=ckptroot)
    scene = dataset.get_scene()

    stages = [
        "microfacet_naive-envmap_sg",
        "microfacet_basis-envmap_ls",
        # "microfacet_basis-envmap_ls-shape_ls",
    ]

    result_root = Path(dataset.result_root)

    for stage in stages:
        print(f"Running stage {stage}...")
        stage_config = configroot / f"{stage}.gin"
        gin.parse_config_file(stage_config)

        if stage == "microfacet_basis-envmap_ls-shape_ls" and contains_board:
            v = scene["mesh.v"]
            v_mask = torch.ones_like(v)
            v_mask[v[:, 1] < 0.01] = 0
            v_mask = v_mask.to(bool)
            gin.bind_parameter("ShapeLS.v_mask", v_mask)

        sub_result_path = result_root / stage
        scene = optimize(scene=scene, dataset=dataset, result_path=sub_result_path)
    shutil.copytree(result_root / stages[-1] / "final", result_root, dirs_exist_ok=True)
    print(f"Final results are written to {result_root}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("configroot", type=str, help="gin config directory")
    parser.add_argument("ckptroot", type=str, help="gin config directory")
    args = parser.parse_args()

    configroot = Path(args.configroot)
    ckptroot = Path(args.ckptroot)

    gin.parse_config_file(configroot / "pipeline.gin")

    pipeline(configroot=configroot, ckptroot=ckptroot)
