# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ckptroot", default="results/stanford_orb", type=Path)
parser.add_argument("--dataroot", default="data/stanford_orb", type=Path)
args = parser.parse_args()


for CAM in sorted(
    glob.glob(str(args.dataroot / "blender_HDR" / "*" / "cameras_novel" / "*json"))
):
    scene_name = Path(CAM).parents[1].stem
    with open(CAM) as f:
        meta = json.load(f)
        assert len(meta["frames"]) == 1
        stem = meta["frames"][0]["path"]
        env_scene_name = meta["frames"][0]["scene_name"]

    # distillation
    # CKPT_GEO = str(args.ckptroot / scene_name / 'neural_distillation' / 'mesh.obj')
    # CKPT_ALBEDO = str(args.ckptroot / scene_name / 'neural_distillation' / 'albedo.exr')
    # CKPT_ROUGH = str(args.ckptroot / scene_name / 'neural_distillation' / 'roughness.exr')

    # pbir
    CKPT_GEO = str(args.ckptroot / scene_name / "pbir" / "mesh.obj")
    CKPT_ALBEDO = str(args.ckptroot / scene_name / "pbir" / "diffuse.exr")
    CKPT_ROUGH = str(args.ckptroot / scene_name / "pbir" / "roughness.exr")

    # run blender
    envmap_name = f"{env_scene_name}_{stem}_for_{scene_name}.exr"
    CKPT_ENV = str(
        args.dataroot
        / "ground_truth"
        / env_scene_name
        / "env_map_for_blender"
        / envmap_name
    )
    assert os.path.isfile(CKPT_GEO)
    assert os.path.isfile(CKPT_ALBEDO)
    assert os.path.isfile(CKPT_ROUGH)
    assert os.path.isfile(CKPT_ENV)
    subprocess.run(
        [
            "python",
            "scripts/relit/relit.py",
            CKPT_GEO,
            CKPT_ALBEDO,
            CKPT_ROUGH,
            CAM,
            "--lgt_paths",
            CKPT_ENV,
            "--render_exr",
            "--with_bg",
        ]
    )
