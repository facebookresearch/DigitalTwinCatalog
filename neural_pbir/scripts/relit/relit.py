# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("mesh_path", help="path to mesh")
parser.add_argument("albedo_path", help="path to albedo")
parser.add_argument("rough_path", help="path to roughness")
parser.add_argument("camera_path", help="path to camera json")
parser.add_argument(
    "--lgt_paths", nargs="*", default=[], help="path to environment map"
)
parser.add_argument("--debug", action="store_true", help="run only 1 example")
parser.add_argument("--split", default="test", help="data split to render")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument(
    "--filter_type",
    default="BLACKMAN_HARRIS",
    choices=["BOX", "GAUSSIAN", "BLACKMAN_HARRIS"],
)
parser.add_argument("--filter_width", default=1.5, type=float)
parser.add_argument(
    "--with_bg", action="store_true", help="render bg pixel from environment map"
)
parser.add_argument("--F0", default=0.04, type=float)
parser.add_argument("--render_exr", action="store_true")
args = parser.parse_args()

cur_dir = Path(__file__).resolve().parent

# Expand lgt_paths using glob and pathlib, and join with comma
lgt_path_list = []
for path in args.lgt_paths:
    lgt_path_list.extend(glob.glob(str(path)))
lgt_path_list = sorted([str(Path(p).resolve()) for p in lgt_path_list])
lgt_paths_str = ",".join(lgt_path_list)

# Use pathlib for all file checks and paths
mesh_path = Path(args.mesh_path).resolve()
albedo_path = Path(args.albedo_path).resolve()
rough_path = Path(args.rough_path).resolve()
camera_path = Path(args.camera_path).resolve()

assert mesh_path.is_file(), f"{mesh_path} not found !!"
assert albedo_path.is_file(), f"{albedo_path} not found !!"
assert rough_path.is_file(), f"{rough_path} not found !!"
assert camera_path.is_file(), f"{camera_path} not found !!"
if lgt_paths_str:
    # Only check if not empty
    for lgt_path in lgt_paths_str.split(","):
        assert Path(lgt_path).is_file(), f"{lgt_path} not found !!"

subprocess.run(
    [
        "blender",
        str(cur_dir / "relit.blend"),
        "--background",
        "--python",
        str(cur_dir / "blender_script.py"),
        "--",
        "1" if args.debug else "0",
        "0" if args.no_gpu else "1",
        str(mesh_path),
        str(albedo_path),
        str(rough_path),
        lgt_paths_str,
        str(camera_path),
        args.split,
        args.filter_type,
        str(args.filter_width),
        "1" if args.with_bg else "0",
        str(args.F0),
        "1" if args.render_exr else "0",
    ]
)
