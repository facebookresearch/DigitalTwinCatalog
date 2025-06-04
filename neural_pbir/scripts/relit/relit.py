# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import subprocess

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

cur_dir = os.path.dirname(os.path.abspath(__file__))

lgt_paths = []
for path in args.lgt_paths:
    lgt_paths.extend(glob.glob(path))
lgt_paths = ",".join(sorted(lgt_paths))

assert os.path.isfile(args.mesh_path), f"{args.mesh_path} not found !!"
assert os.path.isfile(args.albedo_path), f"{args.albedo_path} not found !!"
assert os.path.isfile(args.rough_path), f"{args.rough_path} not found !!"
assert os.path.isfile(args.camera_path), f"{args.camera_path} not found !!"
assert os.path.isfile(lgt_paths), f"{lgt_paths} not found !!"

subprocess.run(
    [
        "blender",
        os.path.join(cur_dir, "relit.blend"),
        "--background",
        "--python",
        os.path.join(cur_dir, "blender_script.py"),
        "--",
        "1" if args.debug else "0",
        "0" if args.no_gpu else "1",
        args.mesh_path,
        args.albedo_path,
        args.rough_path,
        lgt_paths,
        args.camera_path,
        args.split,
        args.filter_type,
        str(args.filter_width),
        "1" if args.with_bg else "0",
        str(args.F0),
        "1" if args.render_exr else "0",
    ]
)
