# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import glob
from pathlib import Path

import cv2
import numpy as np
import torch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("glob_path", default="Path to the reconstructed envmaps.")
    args = parser.parse_args()

    for path in glob.glob(args.glob_path):
        env = cv2.imread(path, -1)  # HDR; BGR
        HW = env.shape[:2]
        # our 2 blender
        env = np.roll(env, HW[1] // 4, axis=1)
        # save result
        in_path = Path(path)
        out_path = in_path.parent / f"{in_path.stem}_for_blender.exr"
        cv2.imwrite(str(out_path), env)
        print("Save to", out_path)
