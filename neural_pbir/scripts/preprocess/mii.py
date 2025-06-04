# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, copy, glob, json, os, random, shutil, sys, time
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm, trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datadir", type=Path)
    args = parser.parse_args()

    assert args.datadir.exists(), f"Folder not found: {args.datadir}"

    # load meta
    new_meta = {
        "split": {"train": [], "test": []},
        "frames": [],
    }
    idx = 0
    H, W, camera_angle_x = None, None, None
    for splitname in ["train", "test"]:
        # load camera poses
        with open(
            os.path.join(args.datadir, f"transforms_{splitname}.json"), "r"
        ) as fp:
            meta = json.load(fp)

        for frame in meta["frames"]:
            if (
                splitname == "train"
                or not (args.datadir / f'{frame["file_path"]}_rgba.png').exists()
            ):
                new_frame = {
                    "path": f'{frame["file_path"]}_rgb.exr',
                    "mask": f'{frame["file_path"]}_mask.png',
                }
            else:
                new_frame = {
                    "path": f'{frame["file_path"]}_rgba.png',
                    "mask_from_alpha": True,
                }
            c2w = np.array(frame["transform_matrix"]).astype(np.float32)
            c2w[:, [1, 2]] *= -1  # to neural_surface internal coordinate system
            c2w[[1, 2]] = np.stack([c2w[2], -c2w[1]])  # swap yz
            c2w[:, :2] *= -1  # to pbir coordinate system
            new_frame["to_world"] = c2w.tolist()
            new_meta["frames"].append(new_frame)
            new_meta["split"][splitname].append(idx)
            idx += 1

        if camera_angle_x is None:
            H, W = imageio.imread(args.datadir / new_frame["path"]).shape[:2]
            camera_angle_x = float(meta["camera_angle_x"])
        else:
            assert camera_angle_x == float(meta["camera_angle_x"])

    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    new_meta["fx"] = focal
    new_meta["fy"] = focal
    new_meta["cx"] = 0.5 * W
    new_meta["cy"] = 0.5 * H
    new_meta["aabb"] = [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]

    # save result
    outpath = args.datadir / "cameras.json"
    with open(outpath, "w") as f:
        json.dump(new_meta, f, indent=4)
    print("Result is saved to", outpath)
