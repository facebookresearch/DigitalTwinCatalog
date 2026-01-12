# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import glob
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import imageio
import imageio.v3 as iio
import numpy as np
import pyexr
from tqdm import tqdm, trange


def cv2_downsize(f: np.ndarray, downsize_factor=None):
    if downsize_factor is not None:
        f = cv2.resize(
            f,
            (0, 0),
            fx=1 / downsize_factor,
            fy=1 / downsize_factor,
            interpolation=cv2.INTER_AREA,
        )
    return f


def load_rgb_exr(path: str, downsize_factor=None) -> np.ndarray:
    # NO correction
    f = pyexr.open(path).get()
    assert f.dtype == np.float32, f.dtype
    assert len(f.shape) == 3 and f.shape[2] == 3, f.shape
    f = cv2_downsize(f, downsize_factor)
    return f


def load_hdr_rgba(path, downsize_factor=None) -> np.ndarray:
    rgb = load_rgb_exr(path, downsize_factor)
    mask = imageio.imread(
        os.path.join(
            os.path.dirname(path) + "_mask",
            os.path.basename(path).replace(".exr", ".png"),
        )
    )
    assert mask.shape == (2048, 2048), mask.shape
    assert mask.dtype == np.uint8, mask.dtype
    mask = (mask / 255).astype(np.float32)
    mask = cv2_downsize(mask, downsize_factor)
    rgba = np.concatenate([rgb, mask[:, :, None]], axis=2)
    return rgba


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

        outdir = args.datadir / f"{splitname}_512x512"
        outdir.mkdir(exist_ok=True)
        for frame in meta["frames"]:
            fname = Path(frame["file_path"]).stem
            new_frame = {
                "path": f"{splitname}_512x512/{fname}_image.exr",
                "mask": f"{splitname}_512x512/{fname}_mask.png",
            }

            rgba = load_hdr_rgba(
                str(args.datadir / f"{frame['file_path']}.exr"), downsize_factor=4
            )
            pyexr.write(str(args.datadir / new_frame["path"]), rgba[..., :3])
            imageio.imwrite(
                args.datadir / new_frame["mask"], (rgba[..., 3] * 255).astype(np.uint8)
            )

            c2w = np.array(frame["transform_matrix"]).astype(np.float32)
            c2w[:, [1, 2]] *= -1  # to neural_surface internal coordinate system
            # c2w[[1,2]] = np.stack([c2w[2], -c2w[1]])  # swap yz
            c2w[:, :2] *= -1  # to pbir coordinate system
            new_frame["to_world"] = c2w.tolist()
            new_meta["frames"].append(new_frame)
            new_meta["split"][splitname].append(idx)
            idx += 1

        if camera_angle_x is None:
            H, W = iio.imread(args.datadir / new_frame["path"]).shape[:2]
            camera_angle_x = float(meta["camera_angle_x"])
        else:
            # sanity check
            assert camera_angle_x == float(meta["camera_angle_x"])

    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    new_meta["fx"] = focal
    new_meta["fy"] = focal
    new_meta["cx"] = 0.5 * W
    new_meta["cy"] = 0.5 * H
    new_meta["aabb"] = [[-0.6, -0.6, -0.6], [0.6, 0.6, 0.6]]

    # save result
    outpath = args.datadir / "cameras.json"
    with open(outpath, "w") as f:
        json.dump(new_meta, f, indent=4)
    print("Result is saved to", outpath)

    # load novel lighting setup
    novel_dir = args.datadir / "cameras_novel"
    novel_dir.mkdir(exist_ok=True)
    with open(os.path.join(args.datadir, f"transforms_novel.json"), "r") as fp:
        meta = json.load(fp)

    for frame in meta["frames"]:
        fname = Path(frame["file_path"]).stem

        H, W = 512, 512
        camera_angle_x = float(frame["camera_angle_x"])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        c2w = np.array(frame["transform_matrix"]).astype(np.float32)
        c2w[:, [1, 2]] *= -1  # to neural_surface internal coordinate system
        # c2w[[1,2]] = np.stack([c2w[2], -c2w[1]])  # swap yz
        c2w[:, :2] *= -1  # to pbir coordinate system

        new_meta = {
            "split": {"test": [0]},
            "fx": focal,
            "fy": focal,
            "cx": 0.5 * W,
            "cy": 0.5 * H,
            "frames": [
                {
                    "path": fname,
                    "to_world": c2w.tolist(),
                    "scene_name": frame["scene_name"],
                    "file_path": frame["file_path"],
                }
            ],
        }

        # save result
        outpath = novel_dir / f"{frame['scene_name']}_{fname}.json"
        with open(outpath, "w") as f:
            json.dump(new_meta, f, indent=4)
        print("Result is saved to", outpath)
