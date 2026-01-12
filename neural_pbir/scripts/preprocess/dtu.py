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
import numpy as np
from tqdm import tqdm, trange


def load_aabb(datadir):
    # region of interest to extract mesh
    cameras_sphere = np.load(datadir / "cameras_sphere.npz")
    scale_mat = cameras_sphere["scale_mat_0"]
    object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
    object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
    # cameras_large = np.load(os.path.join(cfg_data['datadir'], 'cameras_large.npz'))
    # rescale_mat = np.linalg.inv(cameras_sphere['scale_mat_0']) @ cameras_large['scale_mat_0']
    # object_bbox_min = rescale_mat @ object_bbox_min
    # object_bbox_max = rescale_mat @ object_bbox_max
    object_bbox_min = object_bbox_min[:3] * scale_mat[0, 0] + scale_mat[:3, 3]
    object_bbox_max = object_bbox_max[:3] * scale_mat[0, 0] + scale_mat[:3, 3]
    return [list(object_bbox_min), list(object_bbox_max)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datadir", type=Path)
    args = parser.parse_args()

    assert args.datadir.exists(), f"Folder not found: {args.datadir}"

    all_paths = sorted((args.datadir / "image").glob("*.png"))
    all_paths_mask = sorted((args.datadir / "mask").glob("*.png"))
    assert len(all_paths) > 0
    assert len(all_paths_mask) > 0
    assert len(all_paths) == len(all_paths_mask)

    # load camera parameters
    cameras_sphere = np.load(args.datadir / "cameras_sphere.npz")
    world_mats = np.stack(
        [cameras_sphere["world_mat_%d" % idx] for idx in range(len(all_paths))]
    ).astype(np.float32)

    # compute camera intrinsics and poses
    Ks, poses = [], []
    for world_mat in world_mats:
        P = (world_mat)[:3, :4]
        intrinsic, extrinsics_R, t_vec, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        intrinsic /= intrinsic[2, 2]
        c2w = np.eye(4)
        c2w[:3, :3] = np.linalg.inv(extrinsics_R)
        c2w[:3, [3]] = t_vec[:3] / t_vec[3]
        c2w[:, :2] *= -1  # to pbir coordinate system
        Ks.append(intrinsic.astype(np.float32))
        poses.append(c2w.astype(np.float32))

    assert len(Ks) == len(all_paths)

    Ks = np.stack(Ks)
    poses = np.stack(poses)
    aabb = load_aabb(args.datadir)

    fx = float(Ks[:, 0, 0].mean())
    fy = float(Ks[:, 1, 1].mean())
    cx = float(Ks[:, 0, 2].mean())
    cy = float(Ks[:, 1, 2].mean())

    meta = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "aabb": aabb,
        "frames": [
            {
                "path": f"image/{all_paths[i].stem}.png",
                "mask": f"mask/{all_paths_mask[i].stem}.png",
                "to_world": poses[i].tolist(),
            }
            for i in range(len(poses))
        ],
    }

    # save result
    outpath = args.datadir / "cameras.json"
    with open(outpath, "w") as f:
        json.dump(meta, f, indent=4)
    print("Result is saved to", outpath)
