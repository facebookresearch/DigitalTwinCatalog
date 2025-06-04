# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def env_map_to_cam_to_world_by_convention(envmap, c2w, convention):
    R = c2w[:3, :3]
    H, W = envmap.shape[:2]
    theta, phi = np.meshgrid(
        np.linspace(-0.5 * np.pi, 1.5 * np.pi, W), np.linspace(0.0, np.pi, H)
    )
    viewdirs = np.stack(
        [-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
        axis=-1,
    ).reshape(H * W, 3)  # [H, W, 3]
    viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
    viewdirs = viewdirs.reshape(H, W, 3)
    # This is correspond to the convention of +Z at left, +Y at top
    # -np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)
    coord_y = ((np.arccos(viewdirs[..., 1]) / np.pi * (H - 1) + H) % H).astype(
        np.float32
    )
    coord_x = (
        (
            (np.arctan2(viewdirs[..., 0], -viewdirs[..., 2]) + np.pi)
            / 2
            / np.pi
            * (W - 1)
            + W
        )
        % W
    ).astype(np.float32)
    envmap_remapped = cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)
    if convention == "orb":
        return envmap_remapped
    if convention == "physg":
        # change convention from ours (Left +Z, Up +Y) to physg (Left -Z, Up +Y)
        envmap_remapped_physg = np.roll(envmap_remapped, W // 2, axis=1)
        return envmap_remapped_physg
    if convention == "nerd":
        # change convention from ours (Left +Z-X, Up +Y) to nerd (Left +Z+X, Up +Y)
        envmap_remapped_nerd = envmap_remapped[:, ::-1, :]
        return envmap_remapped_nerd

    assert convention == "invrender", convention
    # change convention from ours (Left +Z-X, Up +Y) to invrender (Left -X+Y, Up +Z)
    theta, phi = np.meshgrid(
        np.linspace(1.0 * np.pi, -1.0 * np.pi, W), np.linspace(0.0, np.pi, H)
    )
    viewdirs = np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1
    )  # [H, W, 3]
    # viewdirs = np.stack([-viewdirs[...,0], viewdirs[...,2], viewdirs[...,1]], axis=-1)
    coord_y = ((np.arccos(viewdirs[..., 1]) / np.pi * (H - 1) + H) % H).astype(
        np.float32
    )
    coord_x = (
        (
            (np.arctan2(viewdirs[..., 0], -viewdirs[..., 2]) + np.pi)
            / 2
            / np.pi
            * (W - 1)
            + W
        )
        % W
    ).astype(np.float32)
    envmap_remapped_Inv = cv2.remap(envmap_remapped, coord_x, coord_y, cv2.INTER_LINEAR)
    return envmap_remapped_Inv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", default="data/stanford_orb/ground_truth", type=Path)
    parser.add_argument("--im_dir", default="data/stanford_orb/blender_HDR/", type=Path)
    args = parser.parse_args()

    for path in args.gt_dir.iterdir():
        if not path.is_dir():
            continue

        scene_name = path.stem

        # parse source scene mesh to relit
        with open(args.im_dir / scene_name / "transforms_novel.json") as f:
            src_scene_set = set(v["scene_name"] for v in json.load(f)["frames"])

        src_c2w = dict()
        for src_scene in src_scene_set:
            src_c2w[src_scene] = dict()
            with open(args.im_dir / src_scene / "transforms_novel.json") as f:
                for v in json.load(f)["frames"]:
                    if v["scene_name"] == scene_name:
                        poseid = Path(v["file_path"]).stem
                        src_c2w[src_scene][poseid] = np.array(v["transform_matrix"])

        # parse target testing frame
        with open(args.im_dir / scene_name / "transforms_test.json") as f:
            meta = json.load(f)

        # process
        indir = path / "env_map"
        outdir = path / "env_map_for_blender"
        # outdir = path / 'env_map_for_neuralpbir'
        outdir.mkdir(exist_ok=True)
        for frame in meta["frames"]:
            envpath = (
                args.gt_dir
                / scene_name
                / "env_map"
                / f"{Path(frame['file_path']).stem}.exr"
            )
            poseid = envpath.stem

            env_ = cv2.imread(str(envpath), -1)  # HDR; BGR
            HW = env_.shape[:2]

            for src_scene in src_scene_set:
                env = np.copy(env_)
                R_target = np.array(frame["transform_matrix"])[:3, :3]
                R_src = src_c2w[src_scene][poseid][:3, :3]

                # to ours
                # env = env_map_to_cam_to_world_by_convention(env, R_target, 'orb')
                env = env_map_to_cam_to_world_by_convention(env, R_src, "orb")
                # env = env_map_to_cam_to_world_by_convention(env, R_target @ R_src.T @ R_target, 'orb')

                # to our
                env = np.roll(env, HW[1] // 2, axis=1)

                # to blender
                env = np.roll(env, HW[1] // 4, axis=1)

                # save result
                out_path = outdir / f"{scene_name}_{poseid}_for_{src_scene}.exr"
                cv2.imwrite(str(out_path), env)
                print("Save to", out_path)
