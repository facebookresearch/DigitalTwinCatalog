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

import imageio
import imageio.v3 as iio
import numpy as np
import pyexr
import pytorch3d
import pytorch3d.io
import pytorch3d.renderer
import pytorch3d.structures
import torch
from tqdm import tqdm, trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("cameras_path", help="camera path", type=Path)
    parser.add_argument("mesh_path", help="mesh path", type=Path)
    parser.add_argument("--split", default="test", help="data split to render")
    args = parser.parse_args()

    mesh = pytorch3d.io.load_obj(args.mesh_path, device="cuda")
    vert = mesh[0]
    faces = mesh[1].verts_idx[None]
    mesh_torch = pytorch3d.structures.Meshes(verts=vert[None], faces=faces)
    mesh_torch.verts_normals_packed()
    normals = mesh_torch.verts_normals_padded()

    with open(args.cameras_path) as f:
        cameras = json.load(f)

    outdir = args.mesh_path.parent / "render_geo"
    outdir.mkdir(exist_ok=True)

    K = torch.tensor(
        [
            [cameras["fx"], 0, cameras["cx"], 0],
            [0, cameras["fy"], cameras["cy"], 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    cxcy = K[[0, 1], [2, 2]]
    image_size = [round(cxcy[1].item() * 2), round(cxcy[0].item() * 2)]

    for frame_idx in tqdm(cameras["split"][args.split]):
        frame = cameras["frames"][frame_idx]
        c2w = torch.tensor(frame["to_world"], dtype=torch.float32, device="cuda")
        c2w[:, :2] *= -1

        # contruct image projection matrix
        extrinsic = c2w.inverse()
        P = K @ extrinsic
        vert_ndc = torch.einsum("ij,nj->ni", P[:3, :3], vert) + P[:3, 3]
        vert_ndc[:, :2] = -(vert_ndc[:, :2] / vert_ndc[:, [2]] - cxcy) / cxcy.min()

        # rasterization
        # vertices should be in camera ndc space
        mesh_torch = pytorch3d.structures.Meshes(verts=vert_ndc[None], faces=faces)
        pix_to_face, zbuf, bary_coords, dists = (
            pytorch3d.renderer.mesh.rasterize_meshes(
                mesh_torch,
                image_size=image_size,
                faces_per_pixel=1,
                perspective_correct=True,
                cull_backfaces=True,
            )
        )

        # interpolate vertices feature to form rasterized 2D feature
        frag = pytorch3d.renderer.mesh.rasterizer.Fragments(
            pix_to_face, zbuf, bary_coords, dists
        )
        mesh_torch.textures = pytorch3d.renderer.mesh.textures.TexturesVertex(
            torch.einsum("dnc,rc->dnr", normals, extrinsic[:3, :3])
        )

        # save results
        rast_normal = mesh_torch.sample_textures(frag).squeeze().cpu().numpy()
        rast_normal[..., 1:] *= -1
        rast_depth = zbuf.squeeze().clamp_min(0).cpu().numpy()

        pyexr.write(str(outdir / f"{Path(frame['path']).stem}.zbuf.exr"), rast_depth)
        pyexr.write(
            str(outdir / f"{Path(frame['path']).stem}.normal.exr"),
            rast_normal,
            channel_names=["X", "Y", "Z"],
        )

        dmin = rast_depth[rast_depth > 0].min()
        dmax = rast_depth.max()
        imageio.imwrite(
            outdir / f"{Path(frame['path']).stem}.zbuf.png",
            np.clip((rast_depth - dmin) / (dmax - dmin) * 255, 0, 255).astype(np.uint8),
        )
        imageio.imwrite(
            outdir / f"{Path(frame['path']).stem}.normal.png",
            np.clip((rast_normal + 1) * 0.5 * 255, 0, 255).astype(np.uint8),
        )
