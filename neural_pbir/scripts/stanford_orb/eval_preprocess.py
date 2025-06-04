#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, copy, glob, json, os, random, shutil, sys, time
import json
from pathlib import Path


def init_scene_info(gt_dir, hdr_dir, scene):
    with open(hdr_dir / scene / "transforms_test.json") as f:
        nv_fp = [
            hdr_dir / scene / f"{frame['file_path']}.exr"
            for frame in json.load(f)["frames"]
        ]

    with open(hdr_dir / scene / "transforms_novel.json") as f:
        nl_fp = [
            hdr_dir / frame["scene_name"] / f"{frame['file_path']}.exr"
            for frame in json.load(f)["frames"]
        ]

    return dict(
        view=[
            dict(
                output_image=None,
                target_image=fp.resolve().as_posix(),
            )
            for fp in nv_fp
        ],
        light=[
            dict(
                output_image=None,
                target_image=fp.resolve().as_posix(),
            )
            for fp in nl_fp
        ],
        geometry=[
            dict(
                output_depth=None,
                target_depth=path.resolve().as_posix(),
                output_normal=None,
                target_normal=(gt_dir / scene / "surface_normal" / f"{path.stem}.npy")
                .resolve()
                .as_posix(),
                target_mask=(hdr_dir / scene / "test_mask" / f"{path.stem}.png")
                .resolve()
                .as_posix(),
            )
            for path in sorted((gt_dir / scene / "z_depth").glob("*npy"))
        ],
        material=[
            dict(
                output_image=None,
                target_image=path.resolve().as_posix(),
                target_mask=(hdr_dir / scene / "test_mask" / f"{path.stem}.png")
                .resolve()
                .as_posix(),
            )
            for path in sorted((gt_dir / scene / "pseudo_gt_albedo").glob("*png"))
        ],
        shape=dict(
            output_mesh=None,
            target_mesh=(gt_dir / scene / "mesh_blender" / "mesh.obj")
            .resolve()
            .as_posix(),
        ),
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args.add_argument("--data_dir", default="data/stanford_orb/", type=Path)
    args.add_argument("--ckpt_dir", default="results/stanford_orb/", type=Path)
    args.add_argument(
        "--stage",
        default="pbir",
        choices=[
            "neural_distillation",
            "pbir",
            "pbir-1",
            "pbir-2",
            "pbir-3",
        ],
    )
    args = args.parse_args()

    gt_dir = args.data_dir / "ground_truth"
    hdr_dir = args.data_dir / "blender_HDR"

    # Packed info
    info = dict()
    for path in gt_dir.iterdir():
        if not path.is_dir():
            continue
        scene = path.stem
        info[scene] = init_scene_info(gt_dir=gt_dir, hdr_dir=hdr_dir, scene=scene)

        """ Organize our own result """
        if args.stage == "pbir-1":
            ours_dir = (
                args.ckpt_dir / scene / "pbir" / "microfacet_naive-envmap_sg" / "final"
            )
        elif args.stage == "pbir-2":
            ours_dir = (
                args.ckpt_dir / scene / "pbir" / "microfacet_basis-envmap_ls" / "final"
            )
        elif args.stage == "pbir-3":
            ours_dir = (
                args.ckpt_dir
                / scene
                / "pbir"
                / "microfacet_basis-envmap_ls-shape_ls"
                / "final"
            )
        else:
            ours_dir = args.ckpt_dir / scene / args.stage

        # shape
        info[scene]["shape"]["output_mesh"] = (
            (ours_dir / "mesh.obj").resolve().as_posix()
        )
        assert os.path.isfile(info[scene]["shape"]["output_mesh"])

        # geometry
        for i in range(len(info[scene]["geometry"])):
            stem = Path(info[scene]["geometry"][i]["target_depth"]).stem
            info[scene]["geometry"][i]["output_depth"] = (
                (ours_dir / "render_geo" / f"{stem}_image.zbuf.exr")
                .resolve()
                .as_posix()
            )
            info[scene]["geometry"][i]["output_normal"] = (
                (ours_dir / "render_geo" / f"{stem}_image.normal.exr")
                .resolve()
                .as_posix()
            )
            assert os.path.isfile(info[scene]["geometry"][i]["output_depth"])
            assert os.path.isfile(info[scene]["geometry"][i]["output_normal"])

        # albedo
        for i in range(len(info[scene]["material"])):
            stem = Path(info[scene]["material"][i]["target_image"]).stem
            info[scene]["material"][i]["output_image"] = (
                (ours_dir / "blender_relit" / f"im_{stem}_image_albedo0029.exr")
                .resolve()
                .as_posix()
            )
            assert os.path.isfile(info[scene]["material"][i]["output_image"])

        # novel view
        for i in range(len(info[scene]["view"])):
            stem = Path(info[scene]["view"][i]["target_image"]).stem
            info[scene]["view"][i]["output_image"] = (
                (ours_dir / "blender_relit" / f"im_{stem}_image_envmap_for_blender.exr")
                .resolve()
                .as_posix()
            )
            assert os.path.isfile(info[scene]["view"][i]["output_image"])

        # relighting
        for i in range(len(info[scene]["light"])):
            stem = Path(info[scene]["light"][i]["target_image"]).stem
            env_scene = Path(info[scene]["light"][i]["target_image"]).parents[1].stem
            info[scene]["light"][i]["output_image"] = (
                (
                    ours_dir
                    / "blender_relit"
                    / f"im_{stem}_{env_scene}_{stem}_for_{scene}.exr"
                )
                .resolve()
                .as_posix()
            )
            assert os.path.isfile(info[scene]["light"][i]["output_image"])

    # Statistic & Sanity check
    for scene_name, scene_meta in info.items():
        stat = f"{scene_name:20s}: "

        cnt_pred = 0
        for item in scene_meta["view"]:
            assert os.path.isfile(item["target_image"])
            if item["output_image"] is not None:
                cnt_pred += 1
                assert os.path.isfile(item["output_image"])
        stat += f"nv={cnt_pred:2d}/{len(scene_meta['view']):2d} "

        cnt_pred = 0
        for item in scene_meta["light"]:
            assert os.path.isfile(item["target_image"])
            if item["output_image"] is not None:
                cnt_pred += 1
                assert os.path.isfile(item["output_image"])
        stat += f"nl={cnt_pred:2d}/{len(scene_meta['light']):2d} "

        cnt_pred = 0
        for item in scene_meta["material"]:
            assert os.path.isfile(item["target_image"])
            assert os.path.isfile(item["target_mask"])
            if item["output_image"] is not None:
                cnt_pred += 1
                assert os.path.isfile(item["output_image"])
        stat += f"mat={cnt_pred:2d}/{len(scene_meta['material']):2d} "

        cnt_pred = 0
        for item in scene_meta["geometry"]:
            assert os.path.isfile(item["target_depth"])
            assert os.path.isfile(item["target_normal"])
            assert os.path.isfile(item["target_mask"])
            if item["output_depth"] is not None:
                cnt_pred += 1
                assert os.path.isfile(item["output_depth"])
                assert os.path.isfile(item["output_normal"])
        stat += f"geo={cnt_pred:2d}/{len(scene_meta['geometry']):2d} "

        assert os.path.isfile(scene_meta["shape"]["target_mesh"])
        if scene_meta["shape"]["output_mesh"] is not None:
            assert os.path.isfile(scene_meta["shape"]["output_mesh"])
            stat += f"w/ mesh"
        else:
            stat += f"w/o mesh"

        print(stat)

    # Write-up
    outpath = args.ckpt_dir / f"eval_inputs_{args.stage}.json"
    with open(outpath, "w") as f:
        json.dump({"info": info}, f, indent=4)
    print("Save to", outpath)
