#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
from pathlib import Path

import mmcv

sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
from neural_surface_recon.run import config_parser, run


def run_template(arg_lst, cfg):
    parser = config_parser()
    args = parser.parse_args(arg_lst + ["--config", "dummy"])
    run(args, cfg)


if __name__ == "__main__":
    hyper_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    hyper_parser.add_argument("dataroot", help="path to your dataroot", type=Path)
    hyper_parser.add_argument(
        "--template", default=os.path.join("configs", "template_aruco_spark.py")
    )
    hyper_parser.add_argument(
        "--scale_photoloss",
        default=None,
        type=float,
        help="scale weight of photometric loss. "
        "set this to larger than one (like 10) to get sharper surface.",
    )
    hyper_parser.add_argument(
        "--scale_grid", default=None, type=float, help="scale geometry grid resolution"
    )
    hyper_parser.add_argument(
        "--scale_iter", default=None, type=float, help="scale training iterations"
    )
    hyper_parser.add_argument(
        "--scale_bs",
        default=None,
        type=float,
        help="scale training batch-size (the number of rays)",
    )
    hyper_parser.add_argument(
        "--scale_lap",
        default=None,
        type=float,
        help="scale laplace regularizer strength"
        " (larger smoother; try 5 first if the geometry is heavily affected by specular reflection)",
    )
    hyper_parser.add_argument(
        "--refine_poses", action="store_true", help="enable camera poses refinement"
    )
    hyper_parser.add_argument(
        "--savemem",
        action="store_true",
        help="disable pre-load all training data to gpu",
    )
    hyper_parser.add_argument("--run_vis_factor", default=2, type=int)
    hyper_parser.add_argument("--render_video_factor", default=1, type=float)
    hyper_parser.add_argument("--dtu_postproc", action="store_true")
    hyper_parser.add_argument("--test_only", action="store_true")
    hyper_args = hyper_parser.parse_args()

    # modify config template
    cfg = mmcv.Config.fromfile(hyper_args.template)
    if hyper_args.scale_photoloss is not None:
        weight_main = cfg.fine_train.weight_main
        new_weight_main = weight_main * hyper_args.scale_photoloss
        print(
            f"==> scaling photometric loss weight from {weight_main} to {new_weight_main}"
        )
        cfg.fine_train.weight_main = new_weight_main
    if hyper_args.scale_grid is not None:
        res = int(round(cfg.fine_model_and_render.num_voxels ** (1 / 3)))
        newres = int(round(res * hyper_args.scale_grid))
        print(f"==> scaling grid resolution from {res}^3 to {newres}^3")
        cfg.fine_model_and_render.num_voxels = newres**3
    if hyper_args.scale_iter is not None:
        N_iters = cfg.fine_train.N_iters
        new_N_iters = int(N_iters * hyper_args.scale_iter)
        print(f"==> scaling training iterations from {N_iters} to {new_N_iters}")
        cfg.fine_train.N_iters = new_N_iters
    if hyper_args.scale_bs is not None:
        N_rand = cfg.fine_train.N_rand
        new_N_rand = int(N_rand * hyper_args.scale_bs)
        print(f"==> scaling training batch-size from {N_rand} to {new_N_rand}")
        cfg.fine_train.N_rand = new_N_rand
    if hyper_args.scale_lap is not None:
        weight_laplace = cfg.fine_train.weight_laplace
        new_weight_laplace = weight_laplace * hyper_args.scale_lap
        print(
            f"==> scaling laplace regularizer strength from {weight_laplace} to {new_weight_laplace}"
        )
        cfg.fine_train.weight_laplace = new_weight_laplace
    if hyper_args.refine_poses:
        print(f"==> enable camera poses refinement")
        cfg.camera_refinement_model.trainable = True
    if hyper_args.savemem:
        print(f"==> disable pre-load all training data to save GPU memory")
        cfg.data.load2gpu_on_the_fly = True

    # gogo
    arg_lst = [
        "--run_vis_period",
        "100",
        "--run_vis_factor",
        str(hyper_args.run_vis_factor),
        "--render_video",
        "--render_normal",
        "--render_depth",
        "--render_fg_only",
        "--render_video_factor",
        str(hyper_args.render_video_factor),
        "--export_surface",
        "--test_only" if hyper_args.test_only else "--no_reload",
    ]
    if hyper_args.dtu_postproc:
        arg_lst.append("--mesh_dtu_clean")
    cfg.results_dir = (
        Path("results")
        / cfg.data.dataset_name
        / hyper_args.dataroot.stem
        / "neural_surface_recon"
    )
    cfg.data.datadir = str(hyper_args.dataroot)
    run_template(arg_lst, cfg)
