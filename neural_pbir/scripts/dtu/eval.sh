#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python data/DTU_official/DTUeval-python/eval.py \
        --data results/dtu/dtu_scan"$1"/neural_surface_recon/mesh_mc_raw.obj \
        --scan $1 --dataset_dir data/DTU_official/ \
        --vis_out_dir results/dtu/dtu_scan"$1"/neural_surface_recon/
