#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

dataroot="data/stanford_orb/blender_HDR"
ckptroot="results/stanford_orb"

# Running 3rd-stage (pbir)
for datadir in $(ls -d "$dataroot"/*)
do
    scene=$(basename "$datadir")
    echo "Processing: $scene"
    python pbir/run.py pbir/configs/template $ckptroot/$scene/
done
