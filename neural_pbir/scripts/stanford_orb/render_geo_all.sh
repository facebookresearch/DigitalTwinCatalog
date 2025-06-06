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
    #python scripts/stanford_orb/render_geo.py $dataroot/$scene/cameras.json $ckptroot/$scene/neural_distillation/mesh.obj
    python scripts/stanford_orb/render_geo.py $dataroot/$scene/cameras.json $ckptroot/$scene/pbir/mesh.obj
    # python scripts/stanford_orb/render_geo.py $dataroot/$scene/cameras.json $ckptroot/$scene/pbir/microfacet_basis-envmap_ls/final/mesh.obj
    # python scripts/stanford_orb/render_geo.py $dataroot/$scene/cameras.json $ckptroot/$scene/pbir/microfacet_naive-envmap_sg/final/mesh.obj
done
