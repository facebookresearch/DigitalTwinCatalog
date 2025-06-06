#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ckptroot="results/stanford_orb"

# Running 3rd-stage (pbir)
for datadir in $(ls -d data/stanford_orb/blender_HDR/*)
do
    scene=$(basename "$datadir")
    echo "Processing: $scene"

    zip -r results.zip results/stanford_orb/$scene/neural_distillation/mesh.obj
    zip -r results.zip results/stanford_orb/$scene/neural_distillation/albedo.exr
    zip -r results.zip results/stanford_orb/$scene/neural_distillation/roughness.exr
    zip -r results.zip results/stanford_orb/$scene/neural_distillation/envmap.exr

    zip -r results.zip results/stanford_orb/$scene/pbir/mesh.obj
    zip -r results.zip results/stanford_orb/$scene/pbir/diffuse.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/roughness.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/specular.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/envmap.exr

    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls/final/mesh.obj
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls/final/diffuse.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls/final/roughness.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls/final/specular.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls/final/envmap.exr

    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_naive-envmap_sg/final/mesh.obj
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_naive-envmap_sg/final/diffuse.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_naive-envmap_sg/final/roughness.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_naive-envmap_sg/final/specular.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_naive-envmap_sg/final/envmap.exr

    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls-shape_ls/final/mesh.obj
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls-shape_ls/final/diffuse.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls-shape_ls/final/roughness.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls-shape_ls/final/specular.exr
    zip -r results.zip results/stanford_orb/$scene/pbir/microfacet_basis-envmap_ls-shape_ls/final/envmap.exr

done
