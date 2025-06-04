# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ckptroot=results/stanford_orb
dataroot=data/stanford_orb

for datadir in $(ls -d "$dataroot"/blender_HDR/*)
do
    scene=$(basename "$datadir")
    echo "Processing: $scene"

    CAM=$dataroot/blender_HDR/$scene/cameras.json

    # distillation
    #CKPT_GEO=$ckptroot/$scene/neural_distillation/mesh.obj
    #CKPT_ALBEDO=$ckptroot/$scene/neural_distillation/albedo.exr
    #CKPT_ROUGH=$ckptroot/$scene/neural_distillation/roughness.exr
    #CKPT_ENV=$ckptroot/$scene/neural_distillation/envmap_for_blender.exr

    # pbir
    CKPT_GEO=$ckptroot/$scene/pbir/mesh.obj
    CKPT_ALBEDO=$ckptroot/$scene/pbir/diffuse.exr
    CKPT_ROUGH=$ckptroot/$scene/pbir/roughness.exr
    CKPT_ENV=$ckptroot/$scene/pbir/envmap_for_blender.exr

    # run blender
    python scripts/relit/relit.py $CKPT_GEO $CKPT_ALBEDO $CKPT_ROUGH $CAM --lgt_paths $CKPT_ENV --render_exr --with_bg
done
