# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

dataroot="data/stanford_orb/blender_HDR"
ckptroot="results/stanford_orb"
command="python neural_surface_recon/run_template.py"
command="$command --template neural_surface_recon/configs/template_stanford_orb.py"
command="$command --savemem --render_video_factor 1"


# # Runing 1st-stage (neural surface)
for datadir in $(ls -d "$dataroot"/*)
do
    scene=$(basename "$datadir")
    echo "Processing: $scene"
    $command $dataroot/$scene/
done

# Runing 2nd-stage (material distillation)
for datadir in $(ls -d "$dataroot"/*)
do
    scene=$(basename "$datadir")
    echo "Processing: $scene"
    python neural_distillation/run.py $ckptroot/$scene/
done

# Running 3rd-stage (pbir)
for datadir in $(ls -d "$dataroot"/*)
do
    scene=$(basename "$datadir")
    echo "Processing: $scene"
    python pbir/run.py pbir/configs/template $ckptroot/$scene/
done
