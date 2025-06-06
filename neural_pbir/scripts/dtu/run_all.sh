#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

dataroot="data/DTU"
command="python neural_surface_recon/run_template.py"
command="$command --template neural_surface_recon/configs/template_dtu.py"
command="$command --savemem"
command="$command --dtu_postproc"
command="$command --run_vis_factor 4"

for scanid in 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122
do
    $command $dataroot/dtu_scan"$scanid"/
    ./scripts/dtu/eval.sh $scanid $scanid
done

python scripts/dtu/stat.py
