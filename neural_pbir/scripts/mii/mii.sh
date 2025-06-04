# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

dataroot="data/Synthetic4Relight"
command="python neural_surface_recon/run_template.py"
command="$command --template neural_surface_recon/configs/template_mii.py"
command="$command --savemem"

$command $dataroot/air_baloons/
$command $dataroot/chair/
$command $dataroot/hotdog/
$command $dataroot/jugs/

ckptroot="results/mii"
python neural_distillation/run.py $ckptroot/air_baloons/
python neural_distillation/run.py $ckptroot/chair/
python neural_distillation/run.py $ckptroot/hotdog/
python neural_distillation/run.py $ckptroot/jugs/
