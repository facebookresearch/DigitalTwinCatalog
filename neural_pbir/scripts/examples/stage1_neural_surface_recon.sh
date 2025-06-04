# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

dataroot="data/neural_pbir_real"
command="python neural_surface_recon/run_template.py"
command="$command --template neural_surface_recon/configs/template_charuco.py"
command="$command --savemem"

$command $dataroot/dinosaur/
$command $dataroot/shoes/
$command $dataroot/plantpot/
$command $dataroot/pumpkin/ --scale_lap 30
$command $dataroot/greenplant/
$command $dataroot/mooncake/
$command $dataroot/keyboard/
$command $dataroot/waterbottle/ --scale_lap 5
$command $dataroot/flower/
$command $dataroot/bootbrush/
$command $dataroot/hammer/ --scale_lap 5
$command $dataroot/keyboard2/ --scale_lap 5
