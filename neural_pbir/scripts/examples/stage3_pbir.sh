# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ckptroot="results/charuco"
command="python pbir/run.py pbir/configs/template"

$command $ckptroot/dinosaur/
$command $ckptroot/shoes/
$command $ckptroot/plantpot/
$command $ckptroot/pumpkin/
$command $ckptroot/greenplant/
$command $ckptroot/mooncake/
$command $ckptroot/keyboard/
$command $ckptroot/waterbottle/
$command $ckptroot/flower/
$command $ckptroot/bootbrush/
$command $ckptroot/hammer/
$command $ckptroot/keyboard2/
