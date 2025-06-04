# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ckptroot="results/charuco"

python neural_distillation/run.py $ckptroot/dinosaur/
python neural_distillation/run.py $ckptroot/shoes/
python neural_distillation/run.py $ckptroot/plantpot/
python neural_distillation/run.py $ckptroot/pumpkin/
python neural_distillation/run.py $ckptroot/greenplant/
python neural_distillation/run.py $ckptroot/mooncake/
python neural_distillation/run.py $ckptroot/keyboard/
python neural_distillation/run.py $ckptroot/waterbottle/
python neural_distillation/run.py $ckptroot/flower/
python neural_distillation/run.py $ckptroot/bootbrush/
python neural_distillation/run.py $ckptroot/hammer/
python neural_distillation/run.py $ckptroot/keyboard2/
