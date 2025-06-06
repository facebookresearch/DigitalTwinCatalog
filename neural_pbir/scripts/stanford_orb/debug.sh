#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# for flip in 0 1
# do
#     for deg in 0 45 90 135 180 225 270 315
#     do
#         python scripts/stanford_orb/postproc_envmap_gt_debug.py data/stanford_orb/ground_truth/ --deg $deg --flip $flip
#         python scripts/stanford_orb/relit_nl.py
#         mv /home/chengs/neural-pbir/data/stanford_orb/ground_truth/car_scene002/env_map_for_blender/car_scene002_0068.exr /home/chengs/neural-pbir/data/stanford_orb/ground_truth/car_scene002/env_map_for_blender/flip"$flip"_deg"$deg".exr
#         mv /home/chengs/neural-pbir/results/stanford_orb/car_scene004/neural_distillation/blender_relit/im_0068_car_scene002_0068.exr /home/chengs/neural-pbir/results/stanford_orb/car_scene004/neural_distillation/blender_relit/flip"$flip"_deg"$deg".exr
#     done
# done

flip=0
for deg in 292.5 315 337.5
do
    python scripts/stanford_orb/postproc_envmap_gt_debug.py data/stanford_orb/ground_truth/ --deg $deg --flip $flip
    python scripts/stanford_orb/relit_nl.py
    mv /home/chengs/neural-pbir/data/stanford_orb/ground_truth/car_scene002/env_map_for_blender/car_scene002_0068.exr /home/chengs/neural-pbir/data/stanford_orb/ground_truth/car_scene002/env_map_for_blender/flip"$flip"_deg"$deg".exr
    mv /home/chengs/neural-pbir/results/stanford_orb/car_scene004/neural_distillation/blender_relit/im_0068_car_scene002_0068.exr /home/chengs/neural-pbir/results/stanford_orb/car_scene004/neural_distillation/blender_relit/flip"$flip"_deg"$deg".exr
done


flip=1
for deg in 112.5 135 157.5
do
    python scripts/stanford_orb/postproc_envmap_gt_debug.py data/stanford_orb/ground_truth/ --deg $deg --flip $flip
    python scripts/stanford_orb/relit_nl.py
    mv /home/chengs/neural-pbir/data/stanford_orb/ground_truth/car_scene002/env_map_for_blender/car_scene002_0068.exr /home/chengs/neural-pbir/data/stanford_orb/ground_truth/car_scene002/env_map_for_blender/flip"$flip"_deg"$deg".exr
    mv /home/chengs/neural-pbir/results/stanford_orb/car_scene004/neural_distillation/blender_relit/im_0068_car_scene002_0068.exr /home/chengs/neural-pbir/results/stanford_orb/car_scene004/neural_distillation/blender_relit/flip"$flip"_deg"$deg".exr
done
