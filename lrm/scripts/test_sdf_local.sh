#! /bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

batch_size_per_gpu=8
num_workers=4
mvencoder_type="plucker"
renderer_type="sdf"
fov=20.377
bbox_radius=0.5
output_image_res=256
output_image_num=4
triplane_token_size=64
triplane_out_token_size=512
image_num_per_batch=8
triplane_dim=32
transformer_depth=24
prediction_type="rgb"
patch_size=8
embed_dim=1024
num_samples_per_ray=256
input_image_res=256
data_path="data"
eva_input_views="0 1 2 3 4 5 6 7"
eva_output_views="8 9 10 11"
sdf_inv_std=200
exp_root="experiments"
exp_name="test_320k_volsdf_res256_sample128"


torchrun test_lrm_triplane.py \
    --exp_root ${exp_root} \
    --exp_name ${exp_name} \
    --batch_size_per_gpu ${batch_size_per_gpu} \
    --num_workers ${num_workers} \
    --data_path ${data_path} \
    --mvencoder_type ${mvencoder_type} \
    --renderer_type ${renderer_type} \
    --fov ${fov} \
    --bbox_radius ${bbox_radius} \
    --triplane_token_size ${triplane_token_size} \
    --triplane_out_token_size ${triplane_out_token_size} \
    --triplane_out_token_size ${triplane_out_token_size} \
    --image_num_per_batch ${image_num_per_batch} \
    --output_image_res ${output_image_res} \
    --prediction_type ${prediction_type} \
    --triplane_dim ${triplane_dim} \
    --transformer_depth ${transformer_depth} \
    --output_image_num ${output_image_num} \
    --seed 1024 \
    --patch_size ${patch_size} \
    --num_samples_per_ray ${num_samples_per_ray} \
    --embed_dim ${embed_dim} \
    --input_image_res ${input_image_res} \
    --dataset_type dtc_dataset \
    --use_weight_norm \
    --eva_input_views ${eva_input_views} \
    --eva_output_views ${eva_output_views} \
    --loss_weights_file "sdf" \
    --checkpoint "experiments/opensrc_320k_volsdf_res256_sample128/checkpoints/last.pth" \
    --sdf_inv_std ${sdf_inv_std} \
    --save_video \
    --save_mesh \
