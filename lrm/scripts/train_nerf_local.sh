#! /bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

epochs=200
batch_size_per_gpu=8
num_workers=4
saveimg_iter_freq=400
saveckp_epoch_freq=1
backup_ckp_epoch_freq=50
mvencoder_type="plucker"
renderer_type="nerf"
fov=20.377
bbox_radius=0.5
output_image_res=128
output_image_num=4
triplane_token_size=64
triplane_out_token_size=512
image_num_per_batch="8 8"
triplane_dim=32
transformer_depth=24
prediction_type="rgb"
patch_size=8
embed_dim=1024
lr=4e-4
min_lr=2e-5
warmup_iters=1000
num_samples_per_ray=128
output_res_range="256 256"
input_image_res=256
data_path="data"
# grandteton, H100 80g, minimum 8 gpus
# zionex_80g, A100 80g
# zionex, A100
# https://www.internalfb.com/interr/wiki/rl/rl_production_capacity/rl_efficiency_program/compute/fblearner_and_mast/allocation_efficiency/using_rl_elastic_capacity_for_training/#using-elastic-capacity
# t16_grandteton
# zion_4s_80g
# Then schedule your pipeline as usual, passing the frl_elastic_gpu as the tenant instead of your regular team/pool tenant.
exp_root="experiments"
exp_name="opensrc_320k_nerf_res256_sample128"


torchrun main_lrm_triplane.py \
    --exp_root ${exp_root} \
    --exp_name ${exp_name} \
    --epochs ${epochs} \
    --warmup_iters ${warmup_iters} \
    --lr ${lr} \
    --min_lr ${min_lr} \
    --batch_size_per_gpu ${batch_size_per_gpu} \
    --num_workers ${num_workers} \
    --saveimg_iter_freq ${saveimg_iter_freq} \
    --saveckp_epoch_freq ${saveckp_epoch_freq} \
    --data_path ${data_path} \
    --backup_ckp_epoch_freq ${backup_ckp_epoch_freq} \
    --mvencoder_type ${mvencoder_type} \
    --renderer_type ${renderer_type} \
    --fov ${fov} \
    --bbox_radius ${bbox_radius} \
    --triplane_token_size ${triplane_token_size} \
    --triplane_out_token_size ${triplane_out_token_size} \
    --triplane_out_token_size ${triplane_out_token_size} \
    --image_num_per_batch ${image_num_per_batch} \
    --output_image_res ${output_image_res} \
    --output_res_range ${output_res_range} \
    --prediction_type ${prediction_type} \
    --triplane_dim ${triplane_dim} \
    --transformer_depth ${transformer_depth} \
    --output_image_num ${output_image_num} \
    --seed 1024 \
    --patch_size ${patch_size} \
    --num_samples_per_ray ${num_samples_per_ray} \
    --embed_dim ${embed_dim} \
    --input_image_res ${input_image_res} \
    --use_adobe_view_selection \
    --auto_resume \
    --dataset_type dtc_dataset \
    --use_weight_norm \
    --loss_weights_file "nerf"
