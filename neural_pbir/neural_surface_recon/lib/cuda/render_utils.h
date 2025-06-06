/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

// CUDA forward declarations
std::vector<torch::Tensor> infer_t_minmax_cuda(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    const float near,
    const float far);

torch::Tensor infer_n_samples_cuda(
    torch::Tensor rays_d,
    torch::Tensor t_min,
    torch::Tensor t_max,
    const float stepdist);

std::vector<torch::Tensor> infer_ray_start_dir_cuda(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor t_min);

std::vector<torch::Tensor> sample_pts_on_rays_cuda(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    const float near,
    const float far,
    const float stepdist);

torch::Tensor sample_bg_pts_on_rays_cuda(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor t_max,
    const float bg_preserve,
    const int N_samples);

torch::Tensor maskcache_lookup_cuda(
    torch::Tensor maskgrid,
    torch::Tensor xyz,
    torch::Tensor xyz2ijk_scale,
    torch::Tensor xyz2ijk_shift);
torch::Tensor maskcache_ray_tracing_cuda(
    torch::Tensor maskgrid,
    torch::Tensor rs,
    torch::Tensor rd,
    const float stepdist,
    torch::Tensor max_n_steps,
    torch::Tensor xyz2ijk_scale,
    torch::Tensor xyz2ijk_shift);

std::vector<torch::Tensor>
raw2alpha_cuda(torch::Tensor density, const float shift, const float interval);
std::vector<torch::Tensor> raw2alpha_nonuni_cuda(
    torch::Tensor density,
    const float shift,
    torch::Tensor interval);

torch::Tensor raw2alpha_backward_cuda(
    torch::Tensor exp,
    torch::Tensor grad_back,
    const float interval);
torch::Tensor raw2alpha_nonuni_backward_cuda(
    torch::Tensor exp,
    torch::Tensor grad_back,
    torch::Tensor interval);

std::vector<torch::Tensor>
alpha2weight_cuda(torch::Tensor alpha, torch::Tensor ray_id, const int n_rays);

torch::Tensor alpha2weight_backward_cuda(
    torch::Tensor alpha,
    torch::Tensor weight,
    torch::Tensor T,
    torch::Tensor alphainv_last,
    torch::Tensor i_start,
    torch::Tensor i_end,
    const int n_rays,
    torch::Tensor grad_weights,
    torch::Tensor grad_last);

std::vector<torch::Tensor> alpha2weight_dense_cuda(torch::Tensor alpha);

torch::Tensor alpha2weight_dense_backward_cuda(
    torch::Tensor alpha,
    torch::Tensor weight,
    torch::Tensor T,
    torch::Tensor alphainv_last,
    torch::Tensor grad_weights,
    torch::Tensor grad_last);

torch::Tensor aggregate_tensorf_val_cuda(
    torch::Tensor xy_feat,
    torch::Tensor z_feat,
    torch::Tensor xz_feat,
    torch::Tensor y_feat,
    torch::Tensor yz_feat,
    torch::Tensor x_feat);

std::vector<torch::Tensor>
segment_cumsum_cuda(torch::Tensor w, torch::Tensor s, torch::Tensor ray_id);

// C++ interface
std::vector<torch::Tensor> infer_t_minmax(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    const float near,
    const float far);

torch::Tensor infer_n_samples(
    torch::Tensor rays_d,
    torch::Tensor t_min,
    torch::Tensor t_max,
    const float stepdist);

std::vector<torch::Tensor> infer_ray_start_dir(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor t_min);

std::vector<torch::Tensor> sample_pts_on_rays(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    const float near,
    const float far,
    const float stepdist);

torch::Tensor sample_bg_pts_on_rays(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor t_max,
    const float bg_preserve,
    const int N_samples);

torch::Tensor maskcache_lookup(
    torch::Tensor maskgrid,
    torch::Tensor xyz,
    torch::Tensor xyz2ijk_scale,
    torch::Tensor xyz2ijk_shift);
torch::Tensor maskcache_ray_tracing(
    torch::Tensor maskgrid,
    torch::Tensor rs,
    torch::Tensor rd,
    const float stepdist,
    torch::Tensor max_n_steps,
    torch::Tensor xyz2ijk_scale,
    torch::Tensor xyz2ijk_shift);

std::vector<torch::Tensor>
raw2alpha(torch::Tensor density, const float shift, const float interval);
std::vector<torch::Tensor> raw2alpha_nonuni(
    torch::Tensor density,
    const float shift,
    torch::Tensor interval);

torch::Tensor raw2alpha_backward(
    torch::Tensor exp,
    torch::Tensor grad_back,
    const float interval);
torch::Tensor raw2alpha_nonuni_backward(
    torch::Tensor exp,
    torch::Tensor grad_back,
    torch::Tensor interval);

std::vector<torch::Tensor>
alpha2weight(torch::Tensor alpha, torch::Tensor ray_id, const int n_rays);

torch::Tensor alpha2weight_backward(
    torch::Tensor alpha,
    torch::Tensor weight,
    torch::Tensor T,
    torch::Tensor alphainv_last,
    torch::Tensor i_start,
    torch::Tensor i_end,
    const int n_rays,
    torch::Tensor grad_weights,
    torch::Tensor grad_last);

std::vector<torch::Tensor> alpha2weight_dense(torch::Tensor alpha);

torch::Tensor alpha2weight_dense_backward(
    torch::Tensor alpha,
    torch::Tensor weight,
    torch::Tensor T,
    torch::Tensor alphainv_last,
    torch::Tensor grad_weights,
    torch::Tensor grad_last);

torch::Tensor aggregate_tensorf_val(
    torch::Tensor xy_feat,
    torch::Tensor z_feat,
    torch::Tensor xz_feat,
    torch::Tensor y_feat,
    torch::Tensor yz_feat,
    torch::Tensor x_feat);

std::vector<torch::Tensor>
segment_cumsum(torch::Tensor w, torch::Tensor s, torch::Tensor ray_id);
