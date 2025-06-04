/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "render_utils.h"

// C++ interface implementation
std::vector<torch::Tensor> infer_t_minmax(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    const float near,
    const float far) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  return infer_t_minmax_cuda(rays_o, rays_d, xyz_min, xyz_max, near, far);
}

torch::Tensor infer_n_samples(
    torch::Tensor rays_d,
    torch::Tensor t_min,
    torch::Tensor t_max,
    const float stepdist) {
  CHECK_INPUT(rays_d);
  CHECK_INPUT(t_min);
  CHECK_INPUT(t_max);
  return infer_n_samples_cuda(rays_d, t_min, t_max, stepdist);
}

std::vector<torch::Tensor> infer_ray_start_dir(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor t_min) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(t_min);
  return infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
}

std::vector<torch::Tensor> sample_pts_on_rays(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    const float near,
    const float far,
    const float stepdist) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  TORCH_CHECK(rays_o.dim() == 2, "rays_o.dim()==2");
  TORCH_CHECK(rays_o.size(1) == 3, "rays_o.size(1)==3");
  return sample_pts_on_rays_cuda(
      rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist);
}

torch::Tensor sample_bg_pts_on_rays(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor t_max,
    const float bg_preserve,
    const int N_samples) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(t_max);
  return sample_bg_pts_on_rays_cuda(
      rays_o, rays_d, t_max, bg_preserve, N_samples);
}

torch::Tensor maskcache_lookup(
    torch::Tensor maskgrid,
    torch::Tensor xyz,
    torch::Tensor xyz2ijk_scale,
    torch::Tensor xyz2ijk_shift) {
  CHECK_INPUT(maskgrid);
  CHECK_INPUT(xyz);
  CHECK_INPUT(xyz2ijk_scale);
  CHECK_INPUT(xyz2ijk_shift);
  TORCH_CHECK(maskgrid.dim() == 3, "maskgrid.dim()==3");
  TORCH_CHECK(xyz.dim() == 2, "xyz.dim()==2");
  TORCH_CHECK(xyz.size(1) == 3, "xyz.size(1)==3");
  return maskcache_lookup_cuda(maskgrid, xyz, xyz2ijk_scale, xyz2ijk_shift);
}

torch::Tensor maskcache_ray_tracing(
    torch::Tensor maskgrid,
    torch::Tensor rs,
    torch::Tensor rd,
    const float stepdist,
    torch::Tensor max_n_steps,
    torch::Tensor xyz2ijk_scale,
    torch::Tensor xyz2ijk_shift) {
  CHECK_INPUT(maskgrid);
  CHECK_INPUT(rs);
  CHECK_INPUT(rd);
  CHECK_INPUT(max_n_steps);
  CHECK_INPUT(xyz2ijk_scale);
  CHECK_INPUT(xyz2ijk_shift);
  TORCH_CHECK(maskgrid.dim() == 3, "maskgrid.dim()==3");
  TORCH_CHECK(rs.dim() == 2, "rs.dim()==2");
  TORCH_CHECK(rs.size(1) == 3, "rs.size(1)==3");
  return maskcache_ray_tracing_cuda(
      maskgrid, rs, rd, stepdist, max_n_steps, xyz2ijk_scale, xyz2ijk_shift);
}

std::vector<torch::Tensor>
raw2alpha(torch::Tensor density, const float shift, const float interval) {
  CHECK_INPUT(density);
  TORCH_CHECK(density.dim() == 1, "Tensor should be flatten");
  return raw2alpha_cuda(density, shift, interval);
}
std::vector<torch::Tensor> raw2alpha_nonuni(
    torch::Tensor density,
    const float shift,
    torch::Tensor interval) {
  CHECK_INPUT(density);
  TORCH_CHECK(density.dim() == 1, "Tensor should be flatten");
  return raw2alpha_nonuni_cuda(density, shift, interval);
}

torch::Tensor raw2alpha_backward(
    torch::Tensor exp,
    torch::Tensor grad_back,
    const float interval) {
  CHECK_INPUT(exp);
  CHECK_INPUT(grad_back);
  return raw2alpha_backward_cuda(exp, grad_back, interval);
}
torch::Tensor raw2alpha_nonuni_backward(
    torch::Tensor exp,
    torch::Tensor grad_back,
    torch::Tensor interval) {
  CHECK_INPUT(exp);
  CHECK_INPUT(grad_back);
  return raw2alpha_nonuni_backward_cuda(exp, grad_back, interval);
}

std::vector<torch::Tensor>
alpha2weight(torch::Tensor alpha, torch::Tensor ray_id, const int n_rays) {
  CHECK_INPUT(alpha);
  CHECK_INPUT(ray_id);
  TORCH_CHECK(alpha.dim() == 1, "Tensor should be flatten");
  TORCH_CHECK(ray_id.dim() == 1, "Tensor should be flatten");
  TORCH_CHECK(alpha.sizes() == ray_id.sizes(), "alpha.sizes()==ray_id.sizes()");
  return alpha2weight_cuda(alpha, ray_id, n_rays);
}

torch::Tensor alpha2weight_backward(
    torch::Tensor alpha,
    torch::Tensor weight,
    torch::Tensor T,
    torch::Tensor alphainv_last,
    torch::Tensor i_start,
    torch::Tensor i_end,
    const int n_rays,
    torch::Tensor grad_weights,
    torch::Tensor grad_last) {
  CHECK_INPUT(alpha);
  CHECK_INPUT(weight);
  CHECK_INPUT(T);
  CHECK_INPUT(alphainv_last);
  CHECK_INPUT(i_start);
  CHECK_INPUT(i_end);
  CHECK_INPUT(grad_weights);
  CHECK_INPUT(grad_last);
  return alpha2weight_backward_cuda(
      alpha,
      weight,
      T,
      alphainv_last,
      i_start,
      i_end,
      n_rays,
      grad_weights,
      grad_last);
}

std::vector<torch::Tensor> alpha2weight_dense(torch::Tensor alpha) {
  CHECK_INPUT(alpha);
  return alpha2weight_dense_cuda(alpha);
}

torch::Tensor alpha2weight_dense_backward(
    torch::Tensor alpha,
    torch::Tensor weight,
    torch::Tensor T,
    torch::Tensor alphainv_last,
    torch::Tensor grad_weights,
    torch::Tensor grad_last) {
  CHECK_INPUT(alpha);
  CHECK_INPUT(weight);
  CHECK_INPUT(T);
  CHECK_INPUT(alphainv_last);
  CHECK_INPUT(grad_weights);
  CHECK_INPUT(grad_last);
  return alpha2weight_dense_backward_cuda(
      alpha, weight, T, alphainv_last, grad_weights, grad_last);
}

torch::Tensor aggregate_tensorf_val(
    torch::Tensor xy_feat,
    torch::Tensor z_feat,
    torch::Tensor xz_feat,
    torch::Tensor y_feat,
    torch::Tensor yz_feat,
    torch::Tensor x_feat) {
  CHECK_INPUT(xy_feat);
  CHECK_INPUT(z_feat);
  CHECK_INPUT(xz_feat);
  CHECK_INPUT(y_feat);
  CHECK_INPUT(yz_feat);
  CHECK_INPUT(x_feat);
  return aggregate_tensorf_val_cuda(
      xy_feat, z_feat, xz_feat, y_feat, yz_feat, x_feat);
}

std::vector<torch::Tensor>
segment_cumsum(torch::Tensor w, torch::Tensor s, torch::Tensor ray_id) {
  CHECK_INPUT(w);
  CHECK_INPUT(s);
  CHECK_INPUT(ray_id);
  return segment_cumsum_cuda(w, s, ray_id);
}
