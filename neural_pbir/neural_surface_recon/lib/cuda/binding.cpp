/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "adam_upd.h"
#include "common.h"
#include "grid_sample.h"
#include "render_utils.h"
#include "sdf_rt.h"
#include "total_variation.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "infer_t_minmax",
      &infer_t_minmax,
      "Inference t_min and t_max of ray-bbox intersection");
  m.def(
      "infer_n_samples",
      &infer_n_samples,
      "Inference the number of points to sample on each ray");
  m.def(
      "infer_ray_start_dir",
      &infer_ray_start_dir,
      "Inference the starting point and shooting direction of each ray");
  m.def("sample_pts_on_rays", &sample_pts_on_rays, "Sample points on rays");
  m.def("sample_bg_pts_on_rays", &sample_bg_pts_on_rays, "Sample points on bg");
  m.def("maskcache_lookup", &maskcache_lookup, "Lookup to skip know freespace");
  m.def(
      "maskcache_ray_tracing",
      &maskcache_ray_tracing,
      "Ray tracing on occupancy grid");
  m.def("raw2alpha", &raw2alpha, "Raw values [-inf, inf] to alpha [0, 1].");
  m.def(
      "raw2alpha_backward",
      &raw2alpha_backward,
      "Backward pass of the raw to alpha");
  m.def(
      "raw2alpha_nonuni",
      &raw2alpha_nonuni,
      "Raw values [-inf, inf] to alpha [0, 1].");
  m.def(
      "raw2alpha_nonuni_backward",
      &raw2alpha_nonuni_backward,
      "Backward pass of the raw to alpha");
  m.def(
      "alpha2weight",
      &alpha2weight,
      "Per-point alpha to accumulated blending weight");
  m.def(
      "alpha2weight_backward",
      &alpha2weight_backward,
      "Backward pass of alpha2weight");
  m.def(
      "alpha2weight_dense",
      &alpha2weight_dense,
      "Per-point alpha to accumulated blending weight (dense ver.)");
  m.def(
      "alpha2weight_dense_backward",
      &alpha2weight_dense_backward,
      "Backward pass of alpha2weight (dense ver.)");
  m.def(
      "aggregate_tensorf_val",
      &aggregate_tensorf_val,
      "aggregate_tensorf_val forward pass");

  m.def("adam_upd", &adam_upd, "Adam update");
  m.def("masked_adam_upd", &masked_adam_upd, "Adam update ignoring zero grad");

  m.def(
      "total_variation_add_grad",
      &total_variation_add_grad,
      "Add total variation grad");
  m.def(
      "order1_filter_add_grad",
      &order1_filter_add_grad,
      "Add total variation (1st order) grad");
  m.def("laplace_add_grad", &laplace_add_grad, "Add laplace filter grad");
  m.def("laplace_lg_add_grad", &laplace_lg_add_grad, "Add laplace filter grad");
  m.def(
      "order3_filter_add_grad",
      &order3_filter_add_grad,
      "Third order filter loss");
  m.def("diff_add_grad", &diff_add_grad, "Add diff grad");
  m.def("eikonal_add_grad", &eikonal_add_grad, "Add eikonal filter grad");

  m.def(
      "grid_sample_3d_second_derivative_to_voxel",
      &grid_sample_3d_second_derivative_to_voxel,
      "");
  m.def(
      "grid_sample_2d_second_derivative_to_voxel",
      &grid_sample_2d_second_derivative_to_voxel,
      "");

  m.def(
      "segment_cumsum",
      &segment_cumsum,
      "Compute segment prefix-sum (cumsum).");

  m.def(
      "sdf_grid_trace_surface",
      &sdf_grid_trace_surface,
      "Ray trace on dense sdf grid.");
}
