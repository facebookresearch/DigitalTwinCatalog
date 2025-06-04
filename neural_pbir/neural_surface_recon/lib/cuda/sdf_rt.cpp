/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sdf_rt.h"

// C++ interface implementation
torch::Tensor sdf_grid_trace_surface(
    torch::Tensor sdfgrid,
    torch::Tensor rs,
    torch::Tensor step) {
  CHECK_INPUT(sdfgrid);
  CHECK_INPUT(rs);
  CHECK_INPUT(step);
  TORCH_CHECK(rs.dim() == 2, "Expect rs.dim()==2");
  TORCH_CHECK(rs.size(1) == 3, "Expect rs.size(1)==3");
  TORCH_CHECK(step.dim() == 2, "Expect step.dim()==2");
  TORCH_CHECK(step.size(1) == 3, "Expect step.size(1)==3");
  TORCH_CHECK(sdfgrid.dim() == 5, "Expect sdfgrid.dim()==5");
  TORCH_CHECK(sdfgrid.size(0) == 1, "Expect sdfgrid.size(0)==1");
  TORCH_CHECK(sdfgrid.size(1) == 1, "Expect sdfgrid.size(1)==1");
  return sdf_grid_trace_surface_cuda(sdfgrid, rs, step);
}
