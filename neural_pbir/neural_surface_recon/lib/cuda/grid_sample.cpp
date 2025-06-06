/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "grid_sample.h"

// C++ interface implementation
torch::Tensor grid_sample_3d_second_derivative_to_voxel(
    torch::Tensor coord,
    torch::Tensor dL_dn,
    const int NB,
    const int NC,
    const int NZ,
    const int NY,
    const int NX) {
  CHECK_INPUT(coord);
  CHECK_INPUT(dL_dn);
  TORCH_CHECK(coord.dim() == 2, "Expect coord.dim()==2");
  TORCH_CHECK(coord.size(1) == 3, "Expect coord.size(1)==3");
  TORCH_CHECK(dL_dn.dim() == 2, "Expect dL_dn.dim()==2");
  TORCH_CHECK(dL_dn.size(1) == 3, "Expect dL_dn.size(1)==3");
  TORCH_CHECK(NB == 1, "Expect NB==1");
  return grid_sample_3d_second_derivative_to_voxel_cuda(
      coord, dL_dn, NB, NC, NZ, NY, NX);
}

torch::Tensor grid_sample_2d_second_derivative_to_voxel(
    torch::Tensor coord,
    torch::Tensor dL_dn,
    const int NB,
    const int NC,
    const int NY,
    const int NX) {
  CHECK_INPUT(coord);
  CHECK_INPUT(dL_dn);
  TORCH_CHECK(coord.dim() == 2, "Expect coord.dim()==2");
  TORCH_CHECK(coord.size(1) == 2, "Expect coord.size(1)==2");
  TORCH_CHECK(dL_dn.dim() == 2, "Expect dL_dn.dim()==2");
  TORCH_CHECK(dL_dn.size(1) == 2, "Expect dL_dn.size(1)==2");
  TORCH_CHECK(NB == 1, "Expect NB==1");
  return grid_sample_2d_second_derivative_to_voxel_cuda(
      coord, dL_dn, NB, NC, NY, NX);
}
