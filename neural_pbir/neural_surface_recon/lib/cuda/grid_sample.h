/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

// CUDA forward declarations
torch::Tensor grid_sample_3d_second_derivative_to_voxel_cuda(
    torch::Tensor coord,
    torch::Tensor dL_dn,
    const int NB,
    const int NC,
    const int NZ,
    const int NY,
    const int NX);

torch::Tensor grid_sample_2d_second_derivative_to_voxel_cuda(
    torch::Tensor coord,
    torch::Tensor dL_dn,
    const int NB,
    const int NC,
    const int NY,
    const int NX);

// C++ interface
torch::Tensor grid_sample_3d_second_derivative_to_voxel(
    torch::Tensor coord,
    torch::Tensor dL_dn,
    const int NB,
    const int NC,
    const int NZ,
    const int NY,
    const int NX);

torch::Tensor grid_sample_2d_second_derivative_to_voxel(
    torch::Tensor coord,
    torch::Tensor dL_dn,
    const int NB,
    const int NC,
    const int NY,
    const int NX);
