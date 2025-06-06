/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

// CUDA forward declarations
torch::Tensor sdf_grid_trace_surface_cuda(
    torch::Tensor sdfgrid,
    torch::Tensor rs,
    torch::Tensor step);

// C++ interface
torch::Tensor sdf_grid_trace_surface(
    torch::Tensor sdfgrid,
    torch::Tensor rs,
    torch::Tensor step);
