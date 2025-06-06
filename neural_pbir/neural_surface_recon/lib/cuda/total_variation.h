/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

// CUDA forward declarations
void total_variation_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float wx,
    float wy,
    float wz,
    bool dense_mode);
void order1_filter_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    bool dense_mode);
void laplace_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    int type,
    bool dense_mode,
    int step);
void laplace_lg_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    bool dense_mode,
    int ks);
void order3_filter_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    int type,
    bool dense_mode);
void diff_add_grad_cuda(torch::Tensor param, torch::Tensor grad, float weight);
void eikonal_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float gamma,
    float weight);

// C++ interface
void total_variation_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float wx,
    float wy,
    float wz,
    bool dense_mode);
void order1_filter_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    bool dense_mode);
void laplace_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    int type,
    bool dense_mode,
    int step);
void laplace_lg_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    bool dense_mode,
    int ks);
void order3_filter_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    int type,
    bool dense_mode);
void diff_add_grad(torch::Tensor param, torch::Tensor grad, float weight);
void eikonal_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float gamma,
    float weight);
