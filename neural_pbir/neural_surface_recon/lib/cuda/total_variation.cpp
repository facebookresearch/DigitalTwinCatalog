/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "total_variation.h"

// C++ interface implementation
void total_variation_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float wx,
    float wy,
    float wz,
    bool dense_mode) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  total_variation_add_grad_cuda(param, grad, wx, wy, wz, dense_mode);
}

void order1_filter_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    bool dense_mode) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  order1_filter_add_grad_cuda(param, grad, lb, ub, weight, dense_mode);
}

void laplace_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    int type,
    bool dense_mode,
    int step) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  laplace_add_grad_cuda(param, grad, lb, ub, weight, type, dense_mode, step);
}

void laplace_lg_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    bool dense_mode,
    int ks) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  laplace_lg_add_grad_cuda(param, grad, lb, ub, weight, dense_mode, ks);
}

void order3_filter_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    int type,
    bool dense_mode) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  order3_filter_add_grad_cuda(param, grad, lb, ub, weight, type, dense_mode);
}

void diff_add_grad(torch::Tensor param, torch::Tensor grad, float weight) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  diff_add_grad_cuda(param, grad, weight);
}

void eikonal_add_grad(
    torch::Tensor param,
    torch::Tensor grad,
    float gamma,
    float weight) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  eikonal_add_grad_cuda(param, grad, gamma, weight);
}
