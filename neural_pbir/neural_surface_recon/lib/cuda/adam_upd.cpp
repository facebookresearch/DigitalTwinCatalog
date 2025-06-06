/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "adam_upd.h"

// C++ interface implementation
void adam_upd(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor step,
    float beta1,
    float beta2,
    float lr,
    float eps,
    float warmup_iter) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  CHECK_INPUT(step);
  adam_upd_cuda(
      param,
      grad,
      exp_avg,
      exp_avg_sq,
      step,
      beta1,
      beta2,
      lr,
      eps,
      warmup_iter);
}

void masked_adam_upd(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor step,
    float beta1,
    float beta2,
    float lr,
    float eps,
    float warmup_iter) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  CHECK_INPUT(step);
  masked_adam_upd_cuda(
      param,
      grad,
      exp_avg,
      exp_avg_sq,
      step,
      beta1,
      beta2,
      lr,
      eps,
      warmup_iter);
}
