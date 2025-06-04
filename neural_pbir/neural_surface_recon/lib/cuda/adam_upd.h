/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

// CUDA forward declarations
void adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor step,
    float beta1,
    float beta2,
    float lr,
    float eps,
    float warmup_iter);

void masked_adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor step,
    float beta1,
    float beta2,
    float lr,
    float eps,
    float warmup_iter);

// C++ interface
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
    float warmup_iter);

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
    float warmup_iter);
