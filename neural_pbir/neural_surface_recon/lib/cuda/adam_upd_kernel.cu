/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__device__ __forceinline__ float lerp(float v0, float v1, float t) {
  return fma(t, v1 - v0, v0);
}

template <typename scalar_t>
__global__ void adam_upd_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const size_t N,
    int64_t* __restrict__ step,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float warmup_iter) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    const float _step = static_cast<float>(++step[index]);
    const float warmup_rate = min(1., _step / max(1., warmup_iter));
    const float step_size = lr * sqrt(1 - pow(beta2, _step)) /
        (1 - pow(beta1, _step)) * warmup_rate;
    const float g = grad[index];
    const float gg = g * g;
    exp_avg[index] = lerp(g, exp_avg[index], beta1);
    exp_avg_sq[index] = lerp(gg, exp_avg_sq[index], beta2);
    param[index] -=
        step_size * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}

template <typename scalar_t>
__global__ void masked_adam_upd_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const size_t N,
    int64_t* __restrict__ step,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float warmup_iter) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N && grad[index] != 0) {
    const float _step = static_cast<float>(++step[index]);
    const float warmup_rate = min(1., _step / max(1., warmup_iter));
    const float step_size = lr * sqrt(1 - pow(beta2, _step)) /
        (1 - pow(beta1, _step)) * warmup_rate;
    const float g = grad[index];
    const float gg = g * g;
    exp_avg[index] = lerp(g, exp_avg[index], beta1);
    exp_avg_sq[index] = lerp(gg, exp_avg_sq[index], beta2);
    param[index] -=
        step_size * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}

void adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor step,
    const float beta1,
    const float beta2,
    const float lr,
    const float eps,
    const float warmup_iter) {
  const size_t N = param.numel();

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  adam_upd_cuda_kernel<float_t><<<blocks, threads>>>(
      param.data<float_t>(),
      grad.data<float_t>(),
      exp_avg.data<float_t>(),
      exp_avg_sq.data<float_t>(),
      N,
      step.data<int64_t>(),
      lr,
      beta1,
      beta2,
      eps,
      warmup_iter);
}

void masked_adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor step,
    const float beta1,
    const float beta2,
    const float lr,
    const float eps,
    const float warmup_iter) {
  const size_t N = param.numel();

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  masked_adam_upd_cuda_kernel<float_t><<<blocks, threads>>>(
      param.data<float_t>(),
      grad.data<float_t>(),
      exp_avg.data<float_t>(),
      exp_avg_sq.data<float_t>(),
      N,
      step.data<int64_t>(),
      lr,
      beta1,
      beta2,
      eps,
      warmup_iter);
}
