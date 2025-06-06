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

template <typename scalar_t, typename bound_t>
__device__ __forceinline__ scalar_t
clamp(const scalar_t v, const bound_t lo, const bound_t hi) {
  return min(max(v, lo), hi);
}

template <typename scalar_t, bool dense_mode>
__global__ void total_variation_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float wx,
    float wy,
    float wz,
    const size_t sz_i,
    const size_t sz_j,
    const size_t sz_k,
    const size_t N) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N && (dense_mode || grad[index] != 0)) {
    const size_t k = index % sz_k;
    const size_t j = index / sz_k % sz_j;
    const size_t i = index / sz_k / sz_j % sz_i;

    float grad_to_add = 0;
    grad_to_add +=
        (k == 0 ? 0 : wz * clamp(param[index] - param[index - 1], -1.f, 1.f));
    grad_to_add +=
        (k == sz_k - 1
             ? 0
             : wz * clamp(param[index] - param[index + 1], -1.f, 1.f));
    grad_to_add +=
        (j == 0 ? 0
                : wy * clamp(param[index] - param[index - sz_k], -1.f, 1.f));
    grad_to_add +=
        (j == sz_j - 1
             ? 0
             : wy * clamp(param[index] - param[index + sz_k], -1.f, 1.f));
    grad_to_add +=
        (i == 0 ? 0
                : wz *
                 clamp(param[index] - param[index - sz_k * sz_j], -1.f, 1.f));
    grad_to_add +=
        (i == sz_i - 1 ? 0
                       : wz *
                 clamp(param[index] - param[index + sz_k * sz_j], -1.f, 1.f));
    grad[index] += grad_to_add;
  }
}

void total_variation_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float wx,
    float wy,
    float wz,
    bool dense_mode) {
  const size_t N = param.numel();
  const size_t sz_i = param.size(2);
  const size_t sz_j = param.size(3);
  const size_t sz_k = param.size(4);
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  wx /= 6;
  wy /= 6;
  wz /= 6;

  if (dense_mode) {
    total_variation_add_grad_cuda_kernel<float_t, true><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        wx,
        wy,
        wz,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else {
    total_variation_add_grad_cuda_kernel<float_t, false><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        wx,
        wy,
        wz,
        sz_i,
        sz_j,
        sz_k,
        N);
  }
}

template <typename scalar_t, bool seven, bool separate>
__global__ void laplace_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float lb,
    float ub,
    bool dense_mode,
    int step,
    float weight,
    const int sz_i,
    const int sz_j,
    const int sz_k,
    const int N) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = index % sz_k;
  const int j = index / sz_k % sz_j;
  const int i = index / sz_k / sz_j % sz_i;
  const int sz_jk = sz_j * sz_k;
  const bool valid = index < N && i - step >= 0 && j - step >= 0 &&
      k - step >= 0 && i + step < sz_i && j + step < sz_j && k + step < sz_k;
  if (valid && (lb < param[index] && param[index] < ub) &&
      (dense_mode || (param[index] != 0))) {
    if (separate) {
      const int i111 = i * sz_jk + j * sz_k + k;
      const int i110 = i * sz_jk + j * sz_k + (k - step);
      const int i101 = i * sz_jk + (j - step) * sz_k + k;
      const int i011 = (i - step) * sz_jk + j * sz_k + k;
      const int i112 = i * sz_jk + j * sz_k + (k + step);
      const int i121 = i * sz_jk + (j + step) * sz_k + k;
      const int i211 = (i + step) * sz_jk + j * sz_k + k;
      const scalar_t V111 = param[i111];
      const scalar_t V110 = param[i110];
      const scalar_t V101 = param[i101];
      const scalar_t V011 = param[i011];
      const scalar_t V112 = param[i112];
      const scalar_t V121 = param[i121];
      const scalar_t V211 = param[i211];

      scalar_t lapk = (V110 + V112 - 2 * V111);
      scalar_t lapj = (V101 + V121 - 2 * V111);
      scalar_t lapi = (V011 + V211 - 2 * V111);
      // L = 0.5 * weight * (lapi * lapi + lapj * lapj + lapk * lapk);
      lapk *= weight;
      lapj *= weight;
      lapi *= weight;
      atomicAdd(grad + i111, -2 * (lapk + lapj + lapi));
      atomicAdd(grad + i110, lapk);
      atomicAdd(grad + i101, lapj);
      atomicAdd(grad + i011, lapi);
      atomicAdd(grad + i112, lapk);
      atomicAdd(grad + i121, lapj);
      atomicAdd(grad + i211, lapi);
    } else if (seven) {
      const int i111 = i * sz_jk + j * sz_k + k;
      const int i110 = i * sz_jk + j * sz_k + (k - step);
      const int i101 = i * sz_jk + (j - step) * sz_k + k;
      const int i011 = (i - step) * sz_jk + j * sz_k + k;
      const int i112 = i * sz_jk + j * sz_k + (k + step);
      const int i121 = i * sz_jk + (j + step) * sz_k + k;
      const int i211 = (i + step) * sz_jk + j * sz_k + k;
      const scalar_t V111 = param[i111];
      const scalar_t V110 = param[i110];
      const scalar_t V101 = param[i101];
      const scalar_t V011 = param[i011];
      const scalar_t V112 = param[i112];
      const scalar_t V121 = param[i121];
      const scalar_t V211 = param[i211];
      scalar_t lap = (V110 + V101 + V011 + V112 + V121 + V211 - 6 * V111);
      // L = 0.5 * weight * lap * lap;
      lap *= weight;
      atomicAdd(grad + i111, lap * -6);
      atomicAdd(grad + i110, lap);
      atomicAdd(grad + i101, lap);
      atomicAdd(grad + i011, lap);
      atomicAdd(grad + i112, lap);
      atomicAdd(grad + i121, lap);
      atomicAdd(grad + i211, lap);
    } else {
      scalar_t V[27];
#pragma unroll
      for (int off = 0; off < 27; ++off) {
        const int offk = off % 3 - step;
        const int offj = off / 3 % 3 - step;
        const int offi = off / 3 / 3 % 3 - step;
        const int off_ind = (i + offi) * sz_jk + (j + offj) * sz_k + (k + offk);
        V[off] = param[off_ind];
      }

      scalar_t lap = 0;
      const scalar_t w[4] = {-88. / 26., 6. / 26., 3. / 26., 2. / 26.};
#pragma unroll
      for (int off = 0; off < 27; ++off) {
        const int offk = off % 3 - step;
        const int offj = off / 3 % 3 - step;
        const int offi = off / 3 / 3 % 3 - step;
        lap += V[off] * w[abs(offk) + abs(offj) + abs(offi)];
      }
      lap *= weight;

#pragma unroll
      for (int off = 0; off < 27; ++off) {
        const int offk = off % 3 - step;
        const int offj = off / 3 % 3 - step;
        const int offi = off / 3 / 3 % 3 - step;
        const int off_ind = (i + offi) * sz_jk + (j + offj) * sz_k + (k + offk);
        // L = 0.5 * lap * lap;
        atomicAdd(grad + off_ind, lap * w[abs(offk) + abs(offj) + abs(offi)]);
      }
    }
  }
}

template <typename scalar_t>
__global__ void laplace_y_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float lb,
    float ub,
    bool dense_mode,
    int step,
    float weight,
    const int sz_i,
    const int sz_j,
    const int sz_k,
    const int N) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = index % sz_k;
  const int j = index / sz_k % sz_j;
  const int i = index / sz_k / sz_j % sz_i;
  const int sz_jk = sz_j * sz_k;
  const bool valid = index < N && j - step >= 0 && j + step < sz_j;
  if (valid && (lb < param[index] && param[index] < ub) &&
      (dense_mode || (param[index] != 0))) {
    const int i111 = i * sz_jk + j * sz_k + k;
    const int i101 = i * sz_jk + (j - step) * sz_k + k;
    const int i121 = i * sz_jk + (j + step) * sz_k + k;
    const scalar_t V111 = param[i111];
    const scalar_t V101 = param[i101];
    const scalar_t V121 = param[i121];

    scalar_t lapj = (V101 + V121 - 2 * V111);
    // L = 0.5 * weight * lapj * lapj;
    lapj *= weight;
    atomicAdd(grad + i111, -2 * lapj);
    atomicAdd(grad + i101, lapj);
    atomicAdd(grad + i121, lapj);
  }
}

template <typename scalar_t>
__global__ void laplace_xz_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float lb,
    float ub,
    bool dense_mode,
    int step,
    float weight,
    const int sz_i,
    const int sz_j,
    const int sz_k,
    const int N) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = index % sz_k;
  const int j = index / sz_k % sz_j;
  const int i = index / sz_k / sz_j % sz_i;
  const int sz_jk = sz_j * sz_k;
  const bool valid = index < N && i - step >= 0 && k - step >= 0 &&
      i + step < sz_i && k + step < sz_k;
  if (valid && (lb < param[index] && param[index] < ub) &&
      (dense_mode || (param[index] != 0))) {
    const int i111 = i * sz_jk + j * sz_k + k;
    const int i110 = i * sz_jk + j * sz_k + (k - step);
    const int i011 = (i - step) * sz_jk + j * sz_k + k;
    const int i112 = i * sz_jk + j * sz_k + (k + step);
    const int i211 = (i + step) * sz_jk + j * sz_k + k;
    const scalar_t V111 = param[i111];
    const scalar_t V110 = param[i110];
    const scalar_t V011 = param[i011];
    const scalar_t V112 = param[i112];
    const scalar_t V211 = param[i211];
    scalar_t lap = (V110 + V011 + V112 + V211 - V111 * 4);
    // L = 0.5 * weight * lap * lap;
    lap *= weight;
    atomicAdd(grad + i111, -lap * 4);
    atomicAdd(grad + i110, lap);
    atomicAdd(grad + i011, lap);
    atomicAdd(grad + i112, lap);
    atomicAdd(grad + i211, lap);
  }
}

void laplace_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    int type,
    bool dense_mode,
    int step) {
  const int N = param.numel();
  const int sz_i = param.size(2);
  const int sz_j = param.size(3);
  const int sz_k = param.size(4);
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  if (type == 3) {
    laplace_add_grad_cuda_kernel<float_t, true, true><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        step,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else if (type == -3) {
    laplace_y_add_grad_cuda_kernel<float_t><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        step,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else if (type == 5) {
    laplace_xz_add_grad_cuda_kernel<float_t><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        step,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else if (type == 7) {
    laplace_add_grad_cuda_kernel<float_t, true, false><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        step,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else if (type == 27) {
    laplace_add_grad_cuda_kernel<float_t, false, false><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        step,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else {
    assert(false);
  }
}

template <typename scalar_t, int ks>
__global__ void laplace_lg_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float lb,
    float ub,
    bool dense_mode,
    float weight,
    const int sz_i,
    const int sz_j,
    const int sz_k,
    const int N) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = index % sz_k;
  const int j = index / sz_k % sz_j;
  const int i = index / sz_k / sz_j % sz_i;
  const int sz_jk = sz_j * sz_k;
  const int center_i = ks / 2;
  const bool valid = index < N;

  float kernel[7];
  if (ks == 3) {
    kernel[0] = kernel[2] = 1;
    kernel[1] = -2;
  } else if (ks == 5) {
    kernel[0] = kernel[4] = 0.18242552;
    kernel[1] = kernel[3] = 0.81757448;
    kernel[2] = -2;
  } else if (ks == 7) {
    kernel[0] = kernel[6] = 0.01475347;
    kernel[1] = kernel[5] = 0.17973411;
    kernel[2] = kernel[4] = 0.80551241;
    kernel[3] = -2;
  }

  if (valid && (lb < param[index] && param[index] < ub) &&
      (dense_mode || (param[index] != 0))) {
    const int i0 = i * sz_jk + j * sz_k + k;

    // k direction
    float lap = kernel[center_i] * param[i0];
#pragma unroll
    for (int step = 1; step <= center_i; ++step) {
      const int il = i * sz_jk + j * sz_k + abs(k - step);
      const int ir =
          i * sz_jk + j * sz_k + ((sz_k - 1) - abs(k + step - (sz_k - 1)));
      lap += kernel[center_i - step] * param[il];
      lap += kernel[center_i + step] * param[ir];
    }
    lap *= weight;
    atomicAdd(grad + i0, kernel[center_i] * lap);
#pragma unroll
    for (int step = 1; step <= center_i; ++step) {
      const int il = i * sz_jk + j * sz_k + abs(k - step);
      const int ir =
          i * sz_jk + j * sz_k + ((sz_k - 1) - abs(k + step - (sz_k - 1)));
      atomicAdd(grad + il, kernel[center_i - step] * lap);
      atomicAdd(grad + ir, kernel[center_i + step] * lap);
    }

    // j direction
    lap = kernel[center_i] * param[i0];
#pragma unroll
    for (int step = 1; step <= center_i; ++step) {
      const int il = i * sz_jk + abs(j - step) * sz_k + k;
      const int ir =
          i * sz_jk + ((sz_j - 1) - abs(j + step - (sz_j - 1))) * sz_k + k;
      lap += kernel[center_i - step] * param[il];
      lap += kernel[center_i + step] * param[ir];
    }
    lap *= weight;
    atomicAdd(grad + i0, kernel[center_i] * lap);
#pragma unroll
    for (int step = 1; step <= center_i; ++step) {
      const int il = i * sz_jk + abs(j - step) * sz_k + k;
      const int ir =
          i * sz_jk + ((sz_j - 1) - abs(j + step - (sz_j - 1))) * sz_k + k;
      atomicAdd(grad + il, kernel[center_i - step] * lap);
      atomicAdd(grad + ir, kernel[center_i + step] * lap);
    }

    // i direction
    lap = kernel[center_i] * param[i0];
#pragma unroll
    for (int step = 1; step <= center_i; ++step) {
      const int il = abs(i - step) * sz_jk + j * sz_k + k;
      const int ir =
          ((sz_i - 1) - abs(i + step - (sz_i - 1))) * sz_jk + j * sz_k + k;
      lap += kernel[center_i - step] * param[il];
      lap += kernel[center_i + step] * param[ir];
    }
    lap *= weight;
    atomicAdd(grad + i0, kernel[center_i] * lap);
#pragma unroll
    for (int step = 1; step <= center_i; ++step) {
      const int il = abs(i - step) * sz_jk + j * sz_k + k;
      const int ir =
          ((sz_i - 1) - abs(i + step - (sz_i - 1))) * sz_jk + j * sz_k + k;
      atomicAdd(grad + il, kernel[center_i - step] * lap);
      atomicAdd(grad + ir, kernel[center_i + step] * lap);
    }
  }
}

void laplace_lg_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    bool dense_mode,
    int ks) {
  const int N = param.numel();
  const int sz_i = param.size(2);
  const int sz_j = param.size(3);
  const int sz_k = param.size(4);
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  assert((ks == 3) || (ks == 5) || (ks == 7));

  if (ks == 3) {
    laplace_lg_add_grad_cuda_kernel<float_t, 3><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else if (ks == 5) {
    laplace_lg_add_grad_cuda_kernel<float_t, 5><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else if (ks == 7) {
    laplace_lg_add_grad_cuda_kernel<float_t, 7><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  }
}

template <typename scalar_t, int type>
__global__ void order3_filter_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float lb,
    float ub,
    bool dense_mode,
    float weight,
    const size_t sz_i,
    const size_t sz_j,
    const size_t sz_k,
    const size_t N) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t k = index % sz_k;
  const size_t j = index / sz_k % sz_j;
  const size_t i = index / sz_k / sz_j % sz_i;
  const size_t sz_jk = sz_j * sz_k;
  const bool valid = index < N && i + 3 < sz_i && j + 3 < sz_j && k + 3 < sz_k;
  if (valid && (lb < param[index] && param[index] < ub) &&
      (dense_mode || (param[index] != 0))) {
    const size_t i000 = i * sz_jk + j * sz_k + (k);
    const size_t i001 = i * sz_jk + j * sz_k + (k + 1);
    const size_t i002 = i * sz_jk + j * sz_k + (k + 2);
    const size_t i003 = i * sz_jk + j * sz_k + (k + 3);
    const size_t i010 = i * sz_jk + (j + 1) * sz_k + k;
    const size_t i020 = i * sz_jk + (j + 2) * sz_k + k;
    const size_t i030 = i * sz_jk + (j + 3) * sz_k + k;
    const size_t i100 = (i + 1) * sz_jk + j * sz_k + k;
    const size_t i200 = (i + 2) * sz_jk + j * sz_k + k;
    const size_t i300 = (i + 3) * sz_jk + j * sz_k + k;

    const scalar_t V000 = param[i000];
    const scalar_t V001 = param[i001];
    const scalar_t V002 = param[i002];
    const scalar_t V003 = param[i003];
    const scalar_t V010 = param[i010];
    const scalar_t V020 = param[i020];
    const scalar_t V030 = param[i030];
    const scalar_t V100 = param[i100];
    const scalar_t V200 = param[i200];
    const scalar_t V300 = param[i300];

    scalar_t W[4];
    if (type == 0) {
      W[0] = -0.5;
      W[1] = 0.5;
      W[2] = 0.5;
      W[3] = -0.5;
    } else if (type == 1) {
      W[0] = 1.0;
      W[1] = -3.0;
      W[2] = 3.0;
      W[3] = -1.0;
    }

    const scalar_t lapk = V000 * W[0] + V001 * W[1] + V002 * W[2] + V003 * W[3];
    const scalar_t lapj = V000 * W[0] + V010 * W[1] + V020 * W[2] + V030 * W[3];
    const scalar_t lapi = V000 * W[0] + V100 * W[1] + V200 * W[2] + V300 * W[3];

    // Lk = 0.5 * weight * lapk * lapk
    // Lj = 0.5 * weight * lapj * lapj
    // Li = 0.5 * weight * lapi * lapi
    const scalar_t dLk_dlapk = weight * lapk;
    const scalar_t dLj_dlapj = weight * lapj;
    const scalar_t dLi_dlapi = weight * lapi;

    atomicAdd(grad + i000, W[0] * (dLk_dlapk + dLj_dlapj + dLi_dlapi));
    atomicAdd(grad + i001, W[1] * dLk_dlapk);
    atomicAdd(grad + i002, W[2] * dLk_dlapk);
    atomicAdd(grad + i003, W[3] * dLk_dlapk);
    atomicAdd(grad + i010, W[1] * dLj_dlapj);
    atomicAdd(grad + i020, W[2] * dLj_dlapj);
    atomicAdd(grad + i030, W[3] * dLj_dlapj);
    atomicAdd(grad + i100, W[1] * dLi_dlapi);
    atomicAdd(grad + i200, W[2] * dLi_dlapi);
    atomicAdd(grad + i300, W[3] * dLi_dlapi);
  }
}

void order3_filter_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    int type,
    bool dense_mode) {
  const size_t N = param.numel();
  const size_t sz_i = param.size(2);
  const size_t sz_j = param.size(3);
  const size_t sz_k = param.size(4);
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  if (type == 0) {
    order3_filter_add_grad_cuda_kernel<float_t, 0><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else if (type == 1) {
    order3_filter_add_grad_cuda_kernel<float_t, 1><<<blocks, threads>>>(
        param.data<float_t>(),
        grad.data<float_t>(),
        lb,
        ub,
        dense_mode,
        weight,
        sz_i,
        sz_j,
        sz_k,
        N);
  } else {
    assert(false);
  }
}

template <typename scalar_t>
__global__ void order1_filter_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float lb,
    float ub,
    bool dense_mode,
    float weight,
    const size_t sz_i,
    const size_t sz_j,
    const size_t sz_k,
    const size_t N) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t k = index % sz_k;
  const size_t j = index / sz_k % sz_j;
  const size_t i = index / sz_k / sz_j % sz_i;
  const size_t sz_jk = sz_j * sz_k;
  const bool valid = index < N && i + 1 < sz_i && j + 1 < sz_j && k + 1 < sz_k;
  if (valid && (lb < param[index] && param[index] < ub) &&
      (dense_mode || (param[index] != 0))) {
    const size_t i000 = (i)*sz_jk + (j)*sz_k + (k);
    const size_t i001 = (i)*sz_jk + (j)*sz_k + (k + 1);
    const size_t i010 = (i)*sz_jk + (j + 1) * sz_k + (k);
    const size_t i100 = (i + 1) * sz_jk + (j)*sz_k + (k);

    const scalar_t V000 = param[i000];
    const scalar_t V001 = param[i001];
    const scalar_t V010 = param[i010];
    const scalar_t V100 = param[i100];

    const scalar_t Dk = V000 - V001;
    const scalar_t Dj = V000 - V010;
    const scalar_t Di = V000 - V100;

    // Lk = 0.5 * weight * dk * dk
    // Lj = 0.5 * weight * dj * dj
    // Li = 0.5 * weight * di * di
    const scalar_t dLk_dDk = weight * Dk;
    const scalar_t dLj_dDj = weight * Dj;
    const scalar_t dLi_dDi = weight * Di;

    atomicAdd(grad + i000, dLk_dDk + dLj_dDj + dLi_dDi);
    atomicAdd(grad + i001, -dLk_dDk);
    atomicAdd(grad + i010, -dLj_dDj);
    atomicAdd(grad + i100, -dLi_dDi);
  }
}

void order1_filter_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float lb,
    float ub,
    float weight,
    bool dense_mode) {
  const size_t N = param.numel();
  const size_t sz_i = param.size(2);
  const size_t sz_j = param.size(3);
  const size_t sz_k = param.size(4);
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  order1_filter_add_grad_cuda_kernel<float_t><<<blocks, threads>>>(
      param.data<float_t>(),
      grad.data<float_t>(),
      lb,
      ub,
      dense_mode,
      weight,
      sz_i,
      sz_j,
      sz_k,
      N);
}

template <typename scalar_t>
__global__ void diff_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float weight,
    const size_t sz_i,
    const size_t sz_j,
    const size_t sz_k,
    const size_t N) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t k = index % sz_k;
  const size_t j = index / sz_k % sz_j;
  const size_t i = index / sz_k / sz_j % sz_i;
  const size_t sz_jk = sz_j * sz_k;
  const bool valid = index < N && i < sz_i - 1 && j < sz_j - 1 && k < sz_k - 1;
  if (valid) {
    const size_t i000 = i * sz_jk + j * sz_k + k;
    const size_t i001 = i * sz_jk + j * sz_k + (k + 1);
    const size_t i010 = i * sz_jk + (j + 1) * sz_k + k;
    const size_t i100 = (i + 1) * sz_jk + j * sz_k + k;
    const scalar_t V000 = param[i000];
    const scalar_t V001 = param[i001];
    const scalar_t V010 = param[i010];
    const scalar_t V100 = param[i100];

    const scalar_t dk = V001 - V000;
    const scalar_t dj = V010 - V000;
    const scalar_t di = V100 - V000;

    const scalar_t norm = sqrt(dk * dk + dj * dj + di * di + 1e-9);
    const scalar_t dL_dV001 = weight * dk / norm;
    const scalar_t dL_dV010 = weight * dj / norm;
    const scalar_t dL_dV100 = weight * di / norm;
    const scalar_t dL_dV000 = -(dL_dV001 + dL_dV010 + dL_dV100);

    atomicAdd(grad + i001, dL_dV001);
    atomicAdd(grad + i010, dL_dV010);
    atomicAdd(grad + i100, dL_dV100);
    atomicAdd(grad + i000, dL_dV000);
  }
}

void diff_add_grad_cuda(torch::Tensor param, torch::Tensor grad, float weight) {
  const size_t N = param.numel();
  const size_t sz_i = param.size(2);
  const size_t sz_j = param.size(3);
  const size_t sz_k = param.size(4);
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  diff_add_grad_cuda_kernel<float_t><<<blocks, threads>>>(
      param.data<float_t>(), grad.data<float_t>(), weight, sz_i, sz_j, sz_k, N);
}

template <typename scalar_t>
__global__ void eikonal_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float gamma,
    float weight,
    const size_t sz_i,
    const size_t sz_j,
    const size_t sz_k,
    const size_t N) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t k = index % sz_k;
  const size_t j = index / sz_k % sz_j;
  const size_t i = index / sz_k / sz_j % sz_i;
  const size_t sz_jk = sz_j * sz_k;
  // const bool valid = index<N && i>0 && j>0 && k>0 && i<sz_i-1 && j<sz_j-1 &&
  // k<sz_k-1;
  const bool valid = index < N && i < sz_i - 1 && j < sz_j - 1 && k < sz_k - 1;
  if (valid) {
    const size_t i111 = i * sz_jk + j * sz_k + k;
    const size_t i110 = i * sz_jk + j * sz_k + (k - 1);
    const size_t i101 = i * sz_jk + (j - 1) * sz_k + k;
    const size_t i011 = (i - 1) * sz_jk + j * sz_k + k;
    const size_t i112 = i * sz_jk + j * sz_k + (k + 1);
    const size_t i121 = i * sz_jk + (j + 1) * sz_k + k;
    const size_t i211 = (i + 1) * sz_jk + j * sz_k + k;
    const scalar_t V111 = param[i111];
    const scalar_t V110 = param[i110];
    const scalar_t V101 = param[i101];
    const scalar_t V011 = param[i011];
    const scalar_t V112 = param[i112];
    const scalar_t V121 = param[i121];
    const scalar_t V211 = param[i211];

    const scalar_t dk[2] = {V110 - V111, V112 - V111};
    const scalar_t dj[2] = {V101 - V111, V121 - V111};
    const scalar_t di[2] = {V011 - V111, V211 - V111};
    scalar_t gg = 0;
    scalar_t gk[2] = {0, 0};
    scalar_t gj[2] = {0, 0};
    scalar_t gi[2] = {0, 0};

    for (int ii = 0; ii < 8; ++ii) {
      const int iik = (ii & 1);
      const int iij = (ii & 2) >> 1;
      const int iii = (ii & 4) >> 2;
      // Loss = 0.5 * (norm - gamma)^2
      const scalar_t norm = sqrt(
          dk[iik] * dk[iik] + dj[iij] * dj[iij] + di[iii] * di[iii] + 1e-9);
      const scalar_t dL_dsqsum = weight * (norm - gamma) / norm;
      const scalar_t dL_dk = dL_dsqsum * dk[iik];
      const scalar_t dL_dj = dL_dsqsum * dj[iij];
      const scalar_t dL_di = dL_dsqsum * di[iii];
      gk[iik] += dL_dk;
      gj[iij] += dL_dj;
      gi[iii] += dL_di;
      gg -= dL_dk + dL_dj + dL_di;
    }

    atomicAdd(grad + i111, gg);
    atomicAdd(grad + i110, gk[0]);
    atomicAdd(grad + i101, gj[0]);
    atomicAdd(grad + i011, gi[0]);
    atomicAdd(grad + i112, gk[1]);
    atomicAdd(grad + i121, gj[1]);
    atomicAdd(grad + i211, gi[1]);
  }
}

void eikonal_add_grad_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    float gamma,
    float weight) {
  const size_t N = param.numel();
  const size_t sz_i = param.size(2);
  const size_t sz_j = param.size(3);
  const size_t sz_k = param.size(4);
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  eikonal_add_grad_cuda_kernel<float_t><<<blocks, threads>>>(
      param.data<float_t>(),
      grad.data<float_t>(),
      gamma,
      weight,
      sz_i,
      sz_j,
      sz_k,
      N);
}
