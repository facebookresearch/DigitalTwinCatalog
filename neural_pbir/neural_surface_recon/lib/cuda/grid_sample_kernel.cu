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

__device__ __forceinline__ int clamp(const int i, const int L, const int R) {
  return min(max(i, L), R);
}

template <typename scalar_t>
__global__ void grid_sample_3d_second_derivative_to_voxel_cuda_kernel(
    scalar_t* __restrict__ coord,
    scalar_t* __restrict__ dL_dn,
    const int NZ,
    const int NY,
    const int NX,
    const int n_pts,
    scalar_t* __restrict__ dL_dV) {
  const auto i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_pt < n_pts) {
    const scalar_t x = (coord[i_pt * 3] + 1) * 0.5 * (NX - 1);
    const scalar_t y = (coord[i_pt * 3 + 1] + 1) * 0.5 * (NY - 1);
    const scalar_t z = (coord[i_pt * 3 + 2] + 1) * 0.5 * (NZ - 1);
    const scalar_t px = x - floor(x);
    const scalar_t py = y - floor(y);
    const scalar_t pz = z - floor(z);

    const scalar_t dL_dnx = dL_dn[i_pt * 3] * 0.5 * (NX - 1);
    const scalar_t dL_dny = dL_dn[i_pt * 3 + 1] * 0.5 * (NY - 1);
    const scalar_t dL_dnz = dL_dn[i_pt * 3 + 2] * 0.5 * (NZ - 1);

    const scalar_t sx[2] = {1 - px, px};
    const scalar_t sy[2] = {1 - py, py};
    const scalar_t sz[2] = {1 - pz, pz};
    const scalar_t dd[2] = {-1, 1};

    const int ix[2] = {clamp((int)x, 0, NX - 1), clamp((int)x + 1, 0, NX - 1)};
    const int iy[2] = {clamp((int)y, 0, NY - 1), clamp((int)y + 1, 0, NY - 1)};
    const int iz[2] = {clamp((int)z, 0, NZ - 1), clamp((int)z + 1, 0, NZ - 1)};

    const int NXY = (NX * NY);
    for (int iii = 0; iii < 8; ++iii) {
      const int i = (iii & 1);
      const int j = (iii & 2) >> 1;
      const int k = (iii & 4) >> 2;
      const scalar_t dnx_dV = dd[i] * sy[j] * sz[k];
      const scalar_t dny_dV = sx[i] * dd[j] * sz[k];
      const scalar_t dnz_dV = sx[i] * sy[j] * dd[k];
      atomicAdd(
          dL_dV + iz[k] * NXY + iy[j] * NX + ix[i],
          dL_dnx * dnx_dV + dL_dny * dny_dV + dL_dnz * dnz_dV);
    }
  }
}

torch::Tensor grid_sample_3d_second_derivative_to_voxel_cuda(
    torch::Tensor coord,
    torch::Tensor dL_dn,
    const int NB,
    const int NC,
    const int NZ,
    const int NY,
    const int NX) {
  const int n_pts = coord.size(0);

  auto dL_dV = torch::zeros({1, 1, NZ, NY, NX}, coord.options());

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  grid_sample_3d_second_derivative_to_voxel_cuda_kernel<float_t>
      <<<blocks, threads>>>(
          coord.data<float_t>(),
          dL_dn.data<float_t>(),
          NZ,
          NY,
          NX,
          n_pts,
          dL_dV.data<float_t>());

  dL_dV = dL_dV.repeat({1, NC, 1, 1, 1});

  return dL_dV;
}

template <typename scalar_t>
__global__ void grid_sample_2d_second_derivative_to_voxel_cuda_kernel(
    scalar_t* __restrict__ coord,
    scalar_t* __restrict__ dL_dn,
    const int NY,
    const int NX,
    const int n_pts,
    scalar_t* __restrict__ dL_dV) {
  const auto i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_pt < n_pts) {
    const scalar_t x = (coord[i_pt * 2] + 1) * 0.5 * (NX - 1);
    const scalar_t y = (coord[i_pt * 2 + 1] + 1) * 0.5 * (NY - 1);
    const scalar_t px = x - floor(x);
    const scalar_t py = y - floor(y);

    const scalar_t dL_dnx = dL_dn[i_pt * 2] * 0.5 * (NX - 1);
    const scalar_t dL_dny = dL_dn[i_pt * 2 + 1] * 0.5 * (NY - 1);

    const scalar_t sx[2] = {1 - px, px};
    const scalar_t sy[2] = {1 - py, py};
    const scalar_t dd[2] = {-1, 1};

    const int ix[2] = {clamp((int)x, 0, NX - 1), clamp((int)x + 1, 0, NX - 1)};
    const int iy[2] = {clamp((int)y, 0, NY - 1), clamp((int)y + 1, 0, NY - 1)};

    for (int iii = 0; iii < 4; ++iii) {
      const int i = (iii & 1);
      const int j = (iii & 2) >> 1;
      const scalar_t dnx_dV = dd[i] * sy[j];
      const scalar_t dny_dV = sx[i] * dd[j];
      atomicAdd(dL_dV + iy[j] * NX + ix[i], dL_dnx * dnx_dV + dL_dny * dny_dV);
    }
  }
}

torch::Tensor grid_sample_2d_second_derivative_to_voxel_cuda(
    torch::Tensor coord,
    torch::Tensor dL_dn,
    const int NB,
    const int NC,
    const int NY,
    const int NX) {
  const int n_pts = coord.size(0);

  auto dL_dV = torch::zeros({1, 1, NY, NX}, coord.options());

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  grid_sample_2d_second_derivative_to_voxel_cuda_kernel<float_t>
      <<<blocks, threads>>>(
          coord.data<float_t>(),
          dL_dn.data<float_t>(),
          NY,
          NX,
          n_pts,
          dL_dV.data<float_t>());

  dL_dV = dL_dV.repeat({1, NC, 1, 1});

  return dL_dV;
}
