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

__device__ __forceinline__ int check_sdf_bound(
    const float x,
    const float y,
    const float z,
    const int NZ,
    const int NY,
    const int NX,
    const float tol = 0.01) {
  return -tol <= x && x <= NX - 1 + tol && -tol <= y && y <= NY - 1 + tol &&
      -tol <= z && z <= NZ - 1 + tol;
}

__device__ __forceinline__ float interp_sdf_grid(
    const float* __restrict__ sdfgrid,
    const float x,
    const float y,
    const float z,
    const int NZ,
    const int NY,
    const int NX) {
  const float px = x - floor(x);
  const float py = y - floor(y);
  const float pz = z - floor(z);
  const float sx[2] = {1 - px, px};
  const float sy[2] = {1 - py, py};
  const float sz[2] = {1 - pz, pz};
  const int ix[2] = {clamp((int)x, 0, NX - 1), clamp((int)x + 1, 0, NX - 1)};
  const int iy[2] = {clamp((int)y, 0, NY - 1), clamp((int)y + 1, 0, NY - 1)};
  const int iz[2] = {clamp((int)z, 0, NZ - 1), clamp((int)z + 1, 0, NZ - 1)};
  // const int ix[2] = {(int)x, (int)x+1};
  // const int iy[2] = {(int)y, (int)y+1};
  // const int iz[2] = {(int)z, (int)z+1};
  const int NXY = (NX * NY);
  float sdf = 0;
  for (int iii = 0; iii < 8; ++iii) {
    const int i = (iii & 1);
    const int j = (iii & 2) >> 1;
    const int k = (iii & 4) >> 2;
    const float w = sx[i] * sy[j] * sz[k];
    sdf += (*(sdfgrid + iz[k] * NXY + iy[j] * NX + ix[i])) * w;
  }
  return sdf;
}

__global__ void sdf_grid_trace_surface_cuda_kernel(
    float* __restrict__ sdfgrid,
    float* __restrict__ rs,
    float* __restrict__ step,
    const int NZ,
    const int NY,
    const int NX,
    const int n_rays,
    float* __restrict__ hitxyz) {
  const auto i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_ray < n_rays) {
    const int offset = i_ray * 3;
    float x = rs[offset];
    float y = rs[offset + 1];
    float z = rs[offset + 2];
    const float dx = step[offset];
    const float dy = step[offset + 1];
    const float dz = step[offset + 2];

    // sdf tracing
    float prev_sdf = interp_sdf_grid(sdfgrid, x, y, z, NZ, NY, NX);
    x += dx, y += dy, z += dz;
    float sdf;
    for (int _ = 0; _ < 8192 && check_sdf_bound(x, y, z, NZ, NY, NX);
         ++_, x += dx, y += dy, z += dz) {
      sdf = interp_sdf_grid(sdfgrid, x, y, z, NZ, NY, NX);
      if (sdf <= 0) {
        const float p = abs(sdf / (prev_sdf - sdf));
        x -= dx * p;
        y -= dy * p;
        z -= dz * p;
        break;
      }
      prev_sdf = sdf;
    }

    // fill-in result
    hitxyz[offset] = x;
    hitxyz[offset + 1] = y;
    hitxyz[offset + 2] = z;
  }
}

torch::Tensor sdf_grid_trace_surface_cuda(
    torch::Tensor sdfgrid,
    torch::Tensor rs,
    torch::Tensor step) {
  const int NZ = sdfgrid.size(2);
  const int NY = sdfgrid.size(3);
  const int NX = sdfgrid.size(4);
  const int n_rays = rs.size(0);

  auto hitxyz = torch::zeros_like(rs);
  if (n_rays == 0) {
    return hitxyz;
  }

  const int threads = 512;
  const int blocks = (n_rays + threads - 1) / threads;
  sdf_grid_trace_surface_cuda_kernel<<<blocks, threads>>>(
      sdfgrid.data<float_t>(),
      rs.data<float_t>(),
      step.data<float_t>(),
      NZ,
      NY,
      NX,
      n_rays,
      hitxyz.data<float_t>());

  return hitxyz;
}
