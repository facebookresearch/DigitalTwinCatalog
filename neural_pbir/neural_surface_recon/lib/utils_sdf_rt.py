# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import neural_pbir_cuda_utils
import torch.nn.functional as F


def sdf_grid_trace_surface(sdfgrid, rs, rd, stepsize, xyz_min, xyz_max, world_size):
    shift = xyz_min
    scale = (world_size - 1) / (xyz_max - xyz_min)
    rs_vox = (rs - shift) * scale
    step_vox = F.normalize(rd * scale, dim=-1) * stepsize
    hitxyz = neural_pbir_cuda_utils.sdf_grid_trace_surface(
        sdfgrid, rs_vox.flip(-1), step_vox.flip(-1)
    ).flip(-1)
    hitxyz = hitxyz / scale + shift
    is_hit = ((xyz_min < hitxyz) & (hitxyz < xyz_max)).all(-1)
    return hitxyz, is_hit
