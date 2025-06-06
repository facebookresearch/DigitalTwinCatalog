# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gin
import numpy as np
import torch
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from largesteps.parameterize import from_differential, to_differential


@gin.configurable
class LargeStepsOptimizer(torch.optim.Optimizer):
    def __init__(self, V, F, lr=0.1, betas=(0.9, 0.999), lmbda=20):
        self.V = V
        self.F = F.cuda()

        self.M = compute_matrix(self.V.detach().clone().cuda(), self.F, lmbda)
        self.u = (
            to_differential(self.M, self.V.detach().clone().cuda())
            .clone()
            .requires_grad_()
        )
        defaults = dict(F=self.F, lr=lr, betas=betas)

        self.optimizer = AdamUniform([self.u], lr=lr, betas=betas)
        super(LargeStepsOptimizer, self).__init__([V], defaults)

    def step(self):
        # build compute graph from u to V
        V = from_differential(self.M, self.u, "Cholesky")
        # propagate gradients from V to u
        V.backward(self.V.grad.cuda())
        # step u
        self.optimizer.step()
        # update param
        self.V.data.copy_(
            from_differential(self.M, self.u, "Cholesky").to(self.V.device)
        )

    def zero_grad(self):
        super(LargeStepsOptimizer, self).zero_grad()
        self.optimizer.zero_grad()


def connect_strip(idx0, idx1, l2r=True, ccw=True, connect_ends=False):
    F = []

    def connect_square(v0_idx, v1_idx, v2_idx, v3_idx):
        # write faces
        # left to right
        # v0-----v3
        # |  \    |
        # |    \  |
        # |      \|
        # v1-----v2
        ##
        # right to left
        # v0-----v3
        # |     / |
        # |   /   |
        # | /     |
        # v1-----v2
        ##

        F.append([v1_idx, v2_idx, v0_idx])
        F.append([v3_idx, v0_idx, v2_idx])

        # if l2r:
        #     if ccw:
        #         F.append([v1_idx, v2_idx, v0_idx])
        #         F.append([v3_idx, v0_idx, v2_idx])
        #     else:
        #         F.append([v1_idx, v0_idx, v2_idx])
        #         F.append([v3_idx, v2_idx, v0_idx])
        # else:
        #     if ccw:
        #         F.append([v0_idx, v1_idx, v3_idx])
        #         F.append([v2_idx, v3_idx, v1_idx])
        #     else:
        #         F.append([v0_idx, v3_idx, v1_idx])
        #         F.append([v2_idx, v1_idx, v3_idx])

    l = len(idx0)
    for i in range(l - 1):
        v0_idx = idx0[i]
        v1_idx = idx1[i]
        v2_idx = idx1[i + 1]
        v3_idx = idx0[i + 1]
        connect_square(v0_idx, v1_idx, v2_idx, v3_idx)

    if connect_ends:
        connect_square(idx0[l - 1], idx1[l - 1], idx0[0], idx1[0])

    F = np.stack(F)

    return F
