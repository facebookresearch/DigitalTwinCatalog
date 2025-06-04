# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Extend Adam optimizer to support sparse update to speedup training.
Note that the SparseAdam in pytorch is not work the same way as this.
"""

import os

import neural_pbir_cuda_utils
import torch
from torch.utils.cpp_extension import load


class MaskedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-15, warmup_iter=100):
        # TODO: check why eps=1e-15 make coarse
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        self.warmup_iter = warmup_iter
        super(MaskedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaskedAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            skip_zero_grad = group["skip_zero_grad"]

            for param in group["params"]:
                if param.grad is not None:
                    state = self.state[param]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = torch.zeros(
                            param.shape, dtype=torch.long, device=param.device
                        )
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                    if skip_zero_grad:
                        neural_pbir_cuda_utils.masked_adam_upd(
                            param,
                            param.grad,
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            state["step"],
                            beta1,
                            beta2,
                            lr,
                            eps,
                            self.warmup_iter,
                        )
                    else:
                        neural_pbir_cuda_utils.adam_upd(
                            param,
                            param.grad,
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            state["step"],
                            beta1,
                            beta2,
                            lr,
                            eps,
                            self.warmup_iter,
                        )
