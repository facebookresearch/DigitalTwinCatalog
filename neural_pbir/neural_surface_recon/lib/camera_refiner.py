# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The code is adapted from:
BARF: Bundle-Adjusting Neural Radiance Fields
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_adam import MaskedAdam


class CamRefiner(nn.Module):
    def __init__(self, poses, config):
        super(CamRefiner, self).__init__()
        self.config = config
        self.num_imgs = len(poses)
        self._poses = poses[:, :3, :4].clone()
        self.register_buffer("t_ori", self._poses[:, :3, 3])
        self.register_buffer("t_ori_mean", self.t_ori.mean(0, keepdim=True))
        self.register_buffer(
            "t_ori_scale", (self.t_ori - self.t_ori_mean).pow(2).mean().sqrt()
        )
        self.t_res = nn.Parameter(torch.zeros([self.num_imgs, 3]))
        self.R_res = nn.Parameter(torch.zeros([self.num_imgs, 6]))
        with torch.no_grad():
            self.R_res.data[:, 0] = 1
            self.R_res.data[:, 4] = 1
        self.optim = MaskedAdam(
            [
                {
                    "params": self.t_res,
                    "lr": config.lrate_t_res,
                    "skip_zero_grad": False,
                },
                {
                    "params": self.R_res,
                    "lr": config.lrate_R_res,
                    "skip_zero_grad": False,
                },
            ]
        )
        self.tick = 0
        self.coldstart_iter = config.coldstart_iter

    def forward(self, rays_o, rays_d, viewdirs, image_id):
        if self.tick < self.coldstart_iter:
            return rays_o, rays_d, viewdirs
        # rotate ray origin and direction
        R, t = self.get_residual_poses()
        R = R[image_id]
        t = t[image_id]
        rays_o = rays_o + t.squeeze(-1)
        rays_d = torch.bmm(R, rays_d.unsqueeze(-1)).squeeze(-1)
        if self.config.detach_viewdirs:
            viewdirs = torch.bmm(R.detach(), viewdirs.unsqueeze(-1)).squeeze(-1)
        else:
            viewdirs = torch.bmm(R, viewdirs.unsqueeze(-1)).squeeze(-1)
        return rays_o, rays_d, viewdirs

    def get_residual_poses(self):
        a1, a2 = self.R_res.split([3, 3], dim=-1)
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        R = torch.stack((b1, b2, b3), dim=-2)
        t = self.t_res
        return R, t

    def get_refined_poses(self):
        R, t = self.get_residual_poses()
        R = torch.bmm(R, self._poses[:, :3, :3])
        t = t.unsqueeze(-1) + self._poses[:, :3, [3]]
        return torch.cat([R, t], dim=-1)

    def step(self):
        self.tick += 1
        if self.tick < self.coldstart_iter:
            return

        if self.config.reg_t_mean_scale > 0:
            t_new = self.t_ori + self.t_res
            t_new_mean = t_new.mean(0, keepdim=True)
            t_new_scale = (t_new - t_new_mean).pow(2).mean().sqrt()
            reg_loss = F.mse_loss(t_new_mean, self.t_ori_mean) + (
                t_new_scale - self.t_ori_scale
            ).pow(2)
            reg_loss.mul(self.config.reg_t_mean_scale).backward()
        if self.config.reg_t > 0:
            self.t_res.pow(2).sum().mul(self.config.reg_t).backward()
        self.optim.step()
        self.optim.zero_grad()
