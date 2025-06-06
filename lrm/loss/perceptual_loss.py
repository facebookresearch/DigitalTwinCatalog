# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import lpips
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, loss_type="zhang"):
        """
        loss_type:
            "zhang": Richard Zhang's networkd learned on BAPPS: https://arxiv.org/abs/1801.03924
        """
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "zhang":
            self.net = lpips.LPIPS(net="vgg")
        else:
            raise ValueError("Unknown perceptual loss type {}".format(loss_type))

        for param in self.net.parameters():
            param.requires_grad = False

    def _resize_mask(self, mask, target):
        _, _, H, W = target.shape
        return F.interpolate(mask, [H, W], mode="area")

    def forward(self, image_1, image_2, mask=None):
        """
        image_1: Float[Tensor, "B C H W"], value range already normalized to [0, 1]
        image_2: Float[Tensor, "B C H W"], value range already normalized to [0, 1]
        mask: Float[Tensor, "B 1 H W"], specifies the regional loss weights.
        """

        if self.loss_type == "zhang":
            if mask is None:
                self.net.spatial = False
                loss = self.net(image_1, image_2).mean()
            else:
                self.net.spatial = True
                loss = self.net(image_1, image_2) * mask  # [B, C, H, W]
                loss = loss.mean()
        else:
            raise NotImplementedError(self.loss_type)

        return loss
