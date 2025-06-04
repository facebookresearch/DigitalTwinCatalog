# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def scale_ls(pred, gt, pred_mask, gt_mask, scale_min=0.1, scale_max=5):
    # Can only handle one back of the data
    with torch.no_grad():
        mean_gt = torch.sum(gt * gt_mask) / torch.clamp(
            torch.sum(gt_mask) * 3, min=1e-6
        )
        mean_pred = torch.sum(pred * pred_mask) / torch.clamp(
            torch.sum(pred_mask) * 3, min=1e-6
        )

        scale = mean_gt / torch.clamp(mean_pred, min=1e-6)
        scale = torch.clamp(scale, min=scale_min, max=scale_max)
    pred = scale * pred
    pred = pred * pred_mask + (1 - pred_mask)
    return pred, scale
