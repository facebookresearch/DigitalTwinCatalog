# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .utils import scale_ls


class triplane_loss_acc:
    def __init__(
        self,
        alpha_thre=1e-5,
        early_stop_eps=1e-3,
        filter_normal=False,
        filter_normal_threshold=0.025,
    ):
        super().__init__()
        self.alpha_thre = alpha_thre
        self.early_stop_eps = early_stop_eps
        self.filter_normal = filter_normal
        self.filter_normal_threshold = filter_normal_threshold

    def forward_and_backward(
        self,
        triplane,
        renderer,
        plane_xy,
        plane_xz,
        plane_yz,
        plane_view,
        rays_o,
        rays_d,
        cams,
        gts,
        depth_masks,
        normal_masks,
        loss_weights_dict,
        loss_func_dict,
        args,
        compute_normal=False,
        compute_numerical_normal=False,
        scaler=(None, None),
        use_occ_grid=False,
        visualize_diff=False,
        aux_loss=None,
        scale_invariant_albedo=True,
    ):
        batch_size, im_num, _, height, width = next(iter(gts.values())).shape
        plane_xy_d = nn.Parameter(torch.clone(plane_xy.detach()))
        plane_xz_d = nn.Parameter(torch.clone(plane_xz.detach()))
        plane_yz_d = nn.Parameter(torch.clone(plane_yz.detach()))
        if plane_view is not None:
            plane_view_d = nn.Parameter(torch.clone(plane_view.detach()))
        else:
            plane_view_d = None
        keys = list(gts.keys())
        loss_keys = list(loss_func_dict.keys())
        preds = {}

        loss_arr = {}

        # Compute forward
        for b in range(0, batch_size):
            if use_occ_grid:
                renderer.update_occupancy_grid(
                    plane_xy_d[b : b + 1, :],
                    plane_xz_d[b : b + 1, :],
                    plane_yz_d[b : b + 1, :],
                    triplane,
                )
                if visualize_diff:
                    res = renderer.occgrid_res
                    print(
                        "Total voxel:",
                        res * res * res,
                        "Full voxel:",
                        torch.sum(renderer.estimator.binaries.float()).item(),
                    )

            for m in range(im_num):
                out = renderer(
                    plane_xy_d[b : b + 1, :],
                    plane_xz_d[b : b + 1, :],
                    plane_yz_d[b : b + 1, :],
                    plane_view_d[b : b + 1, :] if plane_view_d is not None else None,
                    triplane,
                    rays_o[b : b + 1, m : m + 1, :],
                    rays_d[b : b + 1, m : m + 1, :],
                    cams[b : b + 1, m : m + 1, :],
                    compute_normal,
                    compute_numerical_normal,
                )

                for key in out.keys():
                    if key == "gradient" or key == "cos_angle" or key == "density":
                        continue
                    if key in preds and key in keys:
                        preds[key].append(out[key].detach().clone())
                    elif key in keys:
                        preds[key] = [out[key].detach().clone()]

                    if key == "valid_mask":
                        valid_mask = out[key].clone().detach()
                        valid_mask = valid_mask.permute(0, 1, 4, 2, 3).reshape(
                            1, 1, height, width
                        )

                if "mask" in gts:
                    gt_mask = gts["mask"][b, m].clone()
                    gt_mask = gt_mask.reshape(1, 1, height, width)

                if depth_masks is not None:
                    depth_mask = depth_masks[b, m, :].reshape(1, 1, height, width)
                else:
                    depth_mask = None

                if normal_masks is not None:
                    normal_mask = normal_masks[b, m, :].reshape(1, 1, height, width)
                else:
                    normal_mask = None

                if (
                    self.filter_normal
                    and depth_mask is not None
                    and "depth" in preds
                    and "depth" in gts
                ):
                    gt_depth = gts["depth"][b, m].reshape(1, -1, height, width)
                    pred_depth = (
                        out["depth"]
                        .permute(0, 1, 4, 2, 3)
                        .reshape(1, -1, height, width)
                    )
                    filter_normal_mask = (
                        torch.abs(gt_depth - pred_depth) < self.filter_normal_threshold
                    )
                    filter_normal_mask = (filter_normal_mask * depth_mask).detach()
                else:
                    filter_normal_mask = None

                loss_sum = 0
                for key in keys:
                    if key not in preds:
                        continue
                    pred_permute = (
                        out[key].permute(0, 1, 4, 2, 3).reshape(1, -1, height, width)
                    )
                    gt_permute = gts[key][b, m].reshape(1, -1, height, width)
                    if key == "albedo" and scale_invariant_albedo:
                        pred_permute, albedo_scale = scale_ls(
                            (0.5 * (pred_permute + 1)) ** 2.2,
                            (0.5 * (gt_permute + 1)) ** 2.2,
                            pred_mask=out["mask"]
                            .permute(0, 1, 4, 2, 3)
                            .reshape(1, -1, height, width),
                            gt_mask=gt_mask,
                        )
                        pred_permute = pred_permute ** (1 / 2.2)
                        pred_permute = 2 * torch.clamp(pred_permute, 0, 1) - 1
                        preds["albedo"][b * im_num + m] = pred_permute.permute(
                            0, 2, 3, 1
                        ).reshape(1, 1, height, width, -1)

                    for loss_key in loss_keys:
                        if loss_weights_dict[loss_key][key] != 0:
                            func = loss_func_dict[loss_key]
                            if key == "depth":
                                mask = valid_mask
                                if depth_mask is not None:
                                    mask = mask * depth_mask
                                # Smaller mask to avoid path tracing averaging
                                loss = loss_weights_dict[loss_key][key] * func(
                                    gt_permute * mask + (1 - mask),
                                    pred_permute * mask + (1 - mask),
                                )
                            elif key == "numerical_normal" or key == "normal":
                                if key == "numerical_normal":
                                    mask = valid_mask * normal_mask
                                else:
                                    mask = normal_mask
                                if filter_normal_mask is not None:
                                    mask = mask * filter_normal_mask
                                loss = loss_weights_dict[loss_key][key] * func(
                                    gt_permute * mask + (1 - mask),
                                    pred_permute * mask + (1 - mask),
                                )
                            else:
                                loss = loss_weights_dict[loss_key][key] * func(
                                    gt_permute, pred_permute
                                )

                            loss = torch.mean(loss)
                            loss = torch.nan_to_num(
                                loss, nan=0.0, posinf=0.0, neginf=0.0
                            )
                            loss_sum += loss

                            if f"{key}_{loss_key}" in loss_arr:
                                loss_arr[f"{key}_{loss_key}"] += loss
                            else:
                                loss_arr[f"{key}_{loss_key}"] = loss

                if scaler[0] is not None:
                    scaler[0].scale(loss_sum).backward()
                else:
                    loss_sum.backward()

            if use_occ_grid:
                with torch.no_grad():
                    renderer.reset_occupancy_grid()

        for key in preds:
            preds[key] = torch.cat(preds[key], dim=0)
            preds[key] = preds[key].reshape(batch_size, im_num, height, width, -1)

        for _, p in triplane.named_parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / batch_size / im_num)

        grad_xy = (plane_xy_d.grad.to(torch.float32) / batch_size / im_num).reshape(-1)
        grad_xz = (plane_xz_d.grad.to(torch.float32) / batch_size / im_num).reshape(-1)
        grad_yz = (plane_yz_d.grad.to(torch.float32) / batch_size / im_num).reshape(-1)
        if plane_view_d is not None:
            grad_view = (plane_view_d.grad.to(torch.float32) / batch_size).reshape(-1)
            grad = torch.cat([grad_xy, grad_xz, grad_yz, grad_view], dim=0)
        else:
            grad = torch.cat([grad_xy, grad_xz, grad_yz], dim=0)

        if scaler[1] is not None and scaler[0] is not None:
            grad = grad / scaler[0].get_scale()
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        if plane_view is not None:
            plane = torch.cat(
                [
                    plane_xy.reshape(-1),
                    plane_xz.reshape(-1),
                    plane_yz.reshape(-1),
                    plane_view.reshape(-1),
                ],
                dim=0,
            )
        else:
            plane = torch.cat(
                [
                    plane_xy.reshape(-1),
                    plane_xz.reshape(-1),
                    plane_yz.reshape(-1),
                ],
                dim=0,
            )
        plane = plane.to(torch.float32)

        if scaler[1] is not None:
            grad = scaler[1].scale(grad)

        if aux_loss is None:
            plane.backward(grad)
        else:
            loss_plane = nn.functional.mse_loss(
                plane, plane.detach() - grad, reduction="sum"
            )
            loss = aux_loss + loss_plane
            loss.backward()

        for key in loss_arr.keys():
            if torch.is_tensor(loss_arr[key]):
                loss_arr[key] = loss_arr[key].item() / batch_size / im_num

        return preds, loss_arr

    def forward(
        self,
        triplane,
        renderer,
        plane_xy,
        plane_xz,
        plane_yz,
        plane_view,
        rays_o,
        rays_d,
        cams,
        args,
        compute_normal=False,
        compute_numerical_normal=False,
        use_occ_grid=False,
        visualize_diff=False,
    ):
        batch_size, im_num, height, width, _ = rays_o.shape
        preds = {}

        # Compute forward
        with torch.no_grad():
            for b in range(0, batch_size):
                if use_occ_grid:
                    renderer.update_occupancy_grid(
                        plane_xy[b : b + 1, :],
                        plane_xz[b : b + 1, :],
                        plane_yz[b : b + 1, :],
                        triplane,
                    )
                    if visualize_diff:
                        res = renderer.occgrid_res
                        print(
                            "Total voxel:",
                            res * res * res,
                            "Full voxel:",
                            torch.sum(renderer.estimator.binaries.float()).item(),
                        )

                preds_per_batch = {}
                for m in range(im_num):
                    out = renderer(
                        plane_xy[b : b + 1, :],
                        plane_xz[b : b + 1, :],
                        plane_yz[b : b + 1, :],
                        plane_view[b : b + 1, :] if plane_view is not None else None,
                        triplane,
                        rays_o[b : b + 1, m : m + 1, :],
                        rays_d[b : b + 1, m : m + 1, :],
                        cams[b : b + 1, m : m + 1, :],
                        compute_normal,
                        compute_numerical_normal,
                    )
                    for key in out.keys():
                        if key == "gradient" or key == "cos_angle" or key == "density":
                            continue
                        if key in preds_per_batch:
                            preds_per_batch[key].append(out[key])
                        else:
                            preds_per_batch[key] = [out[key]]

                for key in preds_per_batch.keys():
                    if key not in preds:
                        preds[key] = [
                            torch.cat(preds_per_batch[key], dim=0).reshape(
                                1, im_num, height, width, -1
                            )
                        ]
                    else:
                        preds[key].append(
                            torch.cat(preds_per_batch[key], dim=0).reshape(
                                1, im_num, height, width, -1
                            )
                        )

            if use_occ_grid:
                with torch.no_grad():
                    renderer.reset_occupancy_grid()

        for key in preds:
            preds[key] = torch.cat(preds[key], dim=0)

        return preds
