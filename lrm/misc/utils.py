# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import copy
import datetime
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import random

import shutil
import sys
import tempfile
import time
from collections import defaultdict, deque, OrderedDict

import cv2

import matplotlib.cm as cm
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torchvision
import trimesh
from skimage import measure

# from ..data_loader.utils import compute_rays

from .dist_helper import get_rank, is_dist_avail_and_initialized

print_debug_info = False


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class RepeatedDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


def linear_to_srgb(l):
    # s = np.zeros_like(l)
    s = torch.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055 * (l[~m] ** (1.0 / 2.4)) - 0.055
    return s


def _suppress_print(gpu=None):
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    if (gpu is not None and gpu != 0) or (gpu is None and not is_main_process()):
        builtins.print = print_pass


def pytorch_mlp_clip_gradients(model, clip):
    grad_norm = []
    for p in model.parameters():
        if p is not None and p.grad is not None:
            grad_norm.append(p.grad.view(-1))

    if len(grad_norm) > 0:
        grad_norm = torch.concat(grad_norm).norm(2).item()
        clip_coef = clip / (grad_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        return grad_norm
    return None


def clip_gradients(model, clip, check_nan_inf=True, file_name=None):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            if check_nan_inf:
                p.grad.data = torch.nan_to_num(
                    p.grad.data, nan=0.0, posinf=0.0, neginf=0.0
                )
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def filter_weights_with_wrong_size(model, weights):
    new_weights = OrderedDict()
    missing_keys = []
    state_dict = model.state_dict()
    for name, value in weights.items():
        if name in state_dict:
            target_value = state_dict[name]
            if value.size() != target_value.size():
                missing_keys.append(name)
            else:
                new_weights[name] = value
        elif "module." + name in state_dict:
            target_value = state_dict["module." + name]
            if value.size() != target_value.size():
                missing_keys.append(name)
            else:
                new_weights["module." + name] = value
        else:
            new_weights[name] = value

    return new_weights, missing_keys


def load_ddp_state_dict(model, weights, key=None, filter_mismatch=True):
    weights = copy.deepcopy(weights)
    if key is not None and (key == "tridecoder" or key == "triupsampler"):
        all_keys = [k for k in weights.keys() if "dconv2d." in k]
        for k in all_keys:
            new_k = k.replace("dconv2d", "dconv2d_geo")
            weights[new_k] = weights[k].clone()
            new_k = k.replace("dconv2d", "dconv2d_app")
            weights[new_k] = weights[k].clone()

    if key is not None and key == "triupsampler":
        all_keys = [k for k in weights.keys() if "upsampler.dconv2d." in k]
        for k in all_keys:
            new_k = k.replace("upsampler.dconv2d", "dconv2d")
            weights[new_k] = weights[k].clone()

    if isinstance(model, nn.parallel.DistributedDataParallel):
        if filter_mismatch:
            weights, missing_keys = filter_weights_with_wrong_size(model, weights)
            if len(missing_keys) > 0:
                print(
                    "Keys ",
                    missing_keys,
                    " are filtered out due to parameter size mismatch.",
                )
        msg = model.load_state_dict(weights, strict=False)
    elif isinstance(model, torch.optim.Optimizer):
        msg = model.load_state_dict(weights)
    else:
        new_weights = OrderedDict()
        for k, v in weights.items():
            if k[:7] == "module.":
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            else:
                name = k
            new_weights[name] = v
        if filter_mismatch:
            new_weights, missing_keys = filter_weights_with_wrong_size(
                model, new_weights
            )
            if len(missing_keys) > 0:
                print(
                    "Keys ",
                    missing_keys,
                    " are filtered out due to parameter size mismatch.",
                )
        msg = model.load_state_dict(new_weights, strict=False)
    return msg


def restart_from_checkpoint(
    ckp_path, run_variables=None, load_weights_only=False, **kwargs
):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(ckp_path)
    print("Found checkpoint at {}".format(ckp_path))

    with open(ckp_path, "rb") as fb:
        checkpoint = torch.load(fb, map_location="cpu", weights_only=False)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = load_ddp_state_dict(value, checkpoint[key], key=key)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                print(
                    "=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path)
                )
        elif key == "triupsampler":
            try:
                msg = load_ddp_state_dict(
                    value, checkpoint["tridecoder"], key="triupsampler"
                )
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                print(
                    "=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path)
                )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if not load_weights_only and run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_iters,
    start_warmup_value=1e-10,
):
    print(
        f"cosin scheduler - lr: {base_value}, min_lr: {final_value}, epochs: {epochs}, it_per_epoch: {niter_per_ep}, warmup_iters: {warmup_iters}, startup_warmup: {start_warmup_value}"
    )
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        if warmup_iters > epochs * niter_per_ep:
            raise RuntimeError(
                f"warm iterations number is exceeding the total number iterations. Epoch: {epochs}: Iteration/Epoch: {niter_per_ep}"
            )

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert (
        len(schedule) == epochs * niter_per_ep
    ), f"Schedule length {len(schedule)} needs to match epoch {epochs} x iteration per epoch {niter_per_ep}"
    return schedule


def linear_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_iters,
    start_warmup_value=1e-10,
):
    print(
        f"cosin scheduler - lr: {base_value}, min_lr: {final_value}, epochs: {epochs}, it_per_epoch: {niter_per_ep}, warmup_iters: {warmup_iters}, startup_warmup: {start_warmup_value}"
    )
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        if warmup_iters > epochs * niter_per_ep:
            raise RuntimeError(
                f"warm iterations number is exceeding the total number iterations. Epoch: {epochs}: Iteration/Epoch: {niter_per_ep}"
            )

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.linspace(base_value, final_value, len(iters))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert (
        len(schedule) == epochs * niter_per_ep
    ), f"Schedule length {len(schedule)} needs to match epoch {epochs} x iteration per epoch {niter_per_ep}"
    return schedule


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=1000, fmt=None):
        if fmt is None:
            fmt = "{median:.8f} ({global_avg:.8f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None, tb_writer=None, epoch=0):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger
        self.tb_writer = tb_writer
        self.epoch = epoch
        assert self.logger is not None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if self.tb_writer:
                    for k in self.meters:
                        global_step = self.epoch * len(iterable) + i
                        self.tb_writer.add_scalar(
                            f"train/{k}",
                            self.meters[k].global_avg,
                            global_step=global_step,
                            new_style=True,
                        )
                if torch.cuda.is_available():
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
                sys.stdout.flush()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            ),
        )


def is_main_process():
    return get_rank() == 0


def save_on_master(ckpt, model_path, backup_ckp_epoch=-1, topk=-1):
    if not is_main_process():
        return
    basedir = os.path.dirname(model_path)
    if not os.path.isdir(basedir):
        os.makedirs(basedir)
    with open(model_path, "wb") as fp:
        torch.save(ckpt, fp)

    if backup_ckp_epoch >= 0:
        target_path = os.path.join(basedir, f"ckpt_{backup_ckp_epoch}.pth")
        shutil.copy2(model_path, target_path)
    return


def save_image(image, name, is_gamma=False):
    if len(image.shape) == 5:
        batch_size, im_num, _, h, w = image.shape
        image = image.reshape((batch_size * im_num, -1, h, w))
        nrow = im_num
    else:
        batch_size = image.shape[0]
        nrow = batch_size

    if "mask" not in name.split("/")[-1] and "env" not in name.split("/")[-1]:
        image = 0.5 * (image + 1)

    if is_gamma:
        image = linear_to_srgb(image)

    with open(name, "wb") as fp:
        torchvision.utils.save_image(image, fp, nrow=nrow)


def save_image_list(image_list, name):
    im_num = len(image_list)
    batch_size = len(image_list[0])
    with open(name, "w") as fp:
        for n in range(0, batch_size):
            for m in range(0, im_num):
                fp.write(image_list[m][n] + "\n")


def save_single_png(images, name, image_ids=None, is_gamma=False):
    if "mask" not in name:
        images = 0.5 * (images + 1)
    if is_gamma:
        images = linear_to_srgb(images)
    images = images.detach().cpu().numpy()
    images = (255 * np.clip(images, 0, 1)).astype(np.uint8)

    batch_size = images.shape[0]
    for n in range(0, batch_size):
        im = images[n].transpose(1, 2, 0)
        if im.shape[-1] == 3:
            im = np.ascontiguousarray(im[:, :, ::-1])
        else:
            im = np.concatenate([im, im, im], axis=-1)

        buffer = cv2.imencode(".png", im)[1]
        buffer = np.array(buffer).tobytes()
        if image_ids is None:
            im_name = name % n
        else:
            im_name = name % image_ids[n]
        with open(im_name, "wb") as fp:
            fp.write(buffer)


def save_single_exr(images, name, image_ids=None, is_gamma=False):
    images = images.detach().cpu().float().numpy()
    images = images.astype(np.float32)

    batch_size = images.shape[0]
    for n in range(0, batch_size):
        im = images[n].transpose(1, 2, 0)
        if im.shape[-1] == 3:
            im = np.ascontiguousarray(im[:, :, ::-1])
        else:
            im = np.concatenate([im, im, im], axis=-1)
        buffer = cv2.imencode(".exr", im)[1]
        buffer = np.array(buffer).tobytes()
        if image_ids is None:
            im_name = name % n
        else:
            im_name = name % image_ids[n]
        with open(im_name, "wb") as fp:
            fp.write(buffer)


def save_depth(depth, depth_mask, name, depth_min=1.5, depth_max=2.5):
    batch_size, im_num, _, h, w = depth.shape
    depth = depth * depth_mask
    depth = np.clip(depth, depth_min, depth_max)

    cmap = cm.get_cmap("jet")
    depth = (depth.reshape(-1) - depth_min) / (depth_max - depth_min)
    depth = depth.detach().cpu().numpy()
    colors = cmap(depth.flatten())[:, :3]
    colors = colors.reshape(batch_size * im_num, h, w, 3)
    colors = colors.transpose(0, 3, 1, 2)
    colors = torch.from_numpy(colors)
    with open(name, "wb") as fp:
        torchvision.utils.save_image(colors, fp, nrow=im_num)


def save_mesh(path, values, N, threshold, radius, save_volume=False):
    values = values.detach().cpu().numpy()
    values = values.reshape(N, N, N).astype(np.float32)

    if save_volume:
        with open(path + ".npy", "wb") as fp:
            np.save(fp, values)

    radius = radius * 1.05
    try:
        vertices, triangles, normals, _ = measure.marching_cubes(values, threshold)
        print(
            "vertices num %d triangles num %d threshold %.3f"
            % (vertices.shape[0], triangles.shape[0], threshold)
        )

        vertices = vertices / (N - 1.0) * 2 * radius - radius
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=triangles, vertex_normals=normals
        )
        mesh.export(path)
    except:
        print("Failed to extract mesh.")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_params_group_single_model(args, model, freeze_transformer=False):
    upsampler = []
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if param is None:
            continue
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if "upsampler" in name:
            upsampler.append(param)
        else:
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            elif len(name.split(".")) > 3 and "norm" in name.split(".")[2]:
                not_regularized.append(param)
            else:
                regularized.append(param)

    if freeze_transformer:
        lr_t = 0
        lr_c = args.lr
    else:
        lr_t = args.lr
        lr_c = args.lr

    return [
        {"params": regularized, "weight_decay": args.weight_decay, "lr": lr_t},
        {"params": not_regularized, "weight_decay": 0.0, "lr": lr_t},
        {"params": upsampler, "weight_decay": args.weight_decay, "lr": lr_c},
    ]


def get_params_groups(args, **kwargs):
    params_groups = []
    for key, value in kwargs.items():
        if "mvencoder" in key or "tridecoder" in key:
            params_groups += get_params_group_single_model(
                args,
                value,
                freeze_transformer=args.freeze_transformer,
            )
        else:
            params_groups += get_params_group_single_model(args, value)
    return params_groups


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for module in model.modules():
        if isinstance(module, bn_types):
            return True
    return False


def compute_rays(fov, matrix, res):
    if isinstance(res, int):
        res = (res, res)

    matrix = np.array(matrix)
    rays_o = np.zeros((res[0], res[1], 3), dtype=np.float32) + matrix[0:3, 3].reshape(
        1, 1, 3
    )

    x_axis = np.linspace(0.5, res[1] - 0.5, res[1]) / res[1]
    x_axis = 2 * x_axis - 1
    y_axis = np.linspace(0.5, res[0] - 0.5, res[0]) / res[0]
    y_axis = 2 * y_axis - 1

    x, y = np.meshgrid(x_axis, y_axis)
    x = x * np.tan(fov / 2.0)
    y = -y * np.tan(fov / 2.0) / res[1] * res[0]
    uv = np.stack([x, y], axis=-1)

    z = -np.ones((res[0], res[1]))
    rays_d_un = np.stack([x, y, z], axis=-1)
    rays_d = rays_d_un / np.linalg.norm(rays_d_un, axis=-1)[:, :, None]
    rot = matrix[0:3, 0:3][None, None, :, :]
    rays_d_un = np.sum(rot * rays_d_un[:, :, None, :], axis=-1)
    rays_d = np.sum(rot * rays_d[:, :, None, :], axis=-1)

    return rays_o, rays_d, rays_d_un, uv


def create_video_cameras(radius, frame_num, res, elevation=20, fov=60, init_cam=None):
    fov = fov / 180.0 * np.pi
    dist = radius / np.sin(fov / 2.0) * 1.2
    theta = elevation / 180.0 * np.pi
    x_axis = np.array([1.0, 0, 0], dtype=np.float32)
    y_axis = np.array([0, 1.0, 0], dtype=np.float32)
    z_axis = np.array([0, 0, 1.0], dtype=np.float32)

    if init_cam is not None:
        init_cam = init_cam.detach().cpu().numpy()
        init_cam = init_cam.transpose(1, 0)
        inv = np.eye(4, dtype=np.float32)
        inv[0:3, 0:3] = init_cam

    camera_arr, rays_o_arr, rays_d_arr, rays_d_un_arr = [], [], [], []
    for n in range(0, frame_num):
        phi = float(n) / frame_num * np.pi * 2
        origin = (
            np.cos(theta) * np.cos(phi) * x_axis
            + np.cos(theta) * np.sin(phi) * y_axis
            + np.sin(theta) * z_axis
        )
        origin = origin * dist

        target = np.array([0, 0, 0], dtype=np.float32)
        up = np.array([0, 0, 1], dtype=np.float32)
        cam_z_axis = (origin - target) / np.linalg.norm(origin - target)
        cam_y_axis = up - np.sum(cam_z_axis * up) * cam_z_axis
        cam_y_axis = cam_y_axis / np.linalg.norm(cam_y_axis)
        cam_x_axis = np.cross(cam_y_axis, cam_z_axis)

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, 0] = cam_x_axis
        extrinsic[:3, 1] = cam_y_axis
        extrinsic[:3, 2] = cam_z_axis
        extrinsic[:3, 3] = origin
        if init_cam is not None:
            extrinsic = np.matmul(inv, extrinsic)

        camera = np.zeros(20, dtype=np.float32)
        camera[0:16] = extrinsic.reshape(-1)
        camera[16:20] = np.array([fov, fov, 0.5, 0.5])

        rays_o, rays_d, rays_d_un, _ = compute_rays(fov, extrinsic, res)

        camera_arr.append(camera)
        rays_o_arr.append(rays_o)
        rays_d_arr.append(rays_d)
        rays_d_un_arr.append(rays_d_un)

    camera_arr = np.stack(camera_arr, axis=0)[None, :, :].astype(np.float32)
    rays_o_arr = np.stack(rays_o_arr, axis=0)[None, :, :, :].astype(np.float32)
    rays_d_arr = np.stack(rays_d_arr, axis=0)[None, :, :, :].astype(np.float32)
    rays_d_un_arr = np.stack(rays_d_un_arr, axis=0)[None, :, :, :].astype(np.float32)

    return camera_arr, rays_o_arr, rays_d_arr, rays_d_un_arr
