# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
import time
from multiprocessing import Pool

import imageio
import imageio.v3 as iio
import numpy as np

import scipy
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm, trange


def load_data(cfg_data):
    images, masks, poses, render_poses, K, impaths, aabb, split = load_dataset(cfg_data)
    i_train, i_test, i_val = split
    assert images.shape[-1] == 3
    print("Loaded dataset:", images.shape, cfg_data.datadir)
    print("# train:", len(i_train))
    print("# test:", len(i_test), i_test)
    near, far = inward_nearfar_heuristic(poses[:, :3, 3], ratio=0)
    near_clip = near

    # Post-processing intrinsics params.
    # Basically, all frames should share the same intrinsics params.
    # Current code duplicate the intrinsics params for each frame for historical reason. :P
    HW = np.array([im.shape[:2] for im in images])
    H = HW[0, 0]
    W = HW[0, 1]

    if K is None:
        focal = hwf[-1]
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    # Remove views that are too far
    if cfg_data.filter_outlier_views is not None:
        print("Removing training views that are too far.")
        cam_o = poses[i_train, :3, 3]
        cam_o_C = np.median(cam_o, 0)
        cam_o_dist = np.sqrt(((cam_o - cam_o_C) ** 2).sum(-1))
        cam_mask = (
            cam_o_dist
            < cam_o_dist.mean() + cam_o_dist.std() * cfg_data.filter_outlier_views
        )
        i_train = i_train[cam_mask]
        print(
            f"Removed {len(cam_mask)-cam_mask.sum()} out of {len(cam_mask)} trainint views."
        )

    # Bundle all datas
    render_poses = render_poses[..., :4]
    data_dict = dict(
        HW=HW,
        Ks=Ks,
        near=near,
        far=far,
        near_clip=near_clip,
        i_train=i_train,
        i_val=i_val,
        i_test=i_test,
        poses=poses,
        render_poses=render_poses,
        images=images,
        depths=None,
        depthconfs=None,
        masks=masks,
        impaths=impaths,
        aabb=aabb,
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05, far_ratio=1.0):
    dist = np.linalg.norm(cam_o[:, None] - cam_o, axis=-1)
    far = dist.max() * far_ratio
    near = far * ratio
    return near, far


def run_mp(func, inplst, n_thread):
    with Pool(n_thread) as p:
        ret = list(tqdm(p.imap(func, inplst), total=len(inplst)))
    return ret


def _load_a_data(args):
    frame_meta, datadir = args
    pose = np.array(frame_meta["to_world"], dtype=np.float32)
    pose[:, :2] *= -1
    im_path = os.path.join(datadir, frame_meta["path"])
    im = imageio.imread(im_path)
    if im_path.endswith("exr"):
        pass
    else:
        im = (im / 255.0).astype(np.float32)
    if frame_meta.get("mask_from_alpha", False):
        mask = im[..., -1]
        im = im[..., :3]
    elif "mask" in frame_meta:
        mask = imageio.imread(os.path.join(datadir, frame_meta["mask"]))
        mask = (mask / 255.0).astype(np.float32)
        if len(mask.shape) == 3:
            mask = mask.mean(-1)
    else:
        mask = None
    return pose, im, im_path, mask


def load_dataset(cfg_data):
    ts = time.time()
    print("Start loading dataset...")

    datadir = cfg_data.datadir
    assert os.path.exists(datadir), f"Folder not found: {datadir}"

    # load cameras.json
    with open(os.path.join(datadir, cfg_data.config_name)) as f:
        cameras = json.load(f)

    K = np.array(
        [
            [cameras["fx"], 0, cameras["cx"]],
            [0, cameras["fy"], cameras["cy"]],
            [0, 0, 1],
        ]
    )

    frames = cameras["frames"]

    ret = run_mp(
        _load_a_data, [(frames[i], datadir) for i in range(len(frames))], n_thread=32
    )
    poses = [v[0] for v in ret]
    images = [v[1] for v in ret]
    all_paths = [v[2] for v in ret]
    masks = [v[3] for v in ret]

    poses = np.stack(poses, 0)
    images = np.stack(images, 0)
    if masks[0] is not None:
        masks = np.stack(masks, 0)
        fg_alpha = masks[..., None]
        if cfg_data.update_bg_bkgd:
            images = images * fg_alpha + cfg_data.bkgd * (1 - fg_alpha)
    else:
        masks = None

    if cfg_data.linear2srgb:
        for i in range(len(images)):
            if all_paths[i].endswith("exr"):
                images[i] = np.clip(images[i] ** (1 / 2.2), 0, 1)

    aabb = np.array(cameras.get("aabb", [[-1, -1, -1], [1, 1, 1]]))

    # Virtual camera trajectory
    if cfg_data.movie_mode == "circular":
        render_poses = generate_virtual_camera_trajectory(
            poses, aabb, **cfg_data.movie_render_kwargs
        )
    elif cfg_data.movie_mode == "interpolate":
        render_poses = interpolate_poses(poses, **cfg_data.movie_render_kwargs)
    else:
        raise NotImplementedError

    # Load data split
    if "split" in cameras:
        i_train = np.array(cameras["split"]["train"])
        i_test = np.array(cameras["split"]["test"])
        i_val = np.copy(i_test)
    else:
        i_train = np.arange(len(images))
        i_val, i_test = np.empty([2, 0], dtype=int)
    split = [i_train, i_test, i_val]

    print(f"Finish loading dataset in {time.time()-ts:.1f} sec.")
    return images, masks, poses, render_poses, K, all_paths, aabb, split


def normalize(x):
    return x / np.linalg.norm(x)


def generate_virtual_camera_trajectory(
    poses, aabb=None, scale_r=1, shift_x=0, shift_y=0, shift_z=0, flip_up=False
):
    ### generate spiral poses for rendering fly-through movie
    if aabb is not None:
        corner_a = np.copy(aabb[0])
        corner_b = np.copy(aabb[1])
        centroid = (corner_a + corner_b) * 0.5
    else:
        centroid = np.zeros([3])
    radcircle = scale_r * np.linalg.norm(poses[:, :3, 3] - centroid, axis=-1).mean()
    centroid[0] += shift_x
    centroid[1] += shift_y
    centroid[2] += shift_z
    target_y = -shift_y

    if flip_up:
        up = np.array([0, 1.0, 0])
    else:
        up = np.array([0, -1.0, 0])

    render_poses = []
    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), 0, radcircle * np.sin(th)])
        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin + centroid
        # rotate to align with new pitch rotation
        lookat = -np.copy(camorigin)
        lookat[1] = target_y
        lookat = normalize(lookat)
        vec2 = -lookat
        vec1 = normalize(np.cross(vec2, vec0))

        p = np.stack([vec0, vec1, vec2, pos], 1)
        p[:, [1, 2]] *= -1

        render_poses.append(p)

    render_poses = np.stack(render_poses, 0)
    render_poses = np.concatenate(
        [
            render_poses,
            np.broadcast_to(poses[0, :3, -1:], render_poses[:, :3, -1:].shape),
        ],
        -1,
    )

    return render_poses


def interpolate_poses(poses, N_views=200):
    slerp = Slerp(np.linspace(0, 1, len(poses)), Rotation.from_matrix(poses[:, :3, :3]))
    tlerp = scipy.interpolate.interp1d(
        np.linspace(0, 1, len(poses)), poses[:, :3, 3].T, kind="cubic"
    )

    ratio = 0.5 - np.cos(np.linspace(0, 2 * np.pi, N_views)) * 0.5
    render_poses = np.eye(4)[None].repeat(N_views, axis=0)
    render_poses[:, :3, :3] = slerp(ratio).as_matrix()
    render_poses[:, :3, 3] = tlerp(ratio).T
    return render_poses
