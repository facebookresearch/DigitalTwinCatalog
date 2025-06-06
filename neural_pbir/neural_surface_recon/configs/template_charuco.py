# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

_base_ = "./default.py"

data = dict(
    dataset_name="charuco",  # just to name to result folder
    rand_bkgd=True,
    fg_bbox_rule="json",
    movie_mode="circular",
    movie_render_kwargs={"shift_y": 0.1, "scale_r": 1},
)

fine_train = dict(
    N_iters=15000,
    N_rand=8192,
    ray_sampler_fg_rate=0.9,
    weight_rgbper=1e-2,
    weight_entropy_last=1e-4,
    weight_laplace=1e-8,
    pg_scale=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
)

fine_model_and_render = dict(
    num_voxels=300**3,
    on_known_board=True,
    # turn on background model
    num_bg_scale=1,
    bg_scale=16.0,
)
