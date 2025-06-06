# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

_base_ = "./default.py"

data = dict(
    dataset_name="mii",  # just to name to result folder
    rand_bkgd=True,
    update_bg_bkgd=False,  # dont maskout bg pixels w/ bkgd color
    fg_bbox_rule="json",
    movie_mode="circular",
)

fine_train = dict(
    N_iters=25000,
    N_rand=8192,
    ray_sampler="hitbbox",
    weight_rgbper=1e-3,
    weight_entropy_last=0,
    weight_mask=1e-3,
    weight_laplace=1e-8,
    pg_scale=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
)

fine_model_and_render = dict(
    num_voxels=300**3,
    # turn on background model
    num_bg_scale=1,
    bg_scale=16.0,
)
