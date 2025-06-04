# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

results_dir = None

""" Template of data options
"""
data = dict(
    datadir=None,  # path to dataset root folder
    dataset_name=None,  # just to name to result folder
    config_name="cameras.json",  # the filename of the json config
    load2gpu_on_the_fly=False,  # do not load all images into gpu (to save gpu memory)
    bkgd=0,  # use white background (note that some dataset don't provide alpha and with blended bg color)
    update_bg_bkgd=True,  # if fg mask is provided, directly set bg pixels to bkgd
    rand_bkgd=False,  # use random background during training
    movie_mode="circular",  # circular | interpolate
    movie_render_kwargs=dict(),
    filter_outlier_views=None,  # check dataloader.py
    linear2srgb=True,  # convert linear to srgb if input is exr
    ndc=False,  # not supported yet
    # Foreground BBOX rule
    fg_bbox_rule="json",  # json | cam_frustrum | cam_centroid_cuboid
)

""" Template of training options
"""
coarse_train = dict(
    N_iters=0,  # number of optimization steps
    N_rand=8192,  # batch size (number of random rays per optimization step)
    lrate_density=1e-1,  # lr of density voxel grid
    lrate_k0=1e-1,  # lr of color/feature voxel grid
    lrate_sdf_scale=0,
    lrate_rgbnet=1e-3,  # lr of the mlp to preduct view-dependent color
    lrate_decay_factor=0.1,
    lrate_geo_decay_factor=0.1,
    lrate_warmup_iter=100,
    bg_lrate_density=1e-1,  # lr of bg density voxel grid
    bg_lrate_k0=1e-1,  # lr of bg color/feature voxel grid
    bg_lrate_rgbnet=1e-3,
    optim_betas=(0.9, 0.99),
    optim_eps=1e-15,
    ray_sampler="flatten",  # ray sampling strategies. [in_maskcache | hitbbox | flatten | in_fgmask | in_bbox]
    ray_sampler_fg_rate=None,  # ensure the minimum percentage of rays hitting the fg bbox
    main_loss="mse",
    huber_min_c=1 / 255,
    weight_main=1.0,  # photometric loss
    weight_rgbper=0.01,  # per-point rgb loss (encourage color concentration)
    weight_entropy_last=0.01,  # fore/background entropy loss (encourage fg/bg concentration)
    weight_mask=0,  # supervision from external foreground mask
    weight_laplace=0,  # geometric grid laplace regularization
    laplace_dense=True,  # compute laplacian loss densely or only on grid points with gradient!=0
    pg_scale=[],  # checkpoints for progressive scaling
    pg_scale_s=2,
    skip_zero_grad_fields=[],  # the variable name to skip optimizing parameters w/ zero grad in each iteration
    use_coarse_mask=False,
)

fine_train = deepcopy(coarse_train)
fine_train.update(
    dict(
        N_iters=10000,
        N_rand=4096,
        lrate_density=0.003,
        lrate_k0=0.1,
        lrate_geo_decay_factor=0.01,
        main_loss="huber",
        weight_entropy_last=1e-4,
        weight_laplace=1e-8,
        pg_scale=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
        skip_zero_grad_fields=["density", "k0"],
    )
)

""" Template of model and rendering options
"""
camera_refinement_model = dict(
    trainable=False,
    detach_viewdirs=True,
    coldstart_iter=0,
    lrate_t_res=1e-5,
    lrate_R_res=1e-5,
    reg_t_mean_scale=0.1,
    reg_t=0,
)

coarse_model_and_render = dict(
    num_voxels=100**3,  # total number of voxel
    num_voxels_k0_max=300**3,
    num_voxels_bg_max=160**3,
    # explicit component
    density_type="DenseGrid",  # DenseGrid | TensoRFGrid | HashGrid (not tested yet)
    k0_type="DenseGrid",
    density_config=dict(),
    k0_config=dict(),
    bbox_alpha_thres=0.01,  # threshold to find a tighten BBox (used to define iso for marching cube)
    bbox_largest_cc_only=True,
    mask_cache_init_thres=1e-3,  # threshold for initial known free-space
    mask_cache_upd_thres=1e-5,  # threshold for updating known free-space
    sdf_mode=False,
    detach_sharpness=True,
    sdf_scale_init=20,
    sdf_scale_step=1 / 20,
    sdf_scale_max=1000,
    sdf_anneal_step=2000,
    constant_init=False,
    constant_init_val=0,
    sphere_init=True,
    sphere_init_scale=0.25,
    sphere_init_shift=-0.125,
    # implicit component
    rgbnet_dim=0,  # feature voxel grid dim
    rgbnet_depth=3,  # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=128,  # width of the colors MLP
    rgbnet_tcnn=False,  # use tiny-cuda-nn MLP
    posbase_pe=-1,
    viewbase_pe=4,
    # background component
    num_bg_scale=0,
    bg_scale=None,
    # more tricks
    alpha_init=1e-5,  # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-6,  # threshold of alpha value to skip the fine stage sampled point
    world_bound_scale=1,  # rescale the BBox enclosing the scene
    stepsize=0.5,  # sampling stepsize (ratio to voxel length) in volume rendering
    on_known_board=False,
)

fine_model_and_render = deepcopy(coarse_model_and_render)
fine_model_and_render.update(
    dict(
        num_voxels=300**3,
        sdf_mode=True,
        rgbnet_dim=12,
        alpha_init=1e-2,
        mask_cache_upd_thres=0,
        fast_color_thres=1e-4,
    )
)

del deepcopy
