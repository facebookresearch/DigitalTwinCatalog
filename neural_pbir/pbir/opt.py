# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from pathlib import Path
from time import time

import gin
import torch
from irtk.io import write_image, write_mesh
from irtk.loss import l1_loss
from tqdm import tqdm


@gin.configurable
def optimize(
    scene,
    dataset,
    result_path,
    model_class,
    num_epochs,
    max_iter,
    checkpoint_iter,
    render_opt,
    render_vis,
):
    # Make a folder at the result_path
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    # Handle sensors
    num_sensors = len(dataset)
    opt_sensor_ids = torch.arange(num_sensors, dtype=torch.int64)
    vis_sensor_ids = torch.randperm(num_sensors)[0:4]

    # Create the model, which solve one or multiple inverse problems
    model = model_class(scene)

    # Optimization related
    max_iter = min(max_iter, len(opt_sensor_ids) * num_epochs)
    loss_record = []
    iter = 0

    # End configuration and begin optimization
    print("Starting the optimization...")
    pbar = tqdm(total=max_iter)

    for epoch in range(1, num_epochs + 1):
        # Randomly shuffle sensors
        sensor_perm = opt_sensor_ids[torch.randperm(num_sensors)]

        for sensor_id in sensor_perm:
            tar_image = dataset[sensor_id]

            model.zero_grad()
            model.set_data()
            scene.configure()

            t0 = time()

            opt_image = render_opt(scene, sensor_ids=[sensor_id], integrator_id=0)

            t1 = time()
            render_time = t1 - t0

            t0 = time()

            image_loss = l1_loss(tar_image, opt_image)
            reg_loss = model.get_regularization()
            loss = image_loss + reg_loss

            loss.backward()
            model.step()

            t1 = time()
            opt_time = t1 - t0

            loss_record.append(loss.item())

            iter += 1

            pbar.update(1)

            if iter == 1 or iter % checkpoint_iter == 0:
                iter_path = result_path / str(iter)
                iter_path.mkdir(parents=True, exist_ok=True)

                model.write_results(iter_path)

                tar_images_cat = torch.cat(
                    [dataset[id] for id in vis_sensor_ids], dim=1
                )

                with torch.no_grad():
                    vis_images = render_vis(scene, sensor_ids=vis_sensor_ids)
                    vis_images_cat = torch.cat(
                        [vis_image for vis_image in vis_images], dim=1
                    )

                final_image = torch.cat([tar_images_cat, vis_images_cat], dim=0)
                write_image(iter_path / "vis.exr", final_image)

                torch.save(loss_record, result_path / "loss.pt")

            if iter == max_iter:
                pbar.close()

                print("Writing final outputs...")
                final_path = result_path / "final"
                final_path.mkdir(parents=True, exist_ok=True)

                d = scene["mat.d"]
                s = scene["mat.s"]
                r = scene["mat.r"]
                write_image(final_path / "diffuse.exr", d)
                write_image(final_path / "diffuse.png", d)

                write_image(final_path / "specular.exr", s)
                write_image(final_path / "specular.png", s)

                write_image(final_path / "roughness.exr", r)
                write_image(final_path / "roughness.png", r, is_srgb=False)

                envmap = scene["envmap.radiance"]
                write_image(final_path / "envmap.exr", envmap)

                v = scene["mesh.v"]
                f = scene["mesh.f"]
                uv = scene["mesh.uv"]
                fuv = scene["mesh.fuv"]
                write_mesh(final_path / "mesh.obj", v, f, uv, fuv)

                scene.clear_cache()

                print("Done.")

                return scene
