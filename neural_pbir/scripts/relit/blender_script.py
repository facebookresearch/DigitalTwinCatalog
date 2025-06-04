# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np

debug = int(sys.argv[6])
use_gpu = int(sys.argv[7])
mesh_path = sys.argv[8]
albedo_path = sys.argv[9]
roughness_path = sys.argv[10]
lgt_paths = sys.argv[11].split(",")
cameras_path = sys.argv[12]
split = sys.argv[13]
filter_type = sys.argv[14]
filter_width = float(sys.argv[15])
with_bg = int(sys.argv[16])
F0 = float(sys.argv[17])
render_exr = int(sys.argv[18])

assert os.path.isfile(mesh_path), f"{mesh_path} not found."

if use_gpu:
    bpy.context.scene.cycles.device = "GPU"

assert os.path.isfile(albedo_path), f"{albedo_path} not found."
assert os.path.isfile(roughness_path), f"{roughness_path} not found."

mat_root = str(Path(albedo_path).parent)
fp = bpy.path.abspath(f"//{os.path.join(os.path.abspath(mat_root), 'blender_relit')}")
print("Results are dumped to", fp)
os.makedirs(fp, exist_ok=True)

# Read camera poses configuration
with open(cameras_path) as f:
    cameras = json.load(f)

fx = cameras["fx"]
fy = cameras["fy"]
cx = cameras["cx"]
cy = cameras["cy"]
H, W = int(cy * 2), int(cx * 2)
bpy.data.cameras[0].angle = max(
    2 * np.arctan2(W * 0.5, fx), 2 * np.arctan2(H * 0.5, fy)
)
print("Set fov:", bpy.data.cameras[0].angle * 180 / np.pi, "deg")
split_indices = cameras["split"][split]
poses = np.stack([cameras["frames"][idx]["to_world"] for idx in split_indices])

# Convert to blender system
poses[:, :, :2] *= -1
poses[:, [1, 2]] = np.stack([-poses[:, 2], poses[:, 1]], 1)
poses[:, :, [1, 2]] *= -1
camera_locations = []
camera_rotations = []
for pose in poses:
    transform_matrix = mathutils.Matrix(pose)
    camera_locations.append(transform_matrix.to_translation())
    camera_rotations.append(transform_matrix.to_euler())

# Load mesh
print("Loading mesh:", mesh_path)
imported_object = bpy.ops.import_scene.obj(filepath=mesh_path)
obj_object = bpy.context.selected_objects[0]

# Set material
material = obj_object.active_material
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Specular"].default_value = 0.5 * (F0 / 0.04)
bsdf.inputs["Specular Tint"].default_value = 0
bsdf.inputs["Sheen Tint"].default_value = 0
bsdf.inputs["Clearcoat Roughness"].default_value = 0

print("Set albedo:", albedo_path)
albedo = material.node_tree.nodes.new("ShaderNodeTexImage")
albedo.image = bpy.data.images.load(albedo_path)
material.node_tree.links.new(albedo.outputs["Color"], bsdf.inputs["Base Color"])

print("Set roughenss:", roughness_path)
roughness = material.node_tree.nodes.new("ShaderNodeTexImage")
roughness.image = bpy.data.images.load(roughness_path)
roughness.image.colorspace_settings.name = "Non-Color"
material.node_tree.links.new(roughness.outputs["Color"], bsdf.inputs["Roughness"])

# if use_metallic:
#     assert os.path.isfile(metallic_path), f'{metallic_path} not found.'
#     print('Set metallic:', metallic_path)
#     metallic = material.node_tree.nodes.new('ShaderNodeTexImage')
#     metallic.image = bpy.data.images.load(metallic_path)
#     metallic.image.colorspace_settings.name = 'Non-Color'
#     material.node_tree.links.new(metallic.outputs['Color'], bsdf.inputs['Metallic'])


# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of albedo (diffuse color in blender)
if "View Layer" in bpy.context.scene.view_layers:
    bpy.context.scene.view_layers["View Layer"].use_pass_diffuse_color = True
else:
    bpy.context.scene.render.layers["RenderLayer"].use_pass_diffuse_color = True

# Set up rendering
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
if render_exr:
    bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
    bpy.context.scene.render.image_settings.color_depth = "32"
else:
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_depth = "8"


# Create input render layer node.
render_layers = tree.nodes.new("CompositorNodeRLayers")

albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = "Albedo Output"
# albedo_file_output.format.color_management = 'OVERRIDE'
# albedo_file_output.format.view_settings.view_transform = 'Raw'
albedo_file_output.format.file_format = "OPEN_EXR"
albedo_alpha = tree.nodes.new(type="CompositorNodeSetAlpha")
links.new(render_layers.outputs["DiffCol"], albedo_alpha.inputs[0])
links.new(render_layers.outputs["Alpha"], albedo_alpha.inputs[1])
links.new(albedo_alpha.outputs[0], albedo_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = not with_bg

# Create collection for objects not to render with background
objs = [ob for ob in bpy.context.scene.objects if ob.type in ("EMPTY")]
bpy.ops.object.delete({"selected_objects": objs})

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.cycles.pixel_filter_type = filter_type
scene.cycles.filter_width = filter_width

cam = scene.objects["Camera"]

albedo_file_output.base_path = fp

# Prepare envmap
envmap = scene.world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")

############################
# render relit im & albedo
############################
for lgt_path in lgt_paths:
    lgt_name = Path(lgt_path).stem
    if lgt_path != "":
        print("Set environment map:", lgt_path)
        envmap.image = bpy.data.images.load(lgt_path)
        scene.world.node_tree.links.new(
            envmap.outputs[0], scene.world.node_tree.nodes["Background"].inputs["Color"]
        )

    for location, rotation, i in zip(camera_locations, camera_rotations, split_indices):
        cam.location = location
        cam.rotation_euler = rotation

        fname = os.path.splitext(os.path.basename(cameras["frames"][i]["path"]))[0]

        scene.render.filepath = fp + f"/im_{fname}_{lgt_name}"
        albedo_file_output.file_slots[0].path = f"im_{fname}_albedo"
        print(f"Writing: im_{fname}_{lgt_name}")

        bpy.ops.render.render(write_still=True)  # render still

        if debug:
            break


############################
# render roughness
############################
# albedo.image = bpy.data.images.load(roughness_path)
# albedo.image.colorspace_settings.name = 'Non-Color'
# for location, rotation, i in zip(camera_locations, camera_rotations, split_indices):

#     cam.location = location
#     cam.rotation_euler = rotation
#     fname = os.path.splitext(os.path.basename(cameras['frames'][i]['path']))[0]

#     scene.render.filepath = fp + f'/im_{fname}_roughness_dummy'
#     albedo_file_output.file_slots[0].path = f'im_{fname}_roughness'
#     print(f'Writing: im_{fname}_roughness')

#     bpy.ops.render.render(write_still=True)  # render still

#     if debug:
#         break
