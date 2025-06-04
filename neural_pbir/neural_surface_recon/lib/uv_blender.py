# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import bpy


def import_obj(obj_path):
    if os.path.exists(obj_path):
        bpy.ops.import_scene.obj(filepath=obj_path)
    else:
        print(obj_path + " does not exists!")


def export_obj(obj_path):
    bpy.ops.export_scene.obj(
        filepath=obj_path, use_normals=False, use_materials=False, use_selection=True
    )


obj_in_fn = sys.argv[-2]
obj_out_fn = sys.argv[-1]

import_obj(obj_in_fn)
obj = bpy.data.objects[-1]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
# Smoothing
bpy.ops.object.modifier_add(type="SMOOTH")
bpy.context.object.modifiers["Smooth"].factor = 0.2
bpy.context.object.modifiers["Smooth"].iterations = 5
bpy.ops.object.modifier_apply(modifier="Smooth")
# UV Unwrap
bpy.ops.mesh.uv_texture_add()
bpy.ops.object.editmode_toggle()
print("UV mapping using Blender")
bpy.ops.uv.smart_project(island_margin=0.001)
export_obj(obj_out_fn)
