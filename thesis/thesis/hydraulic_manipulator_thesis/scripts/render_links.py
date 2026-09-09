"""Per-link orthographic renders of the FME 500 for a CAD catalog figure.

Run:
  ~/tools/blender/blender -b <URDF.blend> -P scripts/render_links.py

Each named link mesh is isolated, framed in an orthographic camera, and
rendered on a transparent background with the Workbench engine (matte,
no cinematic effects) per AGENTS.publication-figures.md.
"""
import math
import os

import bpy
from mathutils import Vector

OUT = os.path.expanduser("~/rrlab_mulag/thesis/thesis/hydraulic_manipulator_thesis/images/links/")

os.makedirs(OUT, exist_ok=True)

scene = bpy.context.scene

# Link mesh objects grouped by the body they represent. Cylinder barrel and
# rod are shown together as one actuator.
GROUPS = {
    "frontrahmen":  ["Frontrahmen"],
    "drehzapfen":   ["Drehzapfen"],
    "ausleger_i":   ["Ausleger_I"],
    "ausleger_ii":  ["Ausleger_II"],
    "messerkopf_schwenk": ["Messerkopf-Schwenk_base", "Messerkopf-Schwenk_Link1",
                            "Messerkopf-Schwenk_Link2"],
    "messerkopf":   ["Messerkopf"],
    "cylinder":     ["Hebezylinder_Rohr", "Hebezylinder_Kolben"],
}

# hide everything renderable first
for o in scene.objects:
    if o.type == "MESH":
        o.hide_render = True

# camera (orthographic, isometric direction)
cam_data = bpy.data.cameras.new("LinkCam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.new("LinkCam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
view_dir = Vector((1.0, -1.0, 0.55)).normalized()

# lighting
sun_data = bpy.data.lights.new("Sun", type="SUN")
sun_data.energy = 3.2
sun = bpy.data.objects.new("Sun", sun_data)
sun.rotation_euler = (math.radians(52), math.radians(12), math.radians(28))
scene.collection.objects.link(sun)

# workbench matte look
scene.render.engine = "BLENDER_WORKBENCH"
sh = scene.display.shading
sh.light = "STUDIO"
sh.color_type = "SINGLE"
sh.single_color = (0.62, 0.64, 0.66)
sh.show_cavity = True
sh.cavity_type = "BOTH"
sh.curvature_ridge_factor = 1.4
sh.curvature_valley_factor = 1.4
scene.display.render_aa = "16"
scene.render.resolution_x = 1400
scene.render.resolution_y = 1400
scene.render.film_transparent = True
scene.render.image_settings.color_mode = "RGBA"


def frame_group(objs):
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mins = Vector(map(min, mins, w))
            maxs = Vector(map(max, maxs, w))
    center = (mins + maxs) / 2
    diag = (maxs - mins).length
    cam.location = center + view_dir * diag * 3.0
    look = center - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    cam_data.ortho_scale = diag * 1.15


for name, members in GROUPS.items():
    objs = [scene.objects.get(m) for m in members]
    objs = [o for o in objs if o is not None]
    if not objs:
        print("MISSING", name, members)
        continue
    for o in objs:
        o.hide_render = False
    frame_group(objs)
    scene.render.filepath = OUT + f"link_{name}.png"
    bpy.ops.render.render(write_still=True)
    print("WROTE", scene.render.filepath)
    for o in objs:
        o.hide_render = True

print("DONE")
