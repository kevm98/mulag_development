"""Publication renders of the FME 500 for Chapter 3.

Run:
  ~/tools/blender/blender -b <URDF.blend> -P scripts/render_fme500_figures.py

Produces
  images/links/link_*.png        per-part catalog renders, each roll-optimised
                                 to fill a square frame (uniform apparent size)
  images/fme500_cad_iso.png      full arm, isometric, cylinders highlighted
  images/fme500_cad_side.png     full arm, side elevation, cylinders highlighted
  images/fme500_topology.png     full arm, flat silhouette for annotation
  images/fme500_joint_coords.json  normalised image coords of joints/cylinders
"""
import json
import math
import os

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

BASE = "/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis/images/"
LINKS = BASE + "links/"
os.makedirs(LINKS, exist_ok=True)

scene = bpy.context.scene

# ----------------------------------------------------------------- bodies
LINK_GROUPS = {
    "frontrahmen": ["Frontrahmen"],
    "drehzapfen": ["Drehzapfen"],
    "ausleger_i": ["Ausleger_I"],
    "ausleger_ii": ["Ausleger_II"],
    "messerkopf_schwenk": ["Messerkopf-Schwenk_base", "Messerkopf-Schwenk_Link1",
                           "Messerkopf-Schwenk_Link2"],
    "messerkopf": ["Messerkopf"],
    "cylinder": ["Hebezylinder_Rohr", "Hebezylinder_Kolben"],
}

# cylinder i -> (barrel, rod); order = slew, inner boom, outer boom, head pivot
CYLINDERS = [
    ("Ausleger-Schwenkzylinder_Rohr", "Ausleger-Schwenkzylinder_Kolben"),
    ("Hebezylinder_Rohr", "Hebezylinder_Kolben"),
    ("Knickzylinder_Rohr", "Knickzylinder_Kolben"),
    ("Messerkopf-Schwenkzylinder_Rohr", "Messerkopf-Schwenkzylinder_Kolben"),
]
CYL_OBJS = [n for pair in CYLINDERS for n in pair]

JOINT_ARMATURES = {
    "J1": "Drehzapfen_link",
    "J2": "Ausleger_I_link",
    "J3": "Ausleger_II_link",
    "J4": "Messerkopf-Schwenk_base_link",
    "J5": "Messerkopf_link",
}

MESHES = [o for o in scene.objects
          if o.type == "MESH" and "collision" not in o.name.lower()]
for o in scene.objects:
    if o.type == "MESH":
        o.hide_render = True

# ------------------------------------------------------------- appearance
GRAY = (0.62, 0.645, 0.67, 1.0)
ORANGE = (0.93, 0.50, 0.07, 1.0)
for o in MESHES:
    o.color = ORANGE if o.name in CYL_OBJS else GRAY

sun_data = bpy.data.lights.new("Sun", type="SUN")
sun_data.energy = 3.2
sun = bpy.data.objects.new("Sun", sun_data)
sun.rotation_euler = (math.radians(52), math.radians(12), math.radians(28))
scene.collection.objects.link(sun)

scene.render.engine = "BLENDER_WORKBENCH"
sh = scene.display.shading
sh.light = "STUDIO"
sh.show_cavity = True
sh.cavity_type = "BOTH"
sh.curvature_ridge_factor = 1.4
sh.curvature_valley_factor = 1.4
scene.display.render_aa = "32"
scene.render.film_transparent = True
scene.render.image_settings.color_mode = "RGBA"

cam_data = bpy.data.cameras.new("FigCam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.new("FigCam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def world_points(objs):
    pts = []
    for o in objs:
        for c in o.bound_box:
            pts.append(o.matrix_world @ Vector(c))
    return pts


def aim(objs, view_dir, roll=0.0, margin=1.10):
    """Point the ortho camera at objs along view_dir with an optional roll."""
    pts = world_points(objs)
    center = sum(pts, Vector()) / len(pts)
    d = Vector(view_dir).normalized()
    diag = max((p - center).length for p in pts) * 2.0
    cam.location = center + d * diag * 3.0
    quat = (center - cam.location).to_track_quat("-Z", "Y")
    cam.rotation_euler = quat.to_euler()
    if roll:
        cam.rotation_euler.rotate_axis("Z", roll)  # local roll
    bpy.context.view_layer.update()
    # projected extent in camera space
    inv = cam.matrix_world.inverted()
    xs = [(inv @ p).x for p in pts]
    ys = [(inv @ p).y for p in pts]
    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    ext = max(dx, dy)
    # ortho_scale spans the LONGER pixel dimension, so convert the required
    # vertical extent into that unit before taking the maximum.
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    need = max(dx, dy * rx / ry) if rx >= ry else max(dx * ry / rx, dy)
    cam_data.ortho_scale = need * margin
    # recentre so the part sits in the middle of the frame
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    cam.location += cam.matrix_world.to_quaternion() @ Vector((cx, cy, 0.0))
    bpy.context.view_layer.update()
    return ext


def best_roll(objs, view_dir):
    """Roll that lets an elongated part fill a square frame best."""
    best, best_ext = 0.0, 1e9
    for deg in range(0, 90, 5):
        ext = aim(objs, view_dir, math.radians(deg))
        if ext < best_ext - 1e-6:
            best_ext, best = ext, math.radians(deg)
    return best


def set_res(res):
    scene.render.resolution_x, scene.render.resolution_y = res


def render(path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("WROTE", path)


ISO = Vector((1.0, -1.0, 0.5))
SIDE = Vector((1.0, 0.04, 0.14))

# ------------------------------------------------- 1. per-part catalog
for name, members in LINK_GROUPS.items():
    objs = [scene.objects.get(m) for m in members]
    objs = [o for o in objs if o]
    if not objs:
        print("MISSING", name)
        continue
    for o in objs:
        o.hide_render = False
    set_res((1200, 1200))
    roll = best_roll(objs, ISO)
    aim(objs, ISO, roll)
    render(LINKS + f"link_{name}.png")
    for o in objs:
        o.hide_render = True

# ------------------------------------------------- 2. whole-arm views
for o in MESHES:
    o.hide_render = False

sh.color_type = "OBJECT"          # highlights the cylinders in orange
set_res((1500, 1500))
aim(MESHES, ISO, margin=1.06)
render(BASE + "fme500_cad_iso.png")

# cylinder centres projected into THIS iso view (normalised 0..1), for
# precise callout placement in make_ch3_schematics.py -- no more hand-guessed
# leader-line coordinates.
iso_coords = {}
for i, (barrel, rod) in enumerate(CYLINDERS, start=1):
    objs = [scene.objects.get(barrel), scene.objects.get(rod)]
    objs = [o for o in objs if o]
    pts = world_points(objs)
    centre = sum(pts, Vector()) / len(pts)
    v = world_to_camera_view(scene, cam, centre)
    iso_coords[f"C{i}"] = [round(v.x, 4), round(1.0 - v.y, 4)]
with open(BASE + "fme500_iso_coords.json", "w") as fh:
    json.dump({"resolution": [1500, 1500], "points": iso_coords}, fh, indent=1)
print("ISO_COORDS", json.dumps(iso_coords))

set_res((2000, 1250))
aim(MESHES, SIDE, margin=1.06)
render(BASE + "fme500_cad_side.png")

# ------------------------------------------------- 3. topology base view
sh.color_type = "SINGLE"
sh.single_color = (0.78, 0.79, 0.80)
sh.show_cavity = False
res = (2000, 1250)
set_res(res)
aim(MESHES, SIDE, margin=1.14)
render(BASE + "fme500_topology.png")

# joint / cylinder centres projected into that same view (normalised 0..1)
coords = {}
for tag, arm_name in JOINT_ARMATURES.items():
    o = scene.objects.get(arm_name)
    if not o:
        continue
    v = world_to_camera_view(scene, cam, o.matrix_world.translation)
    coords[tag] = [round(v.x, 4), round(1.0 - v.y, 4)]
for i, (barrel, rod) in enumerate(CYLINDERS, start=1):
    objs = [scene.objects.get(barrel), scene.objects.get(rod)]
    objs = [o for o in objs if o]
    pts = world_points(objs)
    centre = sum(pts, Vector()) / len(pts)
    v = world_to_camera_view(scene, cam, centre)
    coords[f"C{i}"] = [round(v.x, 4), round(1.0 - v.y, 4)]
with open(BASE + "fme500_joint_coords.json", "w") as fh:
    json.dump({"resolution": res, "points": coords}, fh, indent=1)
print("COORDS", json.dumps(coords))
print("DONE")
