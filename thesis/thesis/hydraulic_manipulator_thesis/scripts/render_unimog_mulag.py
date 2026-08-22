"""Publication renders of the COMPLETE simulated machine: the Unimog carrier
with the MULAG FME 500 arm mounted, taken from the exact USD asset the
IsaacLab environment loads.

This is deliberately not the arm-only CAD render of Chapter 3: it is the
articulation as PhysX sees it, including the carrier and its wheels, so the
reader can see what "the truck drives while the arm reaches over the verge"
actually looks like.

Source asset
  extensions/rrlab_assets/data/Robots/unimog_mulag_fme500_cylinder_as_actuator.usd

Run (CPU only; does not touch the training GPU):
  ~/tools/blender/blender -b -P scripts/render_unimog_mulag.py

Produces
  images/unimog_mulag_iso.png    front three-quarter, arm deployed to the side
  images/unimog_mulag_rear.png   rear three-quarter, same configuration

A true side elevation was rendered and DISCARDED: in the asset's authored
configuration the arm points along the viewing axis, so it collapses to a
vertical stick and shows less than either view kept here.
"""
import math
import os

import bpy
import mathutils

ASSET = ("/home/rrlab/rrlab_mulag/extensions/rrlab_assets/data/Robots/"
         "unimog_mulag_fme500_cylinder_as_actuator.usd")
OUT = ("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis/"
       "images/")
os.makedirs(OUT, exist_ok=True)

# Publication palette: a real-machine industrial finish with strong visual
# separation between carrier, boom and actuators. Geometry remains the exact
# USD articulation; only material assignment and studio lighting are changed.
ARM = (0.62, 0.18, 0.035, 1.0)       # safety orange boom and head
CYL = (0.90, 0.56, 0.08, 1.0)        # warm yellow hydraulic actuators
BODY = (0.16, 0.25, 0.12, 1.0)       # Unimog-style deep olive carrier
TYRE = (0.13, 0.14, 0.15, 1.0)


def matte(name, rgba, rough=0.55):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = rough
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.25
    return m


def prim_path(obj):
    parts = []
    while obj:
        parts.append(obj.name)
        obj = obj.parent
    return "/".join(reversed(parts))


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=ASSET, import_materials=False)

    m_arm, m_cyl, m_body, m_tyre = (
        matte("arm", ARM), matte("cyl", CYL, 0.42),
        matte("body", BODY), matte("tyre", TYRE, 0.85))

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    for o in meshes:
        p = prim_path(o)
        o.data.materials.clear()
        if "/Cylinder_Parts/" in p:
            o.data.materials.append(m_cyl)
        elif "/Wheels/" in p:
            o.data.materials.append(m_tyre)
        elif "/Arm_Links/" in p:
            o.data.materials.append(m_arm)
        else:
            o.data.materials.append(m_body)
        for poly in o.data.polygons:
            poly.use_smooth = True

    # scene bounds
    mn = mathutils.Vector((1e9,) * 3)
    mx = mathutils.Vector((-1e9,) * 3)
    for o in meshes:
        for c in o.bound_box:
            w = o.matrix_world @ mathutils.Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    centre = (mn + mx) / 2.0
    span = max((mx - mn).x, (mx - mn).y, (mx - mn).z)

    # ground plane, slightly below the wheels, for a contact shadow only
    bpy.ops.mesh.primitive_plane_add(size=span * 6,
                                     location=(centre.x, centre.y, mn.z))
    ground = bpy.context.active_object
    gm = matte("ground", (1.0, 1.0, 1.0, 1.0), 0.9)
    ground.data.materials.append(gm)

    # lighting: one key, one fill, one rim -- no cinematic effects
    def lamp(name, loc, energy, size):
        bpy.ops.object.light_add(type="AREA", location=loc)
        L = bpy.context.active_object
        L.name = name
        L.data.energy = energy
        L.data.size = size
        d = centre - L.location
        L.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return L

    lamp("key", (centre.x + span, centre.y - span, centre.z + span * 1.3),
         span * span * 9.0, span * 0.9)
    lamp("fill", (centre.x - span * 1.2, centre.y - span * 0.8,
                  centre.z + span * 0.5), span * span * 3.0, span * 1.2)
    lamp("rim", (centre.x - span * 0.4, centre.y + span * 1.3,
                 centre.z + span * 0.9), span * span * 3.5, span * 0.9)

    world = bpy.data.worlds.new("w")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (
        0.94, 0.95, 0.96, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"          # never touch the training GPU
    sc.cycles.samples = 128
    sc.cycles.use_denoising = True
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "Standard"

    return centre, span


def shoot(centre, span, name, azim_deg, elev_deg, res=(2200, 1500), zoom=1.06):
    sc = bpy.context.scene
    for c in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(c, do_unlink=True)

    az, el = math.radians(azim_deg), math.radians(elev_deg)
    r = span * 3.0
    loc = mathutils.Vector((
        centre.x + r * math.cos(el) * math.cos(az),
        centre.y + r * math.cos(el) * math.sin(az),
        centre.z + r * math.sin(el)))
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = span * zoom
    cam.rotation_euler = (centre - loc).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam

    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.filepath = os.environ.get("RENDER_OUT", OUT) + name
    bpy.ops.render.render(write_still=True)
    print("WROTE", sc.render.filepath + ".png")


if __name__ == "__main__":
    centre, span = build()
    shoot(centre, span, "unimog_mulag_iso", azim_deg=25, elev_deg=16,
          res=(2400, 1700), zoom=1.22)
    shoot(centre, span, "unimog_mulag_rear", azim_deg=-110, elev_deg=20,
          res=(2400, 1700), zoom=1.22)
