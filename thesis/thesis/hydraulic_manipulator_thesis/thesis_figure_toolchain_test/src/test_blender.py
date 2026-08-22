"""Blender headless render: articulated arm with hydraulic cylinders."""
import bpy
import os
import math

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Materials
def make_material(name, color, metallic=0.3, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat

gray_mat = make_material("Gray", (0.4, 0.4, 0.4, 1.0))
blue_mat = make_material("Blue", (0.2, 0.4, 0.7, 1.0))
red_mat = make_material("Red", (0.8, 0.15, 0.15, 1.0))
yellow_mat = make_material("Yellow", (0.9, 0.7, 0.1, 1.0))

def apply_mat(obj, mat):
    obj.data.materials.append(mat)

# Base platform
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.15))
base = bpy.context.active_object
base.scale = (0.6, 0.4, 0.15)
base.name = "Base"
apply_mat(base, gray_mat)

# Arm link 1 (vertical pillar)
bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=1.2, location=(0, 0, 0.9))
link1 = bpy.context.active_object
link1.name = "Link1"
apply_mat(link1, blue_mat)

# Arm link 2 (horizontal arm)
bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=1.0, location=(0.4, 0, 1.4))
link2 = bpy.context.active_object
link2.name = "Link2"
link2.rotation_euler = (0, math.radians(90), 0)
apply_mat(link2, blue_mat)

# Joints
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(0, 0, 1.5))
joint1 = bpy.context.active_object
joint1.name = "Joint1"
apply_mat(joint1, red_mat)

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(0.9, 0, 1.4))
joint2 = bpy.context.active_object
joint2.name = "Joint2"
apply_mat(joint2, red_mat)

# Hydraulic cylinder 1
bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.8, location=(0.3, 0.15, 0.9))
cyl1 = bpy.context.active_object
cyl1.name = "Cylinder1"
cyl1.rotation_euler = (math.radians(20), 0, 0)
apply_mat(cyl1, yellow_mat)

# Hydraulic cylinder 2
bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.6, location=(0.5, -0.12, 1.2))
cyl2 = bpy.context.active_object
cyl2.name = "Cylinder2"
cyl2.rotation_euler = (0, math.radians(15), 0)
apply_mat(cyl2, yellow_mat)

# Ground plane
bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, -0.01))
ground = bpy.context.active_object
ground.name = "Ground"
apply_mat(ground, gray_mat)

# Lighting
bpy.ops.object.light_add(type='AREA', location=(5, 5, 8))
light1 = bpy.context.active_object
light1.data.energy = 500
light1.data.size = 5

bpy.ops.object.light_add(type='AREA', location=(-3, -2, 4))
light2 = bpy.context.active_object
light2.data.energy = 300
light2.data.size = 3

bpy.ops.object.light_add(type='AREA', location=(0, -5, 3))
light3 = bpy.context.active_object
light3.data.energy = 200
light3.data.size = 4

# Camera
bpy.ops.object.camera_add(location=(3.5, -3.5, 2.8))
cam = bpy.context.active_object
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 5
cam.rotation_euler = (math.radians(60), 0, math.radians(45))
bpy.context.scene.camera = cam

# Render settings
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 2000
bpy.context.scene.render.resolution_y = 1500
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGBA'

output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)

png_path = os.path.join(output_dir, "test_blender.png")
bpy.context.scene.render.filepath = png_path
bpy.ops.render.render(write_still=True)
print(f"  {png_path}  ({os.path.getsize(png_path)} bytes)")

# Save .blend
blend_path = os.path.join(output_dir, "test_blender.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"  {blend_path}  ({os.path.getsize(blend_path)} bytes)")

print("Blender verification complete.")
