#!/usr/bin/env python3
"""PyVista 3D engineering visualization."""
import os, numpy as np
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_PANEL"] = "false"

import pyvista as pv

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

base = pv.Cylinder(center=(0, 0, 0.25), direction=(0, 0, 1), radius=0.3, height=0.5)
link1 = pv.Cylinder(center=(0, 0, 0.75), direction=(0.5, 0.2, 0), radius=0.08, height=1.0)
joint1 = pv.Sphere(center=(0, 0, 0.5), radius=0.12)
link2 = pv.Cylinder(center=(0.5, 0.2, 1.0), direction=(0.3, -0.1, 0), radius=0.07, height=0.8)
joint2 = pv.Sphere(center=(0.25, 0.1, 0.75), radius=0.1)

plotter.add_mesh(base, color="darkgray")
plotter.add_mesh(link1, color="steelblue")
plotter.add_mesh(joint1, color="darkred")
plotter.add_mesh(link2, color="steelblue")
plotter.add_mesh(joint2, color="darkred")

t = np.linspace(0, 2*np.pi, 200)
ref = np.column_stack([0.5*np.cos(t), 0.5*np.sin(t), 1.2 + 0.2*np.sin(2*t)])
rng = np.random.default_rng(42)
meas = ref + rng.normal(0, 0.03, ref.shape)

plotter.add_lines(ref, color="black", width=3, connected=True)
plotter.add_lines(meas, color="red", width=2, connected=True)

plotter.add_axes(line_width=3, xlabel="X", ylabel="Y", zlabel="Z")
plotter.camera_position = "xz"
plotter.camera.parallel_projection = True
plotter.set_background("white")

os.makedirs(OUTPUT, exist_ok=True)
png_path = os.path.join(OUTPUT, "test_pyvista.png")
plotter.screenshot(png_path, scale=2)
print(f"  {png_path}  ({os.path.getsize(png_path)} bytes)")

try:
    svg_path = os.path.join(OUTPUT, "test_pyvista.svg")
    plotter.save_graphic(svg_path)
    print(f"  {svg_path}  ({os.path.getsize(svg_path)} bytes)")
except Exception as e:
    print(f"  SVG: {type(e).__name__}: {e}")

try:
    pdf_path = os.path.join(OUTPUT, "test_pyvista.pdf")
    plotter.save_graphic(pdf_path)
    print(f"  {pdf_path}  ({os.path.getsize(pdf_path)} bytes)")
except Exception as e:
    print(f"  PDF: {type(e).__name__}: {e}")

plotter.close()
print("PyVista verification complete.")
