# Agent Instructions: Publication-Quality Figures

When generating figures for the master's thesis, follow these rules:

## Reproducibility
- Always create reproducible source code for every figure.
- Never fabricate experimental data. Use authoritative sources or mark data as synthetic/verification.
- Use authoritative USD, URDF, CAD, or mesh geometry when available.
- Include commands that regenerate every figure.
- Store editable source files separately from generated output.

## Figure Types and Tools
- Use Blender Python for reproducible 3D robot renders.
- Use PyVista for trajectories, workspaces, meshes, and vector fields.
- Use Matplotlib for quantitative results and plots.
- Use Orthographic views for technical illustrations unless perspective is required.

## Export Standards
- Export line-art figures as PDF and SVG (vector).
- Export raster renders at publication resolution (600 dpi or 2000px+).
- Use transparent backgrounds when compositing.
- Keep annotations as SVG, TikZ, or PDF vector elements.

## Style
- Use consistent coordinate frames, cylinder numbers, and joint mappings.
- Match labels and mathematical notation to the LaTeX thesis.
- Avoid decorative gradients, cinematic effects, excessive reflections, and presentation-style graphics.
- Do not overwrite raw data.

## Environment
- Activate: `conda activate thesis_figures`
- Python: `/home/rrlab/miniconda3/envs/thesis_figures/bin/python`
- Set `MPLCONFIGDIR=/tmp/mplconfig` for matplotlib.
- Use system TeX Live (`/usr/bin/pdflatex`) for TikZ.
