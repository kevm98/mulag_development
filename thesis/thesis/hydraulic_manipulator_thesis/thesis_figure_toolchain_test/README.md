# Thesis Figure Toolchain

Installation and verification workspace for publication-quality engineering figures.

## Installed Applications

| Application | Version | Source |
|---|---|---|
| Blender | 4.0.2 | Ubuntu 24.04 apt |
| OpenSCAD | 2021.01 | Ubuntu 24.04 apt |
| Inkscape | 1.2.2 | Ubuntu 24.04 apt |
| Graphviz | 2.42.2 | Ubuntu 24.04 apt |
| FFmpeg | 6.1.1 | Ubuntu 24.04 apt |
| ImageMagick | 6.9.12-98 | Ubuntu 24.04 apt |
| TeX Live | 2023 (system) | Ubuntu 24.04 apt |
| FreeCAD | NOT INSTALLED | Install via AppImage/Flatpak |

## Installed VS Code Extensions

| Extension | Version |
|---|---|
| jacqueslucke.blender-development | 0.0.31 |
| cesium.gltf-vscode | 2.5.1 |
| ranch-hand-robotics.urdf-editor | 1.0.0 |
| leathong.openscad-language-support | 2.0.1 |
| hediet.vscode-drawio | 1.9.0 |
| james-yu.latex-workshop | (pre-existing) |
| ms-python.python | (pre-existing) |
| ms-toolsai.jupyter | 2025.9.1 |

## Conda Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_figures
```

Python 3.11.15 with packages: numpy, scipy, pandas, matplotlib, scienceplots, pyvista, vtk, trimesh, meshio, networkx, graphviz, schemdraw, pillow, imageio, imageio-ffmpeg, svgwrite, cairosvg, jupyter, ipykernel, pyyaml, h5py.

## Regeneration Commands

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_figures
export MPLCONFIGDIR=/tmp/mplconfig

# Matplotlib
python src/test_matplotlib.py

# PyVista 3D
python src/test_pyvista.py

# Blender
blender --background --python src/test_blender.py

# OpenSCAD (STL only; PNG needs display)
openscad -o output/test_openscad.stl src/test_openscad.scad

# Graphviz
for fmt in pdf svg png; do dot -T$fmt src/test_graphviz.dot -o output/test_graphviz.$fmt; done

# TikZ (use system TeX Live)
PATH=/usr/bin:$PATH latexmk -pdf src/test_tikz.tex
mv test_tikz.pdf output/

# Inkscape
inkscape --export-type=pdf --export-filename=output/test_inkscape.pdf output/test_graphviz.svg
inkscape --export-type=png --export-filename=output/test_inkscape.png --export-dpi=300 output/test_graphviz.svg

# FFmpeg animation
ffmpeg -y -f lavfi -i color=c=white:s=640x480:d=3 -vf "drawtext=text='Test':fontsize=32:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2" -c:v libx264 output/test_animation.mp4
```

## Recommended Tools by Figure Type

| Figure Category | Recommended Tool |
|---|---|
| Quantitative plots | Matplotlib |
| Scientific 3D visualization | PyVista/VTK |
| 3D robot renders | Blender |
| Parametric mechanical geometry | OpenSCAD / FreeCAD (AppImage) |
| Architecture/flow diagrams | TikZ / Draw.io |
| Final vector annotation | Inkscape |

## Known Limitations

- FreeCAD is not available in Ubuntu 24.04 repositories. Install via AppImage or Flatpak.
- OpenSCAD PNG rendering requires a display (X server). STL export works headless.
- `/home/rrlab/.local/share` is read-only; Jupyter kernel installation skipped.
- Matplotlib config dir is read-only; set `MPLCONFIGDIR=/tmp/mplconfig`.
- Blender Cycles needs OpenImageDenoiser; use EEVEE for headless rendering.
- System TeX Live 2023 must be used for standalone.cls (user TinyTeX lacks it).

## Troubleshooting

```bash
# Check conda environment
conda env list
conda activate thesis_figures
python -c "import numpy, matplotlib, pyvista, trimesh; print('OK')"

# Check TeX Live
/usr/bin/pdflatex --version

# Check Blender
blender --version
blender --background --python-expr "import bpy; print('Blender OK')"
```

## Selecting the Python (thesis_figures) Jupyter Kernel

In VS Code, open a `.ipynb` file, click the kernel selector (top-right), and choose `Python (thesis_figures)`.
If the kernel is not listed, set the Python interpreter path to:
`/home/rrlab/miniconda3/envs/thesis_figures/bin/python`

## Blender Development Extension

The `jacqueslucke.blender-development` extension provides autocomplete for `bpy` in VS Code.
Select `thesis_figures` as the Python interpreter for Blender scripts.

## Vector Label Preservation

- Export line-art figures as PDF and SVG.
- Use Inkscape to annotate final vector graphics.
- Store editable source files (`.blend`, `.scad`, `.py`, `.tex`) separately from generated output.
- Match labels and notation to the LaTeX thesis.
