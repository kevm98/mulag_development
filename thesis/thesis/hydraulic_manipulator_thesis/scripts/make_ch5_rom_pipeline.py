#!/usr/bin/env python3
"""ROM network structure as a data-flow pipeline (Chapter 5), in the
stacked-slab / RNN-unit / dense-neuron idiom: raw history -> colored
feature slabs -> normalization -> GRU units over time -> dense fully
connected heads -> physical update -> predicted increments.

Run:  python3 scripts/make_ch5_rom_pipeline.py
"""
from pathlib import Path
import cairosvg

BASE = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis")
IMG, DRAWIO = BASE / "images", BASE / "drawio"
DRAWIO.mkdir(exist_ok=True)
ORANGE, BLUE, TEAL, PURPLE, RED, GRAY = (
    "#C97816", "#2F5DA8", "#16847A", "#7A52A8", "#B64B42", "#69737D")
GREEN = "#4E9A3D"
INK, MUTED = "#1F2933", "#66727E"
FONT = "DejaVu Sans, Arial, sans-serif"


def T(x, y, s, sz=13, f=INK, w="normal", a="middle", rot=0):
    tr = f' transform="rotate({rot} {x} {y})"' if rot else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{sz}" '
            f'fill="{f}" font-weight="{w}" text-anchor="{a}"{tr}>{s}</text>')


def slab(x, y, w, h, fill, stroke, sk=18):
    return (f'<polygon points="{x},{y} {x+w},{y} {x+w-sk},{y+h} {x-sk},{y+h}" '
            f'fill="{fill}" fill-opacity="0.9" stroke="{stroke}" stroke-width="2"/>')


def box(x, y, w, h, fill, stroke, rx=8, sw=1.8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def build():
    W, H = 1560, 360
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 150 {W} {H}">',
         '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" '
         'refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#69737D"/>'
         '</marker></defs>',
         f'<rect x="0" y="150" width="{W}" height="{H}" fill="#FFFFFF"/>']
    yc = 300

    def arrow(x1, x2, y=yc):
        return (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{GRAY}" '
                f'stroke-width="1.8" marker-end="url(#a)"/>')

    # 1. raw history slab
    s.append(slab(60, 250, 90, 120, "#D7EFEC", TEAL))
    s.append(T(96, 315, "history", 12, INK, "bold"))
    s.append(T(96, 333, "H&#8342;", 13, INK, "bold"))
    s.append(T(96, 400, "6 &#215; 24", 11, MUTED))

    # 2. colored feature slabs (3 groups of the 24-D input)
    fx = 190
    groups = [("state  x, x&#775;, &#916;p", BLUE, "#DCE6F5"),
              ("prev. valve  u&#770;", ORANGE, "#FBE7C6"),
              ("command  c", PURPLE, "#E9DEF3")]
    for i, (lab, st, fl) in enumerate(groups):
        yy = 230 + i * 52
        s.append(slab(fx, yy, 150, 44, fl, st))
        s.append(T(fx + 66, yy + 28, lab, 11.5, INK, "bold"))
    s.append(T(fx + 60, 210, "24-D features", 11, MUTED))
    s.append(arrow(150, fx - 10))

    # 3. normalization (thin vertical bar)
    s.append(box(370, 236, 34, 128, "#F1F3F5", GRAY, 6))
    s.append(T(387, 300, "normalize", 12, MUTED, "bold", "middle", -90))
    s.append(arrow(345, 370))

    # 4. GRU units over time
    gx = 445
    s.append(box(gx, 210, 150, 180, "#EAF3FF", BLUE, 12, 1.6))
    s.append(T(gx + 75, 200, "GRU encoder  2 &#215; 128", 12, BLUE, "bold"))
    for i, tl in enumerate(["t&#8722;5", "&#8942;", "t"]):
        yy = 226 + i * 52
        s.append(box(gx + 30, yy, 90, 40, "#CFE2FA", BLUE, 8, 1.6))
        s.append(T(gx + 75, yy + 25, f"RNN {tl}", 12, INK, "bold"))
        # recurrent self loop
        s.append(f'<path d="M {gx+120} {yy+8} q 22 -6 0 -14" fill="none" '
                 f'stroke="{BLUE}" stroke-width="1.3" marker-end="url(#a)"/>')
    s.append(arrow(404, gx))

    # 5. dense fully-connected neuron column
    nx = 720
    hcx = nx + 60
    # hidden latent column (green) with dense connections to head column
    hids = [230 + i * 34 for i in range(5)]
    outs = [246 + i * 34 for i in range(4)]
    for hy in hids:
        for oy in outs:
            s.append(f'<line x1="{nx+18}" y1="{hy}" x2="{hcx+120}" y2="{oy}" '
                     f'stroke="{GRAY}" stroke-width="0.5" opacity="0.4"/>')
    for hy in hids:
        s.append(f'<circle cx="{nx+18}" cy="{hy}" r="10" fill="#Dff0DA" '
                 f'stroke="{GREEN}" stroke-width="1.6"/>')
    s.append(T(nx + 18, 210, "latent h&#8342;", 11, GREEN, "bold"))
    s.append(arrow(gx + 150, nx + 4))

    # heads column
    heads = [("actuation head &#8594; u&#770;", BLUE),
             ("residual heads r&#8346;,r&#7590;,r&#8339;", PURPLE),
             ("load head &#8467;(h&#8342;)", ORANGE)]
    for i, (lab, col) in enumerate(heads):
        yy = 236 + i * 46
        s.append(box(hcx + 120, yy, 210, 36, "#FFFFFF", col, 8, 1.7))
        s.append(T(hcx + 225, yy + 23, lab, 11.5, INK, "bold"))

    # 6. physical update block
    px = 1120
    s.append(box(px, 214, 190, 172, "#EAF7F4", TEAL, 12, 1.8))
    s.append(T(px + 95, 205, "physical update", 12, TEAL, "bold"))
    for i, lab in enumerate(["pressure &#916;p&#770;", "velocity x&#775;&#770;",
                             "position x&#770;"]):
        yy = 226 + i * 48
        s.append(box(px + 20, yy, 150, 36, "#CDEAE4", TEAL, 8, 1.5))
        s.append(T(px + 95, yy + 23, lab, 12, INK, "bold"))
        if i < 2:
            s.append(f'<line x1="{px+95}" y1="{yy+36}" x2="{px+95}" '
                     f'y2="{yy+48}" stroke="{TEAL}" stroke-width="1.6" '
                     f'marker-end="url(#a)"/>')
    s.append(f'<line x1="{hcx+330}" y1="260" x2="{px}" y2="270" '
             f'stroke="{GRAY}" stroke-width="1.8" marker-end="url(#a)"/>')

    # 7. output slab
    s.append(slab(1350, 250, 130, 110, "#F1F3F5", GRAY))
    s.append(T(1406, 300, "&#916;p&#770;, &#916;x&#775;&#770;, &#916;x&#770;",
               13, INK, "bold"))
    s.append(T(1406, 330, "PhysX target", 10.5, MUTED))
    s.append(arrow(px + 190, 1352))

    # bottom note
    s.append(T(780, 470,
               "Colored slabs = the three feature groups of the 24-D input; "
               "dense lines = full connectivity; the GRU state carries "
               "pressure and command history across the six steps.",
               12, MUTED))
    s.append(T(780, 492,
               "Neural parts (blue/purple/orange) propose; the physical "
               "update (teal) disposes, keeping every predicted increment "
               "bounded and interpretable.", 12, MUTED))
    s.append("</svg>")
    out = "\n".join(s)
    (DRAWIO / "fig5_rom_pipeline.svg").write_text(out)
    cairosvg.svg2pdf(bytestring=out.encode(),
                     write_to=str(IMG / "models_rom_pipeline.pdf"))
    print("wrote", IMG / "models_rom_pipeline.pdf")


if __name__ == "__main__":
    build()
