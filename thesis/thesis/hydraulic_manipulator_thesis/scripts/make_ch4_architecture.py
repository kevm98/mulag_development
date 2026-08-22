#!/usr/bin/env python3
"""Layered data-generation architecture for Chapter 4, in the nested
colour-coded stack idiom: two campaigns sitting on the shared Simscape
substrate and feeding the learned-model datasets, with spanning
alignment, audit, and interface bars.

Run:  python3 scripts/make_ch4_architecture.py
"""
from pathlib import Path
import cairosvg

BASE = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis")
IMG = BASE / "images"
DRAWIO = BASE / "drawio"
DRAWIO.mkdir(exist_ok=True)

ORANGE, BLUE, TEAL, PURPLE, RED, GRAY = (
    "#C97816", "#2F5DA8", "#16847A", "#7A52A8", "#B64B42", "#69737D")
INK, MUTED = "#1F2933", "#66727E"
FONT = "DejaVu Sans, Arial, sans-serif"


def rect(x, y, w, h, fill, stroke, sw=2.0, rx=12):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, s, size=15, fill=INK, weight="normal", anchor="middle",
        rot=0):
    tr = f' transform="rotate({rot} {x} {y})"' if rot else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{tr}>'
            f'{s}</text>')


def build():
    W, H = 1440, 880
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']

    # ---- top strip: what the architecture produces ----
    s.append(rect(60, 26, 1150, 50, "#EDEFF1", GRAY, 1.8, 12))
    s.append(txt(635, 57,
                 "Learned-model datasets  &#8212;  inverse dynamics (ID) "
                 "&#183; forward dynamics (FD) &#183; reduced-order model (ROM)",
                 15, INK, "bold"))

    # ---- far-right spanning bar: audit ----
    s.append(rect(1230, 26, 60, 700, "#DCE6F2", BLUE, 1.8, 12))
    s.append(txt(1260, 376, "Audit &amp; Validation", 16, BLUE, "bold",
                 "middle", -90))

    # ---- two campaign headers ----
    s.append(rect(60, 92, 560, 54, "#FBE7C6", ORANGE, 2.0, 10))
    s.append(txt(340, 126,
                 "Closed-loop campaign  &#183;  realistic operation (R1, R3)",
                 15, "#8a5410", "bold"))
    s.append(rect(760, 92, 450, 54, "#F0D2CD", RED, 2.0, 10))
    s.append(txt(985, 126, "Open-loop campaign  &#183;  excitation (R2)",
                 15, "#7d332c", "bold"))

    # ---- LEFT nested group: closed-loop pipeline ----
    s.append(rect(60, 160, 560, 380, "#FEF6E8", ORANGE, 1.6, 14))
    s.append(txt(600, 350, "Randomization", 15, "#8a5410", "bold",
                 "middle", -90))
    cl = ["Latin-hypercube parameter sampling (12-D)",
          "Fixed integral-backstepping controller",
          "Waypoint trajectory sampler (12&#8211;20 poses)",
          "1000-run curriculum, ode23t integration",
          "Per-run logging: state, command, $\\xi$"]
    cl = [c.replace("$\\xi$", "&#958;") for c in cl]
    for i, lab in enumerate(cl):
        y = 178 + i * 70
        s.append(rect(84, y, 470, 54, "#FDECCB", ORANGE, 1.8, 10))
        s.append(txt(319, y + 33, lab, 14.5, INK, "bold"))

    # ---- MIDDLE vertical bar: align & join ----
    s.append(rect(636, 160, 106, 380, "#D7EFEC", TEAL, 1.8, 12))
    s.append(txt(689, 350, "Align &amp; Join", 17, TEAL, "bold",
                 "middle", -90))

    # ---- RIGHT nested group: open-loop pipeline ----
    s.append(rect(760, 160, 450, 300, "#F7E2DF", RED, 1.6, 14))
    ol = ["64 s open-loop excitation schedule",
          "Four plant draws (nom / hyd / mech / full)",
          "Stroke-coverage acceptance gate"]
    for i, lab in enumerate(ol):
        y = 178 + i * 70
        s.append(rect(784, y, 402, 54, "#EBC9C3", RED, 1.8, 12))
        s.append(txt(985, y + 33, lab, 14.5, INK, "bold"))
    # fail/reject block (darker, like the reference's Fail-Safe)
    s.append(rect(760, 476, 450, 54, RED, "#7d332c", 2.0, 12))
    s.append(txt(985, 509, "Rejected export &#8212; duplicated-state audit fail",
                 14.5, "#FFFFFF", "bold"))

    # ---- wide interface bar ----
    s.append(rect(60, 556, 1150, 96, "#9DB8D8", BLUE, 1.8, 14))
    s.append(txt(635, 636, "Simscape model interface", 15, "#16305e", "bold"))
    s.append(rect(150, 574, 380, 44, "#2F5DA8", "#16305e", 1.6, 22))
    s.append(txt(340, 602, "Hydraulic subsystem", 14.5, "#FFFFFF", "bold"))
    s.append(rect(740, 574, 380, 44, "#2F5DA8", "#16305e", 1.6, 22))
    s.append(txt(930, 602, "Multibody subsystem", 14.5, "#FFFFFF", "bold"))

    # ---- foundation row ----
    found = ["CAD / URDF geometry", "ode23t stiff solver",
             "Nominal parameter set"]
    for i, lab in enumerate(found):
        x = 60 + i * 397
        s.append(rect(x, 676, 370, 74, "#1B4A82", "#0f2c50", 2.0, 14))
        s.append(txt(x + 185, 720, lab, 15, "#FFFFFF", "bold"))

    s.append("</svg>")
    out = "\n".join(s)
    (DRAWIO / "fig4_architecture.svg").write_text(out)
    cairosvg.svg2pdf(bytestring=out.encode(),
                     write_to=str(IMG / "ch4_architecture.pdf"))
    print("wrote", IMG / "ch4_architecture.pdf")


if __name__ == "__main__":
    build()
