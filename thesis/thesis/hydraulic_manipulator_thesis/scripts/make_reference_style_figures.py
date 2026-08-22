#!/usr/bin/env python3
"""Generate the replacement thesis diagrams in a restrained paper style.

The drawings deliberately follow the visual language of the local reference
papers: large technical content, sparse annotation, standard signal flow,
light grouping fields, and one muted colour per physical/signal family.
Existing chapter text and captions are not changed.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import cairosvg


BASE = Path(__file__).resolve().parents[1]
IMG = BASE / "images"
SRC = BASE / "drawio" / "reference_style"
SRC.mkdir(parents=True, exist_ok=True)

INK = "#20262D"
MUTED = "#64707C"
LINE = "#8C98A4"
PALE = "#F5F7F8"
BLUE = "#357EC7"
BLUE_L = "#E8F2FC"
TEAL = "#25A18E"
TEAL_L = "#E6F7F3"
OCHRE = "#E29A16"
OCHRE_L = "#FFF3D9"
RED = "#D65F59"
RED_L = "#FCEAE8"
VIOLET = "#8665B5"
VIOLET_L = "#F1EBFA"
FONT = "Arial, Helvetica, DejaVu Sans, sans-serif"


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def text(x, y, value, size=22, color=INK, weight="normal", anchor="middle",
         line=1.22, italic=False):
    lines = str(value).split("\n")
    start = y - (len(lines) - 1) * size * line / 2
    tspans = "".join(
        f'<tspan x="{x}" y="{start + i * size * line:.1f}">{esc(s)}</tspan>'
        for i, s in enumerate(lines)
    )
    style = "italic" if italic else "normal"
    return (f'<text font-family="{FONT}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" font-style="{style}" '
            f'text-anchor="{anchor}">{tspans}</text>')


def rect(x, y, w, h, fill="white", stroke=LINE, sw=2, radius=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def line(x1, y1, x2, y2, color=LINE, sw=2, dash=None, arrow=False):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    mk = ' marker-end="url(#arrow)"' if arrow else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}" fill="none"{ds}{mk}/>' )


def path(d, color=LINE, sw=2, dash=None, arrow=False, fill="none"):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    mk = ' marker-end="url(#arrow)"' if arrow else ""
    return (f'<path d="{d}" stroke="{color}" stroke-width="{sw}" '
            f'fill="{fill}"{ds}{mk}/>' )


def circle(x, y, r, fill="white", stroke=LINE, sw=2):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def arrow_defs():
    return ('<defs><marker id="arrow" markerWidth="10" markerHeight="10" '
            'refX="9" refY="5" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>')


def image_href(path_: Path) -> str:
    mime = "image/png" if path_.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path_.read_bytes()).decode()}"


def save(name: str, width: int, height: int, body: list[str]):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}">'
           f'{arrow_defs()}<rect width="100%" height="100%" fill="white"/>'
           + "".join(body) + "</svg>")
    (SRC / f"{name}.svg").write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(IMG / f"{name}.pdf"))
    print("wrote", IMG / f"{name}.pdf")


def title_band(body, x, y, w, label, color, fill):
    body += [rect(x, y, w, 54, fill, color, 2, 5),
             text(x + 18, y + 34, label, 22, color, "bold", "start")]


def callout(body, x, y, w, h, label, color, target_x, target_y,
            lead_x=None, lead_y=None):
    """Add a compact, high-contrast annotation in the Figure 3.3 style."""
    if lead_x is None:
        lead_x = x + w / 2
    if lead_y is None:
        lead_y = y + h / 2
    body += [
        line(lead_x, lead_y, target_x, target_y, color, 3.5),
        circle(target_x, target_y, 17, "white", color, 4),
        circle(target_x, target_y, 9, color, color, 1),
        rect(x, y, w, h, "white", color, 3, 8),
        text(x + w / 2, y + h / 2 + 7, label, 22, INK, "bold"),
    ]


def actuation_chain():
    W, H = 1600, 800
    b: list[str] = [text(40, 38, "Electro-hydraulic command-to-motion chain",
                         28, INK, "bold", "start"),
                    rect(30, 60, 1540, 680, "white", LINE, 2, 8)]
    href = image_href(IMG / "electro_hydraulic_actuator_channel_render.png")
    b += [f'<image x="45" y="75" width="1510" height="650" '
          f'preserveAspectRatio="xMidYMid meet" href="{href}"/>']
    callout(b, 65, 90, 255, 66, "Valve command", OCHRE,
            455, 245, 320, 123)
    callout(b, 65, 565, 285, 66, "Valve and manifold", OCHRE,
            350, 455, 350, 598)
    callout(b, 610, 88, 245, 66, "Chamber flow", BLUE,
            745, 345, 733, 154)
    callout(b, 1015, 88, 275, 66, "Cylinder pressure", BLUE,
            940, 365, 1100, 154)
    callout(b, 1240, 555, 290, 66, "Piston motion", TEAL,
            1305, 425, 1240, 588)
    b += [
        circle(555, 700, 8, OCHRE, OCHRE, 1),
        text(575, 707, "electrical", 18, OCHRE, "bold", "start"),
        circle(775, 700, 8, BLUE, BLUE, 1),
        text(795, 707, "hydraulic", 18, BLUE, "bold", "start"),
        circle(1005, 700, 8, TEAL, TEAL, 1),
        text(1025, 707, "mechanical", 18, TEAL, "bold", "start"),
    ]
    save("fund_actuation_chain", W, H, b)


def _workflow_legacy():
    W, H = 1600, 680
    b: list[str] = [
        text(40, 42, "Four-step thesis workflow", 30, INK, "bold", "start"),
        text(1560, 42, "from physical model to matched evidence", 19,
             MUTED, "normal", "end"),
    ]
    xs = [25, 420, 815, 1210]
    specs = [
        ("1", "MODEL", "Machine physics", "articulation + hydraulic contract",
         "Chapter 3", OCHRE, OCHRE_L),
        ("2", "LEARN", "Actuator dynamics", "Simscape data to compact models",
         "Chapters 4–5", BLUE, BLUE_L),
        ("3", "TRAIN", "Control policy", "randomized parallel plants",
         "Chapter 6", TEAL, TEAL_L),
        ("4", "EVALUATE", "Matched stacks", "tracking · robustness · adaptation",
         "Chapter 7", VIOLET, VIOLET_L),
    ]
    outcomes = ("Physics prior", "Identified dynamics",
                "Anchored policy", "Matched evidence")
    for x, (number, verb, title_, note, chapter, color, fill) in zip(xs, specs):
        b += [
            rect(x, 78, 365, 475, "white", color, 3, 10),
            rect(x, 78, 365, 78, fill, color, 2.4, 10),
            circle(x + 43, 117, 24, "white", color, 3),
            text(x + 43, 124, number, 19, color, "bold"),
            text(x + 82, 112, verb, 23, color, "bold", "start"),
            text(x + 82, 140, title_, 17, INK, "bold", "start"),
        ]

    # Stage 1: rigid-body linkage with explicit joints and one actuator.
    x = xs[0]
    model_nodes = [(x + 62, 304), (x + 137, 226),
                   (x + 225, 288), (x + 302, 211)]
    for first, second in zip(model_nodes[:-1], model_nodes[1:]):
        b += [line(first[0], first[1], second[0], second[1], OCHRE, 9)]
    for px, py in model_nodes:
        b += [circle(px, py, 14, "white", OCHRE, 6)]
    b += [
        line(x + 95, 307, x + 237, 324, BLUE, 7),
        circle(x + 95, 307, 9, "white", BLUE, 4),
        circle(x + 237, 324, 9, "white", BLUE, 4),
    ]

    # Stage 2: measured trajectories entering a compact learned model.
    x = xs[1]
    b += [
        path(f"M {x+38} 290 C {x+72} 210, {x+105} 355, {x+145} 270 "
             f"S {x+208} 225, {x+225} 275", BLUE, 6),
        circle(x + 54, 260, 8, BLUE, BLUE, 1),
        circle(x + 113, 302, 8, BLUE, BLUE, 1),
        circle(x + 184, 249, 8, BLUE, BLUE, 1),
        line(x + 230, 275, x + 252, 275, LINE, 4, arrow=True),
        rect(x + 268, 205, 68, 140, BLUE_L, BLUE, 4, 9),
        text(x + 302, 264, "LEARNED", 14, BLUE, "bold"),
        text(x + 302, 287, "MODEL", 16, BLUE, "bold"),
    ]

    # Stage 3: policy network controlling a randomized plant family.
    x = xs[2]
    inputs = [(x + 52, 218), (x + 52, 274), (x + 52, 330)]
    hidden = [(x + 137, 235), (x + 137, 313)]
    output = (x + 215, 274)
    for ax, ay in inputs:
        for bx, by in hidden:
            b += [line(ax, ay, bx, by, "#B9DCD5", 2.5)]
    for ax, ay in hidden:
        b += [line(ax, ay, output[0], output[1], TEAL, 3)]
    for px, py in inputs + hidden + [output]:
        b += [circle(px, py, 11, "white", TEAL, 4)]
    b += [line(x + 230, 274, x + 255, 274, LINE, 4, arrow=True)]
    for i, shade in enumerate((TEAL_L, "#D8EFEA", "#C9E8E1")):
        b += [rect(x + 274 + i * 9, 214 + i * 17, 54, 96,
                   shade, TEAL, 2.5, 6)]

    # Stage 4: three controller stacks judged by one common checklist.
    x = xs[3]
    for i, (label, color) in enumerate((("A", BLUE), ("B", TEAL), ("C", VIOLET))):
        yy = 195 + i * 55
        b += [
            rect(x + 43, yy, 115, 45, "white", color, 3, 8),
            text(x + 100, yy + 30, f"STACK {label}", 16, color, "bold"),
            line(x + 164, yy + 22, x + 232, 259, LINE, 3),
        ]
    b += [
        circle(x + 270, 259, 49, VIOLET_L, VIOLET, 5),
        path(f"M {x+245} 259 L {x+263} 278 L {x+297} 236",
             VIOLET, 8),
    ]

    for x in xs:
        number = xs.index(x)
        _, _, title_, note, chapter, color, fill = specs[number]
        b += [
            text(x + 182, 391, outcomes[number], 25, INK, "bold"),
            text(x + 182, 429, note, 17, MUTED),
            rect(x + 72, 474, 221, 46, fill, color, 2, 23),
            text(x + 182, 504, chapter, 17, color, "bold"),
        ]
    for x in [390, 785, 1180]:
        b += [line(x, 315, x + 20, 315, LINE, 4, arrow=True)]
    b += [
        rect(265, 585, 1070, 62, PALE, LINE, 2.4, 30),
        text(800, 624, "One shared cylinder-motion contract across all stages",
             22, INK, "bold"),
    ]
    save("fund_workflow", W, H, b)


def workflow():
    # Figure 2.3 is maintained as native TikZ so that its typography, line
    # weight, and colors stay consistent with the thesis document.
    source = IMG / "fund_workflow.tex"
    if not source.exists():
        raise FileNotFoundError("Figure 2.3 requires its TikZ source")
    print(f"preserved {source}")


def robust_adaptive():
    # Figure 2.4 is maintained directly in Draw.io by the thesis author.
    # Preserve the checked-in vector export during bulk figure regeneration.
    source = SRC / "fund_robust_adaptive.drawio"
    output = IMG / "fund_robust_adaptive.pdf"
    if not source.exists() or not output.exists():
        raise FileNotFoundError(
            "Figure 2.4 requires both its Draw.io source and vector PDF export"
        )
    print("preserved", output)


def topology():
    # Preserve the original repository artwork.  Its large CAD mechanism,
    # direct leader labels, and compact vertical chain are clearer than the
    # later card-based replacement and already match the chapter caption.
    source = BASE / "drawio" / "fig3_1_topology.svg"
    generated_source = SRC / "fme500_topology_annotated.svg"
    generated_source.write_bytes(source.read_bytes())
    cairosvg.svg2pdf(url=str(source),
                     write_to=str(IMG / "fme500_topology_annotated.pdf"))
    print("restored original", IMG / "fme500_topology_annotated.pdf")


def valve_symbol(b, x, y, scale=1.0):
    w, h = 150 * scale, 64 * scale
    for i in range(3):
        b.append(rect(x + i * w / 3, y, w / 3, h, "white", BLUE, 1.7, 0))
    b += [line(x + 9*scale, y + 53*scale, x + 40*scale, y + 11*scale, BLUE, 1.6, arrow=True),
          line(x + 110*scale, y + 11*scale, x + 141*scale, y + 53*scale, BLUE, 1.6, arrow=True),
          line(x + 60*scale, y + 12*scale, x + 60*scale, y + 52*scale, BLUE, 1.4),
          line(x + 90*scale, y + 12*scale, x + 90*scale, y + 52*scale, BLUE, 1.4)]


def cylinder_symbol(b, x, y, w=260, h=82):
    b += [rect(x, y, w, h, "white", INK, 2, 0),
          line(x + w * .48, y + 4, x + w * .48, y + h - 4, INK, 3),
          line(x + w * .48, y + h / 2, x + w + 55, y + h / 2, INK, 3),
          line(x + 30, y + h, x + 30, y + h + 20, RED, 2),
          line(x + w - 30, y + h, x + w - 30, y + h + 20, BLUE, 2)]


def hydraulic_schematic():
    W, H = 1600, 1080
    b: list[str] = []
    b += [text(40, 42, "Four independent electro-hydraulic actuator channels",
               25, INK, "bold", "start"),
          text(1160, 42, "Shared hydraulic power unit", 25, INK, "bold", "start"),
          rect(25, 65, 1085, 980, "white", LINE, 2, 8),
          rect(1135, 65, 440, 980, "white", BLUE, 2.3, 8)]

    b += [text(135, 94, "DRIVEN CAD LINK", 14, TEAL, "bold"),
          text(525, 94, "DOUBLE-ACTING CYLINDER", 14, INK, "bold"),
          text(835, 94, "4/3 VALVE", 14, BLUE, "bold")]
    rows = [
        (110, "C1 · J1", "Slew", "Drehzapfen", "drehzapfen", "u₁"),
        (325, "C2 · J2", "Boom I", "Ausleger I", "ausleger_i", "u₂"),
        (540, "C3 · J3", "Boom II", "Ausleger II", "ausleger_ii", "u₃"),
        (755, "C4 · J4", "Head pivot", "Messerkopf-Schwenk", "messerkopf_schwenk", "u₄"),
    ]
    rail_p, rail_t = 1035, 1070
    b += [line(rail_p, 95, rail_p, 975, RED, 5),
          line(rail_t, 95, rail_t, 975, BLUE, 5),
          rect(1007, 73, 55, 38, RED_L, RED, 1.8, 5),
          text(1035, 99, "P", 20, RED, "bold"),
          rect(1067, 73, 55, 38, BLUE_L, BLUE, 1.8, 5),
          text(1094, 99, "T", 20, BLUE, "bold")]

    for y, ident, role, german, thumb, command in rows:
        # One compact, self-contained card per physical actuator.
        b += [rect(45, y, 960, 200, PALE, LINE, 1.7, 7)]
        href = image_href(IMG / "links" / f"link_{thumb}.png")
        b += [rect(60, y + 18, 150, 150, "white", TEAL, 1.8, 6),
              f'<image x="65" y="{y+23}" width="140" height="140" '
              f'preserveAspectRatio="xMidYMid meet" href="{href}"/>',
              text(230, y + 48, ident, 20, BLUE, "bold", "start"),
              text(230, y + 81, role, 22, INK, "bold", "start"),
              text(230, y + 116, german, 15, MUTED, "normal", "start")]

        # Asymmetric double-acting cylinder with visible chambers and rod.
        cx, cy, cw, ch = 390, y + 42, 270, 82
        piston = cx + 135
        b += [rect(cx, cy, cw, ch, "white", INK, 2, 0),
              f'<rect x="{cx+3}" y="{cy+3}" width="{piston-cx-4}" height="{ch-6}" '
              f'fill="{BLUE_L}" stroke="none"/>',
              f'<rect x="{piston+4}" y="{cy+3}" width="{cx+cw-piston-7}" height="{ch-6}" '
              f'fill="{RED_L}" stroke="none"/>',
              line(piston, cy + 3, piston, cy + ch - 3, INK, 4),
              line(piston, cy + ch/2, cx + cw + 62, cy + ch/2, INK, 4),
              text(cx + 68, cy + 48, "p₁", 17, BLUE, "bold"),
              text(cx + 205, cy + 48, "p₂", 17, RED, "bold")]

        # Valve and short, separated A/B working lines.
        vx, vy = 770, y + 47
        valve_symbol(b, vx, vy, .88)
        b += [path(f"M {cx+35} {cy+ch} L {cx+35} {y+162} L {vx+25} {y+162} L {vx+25} {vy+56}",
                   BLUE, 2.6),
              path(f"M {cx+cw-35} {cy+ch} L {cx+cw-35} {y+180} L {vx+105} {y+180} L {vx+105} {vy+56}",
                   RED, 2.6),
              text(vx + 18, y + 155, "A", 14, BLUE, "bold"),
              text(vx + 112, y + 177, "B", 14, RED, "bold"),
              path(f"M {vx+132} {y+77} L {rail_p} {y+77}", RED, 2.6),
              path(f"M {vx+132} {y+104} L {rail_t} {y+104}", BLUE, 2.6),
              line(955, y + 39, 902, y + 72, VIOLET, 2.5, arrow=True),
              text(970, y + 35, command, 19, VIOLET, "bold")]

    b += [rect(55, 982, 940, 46, OCHRE_L, OCHRE, 1.8, 5),
          text(75, 1011, "J5 is passive — no fifth valve or hydraulic cylinder branch.",
               17, OCHRE, "bold", "start")]

    # Recognizable power-unit symbols share one uncluttered P/T backbone.
    hp_p, hp_t = 1175, 1215
    b += [line(hp_p, 120, hp_p, 910, RED, 5),
          line(hp_t, 120, hp_t, 910, BLUE, 5),
          text(hp_p, 103, "P", 21, RED, "bold"),
          text(hp_t, 103, "T", 21, BLUE, "bold"),
          line(rail_p, 130, hp_p, 130, RED, 4),
          line(rail_t, 155, hp_t, 155, BLUE, 4)]

    # Gas-charged accumulator.
    b += [line(hp_p, 205, 1270, 205, RED, 2.5),
          rect(1270, 155, 75, 100, BLUE_L, BLUE, 2.3, 34),
          line(1274, 205, 1341, 205, BLUE, 2),
          text(1380, 190, "Gas accumulator", 19, INK, "bold", "start"),
          text(1380, 218, "pressure-energy buffer", 15, MUTED, "normal", "start")]

    # Supply pressure sensor.
    b += [line(hp_p, 330, 1270, 330, RED, 2.5),
          circle(1305, 330, 35, "white", BLUE, 2.3),
          line(1305, 330, 1323, 312, BLUE, 2.2),
          text(1380, 320, "Pressure sensor", 19, INK, "bold", "start"),
          text(1380, 348, "measured supply pₛ", 15, MUTED, "normal", "start")]

    # Relief valve bridges pressure to tank without crossing other branches.
    b += [line(hp_p, 455, 1270, 455, RED, 2.5),
          rect(1270, 420, 70, 70, "white", RED, 2.2, 3),
          line(1282, 478, 1328, 432, RED, 2.2, arrow=True),
          path(f"M 1340 455 L 1360 455 L 1360 500 L {hp_t} 500", BLUE, 2.5),
          text(1380, 438, "Pressure-relief valve", 19, INK, "bold", "start"),
          text(1380, 466, "P → T overpressure path", 15, MUTED, "normal", "start")]

    # Prime mover and variable-speed pump.
    b += [circle(1300, 620, 40, OCHRE_L, OCHRE, 2.4),
          text(1300, 627, "M", 22, OCHRE, "bold"),
          line(1340, 620, 1375, 620, INK, 3),
          circle(1415, 620, 40, BLUE_L, BLUE, 2.4),
          f'<polygon points="1400,602 1400,638 1432,620" fill="{BLUE}"/>',
          line(1415, 580, 1415, 545, RED, 2.5),
          path(f"M 1415 545 L {hp_p} 545", RED, 2.5, None, True),
          text(1360, 690, "Variable-speed motor + pump", 16, INK, "bold"),
          text(1360, 716, "flow source  Qp = Dp ω", 15, MUTED)]

    # Reservoir and suction filter close the return/suction loop.
    b += [path("M 1260 830 L 1260 910 L 1455 910 L 1455 830", BLUE, 2.7),
          line(1260, 850, 1455, 850, BLUE, 1.8, "8 6"),
          f'<polygon points="1288,850 1310,828 1332,850 1310,872" fill="{BLUE_L}" '
          f'stroke="{BLUE}" stroke-width="2"/>',
          path(f"M {hp_t} 780 L {hp_t} 830 L 1288 830", BLUE, 2.5, None, True),
          path("M 1332 850 L 1515 850 L 1515 620 L 1457 620", BLUE, 2.5, None, True),
          text(1360, 945, "Reservoir + suction filter", 18, INK, "bold"),
          text(1360, 972, "return conditioning and pump inlet", 15, MUTED)]
    save("fme500_hydraulic_schematic", W, H, b)


def actuator_channel():
    W, H = 1600, 1100
    b: list[str] = []
    b += [text(45, 42, "One physical electro-hydraulic actuator channel",
               27, INK, "bold", "start"),
          text(1555, 42, "instrumented asymmetric double-acting cylinder",
               18, MUTED, "normal", "end"),
          rect(30, 65, 1540, 855, "white", LINE, 2, 8)]

    # The central component is deliberately a realistic product-style cutaway.
    # Mathematical notation is kept as vector artwork, outside the raster, so it
    # stays sharp in print and cannot be hallucinated by the image generator.
    href = image_href(IMG / "electro_hydraulic_actuator_channel_render.png")
    b += [f'<image x="40" y="70" width="1520" height="845" '
          f'preserveAspectRatio="xMidYMid meet" href="{href}"/>']

    def callout(x, y, w, title_, sub, col, px, py):
        b.extend([rect(x, y, w, 70, "white", col, 2.2, 6),
                  text(x + 14, y + 27, title_, 18, col, "bold", "start"),
                  text(x + 14, y + 53, sub, 14, MUTED, "normal", "start"),
                  path(f"M {x+w/2} {y+70} L {px} {py}", col, 2.2, None, True)])

    callout(55, 82, 275, "Electrical command  u", "coil drives spool position  xᵥ",
            VIOLET, 172, 250)
    callout(365, 82, 330, "4/3 proportional valve", "machined manifold; ports P, T, A, B",
            OCHRE, 360, 330)
    callout(925, 82, 285, "Cap-end sensing", "pressure p₁; working flow Q₁",
            BLUE, 990, 258)
    callout(1260, 82, 285, "Rod-end sensing", "pressure p₂; working flow Q₂",
            RED, 1275, 262)

    # Direct physical annotations on high-contrast component regions.
    b += [rect(825, 395, 250, 54, BLUE_L, BLUE, 2, 5),
          text(950, 429, "Chamber 1   p₁, Q₁, A₁", 18, BLUE, "bold"),
          rect(1110, 395, 250, 54, RED_L, RED, 2, 5),
          text(1235, 429, "Chamber 2   p₂, Q₂, A₂", 18, RED, "bold"),
          circle(650, 344, 20, BLUE_L, BLUE, 2.5),
          text(650, 351, "A", 17, BLUE, "bold"),
          text(685, 349, "cap-end working line", 16, BLUE, "bold", "start"),
          circle(650, 700, 20, RED_L, RED, 2.5),
          text(650, 707, "B", 17, RED, "bold"),
          text(685, 705, "rod-end working line", 16, RED, "bold", "start"),
          line(1050, 748, 1395, 748, TEAL, 3.2, arrow=True),
          text(1220, 724, "measured stroke x and velocity ẋ", 18, TEAL, "bold"),
          text(1100, 814, "integrated linear position sensor", 16, TEAL, "bold")]

    # Compact notation band: mechanical chain on the left, force identity on
    # the right. This replaces the former block diagram without losing any of
    # the variables used by the chapter equations and caption.
    b += [rect(30, 940, 930, 125, PALE, LINE, 2, 7),
          text(55, 973, "MECHANICAL INTERFACE", 17, TEAL, "bold", "start"),
          text(55, 1007,
               "link k  ↔  barrel clevis  —  prismatic stroke x  —  rod clevis  ↔  link k+1",
               18, INK, "bold", "start"),
          text(55, 1039, "revolute – prismatic – revolute chain; axial load acts along the rod",
               16, MUTED, "normal", "start"),
          rect(985, 940, 585, 125, TEAL_L, TEAL, 2.2, 7),
          text(1010, 973, "HYDRAULIC → MECHANICAL COUPLING", 17, TEAL, "bold", "start"),
          text(1278, 1010, "Fₕ = p₁ A₁ − p₂ A₂", 25, INK, "bold"),
          text(1278, 1043, "net motion opposed by Ffric(ẋ)", 16, MUTED)]
    save("fme500_actuator_channel_detailed", W, H, b)


def models_overview():
    W, H = 1500, 790
    b: list[str] = []
    b += [rect(45, 300, 250, 145, VIOLET_L, VIOLET, 2.5, 8),
          text(170, 348, "Desired cylinder", 23, INK, "bold"),
          text(170, 382, "motion command", 23, INK, "bold"),
          text(170, 420, "same action contract", 17, MUTED)]
    b += [text(400, 72, "PPO–ID–FD path", 24, BLUE, "bold", "start"),
          rect(380, 95, 760, 235, "white", BLUE, 2.4, 8),
          rect(440, 160, 255, 100, BLUE_L, BLUE, 2, 6),
          text(568, 198, "Inverse dynamics", 23, INK, "bold"),
          text(568, 234, "desired motion → valve", 17, MUTED),
          line(695, 210, 805, 210, LINE, 3, arrow=True),
          rect(815, 160, 255, 100, "white", BLUE, 2, 6),
          text(942, 198, "Forward dynamics", 23, INK, "bold"),
          text(942, 234, "valve + state → increment", 17, MUTED)]
    b += [text(400, 420, "PPO–ROM path", 24, TEAL, "bold", "start"),
          rect(380, 445, 760, 235, "white", TEAL, 2.4, 8),
          rect(505, 510, 510, 105, TEAL_L, TEAL, 2, 6),
          text(760, 548, "Command-conditioned grey-box ROM", 23, INK, "bold"),
          text(760, 586, "history + desired motion → physical update", 17, MUTED)]
    b += [line(295, 345, 430, 210, LINE, 3, arrow=True),
          line(295, 400, 495, 560, LINE, 3, arrow=True),
          rect(1220, 300, 240, 145, PALE, LINE, 2.5, 8),
          text(1340, 350, "Bounded PhysX", 23, INK, "bold"),
          text(1340, 385, "position target", 23, INK, "bold"),
          text(1340, 422, "shared plant interface", 17, MUTED),
          line(1070, 210, 1220, 345, LINE, 3, arrow=True),
          line(1015, 560, 1220, 400, LINE, 3, arrow=True)]
    b += [path("M 1340 445 L 1340 735 L 170 735 L 170 445", VIOLET, 2.3, "10 7", True),
          text(755, 763, "measured cylinder state and pressure feedback — identical for both paths", 19, VIOLET)]
    save("models_overview", W, H, b)


def acceptance_flow():
    W, H = 1500, 810
    b: list[str] = []
    gates = [(45, "1", "Offline accuracy", "one-step + rollout"),
             (405, "2", "Input authority", "command ablation"),
             (765, "3", "In-simulator gates", "stroke + pressure"),
             (1125, "4", "Task-space gate", "tracking metrics")]
    for x, n, ttl, sub in gates:
        b += [circle(x + 145, 95, 31, BLUE_L, BLUE, 2.5),
              text(x + 145, 103, n, 21, BLUE, "bold"),
              rect(x, 145, 290, 145, "white", BLUE, 2.2, 7),
              text(x + 145, 195, ttl, 22, INK, "bold"),
              text(x + 145, 235, sub, 18, MUTED),
              text(x + 145, 273, "PASS", 17, TEAL, "bold")]
    for x in [335, 695, 1055]:
        b += [line(x, 216, x + 60, 216, TEAL, 3, arrow=True)]
    repairs = [(45, "revise objective / architecture"),
               (405, "collect identifiable excitation"),
               (765, "repair runtime contract"),
               (1125, "retune controller layer")]
    for x, lab in repairs:
        b += [line(x + 145, 290, x + 145, 405, RED, 2.5, arrow=True),
              text(x + 170, 347, "NO", 17, RED, "bold", "start"),
              rect(x, 420, 290, 95, RED_L, RED, 2, 7),
              text(x + 145, 468, lab, 18, INK, "bold"),
              path(f"M {x+145} 515 L {x+145} 595 L 20 595 L 20 216 L 35 216", RED, 1.8, "8 7", True)]
    b += [line(1270, 290, 1270, 650, TEAL, 3, arrow=True),
          rect(1040, 665, 460, 100, TEAL_L, TEAL, 2.5, 50),
          text(1270, 705, "Promote checkpoint", 24, INK, "bold"),
          text(1270, 739, "artifact + SHA-256 provenance", 17, MUTED)]
    save("models_acceptance_flow", W, H, b)


def nn_architectures():
    W, H = 1500, 900
    b: list[str] = []
    b += [text(45, 45, "(a) Residual MLP used by ID and FD", 25, INK, "bold", "start"),
          rect(35, 70, 1430, 330, "white", BLUE, 2.2, 8)]
    layers = [(90, 170, 180, 115, "Input", "28-D ID / 20-D FD", VIOLET, VIOLET_L),
              (340, 170, 190, 115, "Projection", "width 512", BLUE, BLUE_L),
              (620, 135, 420, 185, "Residual block ×4", "LayerNorm → SiLU → Dropout\nLinear → identity addition", BLUE, "white"),
              (1130, 170, 240, 115, "Linear head", "4-D ID / 12-D FD", TEAL, TEAL_L)]
    for x, y, w, h, ttl, sub, col, fill in layers:
        b += [rect(x, y, w, h, fill, col, 2.2, 7),
              text(x+w/2, y+42, ttl, 22, INK, "bold"),
              text(x+w/2, y+85 if "\n" not in sub else y+118, sub, 17, MUTED)]
    for x1, x2 in [(270,340),(530,620),(1040,1130)]:
        b += [line(x1, 228, x2-10, 228, LINE, 3, arrow=True)]
    b += [path("M 650 135 C 720 78 940 78 1010 135", TEAL, 2.5, "10 6", True),
          text(830, 92, "identity skip", 18, TEAL, "bold")]

    b += [text(45, 485, "(b) Recurrent encoder used by the ROM", 25, INK, "bold", "start"),
          rect(35, 510, 1430, 350, "white", TEAL, 2.2, 8),
          rect(85, 615, 250, 130, VIOLET_L, VIOLET, 2.2, 7),
          text(210, 655, "History Hk", 23, INK, "bold"),
          text(210, 693, "6 steps × 24 features", 18, MUTED),
          text(210, 724, "0.25 s causal window", 17, MUTED)]
    xs = [470, 650, 830]
    labs = ["t−5", "…", "t"]
    for x, lab in zip(xs, labs):
        b += [rect(x, 620, 115, 105, TEAL_L, TEAL, 2.2, 12),
              text(x+57, 655, "GRU", 22, INK, "bold"),
              text(x+57, 695, lab, 18, MUTED)]
    b += [line(335, 680, 460, 680, LINE, 3, arrow=True),
          line(585, 680, 640, 680, TEAL, 3, arrow=True),
          line(765, 680, 820, 680, TEAL, 3, arrow=True),
          text(650, 775, "2 layers × 128 hidden units", 18, TEAL, "bold"),
          line(945, 680, 1055, 680, LINE, 3, arrow=True),
          rect(1065, 590, 325, 180, "white", LINE, 2, 7)]
    for i, (lab, col) in enumerate([("actuation head", BLUE), ("residual heads", VIOLET), ("load head", OCHRE)]):
        y = 615 + i*55
        b += [line(1095, y, 1120, y, col, 4), text(1140, y+6, lab, 19, INK, "bold", "start")]
    save("models_nn_architectures", W, H, b)


def rom_pipeline():
    W, H = 1500, 720
    b: list[str] = []
    stages = [(40, 225, 200, 200, "History Hk", "6 × 24\n0.25 s", VIOLET, VIOLET_L),
              (290, 175, 255, 300, "Feature groups", "state x, xdot, Δp\nprevious valve uhat\ndesired command", VIOLET, "white"),
              (600, 225, 210, 200, "GRU encoder", "2 × 128\nlatent hk", BLUE, BLUE_L),
              (865, 150, 255, 350, "Neural heads", "actuation → uhat\nresiduals rp, rv, rx\nload estimate l(hk)", BLUE, "white"),
              (1175, 190, 285, 270, "Physical update", "orifice flow\npressure dynamics\nvelocity + position", TEAL, TEAL_L)]
    for x, y, w, h, ttl, sub, col, fill in stages:
        b += [rect(x, y, w, h, fill, col, 2.4, 8),
              text(x+w/2, y+54, ttl, 23, INK, "bold"),
              line(x+25, y+82, x+w-25, y+82, col, 1.4),
              text(x+w/2, y+h/2+35, sub, 18, MUTED)]
    for x1, x2, y in [(240,290,325),(545,600,325),(810,865,325),(1120,1175,325)]:
        b += [line(x1, y, x2-10, y, LINE, 3, arrow=True)]
    b += [text(750, 75, "Command-conditioned reduced-order model", 28, INK, "bold"),
          text(750, 112, "neural proposal followed by an explicit hydraulic state update", 20, MUTED),
          path("M 1318 460 L 1318 610 L 700 610 L 700 425", TEAL, 2.3, "10 7", True),
          text(1005, 638, "updated pressure and motion form the next causal history", 19, TEAL),
          rect(565, 660, 370, 45, PALE, LINE, 1.5, 20),
          text(750, 689, "output: Δphat, Δxdothat, Δxhat", 19, INK, "bold")]
    save("models_rom_pipeline", W, H, b)


# ---------------------------------------------------------------------------
# Paper-style 3-D visual language used for the system and learned-model plates.
# These definitions intentionally replace the flatter first-pass versions above.
def cuboid(body, x, y, w, h, depth, fill, side, stroke, title_, sub="",
           title_size=21, sub_size=16):
    d = depth
    body += [
        f'<polygon points="{x},{y} {x+d},{y-d} {x+w+d},{y-d} {x+w},{y}" '
        f'fill="{side}" stroke="{stroke}" stroke-width="1.8"/>',
        f'<polygon points="{x+w},{y} {x+w+d},{y-d} {x+w+d},{y+h-d} {x+w},{y+h}" '
        f'fill="{side}" fill-opacity="0.78" stroke="{stroke}" stroke-width="1.8"/>',
        rect(x, y, w, h, fill, stroke, 2.2, 3),
        text(x+w/2, y+h*.40, title_, title_size, INK, "bold"),
    ]
    if sub:
        body.append(text(x+w/2, y+h*.72, sub, sub_size, MUTED))


def image_panel(body, path_, x, y, w, h, stroke=LINE, label=""):
    href = image_href(path_)
    body += [rect(x-4, y-4, w+8, h+8, "white", stroke, 2, 5),
             f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
             f'preserveAspectRatio="xMidYMid meet" href="{href}"/>']
    if label:
        body += [rect(x+10, y+h-42, w-20, 32, "white", "white", 0, 3),
                 text(x+w/2, y+h-20, label, 16, INK, "bold")]


def control_architecture():
    W, H = 1600, 1080
    b: list[str] = []
    b += [text(45, 38, "Reference, task policy and actuator realization", 27, INK, "bold", "start")]
    # Operating context and reference generation.
    b += [rect(35, 60, 1530, 190, PALE, LINE, 2, 8)]
    image_panel(b, IMG / "fme500_operation.jpg", 55, 78, 270, 150, OCHRE,
                "roadside operating context")
    cuboid(b, 390, 105, 260, 95, 16, OCHRE_L, "#F1D593", OCHRE,
           "Reference generator", "verge line + cut height")
    cuboid(b, 730, 105, 260, 95, 16, BLUE_L, "#C7DDF3", BLUE,
           "Vehicle layer", "constant road speed")
    cuboid(b, 1160, 105, 300, 95, 16, VIOLET_L, "#D8CBEA", VIOLET,
           "DLS teacher", "privileged; training only")
    b += [line(325, 153, 380, 153, LINE, 3, arrow=True),
          line(650, 153, 720, 153, LINE, 3, arrow=True),
          path("M 1310 200 L 1310 325 L 1180 325", VIOLET, 2.2, "9 7", True),
          text(1300, 278, "distillation / identification", 16, VIOLET)]

    # Task policy with visible layered MLP slabs.
    b += [rect(300, 290, 980, 190, "white", BLUE, 2.5, 8),
          text(330, 325, "TASK POLICY", 20, BLUE, "bold", "start"),
          text(330, 360, "terrain profile · task errors · cylinder state · model error", 18, MUTED, "normal", "start")]
    for i, (x, h, col, side) in enumerate([(520,92,VIOLET_L,"#D9C9EC"),
                                            (690,120,BLUE_L,"#C7DDF3"),
                                            (870,120,BLUE_L,"#C7DDF3"),
                                            (1050,82,TEAL_L,"#BFE8DF")]):
        cuboid(b, x, 365-(h-82)/2, 100, h, 14, col, side,
               [VIOLET,BLUE,BLUE,TEAL][i], ["66-D input","MLP","MLP","4-D action"][i],
               ["","hidden","hidden","desired motion"][i], 18, 14)
        if i:
            b.append(line(x-55, 406, x-12, 406, LINE, 2.5, arrow=True))
    b += [line(860, 200, 860, 288, LINE, 3, arrow=True)]

    # Three learned actuator alternatives, arranged as the experimental variable.
    b += [text(45, 535, "LEARNED ACTUATOR REALIZATION — one lane active per experiment", 20, TEAL, "bold", "start")]
    lanes = [(45, "PPO–ROM", TEAL, TEAL_L),
             (555, "PPO–ID–FD", BLUE, BLUE_L),
             (1065, "PPO–(AAM)–ID–FD", VIOLET, VIOLET_L)]
    for x, name, col, fill in lanes:
        b += [rect(x, 555, 470, 240, "white", col, 2.4, 8),
              text(x+25, 590, name, 20, col, "bold", "start")]
    cuboid(b, 125, 635, 310, 100, 16, TEAL_L, "#BFE8DF", TEAL,
           "Command-conditioned ROM", "GRU + hydraulic update")
    cuboid(b, 605, 635, 150, 100, 15, BLUE_L, "#C7DDF3", BLUE,
           "ID", "motion → valve")
    cuboid(b, 835, 635, 150, 100, 15, BLUE_L, "#C7DDF3", BLUE,
           "FD", "valve → state")
    b += [line(770, 685, 825, 685, LINE, 2.5, arrow=True)]
    cuboid(b, 1100, 620, 120, 85, 13, VIOLET_L, "#D8CBEA", VIOLET,
           "ID", "frozen", 18, 14)
    cuboid(b, 1270, 610, 120, 72, 13, OCHRE_L, "#F1D593", OCHRE,
           "AAM", "zhat", 18, 14)
    cuboid(b, 1270, 705, 120, 72, 13, VIOLET_L, "#D8CBEA", VIOLET,
           "adapter", "bounded", 18, 14)
    cuboid(b, 1430, 620, 100, 85, 13, BLUE_L, "#C7DDF3", BLUE,
           "FD", "frozen", 18, 14)
    b += [line(1220, 663, 1260, 663, LINE, 2.2, arrow=True),
          line(1390, 663, 1420, 663, LINE, 2.2, arrow=True),
          line(1330, 682, 1330, 695, OCHRE, 2.2, arrow=True)]
    for x in [280, 790, 1300]:
        b += [path(f"M 860 480 L 860 515 L {x} 515 L {x} 555", LINE, 2.1, None, True)]

    # Shared target and plant. Both real context and exact simulated mechanism are visible.
    b += [rect(35, 845, 1530, 190, PALE, LINE, 2, 8)]
    cuboid(b, 130, 900, 270, 90, 15, RED_L, "#F0C5C2", RED,
           "Bounded PhysX target", "same for all stacks")
    image_panel(b, IMG / "unimog_mulag_iso.png", 960, 860, 560, 160, TEAL,
                "USD / PhysX plant")
    b += [line(400, 945, 940, 945, LINE, 3, arrow=True),
          path("M 1240 1020 L 1240 1055 L 520 1055 L 520 480", VIOLET, 2.2, "10 7", True),
          text(875, 1048, "measured cylinder state and actuator-model error", 17, VIOLET)]
    for x in [280, 790, 1300]:
        b += [path(f"M {x} 795 L {x} 825 L 265 825 L 265 890", LINE, 2.1, None, True)]
    save("fme500_control_architecture", W, H, b)


def models_overview():
    W, H = 1550, 860
    b: list[str] = []
    b += [text(45, 45, "Two learned actuator representations around one common plant contract",
               27, INK, "bold", "start")]
    cuboid(b, 55, 330, 250, 150, 18, VIOLET_L, "#D8CBEA", VIOLET,
           "Desired motion", "same policy action")

    b += [rect(365, 85, 690, 290, "white", BLUE, 2.5, 9),
          text(400, 125, "PPO–ID–FD", 21, BLUE, "bold", "start")]
    cuboid(b, 430, 185, 235, 115, 18, BLUE_L, "#C7DDF3", BLUE,
           "Inverse dynamics", "motion → valve")
    cuboid(b, 760, 185, 235, 115, 18, BLUE_L, "#C7DDF3", BLUE,
           "Forward dynamics", "valve + state → increment")
    b += [line(680, 242, 748, 242, LINE, 3, arrow=True)]

    b += [rect(365, 465, 690, 270, "white", TEAL, 2.5, 9),
          text(400, 505, "PPO–ROM", 21, TEAL, "bold", "start")]
    # Stacked input and recurrent slabs make the model visibly three-dimensional.
    for i, (lab, col, side) in enumerate([("history",VIOLET_L,"#D8CBEA"),
                                          ("GRU",BLUE_L,"#C7DDF3"),
                                          ("physics",TEAL_L,"#BFE8DF")]):
        cuboid(b, 450+i*190, 565, 135, 100, 20, col, side,
               [VIOLET,BLUE,TEAL][i], lab, ["6×24","2×128","bounded update"][i], 19, 14)
        if i:
            b += [line(405+i*190, 615, 438+i*190, 615, LINE, 2.5, arrow=True)]

    b += [line(305, 372, 420, 242, LINE, 3, arrow=True),
          line(305, 440, 438, 615, LINE, 3, arrow=True)]
    image_panel(b, IMG / "unimog_mulag_iso.png", 1130, 190, 370, 360, TEAL,
                "shared USD / PhysX plant")
    cuboid(b, 1150, 610, 320, 100, 16, RED_L, "#F0C5C2", RED,
           "Bounded position target", "identical plant interface")
    b += [line(995, 242, 1120, 330, LINE, 3, arrow=True),
          line(965, 615, 1138, 660, LINE, 3, arrow=True),
          path("M 1310 550 L 1310 805 L 180 805 L 180 480", VIOLET, 2.2, "10 7", True),
          text(760, 835, "measured state and pressure feedback", 18, VIOLET)]
    save("models_overview", W, H, b)


def nn_architectures():
    W, H = 1550, 930
    b: list[str] = []
    b += [text(45, 45, "(a) Residual MLP used by ID and FD", 25, INK, "bold", "start"),
          rect(35, 70, 1480, 350, "white", BLUE, 2.4, 8)]
    specs = [(100, 190, 100, 125, VIOLET_L, "#D8CBEA", VIOLET, "input", "28 / 20"),
             (315, 165, 120, 175, BLUE_L, "#C7DDF3", BLUE, "projection", "512"),
             (565, 125, 150, 255, BLUE_L, "#C7DDF3", BLUE, "residual", "512"),
             (810, 125, 150, 255, BLUE_L, "#C7DDF3", BLUE, "residual", "512"),
             (1055, 150, 130, 205, BLUE_L, "#C7DDF3", BLUE, "×4 blocks", "LN · SiLU · drop"),
             (1320, 190, 105, 125, TEAL_L, "#BFE8DF", TEAL, "output", "4 / 12")]
    for x,y,w,h,fill,side,col,ttl,sub in specs:
        cuboid(b,x,y,w,h,22,fill,side,col,ttl,sub,18,14)
    for x1,x2 in [(200,303),(435,553),(715,798),(960,1043),(1185,1308)]:
        b += [line(x1, 252, x2, 252, LINE, 2.8, arrow=True)]
    b += [path("M 585 122 C 690 70 945 70 1075 145", TEAL, 2.4, "10 6", True),
          text(820, 88, "identity skip", 17, TEAL, "bold")]

    b += [text(45, 495, "(b) Recurrent encoder used by the ROM", 25, INK, "bold", "start"),
          rect(35, 520, 1480, 370, "white", TEAL, 2.4, 8)]
    # Three feature slabs, then recurrent 3-D units and three output heads.
    for i, (lab, col, side) in enumerate([("state",BLUE_L,"#C7DDF3"),
                                          ("previous valve",OCHRE_L,"#F1D593"),
                                          ("command",VIOLET_L,"#D8CBEA")]):
        cuboid(b, 100+i*22, 635-i*22, 210, 95, 24, col, side,
               [BLUE,OCHRE,VIOLET][i], lab, "history feature plane", 18, 13)
    b += [text(105, 790, "state", 15, BLUE, "bold", "start"),
          text(175, 790, "· previous valve", 15, OCHRE, "bold", "start"),
          text(315, 790, "· command", 15, VIOLET, "bold", "start")]
    b += [line(365, 680, 465, 680, LINE, 2.8, arrow=True)]
    for i, lab in enumerate(["t−5", "…", "t"]):
        cuboid(b, 485+i*180, 620, 100, 125, 20, TEAL_L, "#BFE8DF", TEAL,
               "GRU", lab, 19, 15)
        if i:
            b += [line(425+i*180, 680, 473+i*180, 680, TEAL, 2.6, arrow=True)]
    b += [text(680, 790, "2 layers × 128 hidden units", 17, TEAL, "bold"),
          line(945, 680, 1045, 680, LINE, 2.8, arrow=True)]
    for i, (lab, col, fill, side) in enumerate([("actuation",BLUE,BLUE_L,"#C7DDF3"),
                                                ("residual",VIOLET,VIOLET_L,"#D8CBEA"),
                                                ("load",OCHRE,OCHRE_L,"#F1D593")]):
        cuboid(b, 1065, 565+i*100, 290, 70, 16, fill, side, col,
               lab + " head", "dense output", 18, 13)
        b += [line(955, 680, 1053, 600+i*100, col, 2.0, arrow=True)]
    save("models_nn_architectures", W, H, b)


def rom_pipeline():
    W, H = 1600, 1000
    b: list[str] = [text(40,42,"Command-conditioned reduced-order actuator",28,INK,"bold","start"),
                    text(40,78,"neural proposal above; explicit hydraulic realization below",18,MUTED,"normal","start"),
                    rect(30,105,1540,390,"white",BLUE,2.3,8)]
    # History is shown as a real temporal stack rather than one generic input box.
    for i,(lab,col,side,stroke) in enumerate([("state + Δp",BLUE_L,"#C7DDF3",BLUE),
                                              ("previous valve",OCHRE_L,"#F1D593",OCHRE),
                                              ("desired command",VIOLET_L,"#D8CBEA",VIOLET)]):
        cuboid(b,70+i*24,250-i*24,270,125,28,col,side,stroke,lab,"6-step causal history",19,14)
    b += [text(205,425,"Hₖ: 6 × 24 samples · 0.25 s",18,VIOLET,"bold"),
          line(390,305,475,305,LINE,3,arrow=True)]
    cuboid(b,500,190,220,235,30,BLUE_L,"#C7DDF3",BLUE,"GRU encoder","2 layers × 128\nlatent hₖ",22,16)
    for i,(lab,col,fill,side) in enumerate([("actuation head",BLUE,BLUE_L,"#C7DDF3"),
                                            ("residual head",VIOLET,VIOLET_L,"#D8CBEA"),
                                            ("load head",OCHRE,OCHRE_L,"#F1D593")]):
        cuboid(b,820,145+i*115,270,78,18,fill,side,col,lab,"dense output",18,13)
        b += [line(748,307,808,184+i*115,col,2.3,arrow=True)]
    cuboid(b,1210,205,280,165,26,TEAL_L,"#BFE8DF",TEAL,
           "physical proposal","valve · residuals · load",21,15)
    for yy,col in [(184,BLUE),(299,VIOLET),(414,OCHRE)]:
        b += [path(f"M 1110 {yy} L 1160 {yy} L 1160 288 L 1198 288",col,2.2,None,True)]

    b += [rect(30,535,1540,420,"white",TEAL,2.3,8),
          text(55,572,"EXPLICIT HYDRAULIC UPDATE",18,TEAL,"bold","start")]
    cuboid(b,70,650,300,180,28,TEAL_L,"#BFE8DF",TEAL,
           "state update","orifice flow → pressure\nforce → velocity → position",21,15)
    href = image_href(IMG / "electro_hydraulic_actuator_channel_render.png")
    b += [rect(430,590,1090,300,"white",TEAL,2.0,7),
          f'<image x="445" y="605" width="1060" height="270" '
          f'preserveAspectRatio="xMidYMid meet" href="{href}"/>',
          line(1490,288,1490,515,LINE,2.8),
          path("M 1490 515 L 220 515 L 220 638",LINE,3,None,True),
          line(398,740,420,740,TEAL,3,arrow=True),
          rect(500,895,245,42,BLUE_L,BLUE,1.5,20),text(622,922,"pressure increment  Δp̂",16,BLUE,"bold"),
          rect(775,895,270,42,OCHRE_L,OCHRE,1.5,20),text(910,922,"velocity increment  Δẋ̂",16,OCHRE,"bold"),
          rect(1075,895,245,42,TEAL_L,TEAL,1.5,20),text(1197,922,"position increment  Δx̂",16,TEAL,"bold"),
          path("M 1320 915 L 1450 915 L 1450 970 L 210 970 L 210 425",TEAL,2.3,"10 7",True),
          text(820,985,"updated pressure and motion become the next causal history",16,TEAL)]
    save("models_rom_pipeline", W, H, b)


if __name__ == "__main__":
    actuation_chain()
    workflow()
    robust_adaptive()
    topology()
    hydraulic_schematic()
    actuator_channel()
    control_architecture()
    models_overview()
    acceptance_flow()
    nn_architectures()
    rom_pipeline()
