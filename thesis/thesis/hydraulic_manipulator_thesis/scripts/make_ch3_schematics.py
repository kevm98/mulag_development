#!/usr/bin/env python3
"""Build the annotated Chapter-3 schematics as vector PDFs.

  fme500_topology_annotated.pdf  Fig 3.1  render + joint/link/cylinder callouts
                                          + abstract kinematic chain
  fme500_cad_iso_annot.pdf       Fig 3.2a isometric view with actuator callouts
  fme500_hydraulic_schematic.pdf Fig 3.4  four-actuator hydraulic architecture

Run:  python3 scripts/make_ch3_schematics.py
"""
import base64
import json
from pathlib import Path

import cairosvg

BASE = Path(Path.home() / "rrlab_mulag/thesis/thesis/hydraulic_manipulator_thesis")
IMG = BASE / "images"
DRAWIO = BASE / "drawio"
DRAWIO.mkdir(exist_ok=True)

# shared palette (matches the rest of the thesis)
ORANGE, BLUE, TEAL, PURPLE, RED, GRAY = (
    "#C97816", "#2F5DA8", "#16847A", "#7A52A8", "#B64B42", "#69737D")
INK, MUTED = "#1F2933", "#66727E"
FONT = "DejaVu Sans, Arial, sans-serif"

LINKS = [
    ("L0", "Front frame", "carrier mount"),
    ("L1", "Slew column", "Drehzapfen"),
    ("L2", "Inner boom", "Ausleger I"),
    ("L3", "Outer boom", "Ausleger II"),
    ("L4", "Head carrier", "Messerkopf-Schwenk"),
    ("L5", "Mowing head", "Messerkopf"),
]
# cylinder i actuates joint i
CYL = [
    ("C1", "slew"), ("C2", "lift"), ("C3", "knuckle"), ("C4", "head pivot"),
]


def embed(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def txt(x, y, s, size=13, fill=INK, weight="normal", anchor="middle",
        style="normal"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" '
            f'font-style="{style}">{s}</text>')


def marker(mid, color):
    return (f'<marker id="{mid}" markerWidth="9" markerHeight="9" refX="9" '
            f'refY="4.5" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L9,4.5 L0,9 z" fill="{color}"/></marker>')


# =====================================================================
# Figure 3.1 — annotated kinematic topology
# =====================================================================
def topology():
    meta = json.loads((IMG / "fme500_joint_coords.json").read_text())
    P = meta["points"]
    W, H = 1540, 800
    # render placement
    rx, ry, rw = 30, 70, 980
    rh = rw * meta["resolution"][1] / meta["resolution"][0]

    def px(tag):
        nx, ny = P[tag]
        return rx + nx * rw, ry + ny * rh

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}"><defs>',
         marker("aG", GRAY), marker("aO", ORANGE), marker("aB", BLUE),
         '</defs>', f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
         f'<image x="{rx}" y="{ry}" width="{rw}" height="{rh}" '
         f'href="{embed(IMG / "fme500_topology.png")}"/>',
         txt(30, 40, "(a)  Mechanism with joint and actuator labels", 16,
             INK, "bold", "start")]

    # ---- link labels: (anchor on the body) -> (label in clear space)
    link_at = {
        "L0": ((0.88, 0.80), (0.93, 0.95)),
        "L1": ((0.80, 0.70), (0.95, 0.60)),
        "L2": ((0.62, 0.33), (0.80, 0.12)),
        "L3": ((0.30, 0.30), (0.12, 0.12)),
        "L4": ((0.20, 0.68), (0.44, 0.70)),
        "L5": ((0.14, 0.86), (0.40, 0.92)),
    }
    for tag, name, german in LINKS:
        (ax, ay), (lx_, ly_) = link_at[tag]
        ax, ay = rx + ax * rw, ry + ay * rh
        x, y = rx + lx_ * rw, ry + ly_ * rh
        s.append(f'<line x1="{x}" y1="{y}" x2="{ax}" y2="{ay}" stroke="{GRAY}" '
                 f'stroke-width="1.2" marker-end="url(#aG)"/>')
        s.append(f'<circle cx="{ax}" cy="{ay}" r="3.4" fill="{GRAY}"/>')
        s.append(f'<rect x="{x-62}" y="{y-19}" width="124" height="38" rx="7" '
                 f'fill="#FFFFFF" stroke="{GRAY}" stroke-width="1.4"/>')
        s.append(txt(x, y - 3, f"{tag} &#183; {name}", 12, INK, "bold"))
        s.append(txt(x, y + 12, german, 10.5, MUTED))

    # ---- joints
    for tag in ("J1", "J2", "J3", "J4", "J5"):
        if tag not in P:
            continue
        x, y = px(tag)
        passive = tag == "J5"
        col = PURPLE if passive else BLUE
        fill = "#FFFFFF" if passive else "#EAF0FA"
        dash = ' stroke-dasharray="5,3"' if passive else ""
        s.append(f'<circle cx="{x}" cy="{y}" r="14" fill="{fill}" '
                 f'stroke="{col}" stroke-width="2.6"{dash}/>')
        s.append(f'<circle cx="{x}" cy="{y}" r="3.6" fill="{col}"/>')
    joint_lab = {"J1": (52, 40), "J2": (52, -18), "J3": (0, -46),
                 "J4": (-86, -30), "J5": (-92, 22)}
    for tag, (dx, dy) in joint_lab.items():
        if tag not in P:
            continue
        x, y = px(tag)
        col = PURPLE if tag == "J5" else BLUE
        s.append(f'<line x1="{x + dx*0.30}" y1="{y + dy*0.30}" '
                 f'x2="{x + dx*0.72}" y2="{y + dy*0.72}" stroke="{col}" '
                 f'stroke-width="1.3"/>')
        s.append(f'<circle cx="{x+dx}" cy="{y+dy}" r="14" fill="#FFFFFF" '
                 f'stroke="{col}" stroke-width="1.6"/>')
        s.append(txt(x + dx, y + dy + 5, tag, 13, col, "bold"))

    # ---- cylinders (anchor at the cylinder, pill in clear space)
    cyl_lab = {"C1": (0.60, 0.82), "C2": (0.83, 0.42),
               "C3": (0.34, 0.09), "C4": (0.05, 0.47)}
    for (tag, role) in CYL:
        if tag not in P:
            continue
        x, y = px(tag)
        lx_, ly_ = cyl_lab[tag]
        lx, ly = rx + lx_ * rw, ry + ly_ * rh
        s.append(f'<line x1="{lx}" y1="{ly}" x2="{x}" y2="{y}" stroke="{ORANGE}" '
                 f'stroke-width="1.6" marker-end="url(#aO)"/>')
        s.append(f'<rect x="{lx-48}" y="{ly-14}" width="96" height="28" '
                 f'rx="14" fill="#FFF4E5" stroke="{ORANGE}" stroke-width="1.8"/>')
        s.append(txt(lx, ly + 5, f"{tag} {role}", 11.5, ORANGE, "bold"))

    # ---- legend
    ly = ry + rh + 26
    s.append(f'<circle cx="{rx+16}" cy="{ly}" r="9" fill="#EAF0FA" '
             f'stroke="{BLUE}" stroke-width="2.2"/>')
    s.append(txt(rx + 32, ly + 5, "actuated revolute joint", 12, MUTED, "normal", "start"))
    s.append(f'<circle cx="{rx+236}" cy="{ly}" r="9" fill="#FFFFFF" '
             f'stroke="{PURPLE}" stroke-width="2.2" stroke-dasharray="5,3"/>')
    s.append(txt(rx + 252, ly + 5, "passive joint (swivel-link coupled)", 12,
                 MUTED, "normal", "start"))
    s.append(f'<rect x="{rx+526}" y="{ly-9}" width="26" height="18" rx="9" '
             f'fill="#FFF4E5" stroke="{ORANGE}" stroke-width="1.6"/>')
    s.append(txt(rx + 562, ly + 5, "hydraulic cylinder", 12, MUTED, "normal", "start"))

    # =============== (b) abstract chain =================
    cx0 = 1090
    s.append(txt(cx0 - 30, 40, "(b)  Kinematic chain", 16, INK, "bold", "start"))
    bx, bw, bh, gap = cx0 + 40, 210, 52, 44
    top = 92
    for i, (tag, name, _) in enumerate(LINKS):
        y = top + i * (bh + gap)
        s.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="{bh}" rx="9" '
                 f'fill="#F1F3F5" stroke="{GRAY}" stroke-width="1.8"/>')
        s.append(txt(bx + bw / 2, y + 22, tag, 13, INK, "bold"))
        s.append(txt(bx + bw / 2, y + 39, name, 11.5, MUTED))
        if i < len(LINKS) - 1:
            jy = y + bh + gap / 2
            passive = i == 4
            col = PURPLE if passive else BLUE
            s.append(f'<line x1="{bx+bw/2}" y1="{y+bh}" x2="{bx+bw/2}" '
                     f'y2="{y+bh+gap}" stroke="{col}" stroke-width="2.2"/>')
            s.append(f'<circle cx="{bx+bw/2}" cy="{jy}" r="13" '
                     f'fill="{"#FFFFFF" if passive else "#EAF0FA"}" '
                     f'stroke="{col}" stroke-width="2.4"'
                     f'{" stroke-dasharray=\'5,3\'" if passive else ""}/>')
            s.append(txt(bx + bw / 2 + 34, jy + 5, f"J{i+1}", 12, col, "bold"))
            if not passive:
                # cylinder bracket on the left
                s.append(f'<path d="M {bx-16} {y+bh-6} L {bx-40} {y+bh-6} '
                         f'L {bx-40} {y+bh+gap+6} L {bx-16} {y+bh+gap+6}" '
                         f'fill="none" stroke="{ORANGE}" stroke-width="2"/>')
                s.append(txt(bx - 74, jy + 5, f"C{i+1}", 12, ORANGE, "bold"))
    s.append(txt(bx + bw / 2, top + 6 * (bh + gap) + 6,
                 "J5 carries no cylinder: the swivel links", 11.5, MUTED))
    s.append(txt(bx + bw / 2, top + 6 * (bh + gap) + 22,
                 "constrain it to the head-carrier motion", 11.5, MUTED))
    s.append("</svg>")
    out = "\n".join(s)
    (DRAWIO / "fig3_1_topology.svg").write_text(out)
    cairosvg.svg2pdf(bytestring=out.encode(),
                     write_to=str(IMG / "fme500_topology_annotated.pdf"))
    print("wrote topology")


# =====================================================================
# Figure 3.2(a) — isometric render with actuator callouts
# =====================================================================
def iso_annotated():
    W, H = 1080, 1120
    rw = 1040
    rh = rw  # square render
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}"><defs>', marker("aO", ORANGE), '</defs>',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
         f'<image x="20" y="20" width="{rw}" height="{rh}" '
         f'href="{embed(IMG / "fme500_cad_iso.png")}"/>']
    # callouts: anchor = actual cylinder centre projected by Blender
    # (fme500_iso_coords.json, world_to_camera_view -- not hand-guessed),
    # label position chosen to keep the pill clear of the mechanism.
    iso_pts = json.loads((IMG / "fme500_iso_coords.json").read_text())["points"]
    label_at = {
        "C1": (0.90, 0.30),   # slew cylinder
        "C2": (0.86, 0.10),   # lift
        "C3": (0.06, 0.30),   # knuckle
        "C4": (0.03, 0.76),   # head pivot
    }
    call = [(tag, *iso_pts[tag], *label_at[tag]) for tag in ("C1", "C2", "C3", "C4")]
    for tag, px_, py_, lx_, ly_ in call:
        x, y = 20 + px_ * rw, 20 + py_ * rh
        lx, ly = 20 + lx_ * rw, 20 + ly_ * rh
        s.append(f'<line x1="{lx}" y1="{ly}" x2="{x}" y2="{y}" stroke="{ORANGE}" '
                 f'stroke-width="1.8" marker-end="url(#aO)"/>')
        s.append(f'<rect x="{lx-46}" y="{ly-16}" width="92" height="32" rx="16" '
                 f'fill="#FFF4E5" stroke="{ORANGE}" stroke-width="2"/>')
        s.append(txt(lx, ly + 5, tag, 15, ORANGE, "bold"))
    s.append(f'<rect x="20" y="{H-72}" width="{rw}" height="52" rx="10" '
             f'fill="#FFF9F0" stroke="{ORANGE}" stroke-width="1.5"/>')
    s.append(f'<rect x="44" y="{H-56}" width="30" height="20" rx="10" '
             f'fill="{ORANGE}"/>')
    s.append(txt(88, H - 40,
                 "highlighted: the four double-acting hydraulic cylinders "
                 "(C1&#8211;C4) that actuate joints J1&#8211;J4",
                 14, INK, "normal", "start"))
    s.append("</svg>")
    out = "\n".join(s)
    (DRAWIO / "fig3_2a_iso.svg").write_text(out)
    cairosvg.svg2pdf(bytestring=out.encode(),
                     write_to=str(IMG / "fme500_cad_iso_annot.pdf"))
    print("wrote iso")


# =====================================================================
# Figure 3.4 - hydraulic architecture: four actuator channels
# =====================================================================
def hydraulic_schematic():
    W, H = 1560, 1000
    RED, BLU = "#C0392B", "#2F5DA8"
    rows = [
        ("C1", "J1", "Slew", "drehzapfen", "Drehzapfen"),
        ("C2", "J2", "Lift", "ausleger_i", "Ausleger I"),
        ("C3", "J3", "Knuckle", "ausleger_ii", "Ausleger II"),
        ("C4", "J4", "Head pivot", "messerkopf_schwenk", "Messerkopf-Schwenk"),
    ]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}"><defs>', marker("aR", RED), marker("aB", BLU),
         marker("aP", PURPLE), marker("aO", ORANGE), '</defs>',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']

    # ---------------- actuator channels -----------------
    s.append(f'<rect x="24" y="52" width="1000" height="880" rx="16" '
             f'fill="#FCFDFE" stroke="#B8C1CA" stroke-width="1.6"/>')
    s.append(txt(48, 40, "Four independent electro-hydraulic actuator channels",
                 16, INK, "bold", "start"))
    y0, dy = 108, 196
    for i, (cyl, joint, role, img, german) in enumerate(rows):
        y = y0 + i * dy
        # driven link thumbnail
        s.append(f'<image x="52" y="{y-58}" width="118" height="118" '
                 f'href="{embed(IMG / f"links/link_{img}.png")}"/>')
        s.append(txt(111, y + 72, f"{joint} &#183; {role}", 12, INK, "bold"))
        s.append(txt(111, y + 88, german, 10.5, MUTED))
        # cylinder symbol
        cx0, cw, ch = 210, 250, 62
        s.append(f'<rect x="{cx0}" y="{y-ch/2}" width="{cw}" height="{ch}" '
                 f'rx="4" fill="#FFF4E5" stroke="{ORANGE}" stroke-width="2.4"/>')
        pist = cx0 + cw * 0.46
        s.append(f'<rect x="{pist}" y="{y-ch/2}" width="13" height="{ch}" '
                 f'fill="{ORANGE}"/>')
        s.append(f'<line x1="{pist+13}" y1="{y}" x2="{cx0+cw+42}" y2="{y}" '
                 f'stroke="{ORANGE}" stroke-width="7"/>')
        s.append(txt(cx0 + 52, y + 5, "A", 13, RED, "bold"))
        s.append(txt(cx0 + cw - 40, y + 5, "B", 13, BLU, "bold"))
        s.append(txt(cx0 + cw / 2, y - ch / 2 - 12, cyl, 14, ORANGE, "bold"))
        # 4/3 valve symbol (three stacked squares)
        vx, vy, vs_ = 570, y - 33, 22
        for k in range(3):
            fillc = "#EEF4FF" if k == 1 else "#FFFFFF"
            s.append(f'<rect x="{vx + k*vs_*2}" y="{vy}" width="{vs_*2}" '
                     f'height="{vs_*3}" fill="{fillc}" stroke="{BLU}" '
                     f'stroke-width="1.6"/>')
        s.append(f'<line x1="{vx+10}" y1="{vy+40}" x2="{vx+34}" y2="{vy+26}" '
                 f'stroke="{BLU}" stroke-width="1.6" marker-end="url(#aB)"/>')
        s.append(f'<line x1="{vx+56}" y1="{vy+24}" x2="{vx+78}" y2="{vy+42}" '
                 f'stroke="{BLU}" stroke-width="1.4"/>')
        s.append(f'<line x1="{vx+56}" y1="{vy+42}" x2="{vx+78}" y2="{vy+24}" '
                 f'stroke="{BLU}" stroke-width="1.4"/>')
        s.append(f'<line x1="{vx+100}" y1="{vy+26}" x2="{vx+124}" y2="{vy+40}" '
                 f'stroke="{BLU}" stroke-width="1.6" marker-end="url(#aB)"/>')
        s.append(txt(vx + 66, vy + 84, "4/3 proportional valve", 11, MUTED))
        # command input
        s.append(f'<line x1="{vx+66}" y1="{vy-30}" x2="{vx+66}" y2="{vy-4}" '
                 f'stroke="{PURPLE}" stroke-width="2" marker-end="url(#aP)"/>')
        s.append(txt(vx + 66, vy - 36, f"u{i+1}", 12, PURPLE, "bold"))
        # cylinder <-> valve lines
        s.append(f'<path d="M {cx0} {y-16} L {cx0-16} {y-16} L {cx0-16} {y+80} '
                 f'L {vx+22} {y+80} L {vx+22} {vy+66}" fill="none" '
                 f'stroke="{RED}" stroke-width="2"/>')
        s.append(f'<path d="M {cx0+cw} {y+18} L {vx+110} {y+18} L {vx+110} '
                 f'{vy+66}" fill="none" stroke="{BLU}" stroke-width="2"/>')
        # rails
        s.append(f'<line x1="{vx+22}" y1="{vy+66}" x2="{vx+22}" y2="{vy+96}" '
                 f'stroke="{RED}" stroke-width="2"/>')
        s.append(f'<line x1="{vx+110}" y1="{vy+66}" x2="{vx+110}" y2="{vy+104}" '
                 f'stroke="{BLU}" stroke-width="2"/>')
        s.append(f'<line x1="{vx+22}" y1="{vy+96}" x2="1090" y2="{vy+96}" '
                 f'stroke="{RED}" stroke-width="2"/>')
        s.append(f'<line x1="{vx+110}" y1="{vy+104}" x2="1150" y2="{vy+104}" '
                 f'stroke="{BLU}" stroke-width="2"/>')
        s.append(f'<circle cx="1090" cy="{vy+96}" r="4.5" fill="{RED}"/>')
        s.append(f'<circle cx="1150" cy="{vy+104}" r="4.5" fill="{BLU}"/>')

    # passive joint note
    s.append(f'<rect x="52" y="890" width="950" height="30" rx="8" '
             f'fill="#F6F0FB" stroke="{PURPLE}" stroke-width="1.4"/>')
    s.append(txt(70, 910, "J5 (mowing-head rotation) carries no cylinder: the "
                 "swivel-link pair couples it to the head carrier, so four "
                 "cylinders drive five joints.", 12.5, PURPLE, "normal", "start"))

    # ---------------- power unit -----------------
    s.append(f'<rect x="1060" y="52" width="476" height="880" rx="16" '
             f'fill="#F7FAFF" stroke="{BLU}" stroke-width="1.8"/>')
    s.append(txt(1084, 40, "Hydraulic power unit", 16, BLU, "bold", "start"))
    # common rails
    s.append(f'<line x1="1090" y1="120" x2="1090" y2="820" stroke="{RED}" '
             f'stroke-width="3.4"/>')
    s.append(f'<line x1="1150" y1="120" x2="1150" y2="860" stroke="{BLU}" '
             f'stroke-width="3.4"/>')
    s.append(txt(1090, 112, "P", 14, RED, "bold"))
    s.append(txt(1150, 112, "T", 14, BLU, "bold"))

    def label(x, y, title, sub, col, anchor="start"):
        s.append(txt(x, y - 2, title, 13, INK, "bold", anchor))
        s.append(txt(x, y + 15, sub, 10.5, MUTED, "normal", anchor))

    # ---- ISO 1219-style component glyphs (not generic boxes) ----
    def accumulator(cx, cy, col):
        # gas-charged bottle: circle piston chamber on a rectangular shell
        s.append(f'<rect x="{cx-17}" y="{cy-36}" width="34" height="72" rx="17" '
                 f'fill="#FFFFFF" stroke="{col}" stroke-width="2.2"/>')
        s.append(f'<line x1="{cx-17}" y1="{cy}" x2="{cx+17}" y2="{cy}" '
                 f'stroke="{col}" stroke-width="2"/>')
        for k in range(3):
            yy = cy - 24 + k * 12
            s.append(f'<line x1="{cx-9}" y1="{yy}" x2="{cx+9}" y2="{yy}" '
                     f'stroke="{col}" stroke-width="1.3"/>')
        s.append(f'<line x1="{cx}" y1="{cy+36}" x2="{cx}" y2="{cy+52}" '
                 f'stroke="{col}" stroke-width="2"/>')

    def relief_valve(cx, cy, col):
        # square envelope, spring-loaded ball-and-seat with adjustable arrow
        s.append(f'<rect x="{cx-26}" y="{cy-22}" width="52" height="44" '
                 f'fill="#FFFFFF" stroke="{col}" stroke-width="2.2"/>')
        s.append(f'<line x1="{cx-18}" y1="{cy+12}" x2="{cx+4}" y2="{cy-10}" '
                 f'stroke="{col}" stroke-width="1.8"/>')
        s.append(f'<line x1="{cx+4}" y1="{cy-10}" x2="{cx-2}" y2="{cy-16}" '
                 f'stroke="{col}" stroke-width="1.8"/>')
        s.append(f'<line x1="{cx+4}" y1="{cy-10}" x2="{cx+10}" y2="{cy-4}" '
                 f'stroke="{col}" stroke-width="1.8"/>')
        # spring
        pts = " ".join(f'{cx-4+((-1)**k)*6},{cy-4-k*5}' for k in range(6))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                 f'stroke-width="1.4"/>')
        s.append(f'<line x1="{cx}" y1="{cy-22}" x2="{cx}" y2="{cy-34}" '
                 f'stroke="{col}" stroke-width="1.8"/>')
        s.append(f'<line x1="{cx-8}" y1="{cy-34}" x2="{cx+8}" y2="{cy-34}" '
                 f'stroke="{col}" stroke-width="1.8"/>')

    def gauge(cx, cy, col):
        s.append(f'<circle cx="{cx}" cy="{cy}" r="24" fill="#FFFFFF" '
                 f'stroke="{col}" stroke-width="2.2"/>')
        s.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+13}" y2="{cy-13}" '
                 f'stroke="{col}" stroke-width="2"/>')
        for a in range(-60, 90, 30):
            import math as _m
            x1 = cx + 17 * _m.cos(_m.radians(a))
            y1 = cy - 17 * _m.sin(_m.radians(a))
            x2 = cx + 22 * _m.cos(_m.radians(a))
            y2 = cy - 22 * _m.sin(_m.radians(a))
            s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                     f'y2="{y2:.1f}" stroke="{col}" stroke-width="1.2"/>')

    def var_pump(cx, cy, col):
        s.append(f'<circle cx="{cx}" cy="{cy}" r="26" fill="#FFFFFF" '
                 f'stroke="{col}" stroke-width="2.4"/>')
        s.append(f'<polygon points="{cx-11},{cy-10} {cx-11},{cy+10} '
                 f'{cx+11},{cy}" fill="{col}"/>')
        s.append(f'<line x1="{cx-30}" y1="{cy-30}" x2="{cx-14}" y2="{cy-14}" '
                 f'stroke="{col}" stroke-width="1.8" marker-end="url(#aB)"/>')

    def prime_mover(cx, cy, col):
        s.append(f'<circle cx="{cx}" cy="{cy}" r="26" fill="#FFFFFF" '
                 f'stroke="{col}" stroke-width="2.4"/>')
        s.append(txt(cx, cy + 6, "M", 20, col, "bold"))

    def filt(cx, cy, col):
        s.append(f'<polygon points="{cx},{cy-24} {cx+22},{cy} {cx},{cy+24} '
                 f'{cx-22},{cy}" fill="#FFFFFF" stroke="{col}" stroke-width="2.2"/>')
        for k in range(-2, 3):
            s.append(f'<line x1="{cx-14}" y1="{cy+4*k-6}" x2="{cx+14}" '
                     f'y2="{cy+4*k+6}" stroke="{col}" stroke-width="1.1"/>')

    def reservoir(cx, cy, col):
        s.append(f'<path d="M {cx-40} {cy-16} L {cx-40} {cy+16} L {cx+40} '
                 f'{cy+16} L {cx+40} {cy-16}" fill="#FFFFFF" stroke="{col}" '
                 f'stroke-width="2.2"/>')
        s.append(f'<line x1="{cx-40}" y1="{cy-16}" x2="{cx+40}" y2="{cy-16}" '
                 f'stroke="{col}" stroke-width="1.2" stroke-dasharray="4,3"/>')

    cxg = 1210  # glyph column
    lxg = 1250  # label column

    accumulator(cxg, 172, TEAL)
    label(lxg, 172, "Accumulator", "gas-charged, damps ripple", TEAL)
    s.append(f'<line x1="{cxg}" y1="172" x2="1090" y2="172" stroke="{RED}" stroke-width="2"/>')

    relief_valve(cxg, 288, RED)
    label(lxg, 288, "Pressure-relief valve", "caps supply pressure", RED)
    s.append(f'<line x1="{cxg}" y1="288" x2="1090" y2="288" stroke="{RED}" stroke-width="2"/>')
    s.append(f'<path d="M {cxg+26} {318} L {cxg+26} {340} L 1150 340" fill="none" stroke="{BLU}" stroke-width="2"/>')

    gauge(cxg, 400, PURPLE)
    label(lxg, 400, "Pressure sensor", "supply feedback", PURPLE)
    s.append(f'<line x1="{cxg}" y1="400" x2="1090" y2="400" stroke="{PURPLE}" stroke-width="1.8" stroke-dasharray="6,4"/>')

    var_pump(cxg, 520, BLU)
    label(lxg, 520, "Variable-speed pump", "Qp = Dp &#183; &#969;", BLU)
    s.append(f'<path d="M {cxg} 546 L {cxg} 560 L 1090 560" fill="none" stroke="{RED}" stroke-width="2.4" marker-end="url(#aR)"/>')

    prime_mover(cxg, 636, GRAY)
    label(lxg, 636, "Prime mover", "carrier PTO", GRAY)
    s.append(f'<line x1="{cxg}" y1="610" x2="{cxg}" y2="546" stroke="{GRAY}" stroke-width="2"/>')

    filt(cxg, 736, GRAY)
    label(lxg, 736, "Suction filter", "return conditioning", GRAY)

    reservoir(cxg, 826, BLU)
    label(lxg, 826, "Reservoir", "tank", BLU)
    s.append(f'<path d="M {cxg} 810 L {cxg} 760" fill="none" stroke="{GRAY}" stroke-width="2"/>')
    s.append(f'<path d="M {cxg+30} 826 L 1150 826 L 1150 860" fill="none" stroke="{BLU}" stroke-width="2.4"/>')
    s.append(f'<path d="M {cxg-30} 826 L {cxg-30} 700" fill="none" stroke="{GRAY}" stroke-width="1.4" stroke-dasharray="3,3"/>')

    # legend
    leg = [("pressure line P", RED), ("return line T", BLU),
           ("valve command", PURPLE), ("cylinder body", ORANGE)]
    for i, (lab, col) in enumerate(leg):
        x = 60 + i * 250
        s.append(f'<line x1="{x}" y1="962" x2="{x+34}" y2="962" stroke="{col}" '
                 f'stroke-width="3.4"/>')
        s.append(txt(x + 44, 967, lab, 12, MUTED, "normal", "start"))
    s.append("</svg>")
    out = "\n".join(s)
    (DRAWIO / "fig3_4_hydraulic.svg").write_text(out)
    cairosvg.svg2pdf(bytestring=out.encode(),
                     write_to=str(IMG / "fme500_hydraulic_schematic.pdf"))
    print("wrote hydraulic schematic")


# =====================================================================
# Figure 3.5 - one electro-hydraulic actuator channel, mechanical +
# hydraulic interface, notation matched exactly to Section 3.4's
# equations (p_1, p_2, A_1, A_2, F_h, F_fric, Q_1, Q_2, x_v).
# =====================================================================
def actuator_channel():
    W, H = 1540, 900
    BLU, RED, TEAL2, GRAY2 = BLUE, "#B64B42", TEAL, "#69737D"
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}"><defs>', marker("aG2", GRAY2), marker("aB2", BLU),
         marker("aR2", RED), marker("aT2", TEAL2), marker("aP2", PURPLE),
         '</defs>', f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']

    def band(x, y, w, h, text1, text2, col):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                 f'fill="#FFFFFF" stroke="{col}" stroke-width="1.8"/>')
        s.append(txt(x + 14, y + 22, text1, 13.5, col, "bold", "start"))
        s.append(txt(x + 14, y + 40, text2, 11, MUTED, "normal", "start"))

    band(20, 16, 300, 56, "UPPER LEVEL", "mechanical interface", BLU)
    band(20, 388, 300, 56, "LOWER LEVEL", "hydraulic interface (valve)", TEAL2)

    # ================= MECHANICAL ROW (y ~ 230) =================
    my = 230
    # Link k
    s.append(f'<rect x="60" y="{my-40}" width="150" height="80" rx="8" '
             f'fill="#F1F3F5" stroke="{GRAY2}" stroke-width="2"/>')
    s.append(txt(135, my - 2, "Link k", 15, INK, "bold"))
    s.append(txt(135, my + 16, "rigid body", 11, MUTED))
    # revolute joint 1
    s.append(f'<line x1="210" y1="{my}" x2="260" y2="{my}" stroke="{GRAY2}" stroke-width="2.4"/>')
    s.append(f'<circle cx="278" cy="{my}" r="16" fill="#FFFFFF" stroke="{GRAY2}" stroke-width="2.2"/>')
    s.append(f'<circle cx="278" cy="{my}" r="3.2" fill="{GRAY2}"/>')
    s.append(txt(278, my - 28, "Revolute", 11, MUTED))
    s.append(txt(278, my - 14, "joint", 11, MUTED))
    s.append(f'<line x1="294" y1="{my}" x2="330" y2="{my}" stroke="{GRAY2}" stroke-width="2.4"/>')

    # cylinder barrel with chambers 1 (blue) and 2 (red)
    bx, bw, bh = 330, 320, 90
    s.append(f'<rect x="{bx}" y="{my-bh/2}" width="{bw}" height="{bh}" '
             f'fill="none" stroke="{GRAY2}" stroke-width="2.4"/>')
    piston_x = bx + bw * 0.55
    s.append(f'<rect x="{bx+3}" y="{my-bh/2+3}" width="{piston_x-bx-3}" '
             f'height="{bh-6}" fill="{BLU}" fill-opacity="0.14"/>')
    s.append(f'<rect x="{piston_x}" y="{my-bh/2+3}" width="{bx+bw-piston_x-3}" '
             f'height="{bh-6}" fill="{RED}" fill-opacity="0.14"/>')
    s.append(f'<rect x="{piston_x-6}" y="{my-bh/2}" width="12" height="{bh}" fill="{GRAY2}"/>')
    s.append(txt(bx + (piston_x - bx) / 2, my - 4, "Chamber 1", 12.5, BLU, "bold"))
    s.append(txt(bx + (piston_x - bx) / 2, my + 14, f"(area A&#8321;)", 10.5, MUTED))
    s.append(txt(piston_x + (bx + bw - piston_x) / 2, my - 4, "Chamber 2", 12.5, RED, "bold"))
    s.append(txt(piston_x + (bx + bw - piston_x) / 2, my + 14, f"(area A&#8322;)", 10.5, MUTED))
    # stroke dimension
    s.append(f'<line x1="{bx+10}" y1="{my-bh/2-22}" x2="{bx+bw-10}" y2="{my-bh/2-22}" '
             f'stroke="{INK}" stroke-width="1.2" marker-end="url(#aG2)"/>')
    s.append(f'<line x1="{bx+bw-10}" y1="{my-bh/2-22}" x2="{bx+10}" y2="{my-bh/2-22}" '
             f'stroke="{INK}" stroke-width="1.2" marker-end="url(#aG2)"/>')
    s.append(txt(bx + bw / 2, my - bh / 2 - 30, "x (stroke)", 12, INK, "bold"))
    s.append(f'<line x1="{bx+10}" y1="{my-bh/2-34}" x2="{bx+10}" y2="{my-bh/2-14}" stroke="{GRAY2}" stroke-width="1"/>')
    s.append(f'<line x1="{bx+bw-10}" y1="{my-bh/2-34}" x2="{bx+bw-10}" y2="{my-bh/2-14}" stroke="{GRAY2}" stroke-width="1"/>')
    # barrel attachment tick
    s.append(f'<line x1="{bx+30}" y1="{my-bh/2}" x2="{bx+30}" y2="{my-bh/2-16}" stroke="{GRAY2}" stroke-width="1.6"/>')
    s.append(txt(bx + 30, my - bh / 2 - 46, "Cylinder barrel", 11, MUTED))
    s.append(txt(bx + 30, my - bh / 2 - 32, "attachment", 11, MUTED))

    # piston rod
    rod_x2 = bx + bw + 150
    s.append(f'<rect x="{bx+bw}" y="{my-6}" width="{rod_x2-bx-bw}" height="12" '
             f'fill="#D9DCE0" stroke="{GRAY2}" stroke-width="1.6"/>')
    s.append(f'<line x1="{bx+bw+40}" y1="{my-bh/2}" x2="{bx+bw+40}" y2="{my-bh/2-16}" stroke="{GRAY2}" stroke-width="1.6"/>')
    s.append(txt(bx + bw + 40, my - bh / 2 - 46, "Piston rod", 11, MUTED))
    s.append(txt(bx + bw + 40, my - bh / 2 - 32, "attachment", 11, MUTED))
    # axial force / friction arrows -- placed well clear of the rod-attachment
    # callout above and the prismatic-joint callout below
    s.append(f'<line x1="{rod_x2-10}" y1="{my-72}" x2="{rod_x2+50}" y2="{my-72}" '
             f'stroke="{TEAL2}" stroke-width="2.2" marker-end="url(#aT2)"/>')
    s.append(txt(rod_x2 + 20, my - 80, "F&#8462; (axial force)", 12, TEAL2, "bold"))
    s.append(f'<line x1="{rod_x2+50}" y1="{my+58}" x2="{rod_x2-10}" y2="{my+58}" '
             f'stroke="{GRAY2}" stroke-width="1.8" marker-end="url(#aG2)"/>')
    s.append(txt(rod_x2 + 20, my + 74, "F_fric(x&#775;) friction", 11.5, MUTED))

    # prismatic joint (sliding block, hatched)
    pj_x = rod_x2
    s.append(f'<rect x="{pj_x}" y="{my-22}" width="60" height="44" fill="#EEF0F2" '
             f'stroke="{GRAY2}" stroke-width="2"/>')
    for k in range(6):
        xx = pj_x + 4 + k * 9
        s.append(f'<line x1="{xx}" y1="{my+18}" x2="{xx-8}" y2="{my+26}" stroke="{GRAY2}" stroke-width="1.1"/>')
    s.append(txt(pj_x + 30, my - 30, "Prismatic joint", 11, MUTED))
    s.append(txt(pj_x + 30, my - 40, "(sliding)", 11, MUTED))
    s.append(f'<line x1="{pj_x+60}" y1="{my}" x2="{pj_x+96}" y2="{my}" stroke="{GRAY2}" stroke-width="2.4"/>')

    # revolute joint 2
    rj2 = pj_x + 112
    s.append(f'<circle cx="{rj2}" cy="{my}" r="16" fill="#FFFFFF" stroke="{GRAY2}" stroke-width="2.2"/>')
    s.append(f'<circle cx="{rj2}" cy="{my}" r="3.2" fill="{GRAY2}"/>')
    s.append(txt(rj2, my - 28, "Revolute", 11, MUTED))
    s.append(txt(rj2, my - 14, "joint", 11, MUTED))
    s.append(f'<line x1="{rj2+16}" y1="{my}" x2="{rj2+52}" y2="{my}" stroke="{GRAY2}" stroke-width="2.4"/>')

    # Link k+1
    lkx = rj2 + 52
    s.append(f'<rect x="{lkx}" y="{my-40}" width="150" height="80" rx="8" '
             f'fill="#F1F3F5" stroke="{GRAY2}" stroke-width="2"/>')
    s.append(txt(lkx + 75, my - 2, "Link k+1", 15, INK, "bold"))
    s.append(txt(lkx + 75, my + 16, "rigid body", 11, MUTED))
    s.append(f'<line x1="{lkx+150}" y1="{my}" x2="{lkx+190}" y2="{my}" '
             f'stroke="{GRAY2}" stroke-width="2.2" marker-end="url(#aG2)"/>')
    s.append(txt(lkx + 200, my - 6, "F&#8343;", 13, INK, "bold", "start"))
    s.append(txt(lkx + 200, my + 12, "(external load)", 10.5, MUTED, "normal", "start"))

    # ================= chamber -> valve port lines =================
    p1x, p2x = bx + (piston_x - bx) / 2, piston_x + (bx + bw - piston_x) / 2
    vy0 = 470
    s.append(f'<path d="M {p1x} {my+bh/2} L {p1x} {vy0}" fill="none" stroke="{BLU}" stroke-width="2.2"/>')
    s.append(f'<path d="M {p2x} {my+bh/2} L {p2x} {vy0}" fill="none" stroke="{RED}" stroke-width="2.2"/>')
    mid_y = (my + bh / 2 + vy0) / 2
    s.append(txt(p1x - 14, mid_y, "p&#8321;, Q&#8321;", 14, BLU, "bold", "end"))
    s.append(txt(p2x + 14, mid_y, "p&#8322;, Q&#8322;", 14, RED, "bold", "start"))

    # ================= 4/3 proportional valve =================
    vx0 = p1x - 80
    vw = (p2x + 80) - vx0
    s.append(f'<rect x="{vx0}" y="{vy0}" width="{vw}" height="90" fill="#FFFFFF" '
             f'stroke="{TEAL2}" stroke-width="2.2"/>')
    third = vw / 3
    for k in (1, 2):
        s.append(f'<line x1="{vx0+third*k}" y1="{vy0}" x2="{vx0+third*k}" '
                 f'y2="{vy0+90}" stroke="{TEAL2}" stroke-width="1.6"/>')
    # arrows in centre cell (crossed, closed-centre) and side cells (through)
    cxm = vx0 + 1.5 * third
    s.append(f'<line x1="{vx0+third+14}" y1="{vy0+16}" x2="{vx0+2*third-14}" y2="{vy0+74}" stroke="{TEAL2}" stroke-width="1.4"/>')
    s.append(f'<line x1="{vx0+third+14}" y1="{vy0+74}" x2="{vx0+2*third-14}" y2="{vy0+16}" stroke="{TEAL2}" stroke-width="1.4"/>')
    s.append(f'<line x1="{vx0+18}" y1="{vy0+74}" x2="{vx0+18}" y2="{vy0+16}" stroke="{TEAL2}" stroke-width="1.4" marker-end="url(#aT2)"/>')
    s.append(f'<line x1="{vx0+third-18}" y1="{vy0+74}" x2="{vx0+third-18}" y2="{vy0+16}" stroke="{TEAL2}" stroke-width="1.4" marker-end="url(#aT2)"/>')
    s.append(f'<line x1="{vx0+2*third+18}" y1="{vy0+74}" x2="{vx0+2*third+18}" y2="{vy0+16}" stroke="{TEAL2}" stroke-width="1.4" marker-end="url(#aT2)"/>')
    s.append(f'<line x1="{vx0+3*third-18}" y1="{vy0+74}" x2="{vx0+3*third-18}" y2="{vy0+16}" stroke="{TEAL2}" stroke-width="1.4" marker-end="url(#aT2)"/>')
    # springs at both ends
    for side in (vx0 - 24, vx0 + vw):
        pts = " ".join(f'{side+((-1)**k)*8+(24 if side>vx0 else 0)},{vy0+18+k*11}' for k in range(6))
    s.append(f'<rect x="{vx0-26}" y="{vy0+10}" width="26" height="70" fill="none" stroke="{TEAL2}" stroke-width="1.4"/>')
    s.append(f'<rect x="{vx0+vw}" y="{vy0+10}" width="26" height="70" fill="none" stroke="{TEAL2}" stroke-width="1.4"/>')
    # port labels A / B
    s.append(f'<circle cx="{p1x}" cy="{vy0}" r="5" fill="#FFFFFF" stroke="{BLU}" stroke-width="1.8"/>')
    s.append(f'<circle cx="{p2x}" cy="{vy0}" r="5" fill="#FFFFFF" stroke="{RED}" stroke-width="1.8"/>')
    s.append(txt(p1x, vy0 - 12, "A", 13, BLU, "bold"))
    s.append(txt(p2x, vy0 - 12, "B", 13, RED, "bold"))
    s.append(txt(vx0 + vw / 2, vy0 + 112, "4/3 proportional directional valve", 12.5, MUTED))
    # command + spool position
    s.append(f'<line x1="{vx0-140}" y1="{vy0+45}" x2="{vx0-30}" y2="{vy0+45}" '
             f'stroke="{PURPLE}" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#aP2)"/>')
    s.append(txt(vx0 - 150, vy0 + 38, "u (command)", 12, PURPLE, "bold", "end"))
    s.append(txt(vx0 - 150, vy0 + 58, "x&#8348; (spool position)", 11, MUTED, "normal", "end"))
    # P / T ports at bottom
    px_, tx_ = vx0 + 24, vx0 + vw - 24
    s.append(f'<circle cx="{px_}" cy="{vy0+90}" r="5" fill="#FFFFFF" stroke="{BLU}" stroke-width="1.8"/>')
    s.append(f'<circle cx="{tx_}" cy="{vy0+90}" r="5" fill="#FFFFFF" stroke="{RED}" stroke-width="1.8"/>')
    s.append(txt(px_, vy0 + 108, "P", 13, BLU, "bold"))
    s.append(txt(tx_, vy0 + 108, "T", 13, RED, "bold"))
    s.append(f'<line x1="{px_}" y1="{vy0+90}" x2="{px_}" y2="{vy0+140}" stroke="{BLU}" stroke-width="2.2"/>')
    s.append(f'<line x1="{tx_}" y1="{vy0+90}" x2="{tx_}" y2="{vy0+140}" stroke="{RED}" stroke-width="2.2"/>')

    # pump (ISO glyph, matches Fig 3.4) and reservoir
    s.append(f'<circle cx="{px_}" cy="{vy0+168}" r="24" fill="#FFFFFF" stroke="{BLU}" stroke-width="2.2"/>')
    s.append(f'<polygon points="{px_-10},{vy0+158} {px_-10},{vy0+178} {px_+10},{vy0+168}" fill="{BLU}"/>')
    s.append(txt(px_, vy0 + 210, "Pressure supply", 12, INK, "bold"))
    s.append(txt(px_, vy0 + 226, "(from pump / accumulator)", 10.5, MUTED))
    s.append(f'<path d="M {tx_-36} {vy0+150} L {tx_-36} {vy0+178} L {tx_+36} '
             f'{vy0+178} L {tx_+36} {vy0+150}" fill="#FFFFFF" stroke="{RED}" stroke-width="2.2"/>')
    s.append(f'<line x1="{tx_-36}" y1="{vy0+150}" x2="{tx_+36}" y2="{vy0+150}" '
             f'stroke="{RED}" stroke-width="1.1" stroke-dasharray="4,3"/>')
    s.append(txt(tx_, vy0 + 210, "Tank return", 12, INK, "bold"))
    s.append(txt(tx_, vy0 + 226, "(to reservoir)", 10.5, MUTED))

    # ================= legend =================
    lx, ly = 1160, 40
    s.append(f'<rect x="{lx}" y="{ly}" width="330" height="150" rx="8" fill="#FFFFFF" stroke="{GRAY2}" stroke-width="1.6"/>')
    s.append(txt(lx + 16, ly + 26, "LEGEND", 13, INK, "bold", "start"))
    leg = [("Mechanical connection", GRAY2, False), ("Chamber 1 line (p&#8321;,Q&#8321;)", BLU, False),
           ("Chamber 2 line (p&#8322;,Q&#8322;)", RED, False),
           ("Motion / force (F&#8462;, F_fric)", TEAL2, False),
           ("Command signal (u)", PURPLE, True)]
    for i, (lab, col, dash) in enumerate(leg):
        yy = ly + 48 + i * 22
        s.append(f'<line x1="{lx+16}" y1="{yy}" x2="{lx+50}" y2="{yy}" stroke="{col}" '
                 f'stroke-width="2.4"{" stroke-dasharray=\"5,4\"" if dash else ""}/>')
        s.append(txt(lx + 58, yy + 4, lab, 11.5, MUTED, "normal", "start"))

    # ================= key variables =================
    kx, ky = 1160, 210
    s.append(f'<rect x="{kx}" y="{ky}" width="330" height="330" rx="8" fill="#FFFFFF" stroke="{GRAY2}" stroke-width="1.6"/>')
    s.append(txt(kx + 16, ky + 26, "KEY VARIABLES", 13, INK, "bold", "start"))
    kv = [("x, x&#775;", "cylinder stroke and velocity"),
          ("p&#8321;, p&#8322;", "chamber pressures"),
          ("Q&#8321;, Q&#8322;", "chamber flows"),
          ("A&#8321;, A&#8322;", "chamber (piston) areas"),
          ("F&#8462;", "hydraulic force (axial)"),
          ("F_fric(x&#775;)", "friction force"),
          ("F&#8343;", "external load"),
          ("u", "command to valve"),
          ("x&#8348;", "valve spool position")]
    for i, (sym, mean) in enumerate(kv):
        yy = ky + 48 + i * 30
        s.append(txt(kx + 16, yy, sym, 12.5, TEAL2, "bold", "start"))
        s.append(txt(kx + 100, yy, mean, 11, MUTED, "normal", "start"))

    # ================= force-equation box =================
    ex, ey, ew, eh = 300, 780, 450, 90
    s.append(f'<rect x="{ex}" y="{ey}" width="{ew}" height="{eh}" rx="10" '
             f'fill="#F7FAFF" stroke="{BLU}" stroke-width="2"/>')
    s.append(txt(ex + 16, ey + 24, "Cylinder force", 12.5, INK, "bold", "start"))
    s.append(txt(ex + 16, ey + 46, "F&#8462; = p&#8321;A&#8321; &#8722; p&#8322;A&#8322; &#8722; F_fric(x&#775;)",
                 13, BLU, "bold", "start"))
    s.append(txt(ex + 16, ey + 66, "x&#775;&#8348; = (u &#8722; x&#8348;) / &#964;&#8348;  (spool lag)",
                 12, MUTED, "normal", "start"))

    s.append("</svg>")
    out = "\n".join(s)
    (DRAWIO / "fig3_5_actuator_channel.svg").write_text(out)
    cairosvg.svg2pdf(bytestring=out.encode(),
                     write_to=str(IMG / "fme500_actuator_channel_detailed.pdf"))
    print("wrote actuator channel")


if __name__ == "__main__":
    topology()
    iso_annotated()
    hydraulic_schematic()
    actuator_channel()
