#!/usr/bin/env python3
"""Neuron-and-connection style NN diagrams for Chapter 5 (Fig 5.2),
in the spirit of Liu et al. (2025, JMST) Fig. 2: colored stacked input
feature blocks, explicit neuron circles, fan-out connection lines
between layers, and labeled input/output dimensions - not abstract
process boxes.

Run:  python3 scripts/make_ch5_nn_diagram.py
"""
from pathlib import Path
import cairosvg

BASE = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis")
IMG = BASE / "images"
DRAWIO = BASE / "drawio"

ORANGE, BLUE, TEAL, PURPLE, RED, GRAY = (
    "#C97816", "#2F5DA8", "#16847A", "#7A52A8", "#B64B42", "#69737D")
INK, MUTED = "#1F2933", "#66727E"
FONT = "DejaVu Sans, Arial, sans-serif"


def txt(x, y, s, size=13, fill=INK, weight="normal", anchor="middle",
        style="normal"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" '
            f'font-style="{style}">{s}</text>')


def marker(mid, color):
    return (f'<marker id="{mid}" markerWidth="8" markerHeight="8" refX="8" '
            f'refY="4" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L8,4 L0,8 z" fill="{color}"/></marker>')


def neuron_col(cx, y0, n, r, col, fill="#FFFFFF"):
    """Return (svg_fragments, [(x,y), ...]) for n stacked neuron circles."""
    frags, pts = [], []
    step = max(2.4 * r, 26)
    top = y0 - (n - 1) * step / 2
    for i in range(n):
        y = top + i * step
        frags.append(f'<circle cx="{cx}" cy="{y:.1f}" r="{r}" fill="{fill}" '
                      f'stroke="{col}" stroke-width="1.8"/>')
        pts.append((cx, y))
    return frags, pts


def fan(pts_a, pts_b, col, width=0.6, opacity=0.55, stride=1):
    frags = []
    for ax, ay in pts_a[::stride]:
        for bx, by in pts_b[::stride]:
            frags.append(f'<line x1="{ax}" y1="{ay:.1f}" x2="{bx}" '
                          f'y2="{by:.1f}" stroke="{col}" stroke-width="{width}" '
                          f'opacity="{opacity}"/>')
    return frags


def build():
    W, H = 1720, 820
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}"><defs>', marker("aB", BLUE), marker("aT", TEAL),
         marker("aP", PURPLE), '</defs>', f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']

    # =====================================================================
    # (a) ID / FD residual trunk, drawn as neurons with a residual skip
    # =====================================================================
    s.append(txt(40, 40, "(a)  ID / FD residual network (neuron view)", 17,
                 INK, "bold", "start"))
    s.append(f'<line x1="40" x2="740" y1="50" y2="50" stroke="{BLUE}" stroke-width="1.3"/>')

    ax = [90, 220, 340, 460, 580, 700]
    counts = [7, 9, 9, 9, 9, 4]
    labels_top = ["input\n28 / 20", "512", "512", "512", "512", "output\n4 / 12"]
    cols_a = [PURPLE, BLUE, BLUE, BLUE, BLUE, TEAL]
    cy_a = 350
    layer_pts = []
    for x, n, col in zip(ax, counts, cols_a):
        frags, pts = neuron_col(x, cy_a, n, 7, col,
                                 fill="#F6F0FB" if col == PURPLE else
                                 ("#ECF8F6" if col == TEAL else "#EEF4FF"))
        layer_pts.append(pts)
    for i in range(len(ax) - 1):
        s.extend(fan(layer_pts[i], layer_pts[i + 1], GRAY, 0.55, 0.4))
    for x, n, col in zip(ax, counts, cols_a):
        frags, _ = neuron_col(x, cy_a, n, 7, col,
                              fill="#F6F0FB" if col == PURPLE else
                              ("#ECF8F6" if col == TEAL else "#EEF4FF"))
        s.extend(frags)
    for x, lab in zip(ax, labels_top):
        for k, line in enumerate(lab.split("\n")):
            s.append(txt(x, cy_a - 130 + k * 15, line, 11.5, MUTED))
    # residual skip arcs (block 2 -> block 5)
    s.append(f'<path d="M 220 {cy_a-100} C 340 {cy_a-170} 460 {cy_a-170} '
             f'580 {cy_a-100}" fill="none" stroke="{TEAL}" stroke-width="2" '
             f'stroke-dasharray="5,4" marker-end="url(#aT)"/>')
    s.append(txt(400, cy_a - 175, "identity skip (&#215;4 residual blocks)", 12, TEAL, "bold"))
    s.append(txt(400, cy_a + 190, "SiLU + LayerNorm + Dropout 0.02 between linear layers",
                 12, MUTED, "normal"))
    s.append(txt(400, cy_a + 220, "shared trunk: ID predicts &#916;u (4-D), FD predicts "
                 "[&#916;x, &#916;xdot, &#916;&#916;p] (12-D)", 12, MUTED, "normal"))

    # =====================================================================
    # (b) ROM recurrent encoder, Liu-et-al. style: colored input slabs
    #     feeding a recurrent neuron column, fanning into heads
    # =====================================================================
    ox = 900
    s.append(txt(ox - 20, 40, "(b)  Command-conditioned ROM encoder (neuron view)",
                 17, INK, "bold", "start"))
    s.append(f'<line x1="{ox-20}" x2="{ox+620}" y1="50" y2="50" stroke="{TEAL}" stroke-width="1.3"/>')

    cy_b = cy_a
    # stacked colored history slabs (parallelogram planes), Liu Fig.2 style
    slab_x = ox + 10
    slab_colors = [("state x, xdot, &#916;p", BLUE), ("prev. valve u&#770;", ORANGE),
                   ("command c", PURPLE)]
    slab_top = cy_b - 90
    for i, (name, col) in enumerate(slab_colors):
        yoff = slab_top + i * 26
        skew = 26
        s.append(f'<polygon points="{slab_x},{yoff} {slab_x+150},{yoff} '
                 f'{slab_x+150-skew},{yoff+58} {slab_x-skew},{yoff+58}" '
                 f'fill="{col}" fill-opacity="0.16" stroke="{col}" stroke-width="1.8"/>')
    s.append(txt(slab_x + 20, slab_top + 150, "history H&#8342;", 13, INK, "bold", "start"))
    s.append(txt(slab_x + 20, slab_top + 168, "6 &#215; 24, 0.25 s", 11, MUTED, "normal", "start"))
    for i, (name, col) in enumerate(slab_colors):
        yy = slab_top - 34 + i * 22
        s.append(f'<rect x="{slab_x}" y="{yy-10}" width="14" height="14" '
                 f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="1.3"/>')
        s.append(txt(slab_x + 20, yy + 2, name, 11.5, MUTED, "normal", "start"))

    gx = [ox + 340, ox + 430, ox + 520]
    gru_pts = []
    for k, x in enumerate(gx):
        frags, pts = neuron_col(x, cy_b, 5, 8, TEAL, "#ECF8F6")
        s.extend(frags)
        gru_pts.append(pts)
        s.append(txt(x, cy_b + 74, f"t-{5-k}" if k < 2 else "t", 11, TEAL, "bold"))
    for i in range(len(gx) - 1):
        s.extend(fan(gru_pts[i], gru_pts[i + 1], TEAL, 0.7, 0.5))
    s.append(txt(gx[1], cy_b - 130, "GRU, 2 &#215; 128, unrolled over 6 steps",
                 12, MUTED))
    s.append(txt(gx[1], cy_b - 112, "(3 of 6 shown)", 12, MUTED))
    s.append(f'<line x1="{slab_x+150}" y1="{cy_b-2}" x2="{gx[0]-10}" y2="{cy_b}" '
             f'stroke="{GRAY}" stroke-width="1.6" marker-end="url(#aB)"/>')

    # heads fed by final hidden state
    hx = gx[-1] + 60
    head_defs = [("actuation head", "&#8594; valve u&#770;", BLUE, cy_b - 60),
                 ("residual heads", "bounded r_p, r_v, r_x", PURPLE, cy_b),
                 ("load head", "&#8467;(h&#8342;)", ORANGE, cy_b + 60)]
    for name, sub, col, y in head_defs:
        s.append(f'<rect x="{hx}" y="{y-18}" width="180" height="38" rx="9" '
                 f'fill="#FFFFFF" stroke="{col}" stroke-width="2"/>')
        s.append(txt(hx + 90, y - 1, name, 12, INK, "bold"))
        s.append(txt(hx + 90, y + 14, sub, 10.5, MUTED))
        s.append(f'<line x1="{gx[-1]+8}" y1="{cy_b}" x2="{hx}" y2="{y}" '
                 f'stroke="{col}" stroke-width="1.4" opacity="0.8"/>')

    s.append(txt(ox + 260, cy_b + 190,
                 "gray fan lines = full connectivity between adjacent layers; "
                 "colored slabs = distinct feature groups in the 24-D input,",
                 12, MUTED))
    s.append(txt(ox + 260, cy_b + 208,
                 "in the style of the multi-feature input stacks used by Liu "
                 "et al. (2025) for hydraulic-manipulator dynamics networks.",
                 12, MUTED))

    s.append("</svg>")
    out = "\n".join(s)
    (DRAWIO / "fig5_nn_neurons.svg").write_text(out)
    cairosvg.svg2pdf(bytestring=out.encode(),
                     write_to=str(IMG / "models_nn_architectures.pdf"))
    print("wrote NN neuron diagram")


if __name__ == "__main__":
    build()
