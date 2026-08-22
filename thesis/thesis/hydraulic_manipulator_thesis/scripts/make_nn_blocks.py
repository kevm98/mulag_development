#!/usr/bin/env python3
"""Isometric block diagrams of the learned networks.

Draws each layer as a stack of 3-D slabs rather than as an abstract box, so the
SHAPE of the tensor is visible: the input carries its feature and time extent,
the hidden stages carry their channel count, and the latent is visibly two
numbers. Dimensions are taken from the trained checkpoints, not chosen for the
picture.

House idiom follows make_ch6_aam_stack.py: a 1000 px canvas included at
\\textwidth, so no font falls below 17 px.

Run:  /home/rrlab/miniconda3/envs/thesis_figures/bin/python scripts/make_nn_blocks.py
"""
from pathlib import Path

import cairosvg

BASE = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis")
IMG = BASE / "images"

INK, MUTED, RULE = "#1F2933", "#66727E", "#B9C2C9"
FONT = "DejaVu Sans, Arial, sans-serif"
MIN_FONT = 17

# face shades: front, top, right
BLUE = ("#AECBEA", "#CFE2F5", "#7FA8D4")
GREEN = ("#B7DFAF", "#D6EFD0", "#8CC482")
TEAL = ("#5FC8C0", "#8FE0DA", "#38A49C")
AMBER = ("#F0CE93", "#F8E4BE", "#D6A857")


def _f(size):
    if size < MIN_FONT:
        raise ValueError(f"font {size} px below the {MIN_FONT} px budget")
    return size


def txt(x, y, s, size=19, fill=INK, weight="normal", anchor="middle"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{_f(size)}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>')


def cuboid(x, y, w, h, d, shades, sw=1.6):
    """Isometric slab: (x, y) is the top-left of the FRONT face."""
    front, top, right = shades
    return (
        f'<polygon points="{x},{y} {x+d},{y-d} {x+w+d},{y-d} {x+w},{y}" '
        f'fill="{top}" stroke="{INK}" stroke-width="{sw}"/>'
        f'<polygon points="{x+w},{y} {x+w+d},{y-d} {x+w+d},{y+h-d} {x+w},{y+h}" '
        f'fill="{right}" stroke="{INK}" stroke-width="{sw}"/>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="{front}" stroke="{INK}" stroke-width="{sw}"/>')


def stack(x, y, w, h, d, shades, n, gap, sw=1.6):
    """n slabs receding along the isometric axis: shows tensor depth."""
    out = []
    for i in reversed(range(n)):
        out.append(cuboid(x + i * gap, y - i * gap, w, h, d, shades, sw))
    return "".join(out)


def group(x, y, w, h, label, sub=""):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" ry="16" '
         f'fill="none" stroke="{RULE}" stroke-width="2"/>')
    s += txt(x + w / 2, y + h + 30, label, 20, INK, "bold")
    if sub:
        s += txt(x + w / 2, y + h + 54, sub, 18, MUTED)
    return s


def arrow(x1, y1, x2, y2, color=MUTED, sw=2.0):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#ah)"/>')


def fan(x1, y1, targets, color=RULE, sw=1.4):
    return "".join(f'<line x1="{x1}" y1="{y1}" x2="{tx}" y2="{ty}" '
                   f'stroke="{color}" stroke-width="{sw}"/>' for tx, ty in targets)


DEFS = (f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>')


def aam_figure():
    W, H = 1000, 470
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">', DEFS,
         f'<rect width="{W}" height="{H}" fill="white"/>']

    top, gh = 92, 236
    mid = top + 150

    # ---- input: 18 channels x 20 ticks ------------------------------
    gx, gw = 34, 196
    s.append(group(gx, top, gw, gh, "observed history",
                   "18 channels × 20 ticks"))
    s.append(txt(gx + gw / 2, top + 26, "q,  q̇,  Δp,  u", 19, MUTED))
    s.append(stack(gx + 34, top + 108, 68, 106, 22, BLUE, 4, 12))

    # ---- four dilated causal blocks ----------------------------------
    # Receding depth grows with the dilation, so the widening temporal reach is
    # visible in the drawing rather than only stated in the label.
    xs, bw = [258, 400, 542, 684], 128
    for i, bx in enumerate(xs):
        d = 2 ** i
        s.append(group(bx, top, bw, gh, f"block {i+1}", f"dilation {d}"))
        # Dilation drawn as the SPACING between taps, which is what it is: the
        # slabs pull apart as the block reaches further back in time.
        gap = (5, 9, 13, 17)[i]
        s.append(stack(bx + 8, top + 108, 46, 100, 18, GREEN, 4, gap))
    s.append(txt((xs[0] + xs[-1] + bw) / 2, top - 46,
                 "causal dilated convolutions", 20, INK, "bold"))
    s.append(txt((xs[0] + xs[-1] + bw) / 2, top - 22,
                 "64 channels, depth 4, receptive field 20 ticks", 18, MUTED))

    # ---- latent -------------------------------------------------------
    lx, lw = 830, 136
    s.append(group(lx, top, lw, gh, "latent", "2 values, 20 Hz"))
    s.append(cuboid(lx + 40, top + 104, 54, 40, 20, TEAL))
    s.append(cuboid(lx + 40, top + 168, 54, 40, 20, TEAL))
    s.append(txt(lx + lw / 2, top + 78, "ẑ", 22, INK, "bold"))

    # ---- flow ----------------------------------------------------------
    s.append(arrow(gx + gw + 3, mid, xs[0] - 5, mid))
    for u, v in zip(xs, xs[1:]):
        s.append(arrow(u + bw + 3, mid, v - 5, mid))
    s.append(arrow(xs[-1] + bw + 3, mid, lx - 5, mid))

    # ---- the property a reader most often gets wrong ------------------
    yb = top + gh + 84
    s.append(f'<path d="M {gx+40} {yb} L {lx+94} {yb}" stroke="{RULE}" '
             f'stroke-width="1.5" stroke-dasharray="6 6" fill="none"/>')
    s.append(txt(W / 2, yb + 26,
                 "every output reads past ticks only, and no state is carried "
                 "between ticks", 18, MUTED))
    s.append("</svg>")
    svg = "".join(s)
    (BASE / "drawio").mkdir(exist_ok=True)
    (BASE / "drawio/nn_aam_blocks.svg").write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(IMG / "ch5_aam_blocks.pdf"))
    print(f"  wrote {IMG/'ch5_aam_blocks.pdf'}")


if __name__ == "__main__":
    aam_figure()
