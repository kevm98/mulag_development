#!/usr/bin/env python3
"""Chapter 6 figure: the PPO-(AAM)-ID-FD stack.

Produces
    images/ch6_aam_stack.pdf

Shows where the estimated latent enters: the causal AAM reads only the
observable history window, and z-hat conditions BOTH the policy observation
and the bounded adapter around the frozen inverse model. The dashed region is
what the stack collapses to when the residual is zero, which is exactly
PPO-ID-FD.

Same house idiom and legibility budget as ``make_ch6_figures.py``: a 1000 px
canvas included at \\textwidth, so no font below 17 px.

Run:  /home/rrlab/miniconda3/envs/thesis_figures/bin/python \
          scripts/make_ch6_aam_stack.py
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

CANVAS, MIN_FONT = 1000, 17


def _check(size):
    if size < MIN_FONT:
        raise ValueError(f"font {size} px below the {MIN_FONT} px budget")
    return size


def rect(x, y, w, h, fill, stroke, sw=2.0, rx=12, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
            f'{d}/>')


def txt(x, y, s, size=19, fill=INK, weight="normal", anchor="middle",
        style="normal"):
    _check(size)
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" font-style="{style}" '
            f'text-anchor="{anchor}">{s}</text>')


def arrow(x1, y1, x2, y2, color=INK, sw=2.2, dash=None, marker="arrowhead"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def path(d, color=INK, sw=2.2, dash=None, marker="arrowhead"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"'
            f'{da} marker-end="url(#{marker})"/>')


def defs():
    out = ['<defs>']
    for name, col in (("arrowhead", INK), ("arrowpurple", PURPLE),
                      ("arrowgray", GRAY), ("arrowteal", TEAL)):
        out.append(
            f'<marker id="{name}" markerWidth="9" markerHeight="7" refX="8.5" '
            'refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" '
            f'fill="{col}"/></marker>')
    out.append('</defs>')
    return "".join(out)


def build():
    W, H = CANVAS, 580
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">', defs(),
         f'<rect width="{W}" height="{H}" fill="white"/>']

    # ---- estimation lane (top) -------------------------------------------
    s.append(rect(58, 36, 250, 84, "#F4F7FC", BLUE, 2.2))
    s.append(txt(183, 68, "observable history", 20, BLUE, "bold"))
    s.append(txt(183, 96, "20 rows × 18, 1.0 s", 18, MUTED))

    s.append(rect(360, 36, 232, 84, "#F4F7FC", BLUE, 2.2))
    s.append(txt(476, 68, "causal AAM", 20, BLUE, "bold"))
    s.append(txt(476, 96, "1-D CNN, 20 Hz", 18, MUTED))
    s.append(arrow(310, 78, 356, 78, BLUE))

    s.append(rect(648, 38, 150, 80, "#F6F2FA", PURPLE, 2.2))
    s.append(txt(723, 72, "&#7825;", 26, PURPLE, "bold"))
    s.append(txt(723, 100, "2 values", 18, MUTED))
    s.append(arrow(594, 78, 644, 78, BLUE))

    # z-hat -> policy observation
    s.append(path("M 690 120 L 690 152 L 183 152 L 183 306", PURPLE, 2.2,
                  marker="arrowpurple"))
    s.append(txt(437, 145, "conditions the policy", 18, PURPLE))

    # z-hat -> adapter
    s.append(path("M 762 120 L 762 176 L 575 176 L 575 210", PURPLE, 2.2,
                  marker="arrowpurple"))
    s.append(txt(786, 160, "conditions the adapter", 18, PURPLE,
                 anchor="start"))

    # ---- the bounded adapter ---------------------------------------------
    s.append(rect(453, 212, 244, 78, "#F6F2FA", PURPLE, 2.2))
    s.append(txt(575, 242, "bounded adapter", 19, PURPLE, "bold"))
    s.append(txt(575, 270, "scale 0.25, clipped", 18, MUTED))

    # ---- the frozen PPO-ID-FD core ---------------------------------------
    s.append(rect(336, 316, 476, 108, "#FBFBFC", GRAY, 2.0, 14, dash="7 6"))

    s.append(rect(356, 328, 158, 84, "#EFF6F5", TEAL, 2.2))
    s.append(txt(435, 360, "frozen ID", 20, TEAL, "bold"))
    s.append(txt(435, 388, "target → valve", 18, MUTED))

    s.append(f'<circle cx="575" cy="370" r="27" fill="#F6F2FA" '
             f'stroke="{PURPLE}" stroke-width="2.2"/>')
    s.append(txt(575, 379, "+", 27, PURPLE, "bold"))
    s.append(arrow(516, 370, 544, 370))
    s.append(path("M 575 290 L 575 339", PURPLE, 2.2, marker="arrowpurple"))

    s.append(rect(634, 328, 162, 84, "#EFF6F5", TEAL, 2.2))
    s.append(txt(715, 360, "frozen FD", 20, TEAL, "bold"))
    s.append(txt(715, 388, "valve → Δx", 18, MUTED))
    s.append(arrow(604, 370, 630, 370))

    s.append(txt(574, 448, "= PPO-ID-FD when the residual is zero", 18, MUTED,
                 style="italic"))

    # ---- policy and plant -------------------------------------------------
    s.append(rect(58, 328, 250, 84, "#FDF6EE", ORANGE, 2.2))
    s.append(txt(183, 360, "PPO policy", 20, ORANGE, "bold"))
    s.append(txt(183, 388, "68 obs → desired motion", 18, MUTED))
    s.append(arrow(310, 370, 352, 370))

    s.append(rect(634, 486, 162, 76, "#F3F4F5", INK, 2.2))
    s.append(txt(715, 516, "PhysX", 20, INK, "bold"))
    s.append(txt(715, 544, "the plant", 18, MUTED))
    s.append(path("M 715 412 L 715 482", INK, 2.2))

    # feedback to the history window, routed clear of every block
    s.append(path("M 630 524 L 22 524 L 22 78 L 54 78", GRAY, 2.2, dash="6 5",
                  marker="arrowgray"))
    s.append(txt(330, 546, "measured state and previous command", 18, MUTED))

    s.append('</svg>')
    return "".join(s), W, H


def main():
    svg, W, H = build()
    (DRAWIO / "ch6_aam_stack.svg").write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(),
                     write_to=str(IMG / "ch6_aam_stack.pdf"),
                     output_width=W, output_height=H)
    print("wrote", IMG / "ch6_aam_stack.pdf")


if __name__ == "__main__":
    main()
