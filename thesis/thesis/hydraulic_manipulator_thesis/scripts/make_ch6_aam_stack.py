#!/usr/bin/env python3
"""Chapter 6 figure: complete R, IF, and AIF control pipelines.

Produces
    images/ch6_aam_stack.pdf

The AIF lane distinguishes simulation-only latent supervision from the
deployable history-to-estimate path.  The true latent never enters the deployed
controller.

Same house idiom and legibility budget as ``make_ch6_figures.py``: a 1000 px
canvas included at \\textwidth, so no font below 17 px.

Run:  /home/rrlab/miniconda3/envs/thesis_figures/bin/python \
          scripts/make_ch6_aam_stack.py
"""
from pathlib import Path
import subprocess

try:
    import cairosvg
except ImportError:  # macOS thesis workstation: use installed Chrome instead
    cairosvg = None

BASE = Path(__file__).resolve().parents[1]
IMG = BASE / "images"
DRAWIO = BASE / "drawio"
DRAWIO.mkdir(exist_ok=True)

ORANGE, BLUE, TEAL, PURPLE, RED, GRAY = (
    "#A58A68", "#7288A5", "#789A94", "#9182A4", "#A97976", "#7B838A")
INK, MUTED = "#1F2933", "#66727E"
FONT = "DejaVu Sans, Arial, sans-serif"

CANVAS, MIN_FONT = 1000, 19


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
                      ("arrowgray", GRAY), ("arrowteal", TEAL),
                      ("arroworange", ORANGE), ("arrowblue", BLUE)):
        out.append(
            f'<marker id="{name}" markerWidth="9" markerHeight="7" refX="8.5" '
            'refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" '
            f'fill="{col}"/></marker>')
    out.append('</defs>')
    return "".join(out)


def build():
    W, H = 1000, 930
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">', defs(),
         f'<rect width="{W}" height="{H}" fill="white"/>']

    s.append(txt(500, 35, "Complete actuator-control pipelines", 27,
                 INK, "bold"))

    def lane(y, h, name, subtitle, fill, edge, label_fill):
        s.append(rect(15, y, 970, h, fill, edge, 1.8, 10))
        s.append(rect(20, y + 22, 80, h - 44, label_fill, edge, 2.2, 8))
        s.append(txt(60, y + 65, name, 25, edge, "bold"))
        s.append(txt(60, y + 94, subtitle, 19, INK, "bold"))

    def block(x, y, w, h, title, equation=None, fill="#FFFFFF", sw=1.9,
              edge=INK, detail_size=26):
        s.append(rect(x, y, w, h, fill, edge, sw, 7))
        if equation:
            multi = isinstance(equation, (tuple, list))
            s.append(txt(x + w / 2, y + 29, title, 20, INK, "bold"))
            if multi:
                for offset, line in zip((58, 86), equation):
                    s.append(txt(x + w / 2, y + offset, line, detail_size,
                                 INK, "500"))
            else:
                s.append(txt(x + w / 2, y + 68, equation, detail_size,
                             INK, "500"))
        else:
            s.append(txt(x + w / 2, y + h / 2 + 7, title, 20,
                         INK, "bold"))

    # R: policy command is realized by the command ROM.
    lane(55, 170, "R", "ROM", "#FFFFFF", ORANGE, "#F4F0EA")
    block(112, 78, 135, 100, "Policy  πR", "cₖ = πᴿ(oₖ)", "#FAF7F2",
          edge=ORANGE, detail_size=24)
    block(285, 78, 185, 100, "Command ROM", ("Δx̂ₖ = fᴿ", "(sₖ,cₖ)"),
          "#FAF7F2", edge=ORANGE)
    block(510, 78, 195, 100, "Target update",
          ("x*ₖ₊₁ = clip", "(xₖ+Δx̂ₖ)"), "#FAF7F2", edge=ORANGE,
          detail_size=25)
    block(745, 78, 190, 100, "PhysX arm", "xₖ₊₁ , oₖ₊₁", "#F4F7F4",
          edge="#859888", detail_size=25)
    for a, b in ((247, 281), (470, 506), (705, 741)):
        s.append(arrow(a, 128, b, 128, ORANGE, marker="arroworange"))
    s.append(path("M 840 180 L 840 204 L 180 204 L 180 182", GRAY,
                  1.8, marker="arrowgray"))

    # IF: inverse valve command followed by strict forward prediction.
    lane(245, 195, "IF", "ID → FD", "#FFFFFF", BLUE, "#EDF1F5")
    block(104, 272, 125, 100, "Policy  πIF", "cₖ = πIF(oₖ)", "#F4F6F8",
          edge=BLUE, detail_size=22)
    block(260, 272, 150, 100, "Inverse model", ("Δuₖ = fID", "(sₖ,cₖ)"),
          "#F4F6F8", edge=BLUE, detail_size=24)
    block(441, 272, 155, 100, "Valve update",
          ("uₖ = clip(", "uₖ₋₁+Δuₖ)"), "#F4F6F8", edge=BLUE,
          detail_size=24)
    block(627, 272, 145, 100, "Forward model", ("Δx̂ₖ = fFD", "(sₖ,uₖ)"),
          "#F4F6F8", edge=BLUE, detail_size=24)
    block(803, 272, 158, 100, "Target + PhysX", "x*ₖ₊₁ → oₖ₊₁",
          "#F4F7F4", edge="#859888", detail_size=23)
    for a, b in ((229, 256), (410, 437), (596, 623), (772, 799)):
        s.append(arrow(a, 322, b, 322, BLUE, marker="arrowblue"))
    s.append(path("M 882 374 L 882 414 L 166 414 L 166 376", GRAY,
                  1.8, marker="arrowgray"))

    # AIF: a deployable causal estimate, plus a clearly isolated training-only
    # target inspired by the user's reference.  True z never reaches policy or
    # plant at deployment.
    lane(465, 440, "AIF", "AAM + IF", "#FFFFFF", PURPLE, "#F2EFF5")
    s.append(rect(108, 486, 330, 68, "#FCFAF5", ORANGE, 1.7, 7,
                  dash="7 6"))
    s.append(txt(126, 512, "TRAINING ONLY", 19, ORANGE, "bold", "start"))
    s.append(txt(273, 541, "true z  →  loss ‖z − ẑ‖²", 24, INK, "500"))

    block(108, 640, 150, 100, "History Hₖ", ("20 samples", "× 18 signals"),
          "#F6F7F8", edge=GRAY, detail_size=24)
    block(292, 640, 150, 100, "Causal AAM", ("ẑₖ = fAAM", "(Hₖ)"),
          "#F1F6F5", 2.1, edge=TEAL, detail_size=24)
    s.append(arrow(258, 690, 288, 690, TEAL, marker="arrowteal"))
    s.append(f'<circle cx="492" cy="690" r="31" fill="#F1F6F5" '
             f'stroke="{TEAL}" stroke-width="2.2"/>')
    s.append(txt(492, 698, "ẑₖ", 23, INK, "bold"))
    s.append(arrow(442, 690, 457, 690, TEAL, marker="arrowteal"))
    s.append(path("M 273 554 L 273 580 L 492 580 L 492 657", GRAY,
                  1.7, dash="6 5", marker="arrowgray"))

    block(558, 552, 165, 100, "Policy  πAIF", "cₖ=πAIF(oₖ,ẑₖ)", "#F5F2F7",
          edge=PURPLE, detail_size=23)
    block(558, 742, 165, 100, "Base inverse",
          ("uᵇᵃˢᵉ =", "fID(sₖ,cₖ)"), "#F4F6F8", edge=BLUE,
          detail_size=24)
    block(766, 635, 205, 110, "Bounded adapter",
          ("uₖ = clip(", "uᵇᵃˢᵉ+0.25 gψ)"), "#F5F2F7", 2.1,
          edge=PURPLE, detail_size=25)
    block(766, 755, 205, 100, "FD + target + PhysX",
          ("Δx̂ₖ=fFD(sₖ,uₖ)", "x*ₖ₊₁ → PhysX"), "#F1F6F6", edge=TEAL,
          detail_size=22)

    s.append(path("M 523 674 L 540 674 L 540 602 L 554 602", PURPLE,
                  2.2, marker="arrowpurple"))
    s.append(path("M 523 706 L 746 706 L 746 674 L 762 674", PURPLE,
                  2.2, marker="arrowpurple"))
    s.append(arrow(640, 656, 640, 738, BLUE, 2.2, marker="arrowblue"))
    s.append(path("M 723 792 L 744 792 L 744 724 L 762 724", BLUE,
                  2.2, marker="arrowblue"))
    s.append(arrow(868, 749, 868, 751, TEAL, 2.2, marker="arrowteal"))
    s.append(path("M 971 805 L 978 805 L 978 868 L 183 868 L 183 744",
                  GRAY, 1.8, marker="arrowgray"))
    s.append(txt(580, 895, "deployment feedback: measured state and previous command",
                 20, MUTED))

    s.append('</svg>')
    return "".join(s), W, H


def main():
    svg, W, H = build()
    svg_path = DRAWIO / "ch6_aam_stack.svg"
    pdf_path = IMG / "ch6_aam_stack.pdf"
    svg_path.write_text(svg)
    if cairosvg is not None:
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path),
                         output_width=W, output_height=H)
    else:
        chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        html_path = DRAWIO / ".ch6_aam_stack_print.html"
        html_path.write_text(
            f'<style>@page{{size:{W}px {H}px;margin:0}}'
            'html,body{margin:0;padding:0}</style>' + svg)
        subprocess.run(
            [str(chrome), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        html_path.unlink()
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
