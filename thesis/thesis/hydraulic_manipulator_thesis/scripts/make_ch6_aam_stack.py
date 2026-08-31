#!/usr/bin/env python3
"""Chapter 6 figure: AF training and deployment pipeline.

Produces
    images/ch6_aam_stack.pdf

The figure separates the frozen backbone from the trainable history-residual
path and embeds the thesis task render as the shared simulation/deployment
plant.  The encoder and residual head are trained concurrently by PPO; there
is no supervised latent target and no inverse model in the AF command path.

The layout uses a short main signal path and a separate adaptation lane so no
connector crosses a block.  Secondary notes live in the thesis caption rather
than inside the figure.

Run:  /home/rrlab/miniconda3/envs/thesis_figures/bin/python \
          scripts/make_ch6_aam_stack.py
"""
from pathlib import Path
import base64
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

CANVAS, MIN_FONT = 1000, 25


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
    return ('<defs><marker id="arrowhead" markerWidth="9" markerHeight="7" '
            'refX="8.5" refY="3.5" orient="auto"><polygon '
            f'points="0 0, 9 3.5, 0 7" fill="{INK}"/></marker></defs>')


def build():
    W, H = 1400, 720
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">', defs(),
         f'<rect width="{W}" height="{H}" fill="white"/>']

    def block(x, y, w, h, title, lines, fill="#FFFFFF", edge=INK, sw=2.6):
        s.append(rect(x, y, w, h, fill, edge, sw, 8))
        s.append(txt(x + w / 2, y + 43, title, 34, INK, "bold"))
        for index, line in enumerate(lines):
            s.append(txt(x + w / 2, y + 91 + 38 * index, line, 30,
                         INK, "500"))

    def image_href(image_path):
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        return f"data:image/png;base64,{encoded}"

    s.append(txt(24, 46, "AF training and deployment", 37, INK, "bold",
                 "start"))

    s.append(rect(20, 76, 1360, 244, "#F3F4F5", "#CDD1D4", 1.6, 0))
    s.append(txt(38, 112, "DEPLOYED PATH", 27, BLUE, "bold", "start"))
    s.append(rect(20, 350, 900, 300, "#F3F4F5", "#CDD1D4", 1.6, 0))
    s.append(txt(38, 388, "TRAINABLE ADAPTATION", 27, RED, "bold", "start"))

    # The final AF actor retains the baseline 66-channel contract.  The
    # flattened causal history is carried beside it in the PPO rollout and is
    # therefore not counted as additional policy-observation channels.
    block(45, 140, 190, 140, "Policy input", ("oₜ", "66 channels"),
          edge=GRAY)
    block(285, 130, 225, 160, "Frozen actor",
          ("u₀ = π(oₜ)", "valve command"), "#F7F9FB", BLUE)
    block(560, 130, 230, 160, "Compose",
          ("uₜ = clip(u₀ + δuₜ)", "bounded output"),
          "#FAF8FC", PURPLE)
    block(840, 130, 230, 160, "Frozen FD",
          ("Δx̂ₜ = fFD(sₜ,uₜ)", "bounded target"),
          "#F4F8F7", TEAL)
    s.append(arrow(239, 210, 281, 210, INK, 3.3))
    s.append(arrow(514, 210, 556, 210, INK, 3.3))
    s.append(arrow(794, 210, 836, 210, INK, 3.3))
    s.append(arrow(1074, 210, 1116, 210, INK, 3.3))

    block(50, 420, 210, 160, "History Hₜ",
          ("past plant state", "valve action"), edge=GRAY)
    block(320, 410, 220, 180, "Encoder φ",
          ("causal history", "feature hₜ"), "#FFF8F7", RED, 3.0)
    block(610, 410, 220, 180, "Residual gψ",
          ("uses oₜ and hₜ", "bounded δuₜ"), "#FFF8F7", RED, 3.0)
    s.append(arrow(264, 500, 316, 500, INK, 3.3))
    s.append(arrow(544, 500, 606, 500, INK, 3.3))
    s.append(path("M 720 406 L 720 340 L 675 340 L 675 294", INK, 3.3))

    # Existing thesis render of the truck-mounted manipulator and the complete
    # verge-following task replaces the generic quadruped image in the reference.
    render = IMG / "ch3_verge_task_render.png"
    s.append(rect(1120, 96, 240, 204, "#FFFFFF", TEAL, 2.8, 8))
    s.append(f'<image x="1130" y="106" width="220" height="126" '
             f'preserveAspectRatio="xMidYMid slice" href="{image_href(render)}"/>')
    s.append(txt(1240, 267, "PhysX task", 30, INK, "bold"))
    s.append(txt(1240, 293, "truck-mounted arm", 25, MUTED))

    block(970, 400, 390, 170, "PPO objective",
          ("reward and safety gates", "update φ and gψ only"),
          "#FFF8F7", RED, 3.0)

    # Solid lines are deployed signals. Dashed lines use dedicated lanes below
    # and outside the blocks for PPO updates and rollout feedback.
    s.append(path("M 1100 574 L 1100 615 L 720 615 L 720 594",
                  MUTED, 3.3, dash="10 8"))
    s.append(path("M 1240 574 L 1240 640 L 430 640 L 430 594",
                  MUTED, 3.3, dash="10 8"))
    s.append(path("M 1364 210 L 1380 210 L 1380 690 L 155 690 L 155 584",
                  MUTED, 3.3, dash="10 8"))

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
            '<meta charset="utf-8">'
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
