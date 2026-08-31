#!/usr/bin/env python3
"""Create the Chapter 6 overview of the non-adaptive R and IF pipelines."""
from pathlib import Path
import subprocess

try:
    import cairosvg
except ImportError:
    cairosvg = None

BASE = Path(__file__).resolve().parents[1]
IMG = BASE / "images"
DRAWIO = BASE / "drawio"
DRAWIO.mkdir(exist_ok=True)
INK, MUTED, RULE, PANEL = "#111111", "#4F5963", "#8A9198", "#F5F5F5"
FONT = "DejaVu Sans, Arial, sans-serif"


def rect(x, y, w, h, fill, stroke, sw=2, rx=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, value, size=22, color=INK, weight="normal", anchor="middle"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}">'
            f'{value}</text>')


def arrow(x1, y1, x2, y2, color=INK):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="3.3" marker-end="url(#arrow)"/>')


def path(points, color=INK, dash=None):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{points}" fill="none" stroke="{color}" '
            f'stroke-width="3.3"{dashed} marker-end="url(#arrow)"/>')


def build():
    width, height = 1400, 760
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="7" '
        'refX="8.5" refY="3.5" orient="auto"><polygon '
        f'points="0 0, 9 3.5, 0 7" fill="{INK}"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(28, 48, "Non-adaptive actuator-control pipelines", 36, INK,
             "bold", "start"),
        text(1372, 48, "solid: command path  |  dashed: feedback", 25,
             MUTED, "normal", "end"),
    ]

    def block(x, y, w, h, title, lines):
        out.append(rect(x, y, w, h, "#FFFFFF", INK, 2.6))
        out.append(text(x + w / 2, y + 43, title, 34, INK, "bold"))
        for index, line in enumerate(lines):
            out.append(text(x + w / 2, y + 88 + 37 * index, line, 30))

    out.append(rect(20, 78, 1360, 278, PANEL, RULE, 1.6, 0))
    out.append(rect(38, 102, 120, 226, "#FFFFFF", INK, 2.6))
    out.extend([text(98, 178, "R", 52, INK, "bold"),
                text(98, 226, "ROM", 30, INK, "bold"),
                text(98, 275, "baseline", 25, MUTED)])
    block(202, 128, 220, 160, "Policy πᴿ", ["cₖ = πᴿ(oₖ)", "desired motion"])
    block(490, 118, 258, 180, "ROM", ["Δx̂ₖ = fᴿ(sₖ,cₖ)", "calibrated plant"])
    block(816, 128, 220, 160, "Target", ["x*ₖ₊₁ = clip", "xₖ + Δx̂ₖ"])
    block(1104, 128, 238, 160, "PhysX arm", ["xₖ₊₁, oₖ₊₁", "shared task"])
    out.extend([arrow(426, 208, 486, 208), arrow(752, 208, 812, 208),
                arrow(1040, 208, 1100, 208),
                path("M 1346 208 L 1360 208 L 1360 338 L 312 338 L 312 292",
                     MUTED, "10 8")])

    out.append(rect(20, 382, 1360, 330, PANEL, RULE, 1.6, 0))
    out.append(rect(38, 406, 120, 278, "#FFFFFF", INK, 2.6))
    out.extend([text(98, 500, "IF", 50, INK, "bold"),
                text(98, 551, "ID -> FD", 27, INK, "bold"),
                text(98, 605, "baseline", 25, MUTED)])
    block(184, 452, 194, 176, "Policy πᴵᶠ", ["cₖ = πᴵᶠ(oₖ)", "motion command"])
    block(418, 442, 208, 196, "Inverse ID", ["Δuₖ = fᴵᴰ(sₖ,cₖ)", "valve change"])
    block(666, 452, 182, 176, "Valve", ["uₖ = clip", "uₖ₋₁ + Δuₖ"])
    block(888, 442, 208, 196, "Forward FD", ["Δx̂ₖ = fᶠᴰ(sₖ,uₖ)", "strict FD"])
    block(1136, 452, 218, 176, "PhysX", ["x*ₖ₊₁ -> arm", "shared task"])
    for start, end in ((382, 414), (630, 662), (852, 884), (1100, 1132)):
        out.append(arrow(start, 540, end, 540))
    out.append(path("M 1358 540 L 1364 540 L 1364 694 L 281 694 L 281 632",
                    MUTED, "10 8"))
    out.append(text(700, 748,
                    "Same observation, action limits and PhysX task.",
                    27, MUTED))
    out.append('</svg>')
    return "".join(out), width, height


def main():
    svg, width, height = build()
    svg_path = DRAWIO / "ch6_r_if_stack.svg"
    pdf_path = IMG / "ch6_r_if_stack.pdf"
    svg_path.write_text(svg)
    if cairosvg is not None:
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path),
                         output_width=width, output_height=height)
    else:
        chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        html_path = DRAWIO / ".ch6_r_if_stack_print.html"
        html_path.write_text(
            '<meta charset="utf-8">'
            f'<style>@page{{size:{width}px {height}px;margin:0}}'
            'html,body{margin:0;padding:0}</style>' + svg)
        subprocess.run(
            [str(chrome), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        html_path.unlink()
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
