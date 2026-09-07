#!/usr/bin/env python3
"""Chapter 6 figures: the teacher, the policy contract, and distillation.

Produces
    images/ch6_teacher_student.pdf   schematic: privileged teacher, shared
                                     action chain, distillation, deployed loop
    images/ch6_observation.pdf       shared 66-channel observation plus the
                                     four AF-only context columns
    images/ch6_action_chain.pdf      raw action -> PhysX cylinder target
    images/ch6_distillation.pdf      measured DAgger history (real data)

The first three are hand-built SVG in the house idiom of
``make_ch4_architecture.py``; the fourth plots the stored distillation
record and fabricates nothing.

Run:  /home/rrlab/miniconda3/envs/thesis_figures/bin/python \
          scripts/make_ch6_figures.py
"""
from pathlib import Path
import json
import base64
import subprocess

try:
    import cairosvg
except ImportError:  # macOS thesis workstation: use installed Chrome instead
    cairosvg = None

BASE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
IMG = BASE / "images" / "ch6"
IMG.mkdir(parents=True, exist_ok=True)
DRAWIO = BASE / "drawio"
DRAWIO.mkdir(exist_ok=True)

DISTILL_REPORT = (
    REPO / "archive/ppo_rom_retired_2026-08-16/PPO_ROM/reports/distillation.json"
)

ORANGE, BLUE, TEAL, PURPLE, RED, GRAY = (
    "#C97816", "#2F5DA8", "#16847A", "#7A52A8", "#B64B42", "#69737D")
INK, MUTED = "#1F2933", "#66727E"
FONT = "DejaVu Sans, Arial, sans-serif"


def rect(x, y, w, h, fill, stroke, sw=2.0, rx=12, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
            f'{d}/>')


def txt(x, y, s, size=15, fill=INK, weight="normal", anchor="middle", rot=0,
        style="normal"):
    tr = f' transform="rotate({rot} {x} {y})"' if rot else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" font-style="{style}" '
            f'text-anchor="{anchor}"{tr}>{s}</text>')


def arrow(x1, y1, x2, y2, color=INK, sw=2.2, dash=None, marker="arrowhead"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def defs():
    return (
        '<defs>'
        '<marker id="arrowhead" markerWidth="9" markerHeight="7" refX="8.5" '
        'refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" '
        f'fill="{INK}"/></marker>'
        '<marker id="arrowgray" markerWidth="9" markerHeight="7" refX="8.5" '
        'refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" '
        f'fill="{GRAY}"/></marker>'
        '<marker id="arrowpurple" markerWidth="9" markerHeight="7" refX="8.5" '
        'refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" '
        f'fill="{PURPLE}"/></marker>'
        '<marker id="arrowblue" markerWidth="9" markerHeight="7" refX="8.5" '
        'refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" '
        f'fill="{BLUE}"/></marker>'
        '<marker id="arroworange" markerWidth="9" markerHeight="7" refX="8.5" '
        'refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" '
        f'fill="{ORANGE}"/></marker>'
        '<marker id="arrowteal" markerWidth="9" markerHeight="7" refX="8.5" '
        'refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" '
        f'fill="{TEAL}"/></marker>'
        '</defs>'
    )


def save(svg, name, W, H):
    svg_path = DRAWIO / f"{name}.svg"
    pdf_path = IMG / f"{name}.pdf"
    svg_path.write_text(svg)
    if cairosvg is not None:
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path),
                         output_width=W, output_height=H)
    else:
        chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        html_path = DRAWIO / f".{name}_print.html"
        html_path.write_text(
            f'<style>@page{{size:{W}px {H}px;margin:0}}'
            'html,body{margin:0;padding:0}</style>' + svg)
        subprocess.run(
            [str(chrome), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        html_path.unlink()
    print("wrote", IMG / f"{name}.pdf")


# --------------------------------------------------------------------------
# LEGIBILITY BUDGET
# The thesis text block is ~15.4 cm.  A figure drawn on a canvas W wide and
# included at \textwidth renders a font of f px as  f/W * 15.4 cm.  Nothing
# below ~7.5 pt (0.265 cm) survives printing, so on a 1000 px canvas no font
# may be smaller than 17 px.  Every figure here obeys that.
# --------------------------------------------------------------------------
CANVAS = 1000
MIN_FONT = 17


def _check(size):
    if size < MIN_FONT:
        raise ValueError(f"font {size} px below the {MIN_FONT} px legibility "
                         f"budget for a {CANVAS} px canvas")
    return size


# --------------------------------------------------------------------------
# Figure 1: teacher, student, and what each is allowed to see
# --------------------------------------------------------------------------
def teacher_student():
    W, H = CANVAS, 1010
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', defs()]

    def T(x, y, t, size, fill=INK, weight="normal", anchor="middle",
          style="normal"):
        return txt(x, y, t, _check(size), fill, weight, anchor, 0, style)

    # ---------------- training time (top) ----------------
    s.append(rect(24, 24, 952, 470, "#FFFFFF", GRAY, 1.8, 16, dash="8 7"))
    s.append(T(500, 56, "TRAINING TIME", 22, MUTED, "bold"))

    s.append(rect(52, 74, 896, 62, "#E7DFF2", PURPLE, 2.2, 10))
    s.append(T(500, 100, "Privileged inputs &#8212; simulator only", 20,
               "#4b3170", "bold"))
    s.append(T(500, 124, "exact terrain height under the cutter  &#183;  "
               "true cutter&#8211;ground contact force", 17, "#4b3170"))

    s.append(rect(52, 154, 896, 214, "#F3EFFA", PURPLE, 2.2, 14))
    s.append(T(500, 182, "Privileged DLS teacher", 22, "#4b3170", "bold"))

    s.append(rect(72, 198, 420, 76, "#FFFFFF", PURPLE, 1.6, 9))
    s.append(T(282, 222, "Contact regulator", 19, "#4b3170", "bold"))
    s.append(T(282, 245, "holds cutter force near 250 N", 17))
    s.append(T(282, 265, "sets the vertical offset", 17))

    s.append(rect(508, 198, 420, 76, "#FFFFFF", PURPLE, 1.6, 9))
    s.append(T(718, 222, "Line follower", 19, "#4b3170", "bold"))
    s.append(T(718, 245, "verge centre line", 17))
    s.append(T(718, 265, "lateral error weighted &#215;3", 17))

    s.append(rect(72, 286, 856, 68, "#FFFFFF", PURPLE, 1.6, 9))
    s.append(T(500, 311, "DLS inverse of the measured cutter Jacobian", 19,
               "#4b3170", "bold"))
    s.append(T(500, 337, "&#916;x = 0.30 &#183; J&#8314;e,   "
               "|&#916;x| &#8804; 3 mm/step,   cylinders 1&#8211;3", 17))

    s.append(arrow(500, 356, 500, 384))
    s.append(rect(52, 388, 896, 44, "#DCE6F2", BLUE, 2.2, 10))
    s.append(T(500, 416, "the SAME action chain the student uses", 19,
               "#1e3f72", "bold"))
    s.append(arrow(500, 434, 500, 452))
    s.append(rect(52, 456, 896, 44, "#E8F1EF", TEAL, 2.2, 10))
    s.append(T(500, 484, "command ROM + PhysX &#8212; the deployment plant",
               19, "#0e5f58", "bold"))

    # BC label arrow, down the left margin, into the student box
    s.append('<path d="M72 330 L36 330 L36 560 L60 560" fill="none" '
             f'stroke="{PURPLE}" stroke-width="2.4" stroke-dasharray="7 6" '
             'marker-end="url(#arrowpurple)"/>')

    # ---------------- student (middle) ----------------
    s.append(T(250, 520, "behaviour-cloning label", 17, PURPLE, "bold"))
    s.append(rect(48, 528, 904, 64, "#FBE7C6", ORANGE, 2.2, 10))
    s.append(T(500, 554, "Student policy, fitted by DAgger on the teacher's "
               "raw action", 20, "#8a5410", "bold"))
    s.append(T(500, 578, "teacher execution annealed 1 &#8594; 0 over the "
               "first 70 % of iterations", 17))
    s.append(arrow(500, 594, 500, 622, ORANGE))
    s.append(T(608, 616, "distilled actor", 17, "#8a5410", "bold"))

    # ---------------- deployment (bottom) ----------------
    s.append(rect(24, 632, 952, 356, "#FFFFFF", GRAY, 1.8, 16, dash="8 7"))
    s.append(T(500, 664, "DEPLOYMENT", 22, MUTED, "bold"))

    s.append(rect(52, 682, 440, 62, "#EDEFF1", GRAY, 2.2, 10, dash="7 6"))
    s.append(T(272, 707, "Privileged inputs", 19, MUTED, "bold"))
    s.append(T(272, 730, "never read", 17, MUTED, "normal", "middle",
               "italic"))
    s.append(rect(508, 682, 440, 62, "#EDEFF1", GRAY, 2.2, 10, dash="7 6"))
    s.append(T(728, 707, "The teacher is GONE", 19, MUTED, "bold"))
    s.append(T(728, 730, "no nominal command is added", 17, MUTED, "normal",
               "middle", "italic"))

    s.append(rect(52, 764, 896, 60, "#FBE7C6", ORANGE, 2.2, 10))
    s.append(T(500, 790, "Policy &#8212; the whole controller", 20,
               "#8a5410", "bold"))
    s.append(T(500, 812, "66 observations &#8594; four desired cylinder "
               "commands", 17))
    s.append(arrow(500, 826, 500, 850))

    s.append(rect(52, 854, 896, 44, "#DCE6F2", BLUE, 2.2, 10))
    s.append(T(500, 882, "the SAME action chain", 19, "#1e3f72", "bold"))
    s.append(arrow(500, 900, 500, 920))
    s.append(rect(52, 924, 896, 44, "#E8F1EF", TEAL, 2.2, 10))
    s.append(T(500, 952, "ROM   |   ID &#8594; FD   |   ID + adapter(z&#770;) "
               "&#8594; FD   &#8594;   PhysX", 19, "#0e5f58", "bold"))

    # deployed feedback loop
    s.append('<path d="M948 946 L976 946 L976 794 L952 794" fill="none" '
             f'stroke="{GRAY}" stroke-width="2.2" stroke-dasharray="7 6" '
             'marker-end="url(#arrowgray)"/>')

    s.append('</svg>')
    save("".join(s), "ch6_teacher_student", W, H)


# --------------------------------------------------------------------------
# Figure 2: shared observation and the AF-only context extension
# --------------------------------------------------------------------------
def observation():
    W, H = CANVAS, 730
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', defs()]

    def T(x, y, t, size, fill=INK, weight="normal", anchor="middle"):
        return txt(x, y, t, _check(size), fill, weight, anchor)

    groups = [
        (0, 9, "terrain profile",
         "nine forward height bins ahead of the cutter",
         "#F1F4F7", "#8EA4B7", INK),
        (9, 3, "task error",
         "cut-height and lateral error, cutter vertical velocity",
         "#F8F3EA", "#B8A078", INK),
        (12, 16, "cylinder state",
         "four extensions, four velocities, low/high travel headroom",
         "#EFF5F4", "#88A8A3", INK),
        (28, 16, "actuator interface",
         "previous action, inferred valve, predicted increment, target lead",
         "#F3F0F6", "#A596B3", INK),
        (44, 10, "pose and context",
         "head alignment and rate, truck speed, scan validity, head axis, "
         "contact force, path preview",
         "#F3F4F5", "#A0A7AD", INK),
        (54, 12, "model error",
         "cylinder tracking residual, actuator prediction residual, "
         "normalized effort",
         "#F8EFEF", "#B99490", INK),
    ]

    x0, base_x1, x1 = 40, 900, 960
    y, bar_h = 100, 62
    scale = (base_x1 - x0) / 66

    s.append(T(500, 44, "The policy observation contract", 26, INK, "bold"))
    s.append(T((x0 + base_x1) / 2, 78, "SHARED BY R, IF AND AF", 17,
               MUTED, "bold"))
    s.append(T((base_x1 + x1) / 2, 78, "AF ONLY", 17, RED, "bold"))

    for start_i, width, _n, _d, fill, stroke, dark in groups:
        gx = x0 + start_i * scale
        gw = width * scale
        s.append(rect(gx, y, gw - 4, bar_h, fill, stroke, 2.2, 8))
        s.append(T(gx + gw / 2 - 2, y + 41, str(width),
                   30 if width >= 9 else 22, dark, "bold"))
        s.append(f'<line x1="{gx}" y1="{y + bar_h + 6}" x2="{gx}" '
                 f'y2="{y + bar_h + 16}" stroke="{MUTED}" stroke-width="1.4"/>')
        s.append(T(gx, y + bar_h + 34, str(start_i), 17, MUTED))
    s.append(f'<line x1="{base_x1}" y1="{y + bar_h + 6}" x2="{base_x1}" '
             f'y2="{y + bar_h + 16}" stroke="{MUTED}" stroke-width="1.4"/>')
    s.append(T(base_x1, y + bar_h + 34, "66", 17, MUTED))

    # The context extension is deliberately unfilled: it is absent from R and
    # IF and is appended only to the AF actor.  Draw the four inputs separately
    # so they cannot be mistaken for another shared observation group.
    context_gap = 2
    context_w = (x1 - base_x1 - 3 * context_gap) / 4
    for index in range(4):
        gx = base_x1 + index * (context_w + context_gap)
        s.append(rect(gx, y, context_w, bar_h, "#FFFFFF", RED, 2.2, 5))
    s.append(f'<line x1="{x1}" y1="{y + bar_h + 6}" x2="{x1}" '
             f'y2="{y + bar_h + 16}" stroke="{MUTED}" stroke-width="1.4"/>')
    s.append(T(x1, y + bar_h + 34, "70", 17, MUTED))

    s.append(rect(665, 226, 295, 86, "#FFFFFF", RED, 2.0, 8))
    s.append(T(682, 251, "plant context &#8212; channels 66:70", 18,
               INK, "bold", "start"))
    s.append(T(682, 275, "four AF-only inputs; payload is the", 17,
               INK, "normal", "start"))
    s.append(T(682, 296, "only non-constant channel here", 17,
               INK, "normal", "start"))

    ry = 226
    for (start_i, width, name, detail, fill, stroke, dark) in groups:
        s.append(rect(40, ry, 22, 22, fill, stroke, 1.8, 5))
        s.append(T(74, ry + 18,
                   f"{name} &#8212; channels {start_i}:{start_i + width}",
                   19, dark, "bold", "start"))
        words = detail.split()
        line, lines = "", []
        for w in words:
            trial = (line + " " + w).strip()
            if len(trial) > 74:
                lines.append(line)
                line = w
            else:
                line = trial
        lines.append(line)
        ty = ry + 40
        for ln in lines:
            s.append(T(74, ty, ln, 17, INK, "normal", "start"))
            ty += 20
        ry = ty + 14

    s.append('</svg>')
    save("".join(s), "ch6_observation", W, H)


# --------------------------------------------------------------------------
# Figure 3: the action chain
# --------------------------------------------------------------------------
def action_chain():
    """PPO actor--critic formulation and bounded cylinder-action contract."""
    W, H = 1200, 700
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', defs()]

    def T(x, y, t, size, fill=INK, weight="normal", anchor="middle"):
        if size < 20:
            size = 20
        return txt(x, y, t, size, fill, weight, anchor)

    s.append(T(600, 40, "PPO formulation and bounded cylinder control", 30,
               INK, "bold"))

    # Observation tensor, drawn as a small stack in the spirit of the supplied
    # architectural references without importing their encoder--decoder model.
    for dx, dy in ((12, -12), (6, -6)):
        s.append(rect(46 + dx, 263 + dy, 150, 126, "#FFFFFF", GRAY, 1.3, 7))
    s.append(rect(46, 263, 150, 126, "#EAF3FC", BLUE, 2.2, 7))
    s.append(T(121, 302, "Observation", 23, INK, "bold"))
    s.append(T(121, 335, "oₖ", 26, BLUE, "bold"))
    s.append(T(121, 366, "66 / 68 channels", 20, INK))

    # Layered actor and critic blocks. The unequal widths communicate the
    # actual 256--128--64 architecture more clearly than a dense node drawing.
    def network(x, y, title, fill, edge, subtitle):
        s.append(rect(x, y, 220, 128, fill, edge, 2.2, 8))
        s.append(T(x + 110, y + 30, title, 23, INK, "bold"))
        xs = (x + 20, x + 87, x + 154)
        for xx, label in zip(xs, ("256", "128", "64")):
            s.append(rect(xx, y + 52, 46, 28, "#FFFFFF", edge, 1.2, 4))
            s.append(T(xx + 23, y + 73, label, 20, INK, "bold"))
        s.append(T(x + 110, y + 110, subtitle, 20, INK))

    network(244, 92, "Actor  πθ", "#F3ECFA", PURPLE, "ELU hidden layers")
    network(244, 422, "Critic  Vφ", "#EDF4FB", BLUE, "training only")

    # Clean branches from one observation; neither branch crosses a box.
    s.append('<path d="M196 326 L218 326 L218 156 L240 156" fill="none" '
             f'stroke="{BLUE}" stroke-width="2.5" marker-end="url(#arrowblue)"/>')
    s.append('<path d="M196 326 L218 326 L218 486 L240 486" fill="none" '
             f'stroke="{BLUE}" stroke-width="2.5" marker-end="url(#arrowblue)"/>')

    s.append(rect(505, 92, 210, 128, "#FFF2E3", ORANGE, 2.2, 8))
    s.append(T(610, 126, "Gaussian policy", 23, INK, "bold"))
    s.append(T(610, 163, "aₖ ~ N(μθ(oₖ), σ²)", 21, INK))
    s.append(T(610, 196, "aₖ ∈ [-1, 1]⁴", 21, INK, "bold"))
    s.append(arrow(468, 156, 501, 156, ORANGE, 2.5, marker="arroworange"))

    # One grouped safety contract gives the stages room while retaining the
    # numerical limits that make the interface scientifically meaningful.
    s.append(rect(758, 70, 390, 176, "#EAF7F4", TEAL, 2.4, 10))
    s.append(T(953, 104, "Bounded action interface", 24, INK, "bold"))
    s.append(T(953, 139, "rate 0.25  →  EMA 0.70  →  travel shield", 21))
    s.append(T(953, 174, "Δx*ₖ = 4 mm · ãₖ", 23, INK, "bold"))
    s.append(T(953, 208, "target lead ≤ 50 / 20 mm", 21, INK))
    s.append(arrow(719, 156, 754, 156, TEAL, 2.5, marker="arrowteal"))

    s.append(rect(798, 300, 310, 88, "#F2ECF9", PURPLE, 2.2, 8))
    s.append(T(953, 335, "Actuator realization", 23, INK, "bold"))
    s.append(T(953, 368, "R   |   IF   |   AF", 22, PURPLE, "bold"))
    s.append(arrow(953, 250, 953, 296, PURPLE, 2.5, marker="arrowpurple"))

    s.append(rect(798, 450, 310, 92, "#EEF8F1", "#5F946D", 2.3, 8))
    s.append(T(953, 486, "PhysX arm and terrain", 23, INK, "bold"))
    s.append(T(953, 520, "oₖ₊₁,  rₖ", 22, "#477A54", "bold"))
    s.append(arrow(953, 392, 953, 446, PURPLE, 2.5, marker="arrowpurple"))

    s.append(rect(515, 438, 185, 92, "#EDF4FB", BLUE, 2.0, 8))
    s.append(T(607, 474, "Value estimate", 21, INK, "bold"))
    s.append(T(607, 506, "Vφ(oₖ)", 22, BLUE, "bold"))
    s.append(arrow(468, 486, 518, 486, GRAY, 2.0, marker="arrowgray"))

    # Reward and next-state feedback are separated vertically and terminate at
    # the correct consumers; this avoids the crossing lines of the old figure.
    s.append('<path d="M798 496 L742 496 L742 582 L354 582 L354 554" fill="none" '
             f'stroke="{GRAY}" stroke-width="2.0" stroke-dasharray="7 6" '
             'marker-end="url(#arrowgray)"/>')
    s.append(T(552, 608, "reward rₖ updates actor and critic during training", 20,
               MUTED))
    s.append('<path d="M500 582 L500 248 L354 248 L354 224" fill="none" '
             f'stroke="{GRAY}" stroke-width="1.8" stroke-dasharray="7 6" '
             'marker-end="url(#arrowgray)"/>')
    s.append('<path d="M953 546 L953 654 L121 654 L121 395" fill="none" '
             f'stroke="{INK}" stroke-width="2.1" marker-end="url(#arrowhead)"/>')
    s.append(T(548, 678, "measured next observation oₖ₊₁", 21, INK))

    s.append('</svg>')
    save("".join(s), "ch6_action_chain", W, H)


# --------------------------------------------------------------------------
# Figure 4: measured distillation history (real stored data)
# --------------------------------------------------------------------------
def distillation():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(DISTILL_REPORT.read_text())
    hist = report["history"]
    it = [h["iteration"] for h in hist]
    prob = [h["teacher_execute_probability"] for h in hist]
    loss = [h["bc_loss"] for h in hist]
    cut = [h["mean_abs_cut_height_error_m"] for h in hist]
    lat = [h["mean_abs_lateral_error_m"] for h in hist]
    contact = [h["ground_contact_fraction"] for h in hist]
    pre = report["preflight"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(3, 1, figsize=(8.2, 10.0))

    ax[0].plot(it, prob, color="#777181", lw=2.4)
    ax[0].fill_between(it, 0, prob, color="#E2DFE7", alpha=0.72)
    ax[0].set_xlabel("DAgger iteration")
    ax[0].set_ylabel("teacher execution probability")
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_title("(a) Teacher-to-student handover", fontsize=13,
                    loc="left", fontweight="bold")
    ax[0].axvline(84, color=GRAY, lw=1.0, ls="--")
    ax[0].annotate("student alone\nfrom here", xy=(84, 0.52),
                   xytext=(50, 0.66), fontsize=11, color=MUTED,
                   arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))

    ax[1].semilogy(it, loss, color="#7189A6", lw=2.2)
    ax[1].set_xlabel("DAgger iteration")
    ax[1].set_ylabel("behaviour-cloning loss")
    ax[1].set_title("(b) Imitation error", fontsize=13, loc="left",
                    fontweight="bold")
    ax[1].axvline(84, color=GRAY, lw=1.0, ls="--")
    ax[1].annotate("minimum near handover", xy=(82, min(loss)),
                   xytext=(48, min(loss) * 2.4), fontsize=11, color=MUTED,
                   arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))

    # Restore the measured task metrics below the two enlarged training plots.
    # Muted hues distinguish quantities without turning the figure into a
    # presentation-style colour chart.
    metric_colors = {
        "cut": "#B69670", "lateral": "#B97E7E", "contact": "#719D96"
    }
    ax[2].plot(it, cut, color=metric_colors["cut"], lw=2.0,
               label="cut-height error [m]")
    ax[2].plot(it, lat, color=metric_colors["lateral"], lw=2.0,
               label="lateral error [m]")
    ax[2].plot(it, contact, color=metric_colors["contact"], lw=2.0,
               label="ground contact [-]")
    for key, value in (
        ("cut", pre["mean_abs_cut_height_error_m"]),
        ("lateral", pre["mean_abs_lateral_error_m"]),
        ("contact", pre["ground_contact_fraction"]),
    ):
        ax[2].axhline(value, color=metric_colors[key], lw=1.25,
                      ls=":", alpha=0.9)
    ax[2].axvline(84, color=GRAY, lw=1.0, ls="--")
    ax[2].set_xlabel("DAgger iteration")
    ax[2].set_ylabel("task metric")
    ax[2].set_ylim(0.0, 1.05)
    ax[2].set_title("(c) Closed-loop task metrics", fontsize=13, loc="left",
                    fontweight="bold", pad=38)
    # Keep every label outside the data region.  The caption identifies the
    # dotted teacher-preflight references, so no explanatory text is needed
    # over the curves themselves.
    ax[2].legend(frameon=False, fontsize=10.2, loc="lower left", ncol=3,
                 bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0,
                 handlelength=1.8, columnspacing=1.2)

    fig.tight_layout(h_pad=2.0)
    out = IMG / "ch6_distillation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def teacher_student():
    """Training-only teacher and deployable closed-loop controller."""
    W, H = 1200, 720
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', defs()]

    def T(x, y, value, size=18, fill=INK, weight="normal", anchor="middle",
          style="normal"):
        return txt(x, y, value, max(size, 21), fill, weight, anchor, 0, style)

    def href(path):
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    s.append(T(600, 38, "Training supervision and deployed control loop", 30,
               INK, "bold"))

    # Everything in this dashed lane is simulator-only and disappears after
    # distillation. The restrained grayscale treatment makes that boundary
    # immediately legible.
    s += [rect(28, 65, 1144, 220, "#FFFFFF", "#D59A55", 1.8, 12,
               dash="8 7"),
          T(52, 96, "TRAINING ONLY", 20, MUTED, "bold", "start"),
          rect(60, 120, 250, 112, "#F7F7F7", INK, 1.8, 8),
          T(185, 153, "Privileged simulation", 20, INK, "bold"),
          T(185, 184, "terrain height", 18, MUTED),
          T(185, 211, "true contact force", 18, MUTED),
          rect(382, 104, 250, 148, "#FFFFFF", INK, 2.0, 8),
          T(507, 138, "DLS teacher", 22, INK, "bold"),
          T(507, 176, "&#916;x* = clip(", 27, INK, "bold"),
          T(507, 207, "g J# W e)", 27, INK, "bold"),
          T(507, 238, "cylinders 1&#8211;4; C4 held", 17, MUTED),
          rect(704, 120, 230, 112, "#F7F7F7", INK, 1.8, 8),
          T(819, 153, "DAgger labels", 20, INK, "bold"),
          T(819, 186, "teacher execution", 18, MUTED),
          T(819, 213, "1 &#8594; 0", 19, INK, "bold"),
          arrow(314, 176, 378, 176, INK, 2.3),
          arrow(636, 176, 700, 176, INK, 2.3)]

    # Deployable loop. The only colour is inside the two simulation renders.
    s += [T(52, 330, "DEPLOYMENT", 20, INK, "bold", "start"),
          rect(42, 365, 166, 112, "#FFFFFF", INK, 2.0, 8),
          T(125, 402, "Observation", 20, INK, "bold"),
          T(125, 438, "o&#8342;", 23, INK, "bold"),
          T(125, 463, "measurable", 17, MUTED),
          rect(258, 365, 166, 112, "#F2F2F2", INK, 2.2, 8),
          T(341, 403, "PPO actor", 21, INK, "bold"),
          T(341, 439, "&#960;&#952;(o&#8342;)", 21, INK),
          T(341, 464, "no nominal input", 17, MUTED),
          arrow(212, 421, 254, 421, INK, 2.4)]

    s += [rect(474, 345, 196, 152, "#FFFFFF", INK, 2.0, 8),
          f'<image x="489" y="354" width="166" height="92" '
          f'preserveAspectRatio="xMidYMid meet" href="{href(IMG / "links/link_cylinder.png")}"/>',
          T(572, 469, "desired cylinder", 19, INK, "bold"),
          T(572, 492, "motion", 19, INK, "bold"),
          arrow(428, 421, 470, 421, INK, 2.4),
          rect(720, 365, 166, 112, "#F7F7F7", INK, 2.0, 8),
          T(803, 401, "Actuator path", 20, INK, "bold"),
          T(803, 438, "R | IF | AF", 20, INK, "bold"),
          T(803, 464, "bounded target", 17, MUTED),
          arrow(674, 421, 716, 421, INK, 2.4)]

    s += [rect(936, 330, 222, 205, "#FFFFFF", INK, 2.0, 8),
          f'<image x="946" y="340" width="202" height="150" '
          f'preserveAspectRatio="xMidYMid meet" href="{href(IMG / "fme500_cad_iso_orange.png")}"/>',
          T(1047, 514, "PhysX arm plant", 19, INK, "bold"),
          arrow(890, 421, 932, 421, INK, 2.4)]

    # The teacher supervises the actor but never enters the deployed signal
    # path. Measured state closes the lower loop with ample clearance.
    s.append('<path d="M819 236 L819 300 L341 300 L341 361" fill="none" '
             f'stroke="{PURPLE}" stroke-width="2.0" stroke-dasharray="7 6" '
             'marker-end="url(#arrowpurple)"/>')
    s.append(T(575, 291, "supervision", 17, MUTED))
    s.append('<path d="M1047 539 L1047 630 L125 630 L125 481" fill="none" '
             f'stroke="{INK}" stroke-width="2.2" marker-end="url(#arrowhead)"/>')
    s.append(T(590, 658, "measured state closes the deployed loop", 18, MUTED))

    s.append('</svg>')
    save("".join(s), "ch6_teacher_student", W, H)


if __name__ == "__main__":
    teacher_student()
    observation()
    action_chain()
    distillation()
