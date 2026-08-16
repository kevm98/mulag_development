#!/usr/bin/env python3
"""Chapter 6 figures: the teacher, the policy contract, and distillation.

Produces
    images/ch6_teacher_student.pdf   schematic: privileged teacher, shared
                                     action chain, distillation, deployed loop
    images/ch6_observation.pdf       the 66-channel observation, by group
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

import cairosvg

BASE = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis")
IMG = BASE / "images"
DRAWIO = BASE / "drawio"
DRAWIO.mkdir(exist_ok=True)

DISTILL_REPORT = Path(
    "/home/rrlab/rrlab_mulag/PPO_ROM/reports/distillation.json"
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
        '</defs>'
    )


def save(svg, name, W, H):
    (DRAWIO / f"{name}.svg").write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(IMG / f"{name}.pdf"),
                     output_width=W, output_height=H)
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
# Figure 2: the 66-channel observation, by group
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
         "#DCE6F2", BLUE, "#1e3f72"),
        (9, 3, "task error",
         "cut-height and lateral error, cutter vertical velocity",
         "#FBE7C6", ORANGE, "#8a5410"),
        (12, 16, "cylinder state",
         "four extensions, four velocities, low/high travel headroom",
         "#E8F1EF", TEAL, "#0e5f58"),
        (28, 16, "actuator interface",
         "previous action, inferred valve, predicted increment, target lead",
         "#E7DFF2", PURPLE, "#4b3170"),
        (44, 10, "pose and context",
         "head alignment and rate, truck speed, scan validity, head axis, "
         "contact force, path preview",
         "#EDEFF1", GRAY, "#4a545e"),
        (54, 12, "model error",
         "cylinder tracking residual, actuator prediction residual, "
         "normalized effort",
         "#F0D2CD", RED, "#7d332c"),
    ]

    x0, x1 = 40, 960
    y, bar_h = 92, 62
    scale = (x1 - x0) / 66

    s.append(T(500, 46, "The 66-value policy observation", 26, INK, "bold"))
    s.append(T(500, 72, "every channel is something a real machine could "
               "measure", 18, MUTED))

    for start_i, width, _n, _d, fill, stroke, dark in groups:
        gx = x0 + start_i * scale
        gw = width * scale
        s.append(rect(gx, y, gw - 4, bar_h, fill, stroke, 2.2, 8))
        s.append(T(gx + gw / 2 - 2, y + 41, str(width),
                   30 if width >= 9 else 22, dark, "bold"))
        s.append(f'<line x1="{gx}" y1="{y + bar_h + 6}" x2="{gx}" '
                 f'y2="{y + bar_h + 16}" stroke="{MUTED}" stroke-width="1.4"/>')
        s.append(T(gx, y + bar_h + 34, str(start_i), 17, MUTED))
    s.append(f'<line x1="{x1}" y1="{y + bar_h + 6}" x2="{x1}" '
             f'y2="{y + bar_h + 16}" stroke="{MUTED}" stroke-width="1.4"/>')
    s.append(T(x1, y + bar_h + 34, "66", 17, MUTED))

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
    W, H = CANVAS, 600
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', defs()]

    def T(x, y, t, size, fill=INK, weight="normal", anchor="middle"):
        return txt(x, y, t, _check(size), fill, weight, anchor)

    s.append(T(500, 44, "From raw policy output to PhysX cylinder target",
               26, INK, "bold"))
    s.append(T(500, 70, "every stage can only REDUCE authority", 18, MUTED))

    stages = [
        ("raw action", "a &#8712; &#8477;&#8308;", "#FBE7C6", ORANGE, "#8a5410"),
        ("rate limit", "|&#916;a| &#8804; 0.25 / step", "#DCE6F2", BLUE, "#1e3f72"),
        ("EMA filter", "&#946; = 0.70", "#DCE6F2", BLUE, "#1e3f72"),
        ("limit shield", "fades over 20 mm", "#DCE6F2", BLUE, "#1e3f72"),
        ("command scale", "&#215; 4 mm per unit", "#E8F1EF", TEAL, "#0e5f58"),
        ("target integrator", "lead &#8804; 18 mm", "#E8F1EF", TEAL, "#0e5f58"),
        ("actuator + PhysX", "ROM | ID&#8594;FD | +adapter", "#E7DFF2",
         PURPLE, "#4b3170"),
    ]

    bw, bh, gap = 216, 78, 18
    rows = [stages[:3], stages[3:6], stages[6:]]
    ytop = 100
    centres = []
    for r, row in enumerate(rows):
        n = len(row)
        total = n * bw + (n - 1) * gap
        x = (W - total) / 2
        yy = ytop + r * (bh + 44)
        for i, (name, detail, fill, stroke, dark) in enumerate(row):
            s.append(rect(x, yy, bw, bh, fill, stroke, 2.2, 10))
            s.append(T(x + bw / 2, yy + 32, name, 19, dark, "bold"))
            s.append(T(x + bw / 2, yy + 58, detail, 17))
            centres.append((x + bw / 2, yy, x, yy + bh))
            if i < n - 1:
                s.append(arrow(x + bw + 2, yy + bh / 2, x + bw + gap - 4,
                               yy + bh / 2))
            x += bw + gap
        if r < len(rows) - 1:
            nxt = ytop + (r + 1) * (bh + 44)
            s.append(f'<path d="M{(W + total) / 2 - bw / 2} {yy + bh} '
                     f'L{(W + total) / 2 - bw / 2} {yy + bh + 22} '
                     f'L{(W - total) / 2 + bw / 2} {yy + bh + 22} '
                     f'L{(W - total) / 2 + bw / 2} {nxt - 4}" fill="none" '
                     f'stroke="{INK}" stroke-width="2.2" '
                     f'marker-end="url(#arrowhead)"/>')

    s.append(rect(40, 440, 920, 140, "#FDF3F2", RED, 1.8, 12))
    s.append(T(500, 470, "The ceiling this implies", 20, "#7d332c", "bold"))
    s.append(T(500, 498, "4 mm of cylinder travel per step is the entire "
               "authority budget, and it bounds", 18))
    s.append(T(500, 522, "the lateral cutter rate at roughly 0.2 m/s. "
               "Reference paths are DERIVED from", 18))
    s.append(T(500, 546, "that ceiling, and a preflight check refuses a run "
               "on a path that demands more.", 18))

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
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.1))

    ax[0].plot(it, prob, color=PURPLE, lw=1.9)
    ax[0].fill_between(it, 0, prob, color=PURPLE, alpha=0.12)
    ax[0].set_xlabel("DAgger iteration")
    ax[0].set_ylabel("teacher execution probability")
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_title("(a) the teacher hands over", fontsize=9.5, loc="left")
    ax[0].axvline(84, color=GRAY, lw=1.0, ls="--")
    ax[0].annotate("student alone\nfrom here", xy=(84, 0.52),
                   xytext=(52, 0.62), fontsize=8, color=MUTED,
                   arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))

    ax[1].semilogy(it, loss, color=BLUE, lw=1.6)
    ax[1].set_xlabel("DAgger iteration")
    ax[1].set_ylabel("behaviour-cloning loss")
    ax[1].set_title("(b) imitation error", fontsize=9.5, loc="left")
    ax[1].axvline(84, color=GRAY, lw=1.0, ls="--")

    ax[2].plot(it, cut, color=ORANGE, lw=1.5, label="cut-height error [m]")
    ax[2].plot(it, lat, color=RED, lw=1.5, label="lateral error [m]")
    ax[2].plot(it, contact, color=TEAL, lw=1.5, label="ground contact [--]")
    ax[2].axhline(pre["mean_abs_lateral_error_m"], color=RED, lw=1.0,
                  ls=":", alpha=0.8)
    ax[2].axhline(pre["ground_contact_fraction"], color=TEAL, lw=1.0,
                  ls=":", alpha=0.8)
    ax[2].set_xlabel("DAgger iteration")
    ax[2].set_ylim(0.0, 1.62)
    ax[2].set_title("(c) task metrics, dotted = teacher", fontsize=9.5,
                    loc="left")
    ax[2].legend(frameon=False, fontsize=7.4, loc="upper right",
                 ncol=1, borderaxespad=0.2, handlelength=1.4)
    ax[2].axvline(84, color=GRAY, lw=1.0, ls="--")

    fig.tight_layout()
    out = IMG / "ch6_distillation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def teacher_student():
    """Image-led training/deployment comparison for Figure 6.1."""
    W, H = 1200, 800
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', defs()]

    def T(x, y, value, size=18, fill=INK, weight="normal", anchor="middle",
          style="normal"):
        return txt(x, y, value, size, fill, weight, anchor, 0, style)

    def href(path):
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    # Training: real task context, privileged DLS teacher, and student actor.
    s += [rect(20, 25, 1160, 405, "#FFFFFF", PURPLE, 2.2, 14),
          T(45, 58, "TRAINING — privileged teacher supervises the student", 24,
            "#4b3170", "bold", "start"),
          rect(40, 80, 300, 275, "#FFFFFF", PURPLE, 1.8, 8),
          f'<image x="50" y="90" width="280" height="210" '
          f'preserveAspectRatio="xMidYMid slice" href="{href(IMG / "fme500_operation.jpg")}"/>',
          rect(55, 302, 270, 45, "#E7DFF2", PURPLE, 1.4, 16),
          T(190, 322, "exact terrain height", 15, "#4b3170", "bold"),
          T(190, 340, "+ true contact force", 15, "#4b3170", "bold")]

    s += [rect(380, 95, 315, 190, "#F3EFFA", PURPLE, 2.2, 10),
          T(538, 128, "PRIVILEGED DLS TEACHER", 19, "#4b3170", "bold"),
          rect(405, 150, 120, 64, "#FFFFFF", PURPLE, 1.5, 8),
          T(465, 176, "contact", 17, "#4b3170", "bold"),
          T(465, 199, "250 N", 16, MUTED),
          rect(550, 150, 120, 64, "#FFFFFF", PURPLE, 1.5, 8),
          T(610, 176, "verge line", 17, "#4b3170", "bold"),
          T(610, 199, "lateral ×3", 16, MUTED),
          T(538, 250, "Δx = 0.30 · J⁺e   |   cylinders 1–3", 17, INK, "bold")]

    s += [rect(780, 95, 360, 190, "#FBE7C6", ORANGE, 2.2, 10),
          T(960, 132, "STUDENT POLICY", 20, "#8a5410", "bold"),
          T(960, 166, "66 measurable observations", 17, INK),
          T(960, 196, "four desired-cylinder actions", 17, INK),
          T(960, 245, "DAgger: teacher execution 1 → 0", 16, MUTED),
          arrow(695, 188, 770, 188, PURPLE, 2.8, marker="arrowpurple"),
          T(733, 174, "BC label", 16, PURPLE, "bold")]

    s += [rect(380, 325, 760, 65, "#DCE6F2", BLUE, 2.2, 10),
          T(760, 353, "SAME ACTION CHAIN", 18, "#1e3f72", "bold"),
          T(760, 378, "rate limit → filter → shield → bounded target", 16, MUTED),
          arrow(538, 285, 538, 315, PURPLE, 2.4, marker="arrowpurple"),
          arrow(960, 285, 960, 315, ORANGE, 2.4)]

    # Deployment: the distilled actor is the only controller that survives.
    s += [rect(20, 465, 1160, 310, "#FFFFFF", TEAL, 2.2, 14),
          T(45, 500, "DEPLOYMENT — the policy is the whole controller", 24,
            "#0e5f58", "bold", "start"),
          rect(45, 525, 250, 55, "#EDEFF1", GRAY, 1.8, 8, dash="7 6"),
          T(170, 549, "teacher gone", 18, MUTED, "bold"),
          T(170, 572, "no nominal command", 16, MUTED, "normal", "middle", "italic"),
          rect(45, 605, 250, 105, "#FBE7C6", ORANGE, 2.2, 10),
          T(170, 642, "DISTILLED ACTOR", 19, "#8a5410", "bold"),
          T(170, 674, "observations → action", 17, INK),
          arrow(300, 657, 390, 657, ORANGE, 3.0)]

    s += [rect(405, 605, 300, 105, "#DCE6F2", BLUE, 2.2, 10),
          T(555, 640, "ACTION CHAIN", 19, "#1e3f72", "bold"),
          T(555, 671, "same bounded contract", 17, INK),
          arrow(710, 657, 770, 657, BLUE, 3.0),
          rect(785, 525, 355, 205, "#FFFFFF", TEAL, 2.0, 8),
          f'<image x="800" y="540" width="325" height="150" '
          f'preserveAspectRatio="xMidYMid meet" href="{href(IMG / "unimog_mulag_iso.png")}"/>',
          T(962, 712, "ROM | ID→FD | ID+AAM→FD → PhysX", 16, "#0e5f58", "bold"),
          arrow(962, 730, 170, 730, GRAY, 2.0, dash="8 6", marker="arrowgray"),
          T(565, 757, "measured plant state closes the deployed loop", 16, MUTED)]

    s.append('</svg>')
    save("".join(s), "ch6_teacher_student", W, H)


if __name__ == "__main__":
    teacher_student()
    observation()
    action_chain()
    distillation()
