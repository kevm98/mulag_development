#!/usr/bin/env python3
"""Regenerate the replacement figures requested for Chapters 2--5.

The diagrams are deliberately source-generated rather than hand-edited
screenshots.  Technical line work is exported as PDF/SVG; the CAD catalog
uses the authoritative URDF renders already stored in ``images/``.  The two
Chapter-4 plots are re-framed from their existing vector PDFs so no data are
changed while their page whitespace is removed.

Run from the repository root with the thesis figure environment::

    ~/miniconda3/envs/thesis_figures/bin/python \
      "thesis/thesis/hydraulic_manipulator_thesis/scripts/\
      make_thesis_replacement_figures.py"
"""
from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

BASE = Path(Path.home() / "rrlab_mulag/thesis/thesis/hydraulic_manipulator_thesis")
IMG = BASE / "images"
DRAWIO = BASE / "drawio"
IMG.mkdir(parents=True, exist_ok=True)
DRAWIO.mkdir(parents=True, exist_ok=True)

INK = "#1F2933"
MUTED = "#66727E"
GRID = "#D7DDE3"
BLUE = "#2F5DA8"
ORANGE = "#C97816"
TEAL = "#16847A"
PURPLE = "#7A52A8"
RED = "#B64B42"
GREEN = "#4E9A3D"
PALE_BLUE = "#EAF0FA"
PALE_ORANGE = "#FFF4E5"
PALE_TEAL = "#EAF7F4"
PALE_PURPLE = "#F1EAF7"
FONT = "DejaVu Sans"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def save_fig(fig: plt.Figure, stem: str, *, svg: bool = False) -> None:
    """Save a tight, publication-resolution PDF plus a preview PNG."""
    fig.savefig(IMG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(IMG / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    if svg:
        fig.savefig(DRAWIO / f"{stem}.svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def ax_off(figsize=(12.0, 5.2), xlim=(0, 12), ylim=(0, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def box(ax, xy, wh, text, *, fc="white", ec=INK, lw=1.5, radius=0.12,
        color=INK, size=10, weight="normal", sub=None, sub_color=MUTED):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.03,rounding_size={radius}",
                           facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2 + (0.09 if sub else 0), text,
            ha="center", va="center", color=color, fontsize=size,
            weight=weight, wrap=True)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.18, sub, ha="center", va="center",
                color=sub_color, fontsize=max(size - 1.5, 7.5), wrap=True)
    return patch


def arrow(ax, start, end, *, color=MUTED, lw=1.8, ls="-", rad=0.0,
          mutation=12, zorder=3):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=mutation,
                            linewidth=lw, linestyle=ls, color=color,
                            connectionstyle=f"arc3,rad={rad}", zorder=zorder)
    ax.add_patch(patch)
    return patch


def section_label(ax, x, y, text, color=INK, size=11):
    ax.text(x, y, text.upper(), ha="left", va="bottom", color=color,
            fontsize=size, weight="bold")
    ax.plot([x, x + 2.0], [y - 0.08, y - 0.08], color=color, lw=1.1)


def fig_2_2_actuation_chain() -> None:
    fig, ax = ax_off(figsize=(13.0, 5.1), xlim=(0, 13), ylim=(0, 5.1))
    ax.text(0.15, 4.82, "From valve command to tool motion", fontsize=15,
            weight="bold", color=INK, ha="left")
    ax.text(0.15, 4.55,
            "The effective gain changes with pressure, direction, and linkage configuration.",
            fontsize=10.5, color=MUTED, ha="left")
    section_label(ax, 0.25, 3.9, "electrical", BLUE)
    section_label(ax, 3.1, 3.9, "hydraulic", ORANGE)
    section_label(ax, 6.85, 3.9, "mechanical", TEAL)

    box(ax, (0.25, 2.15), (1.45, 1.0), "command $u$", fc=PALE_BLUE,
        ec=BLUE, color=BLUE, size=13, weight="bold", sub="normalised spool input")
    box(ax, (2.05, 2.15), (1.55, 1.0), "4/3 valve", fc=PALE_ORANGE,
        ec=ORANGE, color=ORANGE, size=12, weight="bold", sub="dead zone + lag")
    box(ax, (4.0, 2.15), (1.75, 1.0), "chambers", fc=PALE_ORANGE,
        ec=ORANGE, color=ORANGE, size=12, weight="bold", sub="$p_1,p_2$; compressibility")
    box(ax, (6.15, 2.15), (1.65, 1.0), "cylinder", fc=PALE_ORANGE,
        ec=ORANGE, color=ORANGE, size=12, weight="bold", sub=r"$x,\dot{x},F_h$")
    box(ax, (8.25, 2.15), (1.75, 1.0), "linkage", fc=PALE_TEAL,
        ec=TEAL, color=TEAL, size=12, weight="bold", sub="configuration-dependent")
    box(ax, (10.45, 2.15), (2.0, 1.0), "mowing head", fc=PALE_TEAL,
        ec=TEAL, color=TEAL, size=12, weight="bold", sub="task-space motion")
    for a, b, col in [((1.7, 2.65), (2.05, 2.65), BLUE),
                      ((3.6, 2.65), (4.0, 2.65), ORANGE),
                      ((5.75, 2.65), (6.15, 2.65), ORANGE),
                      ((7.8, 2.65), (8.25, 2.65), TEAL),
                      ((10.0, 2.65), (10.45, 2.65), TEAL)]:
        arrow(ax, a, b, color=col, lw=2.2)

    # Flow rails and physical feedback are shown as explicit, readable loops.
    ax.plot([4.05, 5.7], [1.7, 1.7], color=RED, lw=2.0)
    ax.plot([4.05, 4.05], [1.7, 2.15], color=RED, lw=2.0)
    ax.plot([5.7, 5.7], [1.7, 2.15], color=RED, lw=2.0)
    ax.text(4.88, 1.49, r"pressure differential $\Delta p$", ha="center",
            va="top", color=RED, fontsize=9.5)
    arrow(ax, (4.0, 1.9), (3.0, 2.15), color=RED, lw=1.5, ls="--", rad=0.2)
    ax.text(2.73, 1.78, "load pressure throttles flow", ha="center", va="top",
            color=RED, fontsize=8.5, rotation=12)
    arrow(ax, (6.95, 2.15), (5.0, 1.65), color=TEAL, lw=1.5, ls="--", rad=-0.18)
    ax.text(6.55, 1.35, "piston motion consumes flow", ha="center", va="top",
            color=TEAL, fontsize=8.5)

    box(ax, (0.55, 0.45), (2.2, 0.72), "directional gain", fc="#F7F8FA",
        ec=BLUE, color=INK, size=10.5, weight="bold", sub=r"$q \propto x_v\sqrt{|p_s-p_{load}|}$")
    box(ax, (3.45, 0.45), (2.2, 0.72), "fluid storage", fc="#F7F8FA",
        ec=ORANGE, color=INK, size=10.5, weight="bold", sub=r"$\beta/V(x)$ sets pressure buildup")
    box(ax, (6.35, 0.45), (2.2, 0.72), "load coupling", fc="#F7F8FA",
        ec=TEAL, color=INK, size=10.5, weight="bold", sub="gravity, friction, cutting load")
    box(ax, (9.25, 0.45), (2.85, 0.72), "operating point", fc="#F7F8FA",
        ec=PURPLE, color=INK, size=10.5, weight="bold", sub="same command \u2260 same motion")
    save_fig(fig, "fund_actuation_chain_v2", svg=True)


def fig_2_3_workflow() -> None:
    fig, ax = ax_off(figsize=(13.0, 4.7), xlim=(0, 13), ylim=(0, 4.7))
    ax.text(0.15, 4.43, "Division of labour across the thesis", fontsize=15,
            weight="bold", color=INK, ha="left")
    ax.text(0.15, 4.16,
            "High-fidelity hydraulics provide evidence; compact models provide throughput.",
            fontsize=10.5, color=MUTED, ha="left")
    cards = [
        (0.25, BLUE, PALE_BLUE, "1", "Model what is known", "IsaacLab articulation\ntask-space kinematics", "Chapter 3"),
        (3.35, ORANGE, PALE_ORANGE, "2", "Learn the actuator", "Simscape trajectories\nopen-loop excitation", "Chapters 2 + 4"),
        (6.45, TEAL, PALE_TEAL, "3", "Train the policy", "parallel surrogate\nmatched action contract", "Chapter 6"),
        (9.55, PURPLE, PALE_PURPLE, "4", "Evaluate matched", "tracking, delay,\nrobustness, adaptation", "Chapter 7"),
    ]
    for x, col, fc, n, title, body, chapter in cards:
        box(ax, (x, 1.15), (2.35, 2.45), "", fc=fc, ec=col, lw=1.7, radius=0.16)
        ax.add_patch(Circle((x + 0.28, 3.28), 0.18, facecolor=col, edgecolor=col))
        ax.text(x + 0.28, 3.28, n, color="white", ha="center", va="center",
                fontsize=10, weight="bold")
        ax.text(x + 1.18, 3.22, title, color=INK, ha="center", va="center",
                fontsize=12, weight="bold")
        ax.text(x + 1.18, 2.22, body, color=INK, ha="center", va="center",
                fontsize=10.5, linespacing=1.45)
        ax.text(x + 1.18, 1.48, chapter, color=col, ha="center", va="center",
                fontsize=9.5, weight="bold")
        if x < 9.55:
            arrow(ax, (x + 2.42, 2.38), (x + 3.0, 2.38), color=MUTED, lw=2.0)
    ax.plot([1.4, 11.8], [0.68, 0.68], color=GRID, lw=1.3)
    ax.text(6.6, 0.36,
            "one shared contract: desired cylinder motion  →  bounded PhysX target",
            ha="center", va="center", color=MUTED, fontsize=10.5)
    save_fig(fig, "fund_workflow_v2", svg=True)


def fig_2_4_robust_adaptive() -> None:
    fig, ax = ax_off(figsize=(13.0, 5.6), xlim=(0, 13), ylim=(0, 5.6))
    ax.text(0.15, 5.33, "Robustness and adaptation answer different questions",
            fontsize=15, weight="bold", color=INK, ha="left")
    ax.text(0.15, 5.06,
            "Both treatments see the same randomized plant family; only one is allowed to specialise online.",
            fontsize=10.5, color=MUTED, ha="left")
    # Shared family row.
    ax.text(0.3, 4.4, "randomized plant family", color=MUTED, fontsize=10,
            weight="bold", ha="left")
    for i, (lab, col) in enumerate([("nominal", BLUE), ("soft / leaky", ORANGE),
                                    ("stiff / loaded", TEAL), ("wear + load", PURPLE)]):
        x = 0.5 + i * 1.3
        ax.add_patch(Circle((x, 3.95), 0.35, facecolor="white", edgecolor=col,
                            linewidth=2.0))
        ax.text(x, 3.95, f"P{i+1}", ha="center", va="center", color=col,
                fontsize=9, weight="bold")
        ax.text(x, 3.38, lab, ha="center", va="center", color=MUTED, fontsize=8.5)
        arrow(ax, (x, 3.57), (x, 3.0), color=GRID, lw=1.2)
    # left lane
    box(ax, (0.35, 0.8), (5.7, 2.15), "", fc="#F8FAFC", ec=BLUE, lw=1.7,
        radius=0.16)
    ax.text(0.65, 2.62, "(a) robust training", color=BLUE, fontsize=12,
            weight="bold", ha="left")
    box(ax, (0.65, 1.3), (1.65, 0.72), "one fixed model", fc=PALE_BLUE,
        ec=BLUE, color=BLUE, size=11, weight="bold", sub="same response everywhere")
    arrow(ax, (2.4, 1.66), (3.0, 1.66), color=BLUE, lw=1.8)
    box(ax, (3.12, 1.12), (2.45, 1.08), "acceptable family-wide\nresponse band", fc="white",
        ec=BLUE, color=INK, size=10.5, weight="bold", sub="robust, not specialised")
    ax.text(0.65, 0.98, "plant factors vary during training; the deployed mapping is fixed",
            color=MUTED, fontsize=8.8, ha="left")
    # right lane
    box(ax, (6.45, 0.8), (6.15, 2.15), "", fc="#FBF9FD", ec=PURPLE, lw=1.7,
        radius=0.16)
    ax.text(6.75, 2.62, "(b) latent adaptation", color=PURPLE, fontsize=12,
            weight="bold", ha="left")
    box(ax, (6.75, 1.42), (1.45, 0.72), "history", fc=PALE_BLUE, ec=BLUE,
        color=BLUE, size=10.5, weight="bold", sub="states + commands")
    arrow(ax, (8.3, 1.78), (8.72, 1.78), color=MUTED, lw=1.8)
    box(ax, (8.82, 1.42), (1.45, 0.72), "encoder", fc=PALE_PURPLE, ec=PURPLE,
        color=PURPLE, size=10.5, weight="bold", sub="student module")
    arrow(ax, (10.38, 1.78), (10.82, 1.78), color=MUTED, lw=1.8)
    box(ax, (10.92, 1.42), (1.35, 0.72), "$\\hat{z}$", fc=PALE_PURPLE,
        ec=PURPLE, color=PURPLE, size=13, weight="bold", sub="compact context")
    ax.text(6.75, 0.98, "the downstream mapping changes with the inferred plant context",
            color=MUTED, fontsize=8.8, ha="left")
    arrow(ax, (7.4, 3.45), (9.25, 2.94), color=PURPLE, lw=1.4, ls="--", rad=-0.2)
    ax.text(8.1, 3.55, "privileged factors $\\xi$ supervise training only",
            color=PURPLE, fontsize=8.8, ha="center")
    ax.plot([6.6, 12.35], [0.55, 0.55], color=GRID, lw=1.1)
    ax.text(6.6, 0.25,
            "deployment: observable history only  |  no gradient update  |  bounded correction",
            color=MUTED, fontsize=9.0, ha="left")
    save_fig(fig, "fund_robust_adaptive_v2", svg=True)


def crop_rgba(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    alpha = im.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        pad = 40
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))
        im = im.crop(bbox)
    return im


def fig_3_1_topology() -> None:
    fig, ax = ax_off(figsize=(13.0, 6.0), xlim=(0, 13), ylim=(0, 6.0))
    ax.text(0.15, 5.72, "FME 500 mechanism: links, joints, and actuators",
            fontsize=15, weight="bold", color=INK, ha="left")
    ax.text(0.15, 5.45,
            "The fifth revolute joint is passive; cylinders C1--C4 are the learned actuator coordinates.",
            fontsize=10.5, color=MUTED, ha="left")
    img = Image.open(IMG / "fme500_topology.png").convert("RGB")
    ax.imshow(img, extent=(0.2, 8.55, 0.8, 5.15), aspect="auto")
    ax.text(0.25, 5.18, "(a) CAD-derived topology", color=INK, fontsize=11,
            weight="bold", ha="left")
    # right-side chain, deliberately large and sparse
    ax.text(9.05, 5.18, "(b) coordinate chain", color=INK, fontsize=11,
            weight="bold", ha="left")
    names = [("L0", "front frame"), ("L1", "slew column"),
             ("L2", "inner boom"), ("L3", "outer boom"),
             ("L4", "head carrier"), ("L5", "mowing head")]
    x, bw, bh = 9.25, 2.35, 0.48
    for i, (tag, name) in enumerate(names):
        y = 4.55 - i * 0.68
        box(ax, (x, y), (bw, bh), f"{tag}  {name}", fc="#F7F8FA", ec=MUTED,
            color=INK, size=10.5, weight="bold")
        if i < len(names) - 1:
            col = PURPLE if i == 4 else BLUE
            ax.plot([x + bw / 2, x + bw / 2], [y, y - 0.20], color=col, lw=2)
            ax.add_patch(Circle((x + bw / 2, y - 0.31), 0.11,
                                facecolor="white" if i == 4 else PALE_BLUE,
                                edgecolor=col, linewidth=1.8,
                                linestyle="--" if i == 4 else "-"))
            ax.text(x + bw + 0.12, y - 0.31, f"J{i+1}", color=col,
                    fontsize=9.5, weight="bold", va="center")
            if i < 4:
                ax.text(x - 0.23, y - 0.31, f"C{i+1}", color=ORANGE,
                        fontsize=9.5, weight="bold", va="center", ha="right")
    ax.text(9.25, 0.7, "blue: actuated joint   purple dashed: passive joint   orange: cylinder",
            color=MUTED, fontsize=8.8, ha="left")
    save_fig(fig, "fme500_topology_annotated_v2", svg=True)


def fig_3_3_link_catalog() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 9.2))
    axes = axes.ravel()
    entries = [
        ("link_frontrahmen.png", "L0", "front frame"),
        ("link_drehzapfen.png", "L1", "slew column"),
        ("link_ausleger_i.png", "L2", "inner boom"),
        ("link_ausleger_ii.png", "L3", "outer boom"),
        ("link_messerkopf_schwenk.png", "L4", "head carrier"),
        ("link_messerkopf.png", "L5", "mowing head"),
        ("link_cylinder.png", "C1--C4", "double-acting cylinder"),
    ]
    for ax, (file, tag, name) in zip(axes, entries):
        ax.set_facecolor("#F8FAFC")
        im = crop_rgba(IMG / "links" / file)
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID); sp.set_linewidth(1.0)
        ax.text(0.05, 0.95, tag, transform=ax.transAxes, ha="left", va="top",
                color=ORANGE if tag.startswith("C") else BLUE, fontsize=11,
                weight="bold")
        ax.text(0.05, 0.05, name, transform=ax.transAxes, ha="left", va="bottom",
                color=INK, fontsize=10, weight="bold")
    for ax in axes[len(entries):]:
        ax.axis("off")
    fig.suptitle("URDF link catalog — common orthographic render and scale",
                 fontsize=15, weight="bold", x=0.07, ha="left", color=INK)
    fig.text(0.07, 0.925, "The actuator cylinder is shown once as a representative barrel–rod pair.",
             fontsize=10.5, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.055, right=0.98, top=0.88, bottom=0.05, wspace=0.12, hspace=0.18)
    save_fig(fig, "fme500_link_catalog_v2", svg=False)


def fig_3_4_hydraulic_schematic() -> None:
    fig, ax = ax_off(figsize=(13.0, 6.0), xlim=(0, 13), ylim=(0, 6.0))
    ax.text(0.15, 5.73, "System-level hydraulic circuit", fontsize=15,
            weight="bold", color=INK, ha="left")
    ax.text(0.15, 5.46,
            "Four identical proportional channels share the supply and return rails of one power unit.",
            fontsize=10.5, color=MUTED, ha="left")
    # common rails
    ax.plot([9.95, 9.95], [1.0, 4.65], color=RED, lw=3)
    ax.plot([10.45, 10.45], [1.0, 4.65], color=BLUE, lw=3)
    ax.text(9.95, 4.88, "P", color=RED, fontsize=11, weight="bold", ha="center")
    ax.text(10.45, 4.88, "T", color=BLUE, fontsize=11, weight="bold", ha="center")
    box(ax, (9.0, 0.95), (2.4, 0.72), "hydraulic power unit", fc=PALE_BLUE,
        ec=BLUE, color=BLUE, size=11, weight="bold", sub="pump · accumulator · relief · reservoir")
    for i, (c, joint, name) in enumerate([(ORANGE, "J1", "slew"), (BLUE, "J2", "lift"),
                                            (TEAL, "J3", "knuckle"), (PURPLE, "J4", "head pivot")]):
        y = 4.18 - i * 0.92
        ax.text(0.25, y + 0.22, f"C{i+1}", color=ORANGE, fontsize=10.5,
                weight="bold", va="center")
        box(ax, (0.75, y - 0.08), (1.65, 0.58), f"{name} valve", fc=PALE_ORANGE,
            ec=ORANGE, color=INK, size=10.5, weight="bold", sub="4/3 proportional")
        arrow(ax, (2.45, y + 0.21), (3.05, y + 0.21), color=ORANGE, lw=1.7)
        box(ax, (3.1, y - 0.08), (1.75, 0.58), "double-acting\ncylinder", fc=PALE_ORANGE,
            ec=ORANGE, color=INK, size=10, weight="bold")
        arrow(ax, (4.9, y + 0.21), (5.45, y + 0.21), color=TEAL, lw=1.7)
        box(ax, (5.5, y - 0.08), (1.75, 0.58), f"{joint} linkage", fc=PALE_TEAL,
            ec=TEAL, color=INK, size=10.5, weight="bold", sub="mechanical load")
        ax.plot([7.25, 9.95], [y + 0.36, y + 0.36], color=RED, lw=1.5)
        ax.plot([7.25, 10.45], [y + 0.06, y + 0.06], color=BLUE, lw=1.5)
        ax.plot([9.95, 9.95], [y + 0.36, 4.65], color=RED, lw=1.0)
        ax.plot([10.45, 10.45], [y + 0.06, 4.65], color=BLUE, lw=1.0)
        ax.text(7.8, y + 0.43, "A", color=RED, fontsize=8.5, va="bottom")
        ax.text(7.8, y - 0.03, "B", color=BLUE, fontsize=8.5, va="top")
    ax.text(7.45, 5.0, "shared supply / return rails", color=MUTED, fontsize=9.5,
            ha="left")
    ax.text(0.25, 0.38, "No fifth hydraulic branch: J5 is passive and mechanically coupled to the head carrier.",
            color=MUTED, fontsize=9.5, ha="left")
    save_fig(fig, "fme500_hydraulic_schematic_v2", svg=True)


def fig_3_5_actuator_channel() -> None:
    fig, ax = ax_off(figsize=(13.0, 5.9), xlim=(0, 13), ylim=(0, 5.9))
    ax.text(0.15, 5.65, "One electro-hydraulic actuator channel", fontsize=15,
            weight="bold", color=INK, ha="left")
    ax.text(0.15, 5.38,
            "The same cylinder coordinate x is the interface between the rigid-body and fluid submodels.",
            fontsize=10.5, color=MUTED, ha="left")
    # mechanical band
    box(ax, (0.3, 3.35), (12.35, 1.35), "", fc="#F8FAFC", ec=TEAL, lw=1.6, radius=0.16)
    ax.text(0.6, 4.42, "mechanical interface", color=TEAL, fontsize=11.5,
            weight="bold", ha="left")
    box(ax, (0.75, 3.65), (1.55, 0.58), "link k", fc="white", ec=MUTED,
        color=INK, size=11, weight="bold", sub="revolute")
    arrow(ax, (2.4, 3.94), (3.0, 3.94), color=TEAL, lw=1.8)
    box(ax, (3.05, 3.55), (2.25, 0.78), "barrel + rod", fc=PALE_ORANGE,
        ec=ORANGE, color=ORANGE, size=11, weight="bold", sub="prismatic coordinate x")
    arrow(ax, (5.4, 3.94), (6.0, 3.94), color=TEAL, lw=1.8)
    box(ax, (6.05, 3.65), (1.55, 0.58), "link k+1", fc="white", ec=MUTED,
        color=INK, size=11, weight="bold", sub="revolute")
    arrow(ax, (7.75, 3.94), (9.0, 3.94), color=ORANGE, lw=1.5, ls="--")
    ax.text(8.4, 4.20, "$F_h$", color=ORANGE, fontsize=11, weight="bold", ha="center")
    ax.text(9.25, 3.94, "$x,\dot{x}$", color=TEAL, fontsize=11, weight="bold", va="center")
    box(ax, (10.2, 3.55), (1.9, 0.78), "external load", fc=PALE_TEAL,
        ec=TEAL, color=TEAL, size=11, weight="bold", sub="gravity + cutting")
    # hydraulic band
    box(ax, (0.3, 0.55), (12.35, 2.3), "", fc="#FBF9F6", ec=ORANGE, lw=1.6, radius=0.16)
    ax.text(0.6, 2.57, "hydraulic interface", color=ORANGE, fontsize=11.5,
            weight="bold", ha="left")
    box(ax, (0.7, 1.35), (1.35, 0.68), "command $u$", fc=PALE_BLUE,
        ec=BLUE, color=BLUE, size=11, weight="bold", sub="spool lag")
    arrow(ax, (2.15, 1.69), (2.8, 1.69), color=BLUE, lw=1.8)
    box(ax, (2.85, 1.25), (1.75, 0.88), "4/3 valve", fc=PALE_ORANGE,
        ec=ORANGE, color=ORANGE, size=11, weight="bold", sub="ports P · T · A · B")
    arrow(ax, (4.7, 1.69), (5.35, 1.69), color=ORANGE, lw=1.8)
    box(ax, (5.4, 1.25), (2.05, 0.88), "chamber 1 / 2", fc=PALE_ORANGE,
        ec=ORANGE, color=ORANGE, size=11, weight="bold", sub="$p_1,Q_1$  |  $p_2,Q_2$")
    arrow(ax, (7.6, 1.69), (8.15, 1.69), color=ORANGE, lw=1.8)
    box(ax, (8.2, 1.25), (1.65, 0.88), "continuity", fc="white", ec=ORANGE,
        color=INK, size=11, weight="bold", sub=r"$\beta/V(x)$")
    arrow(ax, (10.0, 1.69), (10.55, 1.69), color=TEAL, lw=1.8)
    box(ax, (10.6, 1.25), (1.45, 0.88), "force", fc=PALE_TEAL,
        ec=TEAL, color=TEAL, size=11, weight="bold", sub="$F_h$")
    arrow(ax, (6.4, 3.35), (6.4, 2.18), color=TEAL, lw=1.5, ls="--")
    ax.text(6.65, 2.72, "pressure → force", color=TEAL, fontsize=9.2,
            rotation=90, va="center")
    ax.text(0.62, 0.80, "supply P", color=RED, fontsize=9, ha="left")
    ax.text(2.15, 0.80, "return T", color=BLUE, fontsize=9, ha="left")
    save_fig(fig, "fme500_actuator_channel_detailed_v2", svg=True)


def fig_5_1_model_overview() -> None:
    fig, ax = ax_off(figsize=(13.0, 5.15), xlim=(0, 13), ylim=(0, 5.15))
    ax.text(0.15, 4.87, "Two learned actuator representations, one deployment contract",
            fontsize=15, weight="bold", color=INK, ha="left")
    ax.text(0.15, 4.60,
            "The experimental variable is the actuator representation; the policy, state bus, and PhysX target are shared.",
            fontsize=10.5, color=MUTED, ha="left")
    box(ax, (0.35, 3.60), (2.15, 0.78), "desired command $c_k$", fc=PALE_BLUE,
        ec=BLUE, color=BLUE, size=12, weight="bold", sub="$\u0394x^*$, $\u0394\\dot{x}^*$")
    box(ax, (0.35, 2.25), (2.15, 0.78), "measured state bus", fc="#F7F8FA",
        ec=MUTED, color=INK, size=11.5, weight="bold", sub="$x,\\dot{x},\\Delta p, F$")
    # shared input split
    box(ax, (3.05, 3.02), (1.35, 0.82), "policy", fc=PALE_PURPLE,
        ec=PURPLE, color=PURPLE, size=12, weight="bold", sub="same PPO action")
    arrow(ax, (2.55, 3.98), (3.05, 3.48), color=MUTED, lw=1.7)
    arrow(ax, (2.55, 2.64), (3.05, 3.38), color=MUTED, lw=1.7)
    # upper lane
    box(ax, (4.85, 3.38), (2.15, 0.82), "ID network", fc=PALE_BLUE,
        ec=BLUE, color=BLUE, size=11.5, weight="bold", sub="command → valve")
    arrow(ax, (4.4, 3.42), (4.85, 3.80), color=BLUE, lw=1.8)
    box(ax, (7.35, 3.38), (2.15, 0.82), "FD network", fc=PALE_BLUE,
        ec=BLUE, color=BLUE, size=11.5, weight="bold", sub="valve + state → Δx")
    arrow(ax, (7.0, 3.80), (7.35, 3.80), color=BLUE, lw=1.8)
    ax.text(4.85, 4.37, "PPO–ID–FD lane", color=BLUE, fontsize=11.5,
            weight="bold", ha="left")
    # lower lane
    box(ax, (4.85, 1.52), (4.65, 0.95), "command-conditioned ROM", fc=PALE_TEAL,
        ec=TEAL, color=TEAL, size=12, weight="bold", sub="internal valve head + pressure / velocity / position update")
    arrow(ax, (4.4, 3.16), (4.85, 2.10), color=TEAL, lw=1.8)
    ax.text(4.85, 2.72, "PPO–ROM lane", color=TEAL, fontsize=11.5,
            weight="bold", ha="left")
    # shared endpoint
    box(ax, (10.0, 2.52), (2.45, 1.18), "bounded PhysX target", fc=PALE_ORANGE,
        ec=ORANGE, color=ORANGE, size=12, weight="bold", sub="same cylinder-position interface")
    arrow(ax, (9.55, 3.80), (10.0, 3.20), color=BLUE, lw=1.8)
    arrow(ax, (9.55, 1.99), (10.0, 2.87), color=TEAL, lw=1.8)
    arrow(ax, (12.45, 2.98), (12.75, 2.98), color=ORANGE, lw=1.8)
    ax.text(12.72, 3.22, "plant", color=ORANGE, fontsize=10, ha="right", weight="bold")
    arrow(ax, (11.2, 2.48), (1.45, 2.25), color=MUTED, lw=1.5, ls="--", rad=0.08)
    ax.text(6.75, 0.72, "feedback bus closes the loop; only the actuator representation changes between the two lanes",
            ha="center", va="center", color=MUTED, fontsize=10)
    save_fig(fig, "models_overview_v2", svg=True)


def fig_5_2_acceptance_flow() -> None:
    fig, ax = ax_off(figsize=(13.0, 4.65), xlim=(0, 13), ylim=(0, 4.65))
    ax.text(0.15, 4.38, "Acceptance ladder for every learned model", fontsize=15,
            weight="bold", color=INK, ha="left")
    ax.text(0.15, 4.11,
            "Promotion is a gated evidence decision, not a consequence of one-step supervised accuracy.",
            fontsize=10.5, color=MUTED, ha="left")
    gates = [(0.3, BLUE, PALE_BLUE, "1", "offline accuracy", "one-step + short rollout"),
             (3.35, ORANGE, PALE_ORANGE, "2", "input authority", "zeroed / shuffled command"),
             (6.4, TEAL, PALE_TEAL, "3", "in-simulator gates", "runtime + safety contract"),
             (9.45, PURPLE, PALE_PURPLE, "4", "task-space gate", "tracking + contact")]
    for x, col, fc, n, title, sub in gates:
        box(ax, (x, 2.12), (2.25, 1.15), "", fc=fc, ec=col, lw=1.7, radius=0.16)
        ax.add_patch(Circle((x + 0.28, 3.03), 0.18, facecolor=col, edgecolor=col))
        ax.text(x + 0.28, 3.03, n, ha="center", va="center", color="white",
                fontsize=10, weight="bold")
        ax.text(x + 1.22, 2.87, title, ha="center", va="center", color=INK,
                fontsize=11.5, weight="bold")
        ax.text(x + 1.22, 2.48, sub, ha="center", va="center", color=MUTED,
                fontsize=9.4)
        if x < 9.45:
            arrow(ax, (x + 2.3, 2.70), (x + 2.9, 2.70), color=MUTED, lw=2)
    box(ax, (10.0, 0.55), (2.45, 0.72), "promote checkpoint", fc=PALE_TEAL,
        ec=TEAL, color=TEAL, size=11.5, weight="bold", sub="hash + provenance")
    arrow(ax, (10.55, 2.05), (11.15, 1.30), color=TEAL, lw=1.8)
    # repair loop is intentionally below the ladder, not tangled through it.
    ax.plot([1.4, 1.4, 11.6, 11.6], [2.03, 1.48, 1.48, 2.03], color=RED,
            lw=1.3, ls="--")
    ax.text(6.5, 1.63, "any failure exits to a named repair and re-enters at gate 1",
            ha="center", va="center", color=RED, fontsize=9.5)
    ax.text(0.35, 0.42, "offline score → causal authority → runtime validity → task outcome",
            color=MUTED, fontsize=10, ha="left")
    save_fig(fig, "models_acceptance_flow_v2", svg=True)


def dense_column(ax, x, y0, n, color, radius=0.055, spread=0.27):
    ys = np.linspace(y0 - spread, y0 + spread, n)
    for y in ys:
        ax.add_patch(Circle((x, y), radius, facecolor="white", edgecolor=color,
                            linewidth=0.8))
    return ys


def fig_5_3_nn_architectures() -> None:
    fig, ax = ax_off(figsize=(13.0, 6.0), xlim=(0, 13), ylim=(0, 6.0))
    ax.text(0.15, 5.73, "Network families used by the actuator models", fontsize=15,
            weight="bold", color=INK, ha="left")
    ax.text(0.15, 5.46,
            "The ID/FD pair shares a residual trunk; the ROM carries history through a recurrent latent.",
            fontsize=10.5, color=MUTED, ha="left")
    # MLP panel
    box(ax, (0.3, 0.55), (5.95, 4.45), "", fc="#F8FAFC", ec=BLUE, lw=1.6, radius=0.16)
    ax.text(0.6, 4.68, "(a) residual MLP trunk", color=BLUE, fontsize=12,
            weight="bold", ha="left")
    cols = [(1.0, 4, "input", BLUE), (2.1, 6, "projection", BLUE),
            (3.2, 6, "residual ×4", TEAL), (4.3, 6, "residual ×4", TEAL),
            (5.45, 4, "head", PURPLE)]
    prev = None
    for x, n, lab, col in cols:
        ys = dense_column(ax, x, 2.75, n, col, radius=0.07, spread=0.78 if n > 4 else 0.52)
        ax.text(x, 1.02, lab, ha="center", va="center", color=INK, fontsize=9.4,
                weight="bold")
        if prev is not None:
            for y1 in prev:
                for y2 in ys:
                    ax.plot([prev_x + 0.08, x - 0.08], [y1, y2], color=GRID,
                            lw=0.35, alpha=0.7)
        prev, prev_x = ys, x
    ax.annotate("identity skip", xy=(4.3, 4.18), xytext=(3.2, 4.18),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4,
                                connectionstyle="arc3,rad=0.32"), color=TEAL,
                fontsize=9.2, ha="center")
    ax.text(3.25, 1.45, "Linear → LayerNorm → SiLU → Dropout → Linear",
            ha="center", va="center", color=MUTED, fontsize=9.2)
    ax.text(3.25, 0.85, "only the input and output widths differ between ID and FD",
            ha="center", va="center", color=MUTED, fontsize=8.7)
    # ROM panel
    box(ax, (6.6, 0.55), (6.1, 4.45), "", fc="#FBF9FD", ec=PURPLE, lw=1.6, radius=0.16)
    ax.text(6.9, 4.68, "(b) recurrent ROM encoder", color=PURPLE, fontsize=12,
            weight="bold", ha="left")
    for i, (y, lab, col) in enumerate([(3.9, "state", BLUE), (3.35, "prev. valve", ORANGE),
                                       (2.8, "command", PURPLE)]):
        ax.add_patch(Rectangle((7.0, y), 1.2, 0.32, facecolor=col, alpha=0.18,
                               edgecolor=col, linewidth=1.2))
        ax.text(7.6, y + 0.16, lab, ha="center", va="center", color=INK,
                fontsize=9.4, weight="bold")
    ax.text(7.6, 2.32, "6-step history", ha="center", va="center", color=MUTED,
            fontsize=9.3)
    for i, x in enumerate([8.7, 9.55, 10.4]):
        ys = dense_column(ax, x, 3.1, 5, TEAL, radius=0.065, spread=0.55)
        ax.text(x, 2.25, ["t−5", "…", "t"][i], ha="center", va="center",
                color=TEAL, fontsize=9.2, weight="bold")
        if i:
            for a in prev:
                for b in ys:
                    ax.plot([prev_x + 0.08, x - 0.08], [a, b], color=GRID,
                            lw=0.35, alpha=0.65)
        prev, prev_x = ys, x
    ax.text(9.55, 4.25, "GRU 2 × 128", ha="center", va="center", color=TEAL,
            fontsize=10, weight="bold")
    for y, col, label in [(3.95, BLUE, "actuation head"), (3.25, PURPLE, "residual heads"),
                           (2.55, ORANGE, "load head")]:
        box(ax, (11.1, y - 0.2), (1.25, 0.45), label, fc="white", ec=col,
            color=INK, size=8.7, weight="bold")
        arrow(ax, (10.78, 3.1), (11.1, y), color=col, lw=1.1)
    ax.text(9.6, 1.42, "history → latent hₖ → bounded physics-aware heads",
            ha="center", va="center", color=MUTED, fontsize=9.2)
    ax.text(9.6, 0.85, "the hidden state carries pressure and command transients",
            ha="center", va="center", color=MUTED, fontsize=8.7)
    save_fig(fig, "models_nn_architectures_v2", svg=True)


def fig_5_6_rom_pipeline() -> None:
    fig, ax = ax_off(figsize=(13.0, 4.8), xlim=(0, 13), ylim=(0, 4.8))
    ax.text(0.15, 4.52, "Command-conditioned reduced-order model", fontsize=15,
            weight="bold", color=INK, ha="left")
    ax.text(0.15, 4.25,
            "Neural heads propose; the explicit pressure–velocity–position update disposes.",
            fontsize=10.5, color=MUTED, ha="left")
    box(ax, (0.25, 1.55), (1.2, 1.35), "history $H_k$", fc=PALE_BLUE,
        ec=BLUE, color=BLUE, size=11.5, weight="bold", sub="6 × 24 features")
    arrow(ax, (1.55, 2.22), (1.9, 2.22), color=MUTED, lw=1.8)
    # feature groups
    for y, label, col, fc in [(2.72, "state", BLUE, "#DCE6F5"),
                              (2.18, "prev. valve", ORANGE, "#FBE7C6"),
                              (1.64, "command", PURPLE, "#E9DEF3")]:
        box(ax, (1.95, y - 0.18), (1.35, 0.38), label, fc=fc, ec=col,
            color=INK, size=9.5, weight="bold")
    arrow(ax, (3.42, 2.22), (3.8, 2.22), color=MUTED, lw=1.8)
    box(ax, (3.85, 1.56), (0.85, 1.32), "normalise", fc="#F7F8FA",
        ec=MUTED, color=INK, size=9.5, weight="bold")
    arrow(ax, (4.8, 2.22), (5.15, 2.22), color=MUTED, lw=1.8)
    box(ax, (5.2, 1.38), (1.65, 1.68), "GRU encoder", fc=PALE_TEAL,
        ec=TEAL, color=TEAL, size=11, weight="bold", sub="2 × 128; hₖ")
    arrow(ax, (6.95, 2.22), (7.3, 2.22), color=MUTED, lw=1.8)
    box(ax, (7.35, 1.12), (1.65, 0.46), "actuation head → û", fc=PALE_BLUE,
        ec=BLUE, color=INK, size=9.5, weight="bold")
    box(ax, (7.35, 1.72), (1.65, 0.46), "residual heads", fc=PALE_PURPLE,
        ec=PURPLE, color=INK, size=9.5, weight="bold")
    box(ax, (7.35, 2.32), (1.65, 0.46), "load head", fc=PALE_ORANGE,
        ec=ORANGE, color=INK, size=9.5, weight="bold")
    arrow(ax, (9.15, 2.22), (9.48, 2.22), color=MUTED, lw=1.8)
    box(ax, (9.52, 1.16), (1.85, 2.1), "physical update", fc=PALE_TEAL,
        ec=TEAL, color=TEAL, size=11, weight="bold", sub="orifice flow\npressure\nvelocity\nposition")
    arrow(ax, (11.47, 2.22), (11.82, 2.22), color=MUTED, lw=1.8)
    box(ax, (11.87, 1.48), (0.9, 1.48), "Δx̂", fc=PALE_ORANGE,
        ec=ORANGE, color=ORANGE, size=14, weight="bold", sub="target")
    ax.text(6.5, 0.55,
            "state groups are observable; the previous valve is the model’s recurrent state; all residuals are bounded",
            ha="center", va="center", color=MUTED, fontsize=9.8)
    save_fig(fig, "models_rom_pipeline_v2", svg=True)


def reframe_pdf(stem: str) -> None:
    """Tighten an existing vector figure without changing its data."""
    src = IMG / f"{stem}.pdf"
    dst = IMG / f"{stem}_v2.pdf"
    with tempfile.TemporaryDirectory(prefix="thesisfig_") as td:
        svg = Path(td) / f"{stem}.svg"
        subprocess.run(["pdftocairo", "-svg", str(src), str(svg)], check=True)
        subprocess.run(["inkscape", str(svg), "--export-filename", str(dst),
                        "--export-area-drawing", "--export-text-to-path"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("reframed", dst)


def crop_unimog_views() -> None:
    """Crop the empty render margins; underlying USD render is unchanged."""
    crops = {
        "unimog_mulag_iso": (70, 300, 2350, 1650),
        "unimog_mulag_rear": (45, 245, 2360, 1640),
    }
    for stem, crop in crops.items():
        im = Image.open(IMG / f"{stem}.png")
        im.crop(crop).save(IMG / f"{stem}_v2.png", optimize=True)
        print("cropped", stem)


def main() -> None:
    fig_2_2_actuation_chain()
    fig_2_3_workflow()
    fig_2_4_robust_adaptive()
    fig_3_1_topology()
    fig_3_3_link_catalog()
    fig_3_4_hydraulic_schematic()
    fig_3_5_actuator_channel()
    fig_5_1_model_overview()
    fig_5_2_acceptance_flow()
    fig_5_3_nn_architectures()
    fig_5_6_rom_pipeline()
    reframe_pdf("ch4_closed_loop_trace")
    reframe_pdf("ch4_regimes")
    crop_unimog_views()
    print("replacement figures written to", IMG)


if __name__ == "__main__":
    main()
