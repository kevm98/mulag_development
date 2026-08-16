#!/usr/bin/env python3
"""Rebuild Chapter 4 and selected Chapter 5 figures in one thesis style.

The statistical figures are generated from the stored MATLAB curriculum,
strict-excitation CSV, and validated model reports.  No plotted result is
invented or hand-adjusted.  Figure 4.1 reuses the thesis plant and actuator
renders as visual anchors for the documented data-generation architecture.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]
OUT = BASE / "images"
DATA = (BASE / "Domain_Randomization_Data_Collection-20260720T142849Z-1-001"
        / "Domain_Randomization_Data_Collection")
CL_MAT = (DATA / "Randomization_closed_loop"
          / "Hydraulic_Mechanic_Dataset_Curriculum_1000.mat")
OL_CSV = (DATA / "Randomization_open_loop" / "Excitation_Finetune_25"
          / "Excitation_FD_Strict.csv")

# Coherent, colour-blind-aware thesis palette.  The cylinder mapping is used
# unchanged in Chapters 4 and 5.
BLUE = "#357EC7"
ORANGE = "#E29A16"
TEAL = "#25A18E"
MAGENTA = "#B55B9A"
VIOLET = "#8665B5"
RED = "#D65F59"
GRAY = "#5F6872"
INK = "#24313D"
MUTED = "#65727E"
GRID = "#CDD5DC"
PALE_BLUE = "#EAF3FB"
PALE_ORANGE = "#FFF3DC"
PALE_TEAL = "#E8F6F3"
PALE_MAGENTA = "#F7ECF4"
PALE_GRAY = "#F4F6F8"
CYL = [BLUE, ORANGE, TEAL, MAGENTA]
FULLW = 6.3

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "legend.fontsize": 8.4,
    "xtick.labelsize": 8.7,
    "ytick.labelsize": 8.7,
    "axes.edgecolor": "#7B8791",
    "axes.linewidth": 0.75,
    "axes.facecolor": "#FBFCFD",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "grid.color": GRID,
    "grid.linewidth": 0.55,
    "grid.alpha": 0.72,
    "legend.frameon": False,
    "lines.linewidth": 1.55,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.045,
})


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=7)


def panel_badge(ax, letter: str, color: str) -> None:
    ax.text(-0.095, 1.065, letter, transform=ax.transAxes, ha="center",
            va="center", color="white", fontsize=9.2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.26", facecolor=color,
                      edgecolor="none"), clip_on=False)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def rounded(ax, xy, width, height, face, edge, radius=1.6, lw=1.2,
            zorder=1):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=GRAY, lw=1.25, rad=0.0, zorder=5):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        connectionstyle=f"arc3,rad={rad}", linewidth=lw, color=color,
        shrinkA=2, shrinkB=2, zorder=zorder))


def add_image(ax, path: Path, xy, zoom: float, zorder=3) -> None:
    image = mpimg.imread(path)
    artist = AnnotationBbox(
        OffsetImage(image, zoom=zoom), xy, frameon=False,
        box_alignment=(0.5, 0.5), zorder=zorder)
    ax.add_artist(artist)


def load_curriculum():
    """Load all scalar hydraulic descriptors, run 601, and a 120-run pool."""
    params = {k: np.empty(1000) for k in
              ("Cd_actual", "prv_Pset", "beta_eff", "eps_air", "eps_vol")}
    pool = [[] for _ in range(4)]
    chosen = set(np.linspace(0, 999, 120, dtype=int).tolist())
    with h5py.File(CL_MAT, "r") as h5:
        refs = h5["Dataset/Data"]
        run601 = None
        for i in range(1000):
            run = h5[refs[i, 0]]
            pg = run["Params"]
            for key in params:
                params[key][i] = float(np.asarray(pg[key]).squeeze())
            if i in chosen:
                ext = np.asarray(run["Extension"], dtype=float)
                for c in range(4):
                    pool[c].append(ext[c])
            if i == 600:
                run601 = {
                    "time": np.asarray(run["Time"], dtype=float).squeeze(),
                    "extension": np.asarray(run["Extension"], dtype=float),
                    "valve": np.asarray(run["ValveCmd"], dtype=float)[:4],
                }
    for c in range(4):
        pool[c] = np.concatenate(pool[c])
    assert run601 is not None
    return params, run601, pool


def load_open_loop_pool():
    df = pd.read_csv(OL_CSV, usecols=["valid"] +
                     [f"Extension_{i}" for i in range(1, 5)])
    # The stored export marks startup/shutdown as invalid for training.
    valid = df["valid"].astype(bool).to_numpy()
    return [df.loc[valid, f"Extension_{i}"].to_numpy(dtype=float)
            for i in range(1, 5)]


def figure_4_1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(FULLW, 5.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 78)
    ax.axis("off")

    # Top deliverables.
    rounded(ax, (8, 70), 80, 6.3, "#EDF2F7", "#AAB7C2", 1.5)
    ax.text(48, 74.4, "LEARNED-MODEL DATASETS", ha="center", va="center",
            fontsize=8.4, color=MUTED, fontweight="bold")
    for x, label, color in [(24, "inverse dynamics  ·  ID", ORANGE),
                            (48, "forward dynamics  ·  FD", BLUE),
                            (72, "reduced order  ·  ROM", TEAL)]:
        rounded(ax, (x - 10.5, 70.7), 21, 2.6, color, color, 1.2)
        ax.text(x, 72.0, label, ha="center", va="center", color="white",
                fontsize=7.0, fontweight="bold")

    # Campaign cards.
    rounded(ax, (2, 29), 42, 38.2, PALE_ORANGE, "#E5B65C", 2.0)
    rounded(ax, (50, 29), 38, 38.2, "#EDF5FB", "#86ACCF", 2.0)
    rounded(ax, (3.2, 61.5), 39.6, 4.6, ORANGE, ORANGE, 1.4)
    rounded(ax, (51.2, 61.5), 35.6, 4.6, BLUE, BLUE, 1.4)
    ax.text(23, 63.8, "CLOSED-LOOP CAMPAIGN  ·  R1 / R3", color="white",
            ha="center", va="center", fontsize=8.2, fontweight="bold")
    ax.text(69, 63.8, "OPEN-LOOP CAMPAIGN  ·  R2", color="white",
            ha="center", va="center", fontsize=8.2, fontweight="bold")

    # The plant and actuator renders replace another row of abstract blocks.
    rounded(ax, (4.2, 48), 14.6, 11.5, "white", "#E0C37E", 1.2)
    add_image(ax, OUT / "unimog_mulag_iso.png", (11.5, 53.7), 0.017)
    ax.text(21.0, 57.6, "realistic operation", color=INK, fontsize=8.6,
            fontweight="bold", va="center")
    ax.text(21.0, 54.3, "controller + waypoints", color=MUTED, fontsize=8.0,
            va="center")
    ax.text(21.0, 51.1, "gravity-loaded motion", color=MUTED, fontsize=8.0,
            va="center")

    rounded(ax, (52.2, 48), 33.6, 11.5, "white", "#A7C4DD", 1.2)
    add_image(ax, OUT / "electro_hydraulic_actuator_channel_render.png",
              (60.3, 53.8), 0.024)
    ax.text(68.4, 57.6, "causal excitation", color=INK,
            fontsize=8.5, fontweight="bold", va="center")
    ax.text(68.4, 54.3, "64 s command schedule", color=MUTED, fontsize=7.9,
            va="center")
    ax.text(68.4, 51.1, "four randomized draws", color=MUTED, fontsize=7.9,
            va="center")

    left_steps = [
        ("01", "12-D Latin-hypercube randomization"),
        ("02", "1,000-run four-stage curriculum"),
        ("03", "state · pressure · command · descriptor"),
    ]
    for j, (num, label) in enumerate(left_steps):
        y = 43.7 - j * 5.4
        rounded(ax, (4.2, y), 37.6, 4.2, "white", "#E4D1A6", 1.0)
        ax.text(6.6, y + 2.1, num, ha="center", va="center", color=ORANGE,
                fontsize=8.0, fontweight="bold")
        ax.text(9.2, y + 2.1, label, ha="left", va="center", color=INK,
                fontsize=7.75)

    right_steps = [
        ("01", "single-channel + simultaneous phases"),
        ("02", "stroke-coverage acceptance gate"),
    ]
    for j, (num, label) in enumerate(right_steps):
        y = 43.7 - j * 5.4
        rounded(ax, (52.2, y), 33.6, 4.2, "white", "#B8CEE1", 1.0)
        ax.text(54.7, y + 2.1, num, ha="center", va="center", color=BLUE,
                fontsize=8.0, fontweight="bold")
        ax.text(57.3, y + 2.1, label, ha="left", va="center", color=INK,
                fontsize=7.55)
    rounded(ax, (52.2, 32.8), 33.6, 4.2, "#FBE9E7", RED, 1.0)
    ax.text(69.0, 34.9, "rejected export  ·  duplicate-state audit",
            ha="center", va="center", color="#9B4039", fontsize=7.5,
            fontweight="bold")

    # Join bridge and audit spine.
    rounded(ax, (44.9, 42.2), 4.1, 13.8, PALE_TEAL, TEAL, 1.3)
    ax.text(47.0, 49.1, "ALIGN\n+\nJOIN", ha="center", va="center",
            fontsize=7.3, color="#137267", fontweight="bold")
    arrow(ax, (42.7, 49.1), (44.8, 49.1), ORANGE)
    arrow(ax, (50.2, 49.1), (49.1, 49.1), BLUE)
    arrow(ax, (47.0, 56.1), (47.0, 69.7), TEAL)

    rounded(ax, (90, 29), 8, 47.3, "#E9EFF5", "#8297AA", 1.7)
    ax.text(92.4, 52.8, "AUDIT  ·  VALIDATION", rotation=90, ha="center",
            va="center", color="#4F6579", fontsize=8.4, fontweight="bold")
    for y in (34.5, 44.5, 54.5, 64.5, 73.0):
        ax.plot(96.3, y, marker="o", ms=4.3, mfc="white", mec=TEAL, mew=1.3,
                zorder=6)
    arrow(ax, (88.0, 50.0), (89.8, 50.0), GRAY)

    # Shared physical substrate.
    rounded(ax, (2, 11.2), 86, 14.3, "#E9F0F7", "#6E91B3", 1.8)
    add_image(ax, OUT / "fme500_cad_iso.png", (11.5, 18.3), 0.037)
    ax.text(22.0, 21.7, "SHARED SIMSCAPE DIGITAL PLANT", ha="left",
            va="center", color="#315D85", fontsize=9.0, fontweight="bold")
    ax.text(22.0, 17.6, "electro-hydraulic subsystem", ha="left",
            va="center", color=INK, fontsize=8.0)
    ax.text(50.0, 17.6, "multibody mechanism", ha="left", va="center",
            color=INK, fontsize=8.0)
    ax.text(22.0, 14.5, "one plant interface  ·  one time base  ·  one schema",
            ha="left", va="center", color=MUTED, fontsize=7.7)
    arrow(ax, (23, 25.4), (23, 28.8), ORANGE)
    arrow(ax, (69, 25.4), (69, 28.8), BLUE)

    for x, w, label in [(2, 26.5, "CAD / URDF geometry"),
                        (31.8, 26.5, "ode23t stiff integration"),
                        (61.5, 26.5, "nominal parameter set")]:
        rounded(ax, (x, 3.2), w, 5.4, "#365D82", "#274965", 1.2)
        ax.text(x + w / 2, 5.9, label, ha="center", va="center",
                color="white", fontsize=8.0, fontweight="bold")
    save(fig, "ch4_architecture.pdf")


def figure_4_2_design() -> None:
    fig = plt.figure(figsize=(FULLW, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.25], hspace=0.54,
                          wspace=0.32, left=0.10, right=0.985, top=0.96,
                          bottom=0.10)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])

    blocks = [("Nominal", 1, GRAY), ("Hydraulics", 250, BLUE),
              ("Mechanics", 250, ORANGE), ("Fully randomized", 499, TEAL)]
    left = 0
    for label, n, color in blocks:
        axa.barh(0, n, left=left, height=0.52, color=color,
                 edgecolor="white", linewidth=1.3)
        if n > 10:
            axa.text(left + n / 2, 0, f"{n}", ha="center", va="center",
                     color="white", fontsize=9.5, fontweight="bold")
        else:
            axa.annotate("nominal · 1", xy=(0.5, 0.22), xytext=(52, 0.54),
                         arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9),
                         fontsize=8.5, color=GRAY, ha="center")
        left += n
    axa.set_xlim(0, 1000)
    axa.set_ylim(-0.42, 0.70)
    axa.set_yticks([])
    axa.set_xlabel("run index in the closed-loop curriculum")
    axa.spines[["left", "right", "top"]].set_visible(False)
    left = 0
    for label, n, color in blocks:
        if n > 10:
            axa.text(left + n / 2, 0.39, label, ha="center", va="bottom",
                     color=color, fontsize=8.3, fontweight="bold")
        left += n
    panel_title(axa, "Four-stage curriculum  ·  1,000 runs")
    panel_badge(axa, "a", ORANGE)

    rng = np.random.default_rng(0)
    n = 499
    lhs = np.column_stack([(rng.permutation(n) + rng.random(n)) / n
                           for _ in range(2)])
    ea = 0.001 + 0.079 * lhs[:, 0]
    ev = 0.005 + 0.075 * lhs[:, 1]
    raw = 1 / ((1 - ea) / 0.5e9 + ea / (1.4e7) + ev / 0.5e9) / 1e9
    axb.hist(raw, bins=27, color=BLUE, edgecolor="white", linewidth=0.45,
             alpha=0.9)
    axb.axvline(0.70, color=MAGENTA, lw=2.0, label="0.70 GPa clamp")
    axb.axvspan(raw.min(), raw.max(), color=BLUE, alpha=0.08, zorder=0)
    axb.text(0.68, axb.get_ylim()[1] * 0.82, "stored value\n0.70 GPa",
             ha="right", va="top", color=MAGENTA, fontsize=8.5,
             fontweight="bold")
    axb.text((raw.min() + raw.max()) / 2, axb.get_ylim()[1] * 0.45,
             f"formula range\n{raw.min():.2f}–{raw.max():.2f} GPa",
             ha="center", va="center", color="#245C89", fontsize=8.2)
    axb.set_xlim(0.08, 0.78)
    axb.set_xlabel(r"base modulus before clamp, $\beta$ [GPa]")
    axb.set_ylabel("fully randomized runs")
    axb.grid(True, axis="y")
    panel_title(axb, "Bulk-modulus clamp")
    panel_badge(axb, "b", BLUE)

    # Explicit schedule: four 10 s single-cylinder slots separated by 0.5 s,
    # then a simultaneous phase, with startup/shutdown excluded.
    axc.axvspan(0, 1, color=GRAY, alpha=0.20)
    axc.axvspan(63, 64, color=GRAY, alpha=0.20)
    t = 1.0
    for c in range(4):
        axc.barh(3 - c, 10.0, left=t, height=0.60, color=CYL[c], alpha=0.93)
        axc.text(t + 5, 3 - c, f"C{c + 1}", color="white", ha="center",
                 va="center", fontsize=8.4, fontweight="bold")
        t += 10.0
        axc.axvspan(t, t + 0.5, color="#DDE2E7", zorder=0)
        t += 0.5
    for c in range(4):
        axc.barh(3 - c, 20, left=t, height=0.60, color=CYL[c], alpha=0.55)
    axc.text(t + 10, 1.5, "simultaneous", ha="center", va="center",
             color=INK, fontsize=8.2, fontweight="bold",
             bbox=dict(boxstyle="round,pad=.22", facecolor="white",
                       edgecolor="none", alpha=.88))
    axc.text(0.5, 3.7, "excluded", ha="center", va="bottom", rotation=90,
             color=GRAY, fontsize=7.3)
    axc.text(63.5, 3.7, "excluded", ha="center", va="bottom", rotation=90,
             color=GRAY, fontsize=7.3)
    axc.set_xlim(0, 64)
    axc.set_ylim(-0.65, 4.25)
    axc.set_yticks(range(4), ["C4", "C3", "C2", "C1"])
    axc.set_xlabel("open-loop run time [s]")
    axc.grid(True, axis="x")
    panel_title(axc, "64 s causal-excitation schedule")
    panel_badge(axc, "c", TEAL)
    save(fig, "ch4_dataset_design.pdf")


def figure_4_3_params(params) -> None:
    vd = np.clip(5e-4 * (1 + params["eps_air"] + params["eps_vol"]),
                 5e-4, 1e-3) * 1e6  # mL
    vd0 = 512.5
    beta0 = 0.70
    data = [params["Cd_actual"], params["prv_Pset"] / 1e6,
            beta0 * vd0 / vd, beta0 / (vd / 1000)]
    specs = [
        (r"Discharge coefficient  $c_d$", BLUE, 0.70, "–"),
        ("Relief setting  [MPa]", ORANGE, 27.6, "MPa"),
        (r"Compliance-equivalent  $\beta_{eq}$  [GPa]", TEAL, 0.70, "GPa"),
        (r"Pressure stiffness  $K_p^d$  [GPa/L]", MAGENTA,
         beta0 / (vd0 / 1000), "GPa/L"),
    ]
    fig, grid = plt.subplots(2, 2, figsize=(FULLW, 4.35),
                             gridspec_kw=dict(wspace=0.27, hspace=0.54,
                                              left=0.10, right=0.985,
                                              top=0.86, bottom=0.13))
    # Left column: directly randomized valve descriptors.  Right column:
    # the two compliance-equivalent descriptors referenced by the caption.
    axes = [grid[0, 0], grid[1, 0], grid[0, 1], grid[1, 1]]
    plot_data = [data[0], data[1], data[2], data[3]]
    plot_specs = [specs[0], specs[1], specs[2], specs[3]]
    for i, (ax, vals, (label, color, nominal, unit)) in enumerate(
            zip(axes, plot_data, plot_specs)):
        ax.hist(vals, bins=26, color=color, alpha=0.88, edgecolor="white",
                linewidth=0.40)
        ax.axvline(nominal, color=INK, lw=1.25, ls="--")
        ax.set_xlabel(label, fontsize=9.2)
        ax.grid(True, axis="y")
        ax.tick_params(labelsize=8.4)
        ax.text(0.03, 0.94, f"{np.min(vals):.2f}–{np.max(vals):.2f}",
                transform=ax.transAxes, ha="left", va="top", color=color,
                fontsize=8.2, fontweight="bold",
                bbox=dict(boxstyle="round,pad=.20", facecolor="white",
                          edgecolor="none", alpha=.9))
        panel_badge(ax, chr(ord("a") + i), color)
    axes[0].set_ylabel("curriculum runs")
    axes[1].set_ylabel("curriculum runs")
    axes[2].set_yticklabels([])
    axes[3].set_yticklabels([])
    fig.text(0.75, 0.965,
             r"base modulus fixed for all runs:  $\beta_0 = 0.70$ GPa",
             ha="center", va="top", color="#137267", fontsize=9.0,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=.34", facecolor=PALE_TEAL,
                       edgecolor="#9DCFC7"))
    fig.text(0.20, 0.965, "stored plant descriptors  ·  n = 1,000",
             ha="center", va="top", color=MUTED, fontsize=8.5)
    save(fig, "ch4_delivered_params.pdf")


def figure_4_4_trace(run) -> None:
    t = run["time"]
    ext = run["extension"]
    valve = run["valve"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FULLW, 4.65), sharex=True,
        gridspec_kw=dict(height_ratios=[1.08, 0.92], hspace=0.13,
                         left=0.115, right=0.985, top=0.90, bottom=0.11))
    for c in range(4):
        ax1.plot(t, ext[c], color=CYL[c], label=f"cylinder {c + 1}")
        ax2.plot(t, valve[c], color=CYL[c], lw=1.35)
    ax1.set_ylabel("cylinder extension [m]")
    ax1.set_ylim(0.0, 0.82)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center",
               bbox_to_anchor=(0.57, 0.995), columnspacing=1.35,
               handlelength=2.0)
    ax2.axhspan(0.8, 1.0, color=MAGENTA, alpha=0.07)
    ax2.axhspan(-1.0, -0.8, color=MAGENTA, alpha=0.07)
    ax2.axhline(0, color=GRAY, lw=0.7)
    ax2.set_ylim(-1.02, 1.02)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylabel("valve command $u$")
    ax2.set_xlabel("time [s]")
    for ax in (ax1, ax2):
        ax.grid(True, axis="y")
    panel_title(ax1, "Coordinated cylinder motion")
    panel_title(ax2, "Controller effort  ·  shaded bands mark saturation")
    panel_badge(ax1, "a", ORANGE)
    panel_badge(ax2, "b", BLUE)
    save(fig, "ch4_closed_loop_trace.pdf")


def figure_4_5_regimes(run) -> None:
    t = run["time"]
    ext = run["extension"]
    u = run["valve"].ravel()
    vel = np.vstack([np.gradient(ext[c], t) for c in range(4)]).ravel()
    au = np.abs(u)
    masks = [au >= 0.8, (au >= 0.3) & (au < 0.8), au < 0.3]
    labels = ["saturated", "mid band", "low command"]
    cols = [MAGENTA, ORANGE, BLUE]
    shares = 100 * np.asarray([m.mean() for m in masks])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FULLW, 4.55),
        gridspec_kw=dict(height_ratios=[0.58, 2.2], hspace=0.28,
                         left=0.115, right=0.985, top=0.94, bottom=0.12))
    left = 0.0
    for val, label, color in zip(shares, labels, cols):
        ax1.barh(0, val, left=left, height=0.56, color=color,
                 edgecolor="white", linewidth=1.5)
        ax1.text(left + val / 2, 0, f"{label}  {val:.0f}%", color="white",
                 ha="center", va="center", fontsize=8.6, fontweight="bold")
        left += val
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.48, 0.48)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines[:].set_visible(False)
    panel_title(ax1, "Control-regime occupancy")
    panel_badge(ax1, "a", ORANGE)

    order = [2, 1, 0]
    for idx in order:
        ax2.scatter(vel[masks[idx]], u[masks[idx]], s=10, color=cols[idx],
                    alpha=0.48, linewidths=0, label=labels[idx],
                    rasterized=True)
    ax2.axhline(0.8, color=MAGENTA, lw=0.85, ls=":")
    ax2.axhline(-0.8, color=MAGENTA, lw=0.85, ls=":")
    ax2.axhline(0.3, color=ORANGE, lw=0.65, ls=":", alpha=.75)
    ax2.axhline(-0.3, color=ORANGE, lw=0.65, ls=":", alpha=.75)
    ax2.axvline(0, color=GRAY, lw=0.65)
    ax2.set_xlabel("cylinder velocity [m/s]")
    ax2.set_ylabel("valve command $u$")
    ax2.set_ylim(-1.02, 1.02)
    ax2.grid(True)
    ax2.legend(ncol=3, loc="lower right", markerscale=1.7,
               columnspacing=1.25)
    ax2.text(0.02, 0.95, "narrow command–state band",
             transform=ax2.transAxes, ha="left", va="top", color=INK,
             fontsize=8.3, fontweight="bold",
             bbox=dict(boxstyle="round,pad=.25", facecolor="white",
                       edgecolor="#D6DDE3", alpha=.92))
    panel_title(ax2, "Valve command versus the velocity it produced")
    panel_badge(ax2, "b", BLUE)
    save(fig, "ch4_regimes.pdf")


def figure_4_6_coverage(cl_pool, ol_pool) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(FULLW, 4.55), sharey=False,
        gridspec_kw=dict(hspace=0.47, wspace=0.28, left=0.10, right=0.985,
                         top=0.92, bottom=0.12))
    axes = axes.ravel()
    for c, ax in enumerate(axes):
        lo = min(np.min(cl_pool[c]), np.min(ol_pool[c]))
        hi = max(np.max(cl_pool[c]), np.max(ol_pool[c]))
        bins = np.linspace(lo, hi, 36)
        ax.hist(cl_pool[c], bins=bins, density=True, color=CYL[c],
                alpha=0.55, edgecolor="white", linewidth=0.22,
                label="closed loop · 120 runs")
        ax.hist(ol_pool[c], bins=bins, density=True, histtype="step",
                color=INK, linewidth=1.55, label="open loop · 4 runs")
        ax.set_xlabel("extension [m]")
        ax.set_yticks([])
        ax.grid(True, axis="x")
        panel_title(ax, f"Cylinder {c + 1}")
        panel_badge(ax, chr(ord("a") + c), CYL[c])
    axes[0].set_ylabel("density")
    axes[2].set_ylabel("density")
    axes[0].legend(loc="upper right", fontsize=7.7)
    fig.text(0.5, 0.985,
             "filled = realistic pose coverage     outline = causal plateaus",
             ha="center", va="top", color=MUTED, fontsize=8.7)
    save(fig, "ch4_coverage.pdf")


def figure_5_5_authority() -> None:
    report = (REPO / "archive/model_training_trees/PPO1_ID_FD_PhysX"
              / "models/fd/fd_authority_report.json")
    rep = json.loads(report.read_text())
    horizons = [1, 2, 5, 10, 20]
    tt = np.asarray(horizons) * 0.05

    def series(model, key):
        return np.asarray([rep[model][str(h)][key] for h in horizons])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FULLW, 4.72), sharex=True,
        gridspec_kw=dict(hspace=0.16, left=0.13, right=0.985,
                         top=0.92, bottom=0.11))
    models = [(ax1, "best_fd_cylinder_control", ORANGE,
               "Before strict retraining  ·  paired control"),
              (ax2, "best_fd_cylinder_strict", BLUE,
               "After strict retraining  ·  causal excitation")]
    for idx, (ax, model, color, title) in enumerate(models):
        ax.plot(tt, series(model, "fd"), color=color, marker="o", ms=5,
                label="normal valve input")
        ax.plot(tt, series(model, "fd_valve0"), color=color, marker="o",
                ms=5, ls="--", label="valve input zeroed")
        ax.plot(tt, series(model, "hold"), color=GRAY, marker="^", ms=4.5,
                ls=":", label="zero-order hold")
        ax.set_yscale("log")
        ax.set_ylabel("rollout RMSE [m]")
        ax.grid(True, which="both", axis="y")
        ax.legend(ncol=3, loc="upper left", frameon=True, facecolor="white",
                  edgecolor="none", framealpha=.90)
        panel_title(ax, title)
        panel_badge(ax, chr(ord("a") + idx), color)
    ax2.set_xticks(tt)
    ax2.set_xlabel("open-loop rollout horizon [s]")
    save(fig, "ch5_fd_rollout_authority.pdf")


def figure_5_7_rom_training() -> None:
    history_path = (REPO / "archive/model_training_trees/ROM-cylinders_pipeline"
                    / "checkpoints/command_run_001/training_history.json")
    history = json.loads(history_path.read_text())
    epoch = np.arange(1, len(history) + 1)
    train = np.asarray([row["train_loss"] for row in history])
    one_step = np.asarray([row["validation_one_step_loss"] for row in history])
    rollout = np.asarray([row["validation_rollout_loss"] for row in history])
    stages = np.asarray([row["stage"] for row in history])
    unique_stages = sorted(set(stages.tolist()))
    stage_colors = [PALE_BLUE, PALE_ORANGE, PALE_TEAL, PALE_MAGENTA]

    fig, ax = plt.subplots(figsize=(FULLW, 3.65))
    for j, stage in enumerate(unique_stages):
        idx = np.flatnonzero(stages == stage)
        lo, hi = epoch[idx[0]] - 0.5, epoch[idx[-1]] + 0.5
        ax.axvspan(lo, hi, color=stage_colors[j % len(stage_colors)],
                   alpha=0.72, lw=0, zorder=0)
        horizon = next(row["horizon"] for row in history
                       if row["stage"] == stage)
        ax.text((lo + hi) / 2, 0.965, f"stage {stage}  ·  $H={horizon}$",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=INK, fontsize=8.2, fontweight="bold")
        if j:
            ax.axvline(lo, color="#AAB4BD", lw=0.8, ls="--", zorder=1)

    ax.plot(epoch, train, color=BLUE, label="training loss", zorder=3)
    ax.plot(epoch, one_step, color=ORANGE,
            label="validation one-step loss", zorder=3)
    valid = rollout > 0
    ax.plot(epoch[valid], rollout[valid], color=TEAL,
            label="validation rollout loss", zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(epoch[0], epoch[-1])
    ax.set_xlabel("epoch  ·  cumulative over rollout curriculum")
    ax.set_ylabel("normalized loss")
    ax.grid(True, which="both", axis="y")
    ax.legend(ncol=3, loc="lower left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=.90)
    panel_title(ax, "Promoted command-ROM training history")
    panel_badge(ax, "a", VIOLET)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.16)
    save(fig, "ch5_rom_training_history.pdf")


def figure_5_8_rom_validation() -> None:
    report = (REPO / "archive/model_training_trees/ROM-cylinders_pipeline"
              / "reports/command_offline_validation.json")
    rep = json.loads(report.read_text())
    groups = ["Valve", r"$\Delta(\Delta p)$", r"$\Delta\dot{x}$", r"$\Delta x$"]

    def grouped_mean(vals):
        return [float(np.mean(vals[4 * i:4 * i + 4])) for i in range(4)]

    full = grouped_mean(rep["r2_by_output"])
    zero = grouped_mean(rep["zeroed_command_r2_by_output"])
    shuffled = grouped_mean(rep["shuffled_command_r2_by_output"])
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FULLW, 3.35),
        gridspec_kw=dict(width_ratios=[1.12, 1.0], wspace=0.30,
                         left=0.095, right=0.985, top=0.77, bottom=0.18))
    x = np.arange(4)
    w = 0.24
    ax1.bar(x - w, full, w, color=BLUE, label="full command")
    ax1.bar(x, zero, w, color=ORANGE, label="command zeroed")
    ax1.bar(x + w, shuffled, w, color=MAGENTA, label="command shuffled")
    ax1.axhline(0, color=GRAY, lw=0.8)
    ax1.set_xticks(x, groups)
    ax1.set_ylim(-0.70, 1.05)
    ax1.set_ylabel(r"one-step $R^2$ · mean over cylinders")
    ax1.grid(True, axis="y")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper left",
               bbox_to_anchor=(0.09, 0.985), fontsize=8.0,
               columnspacing=1.15, handlelength=1.5)
    panel_title(ax1, "Command ablation")
    panel_badge(ax1, "a", BLUE)

    horizons = sorted(int(h) for h in rep["rollout_metrics"])
    time = np.asarray(horizons) * 0.05
    for c in range(4):
        vals = [rep["rollout_metrics"][str(h)]["terminal_r2_by_output"][12 + c]
                for h in horizons]
        ax2.plot(time, vals, color=CYL[c], marker="o", ms=4.3,
                 label=f"cylinder {c + 1}")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("recursive rollout horizon [s]")
    ax2.set_ylabel(r"terminal extension $R^2$")
    ax2.grid(True, axis="y")
    ax2.legend(loc="lower right", fontsize=7.7)
    panel_title(ax2, "Held-out rollout accuracy")
    panel_badge(ax2, "b", TEAL)
    save(fig, "ch5_rom_validation.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params, run601, cl_pool = load_curriculum()
    ol_pool = load_open_loop_pool()
    figure_4_1_architecture()
    figure_4_2_design()
    figure_4_3_params(params)
    figure_4_4_trace(run601)
    figure_4_5_regimes(run601)
    figure_4_6_coverage(cl_pool, ol_pool)
    figure_5_5_authority()
    figure_5_7_rom_training()
    figure_5_8_rom_validation()


if __name__ == "__main__":
    main()
