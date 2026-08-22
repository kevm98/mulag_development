#!/usr/bin/env python3
"""Readable, reference-style layouts for the requested Chapter 4/5 plots.

The Chapter-4 trace coordinates are digitized from the existing vector
figure; the Chapter-4 regime panels are re-framed from the existing 1200-dpi
source without altering any samples. Chapter-5 values are loaded from the
validated authority report.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]
OUT = BASE / "images"

COLORS = ["#357EC7", "#E29A16", "#25A18E", "#B55B9A"]
BLUE, ORANGE, TEAL, MAGENTA = COLORS
GRAY = "#5F6872"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.75,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "grid.color": "#CBD2D8",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.75,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


TRACE = {
    "cyl 1": [(0.0,.2953),(1.65,.1762),(3.4,.0403),(5.65,.3738),(6.6,.3611),(8.35,.1053),(9.4,.1669),(10.6,.2531),(11.65,.2552),(13.7,.0446),(15.4,.1034),(16.7,.2202),(18.05,.3162),(20.0,.0693),(21.65,.3243),(22.0,.2698)],
    "cyl 2": [(0.0,.5571),(.95,.5395),(1.85,.5162),(2.75,.4972),(3.65,.4824),(5.75,.7636),(7.05,.6107),(9.1,.3607),(10.55,.4232),(11.45,.4164),(12.8,.482),(13.9,.5758),(15.4,.7022),(16.3,.6603),(18.1,.4416),(19.6,.6094),(21.1,.5884),(22.0,.439)],
    "cyl 3": [(0.0,.4535),(.95,.4371),(1.9,.4521),(2.7,.3955),(3.6,.4038),(4.7,.5332),(5.7,.5966),(6.75,.5371),(7.55,.5906),(8.7,.4192),(9.75,.3919),(10.85,.5257),(11.8,.5646),(12.75,.608),(13.5,.5465),(14.55,.6197),(15.55,.5436),(16.55,.4623),(17.5,.4197),(18.55,.3914),(19.75,.5849),(20.4,.4742),(21.4,.5272),(22.0,.372)],
    "cyl 4": [(0.0,.2859),(.5,.5476),(1.2,.5393),(2.0,.5056),(2.7,.3908),(3.35,.4321),(4.0,.5547),(4.75,.5471),(5.45,.4671),(6.2,.3787),(6.9,.4192),(7.5,.5635),(8.25,.5413),(9.0,.4846),(9.65,.3767),(10.35,.4508),(11.0,.5661),(11.8,.407),(12.35,.5616),(13.15,.5552),(13.9,.5144),(14.55,.3722),(15.35,.3834),(15.9,.5445),(16.6,.538),(17.4,.5053),(18.1,.3917),(18.75,.4323),(19.4,.5543),(20.15,.5458),(20.85,.4665),(21.6,.378),(22.0,.5661)],
}

VALVE = {
    "cyl 1": [(0,-1),(2.2,1),(4.4,-1),(6.6,1),(8.8,1),(11,-1),(13.2,1),(15.4,-1),(17.6,-1),(19.8,1),(22,-1)],
    "cyl 2": [(0,-1),(2.2,1),(4.4,-1),(6.6,-1),(8.8,1),(11,1),(13.2,-1),(15.4,1),(22,1)],
    "cyl 3": [(0,-.6719),(2.2,.3919),(4.4,-.5062),(6.6,.5212),(8.8,-.585),(11,.6833),(13.2,-.6806),(15.4,.6343),(17.6,-.8421),(19.8,.1226),(22,-.8421)],
    "cyl 4": [(0,-.8421),(2.2,.8422),(4.4,-.8421),(6.6,.8422),(8.8,-.8421),(11,-.8059),(13.2,.8422),(15.4,-.8421),(17.6,.8422),(19.8,-.8421),(22,.8422)],
}


def closed_loop_trace():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 4.45), sharex=True,
                                  gridspec_kw={"hspace": .12})
    for (lab, pts), color in zip(TRACE.items(), COLORS):
        a = np.asarray(pts)
        ax1.plot(a[:, 0], a[:, 1], color=color, lw=1.65, label=lab)
    for (lab, pts), color in zip(VALVE.items(), COLORS):
        a = np.asarray(pts)
        ax2.plot(a[:, 0], a[:, 1], color=color, lw=1.45)
    ax1.set_ylabel("extension [m]")
    ax1.set_ylim(0, .82)
    ax1.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.20),
               columnspacing=1.6, handlelength=2.2)
    ax2.set_ylabel("valve command")
    ax2.set_xlabel("time [s]")
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlim(0, 22)
    ax2.set_xticks(np.arange(0, 22.1, 2.5))
    for ax in (ax1, ax2):
        ax.grid(True, axis="y")
    fig.subplots_adjust(left=.12, right=.985, top=.90, bottom=.12)
    fig.savefig(OUT / "ch4_closed_loop_trace.pdf")
    plt.close(fig)


def regimes_reframed():
    source = OUT / "ch4_regimes_v2.pdf"
    raster = Path("/tmp/ch4_regimes_reference_1200.png")
    subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", "1200",
                    str(source), str(raster.with_suffix(""))], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    im = mpimg.imread(raster)
    h, w = im.shape[:2]
    right = im[:, int(w * .385):]
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.35),
                             gridspec_kw={"height_ratios": [.55, 2.2]})
    sizes = np.asarray([23, 45, 31], dtype=float)
    sizes *= 100.0 / sizes.sum()
    labels = ["saturated", "mid", "low control"]
    cols = [MAGENTA, ORANGE, BLUE]
    left_edge = 0.0
    for value, label, color in zip(sizes, labels, cols):
        axes[0].barh(0, value, left=left_edge, height=.58, color=color,
                     edgecolor="white", linewidth=1.6)
        axes[0].text(left_edge + value / 2, 0, f"{label}\n{value:.0f}%",
                     ha="center", va="center", color="white", fontsize=10,
                     fontweight="bold")
        left_edge += value
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(-.55, .55)
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_title("(a) time in each control regime", loc="left", pad=3)
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    axes[1].imshow(right, aspect="auto")
    axes[1].set_axis_off()
    fig.subplots_adjust(left=.06, right=.985, top=.96, bottom=.025, hspace=.13)
    fig.savefig(OUT / "ch4_regimes.pdf", dpi=600)
    plt.close(fig)


def fd_authority():
    report = REPO / "archive/model_training_trees/PPO1_ID_FD_PhysX/models/fd/fd_authority_report.json"
    rep = json.loads(report.read_text())
    horizons = [1, 2, 5, 10, 20]
    tt = np.asarray(horizons) * .05

    def series(model, key):
        return np.asarray([rep[model][str(h)][key] for h in horizons])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 4.7), sharex=True,
                                 gridspec_kw={"hspace": .13})
    paired = "best_fd_cylinder_control"
    strict = "best_fd_cylinder_strict"
    a1.plot(tt, series(paired, "fd"), color=ORANGE, marker="o", ms=4.5,
            label="normal valve input")
    a1.plot(tt, series(paired, "fd_valve0"), color=ORANGE, marker="o", ms=4.5,
            ls="--", label="valve input zeroed")
    a1.plot(tt, series(strict, "hold"), color=GRAY, marker="^", ms=4,
            ls=":", label="zero-order hold")
    a1.set_title("(a) Paired-control forward model", loc="left")
    a1.set_ylabel("rollout RMSE [m]")
    a1.legend(ncol=3, loc="upper left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=.94)

    a2.plot(tt, series(strict, "fd"), color=BLUE, marker="s", ms=4.5,
            label="normal valve input")
    a2.plot(tt, series(strict, "fd_valve0"), color=BLUE, marker="s", ms=4.5,
            ls="--", label="valve input zeroed")
    a2.plot(tt, series(strict, "hold"), color=GRAY, marker="^", ms=4,
            ls=":", label="zero-order hold")
    a2.set_title("(b) Strict-excitation forward model", loc="left")
    a2.set_xlabel("open-loop rollout horizon [s]")
    a2.set_ylabel("rollout RMSE [m]")
    a2.legend(ncol=3, loc="upper left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=.94)
    a2.set_xticks(tt)
    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y")
    fig.subplots_adjust(left=.13, right=.985, top=.95, bottom=.12)
    fig.savefig(OUT / "ch5_fd_rollout_authority.pdf")
    plt.close(fig)


if __name__ == "__main__":
    closed_loop_trace()
    regimes_reframed()
    fd_authority()
