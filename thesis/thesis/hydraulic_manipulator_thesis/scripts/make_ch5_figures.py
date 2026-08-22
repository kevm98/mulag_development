#!/usr/bin/env python3
"""Generate Chapter 5 figures for the thesis from the real PPO1/PPO2 artifacts.

All numbers are read from the repository's validated reports and datasets;
nothing is synthesized.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/rrlab/rrlab_mulag")
OUT = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis/images")
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito subset, CVD-safe, validated (exit 0) with the dataviz validator.
C_CYL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]  # cylinders 1..4
C_BLUE, C_ORANGE, C_GREEN, C_PINK = C_CYL
C_GRAY = "#666666"

# IEEE-style plot conventions: serif/STIX typography, full axis box,
# inward ticks, restrained grid, vector output with embedded fonts.
mpl.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "grid.color": "#c8c8c8",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.7,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "figure.constrained_layout.use": True,
})

FULLW = 6.3  # \textwidth = 455 pt


# ---------------------------------------------------------------- figure 1
def fig_excitation() -> None:
    csv = REPO / "archive/model_training_trees/PPO1_ID_FD_PhysX/training_data/Excitation_FD_Strict.csv"
    df = pd.read_csv(csv)
    run = df[df.run_id == 1]
    t = run["t"].to_numpy()
    win = (t >= 0.0) & (t <= 64.0)
    t = t[win]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(FULLW, 4.6))
    for i in range(4):
        axes[0].plot(t, run[f"ValveCmd_{i+1}"].to_numpy()[win], color=C_CYL[i],
                     lw=1.0, label=f"Cylinder {i+1}")
        axes[1].plot(t, run[f"dP_{i+1}"].to_numpy()[win] / 1e6, color=C_CYL[i], lw=1.0)
        axes[2].plot(t, run[f"Extension_{i+1}"].to_numpy()[win], color=C_CYL[i], lw=1.2)
    axes[0].set_ylabel("Valve command $u$ [--]")
    axes[1].set_ylabel(r"$\Delta p$ [MPa]")
    axes[2].set_ylabel("Extension $x$ [m]")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, axis="y")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    axes[2].set_xlim(0, 64)
    fig.savefig(OUT / "ch5_excitation_dataset.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_fd_authority() -> None:
    rep = json.loads((REPO / "archive/model_training_trees/PPO1_ID_FD_PhysX/models/fd/fd_authority_report.json").read_text())
    horizons = [1, 2, 5, 10, 20]
    tt = [0.05 * h for h in horizons]

    def series(model: str, key: str) -> list[float]:
        return [rep[model][str(h)][key] for h in horizons]

    fig, (ax, ax_ratio) = plt.subplots(1, 2, figsize=(FULLW, 3.15),
                                       gridspec_kw={"width_ratios": [1.55, 0.85]})
    ax.plot(tt, series("best_fd_cylinder_control", "fd"), color=C_ORANGE,
            marker="o", ms=4, label="Paired-control FD")
    ax.plot(tt, series("best_fd_cylinder_control", "fd_valve0"), color=C_ORANGE,
            marker="o", ms=4, ls="--", label="Paired-control FD, valve zeroed")
    ax.plot(tt, series("best_fd_cylinder_strict", "fd"), color=C_BLUE,
            marker="s", ms=4, label="Strict FD")
    ax.plot(tt, series("best_fd_cylinder_strict", "fd_valve0"), color=C_BLUE,
            marker="s", ms=4, ls="--", label="Strict FD, valve zeroed")
    ax.plot(tt, series("best_fd_cylinder_strict", "hold"), color=C_GRAY,
            marker="^", ms=4, ls=":", label="Hold (zero-order) baseline")
    ax.set_yscale("log")
    ax.set_xlabel("Open-loop rollout horizon [s]")
    ax.set_ylabel("Extension rollout RMSE [m]")
    ax.set_xticks(tt)
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", ncol=1, fontsize=7.2)
    ax.set_title("(a) Long-horizon rollout", fontsize=9)

    # A compact second view makes the causal criterion explicit: the
    # valve-zero penalty should be close to one for the paired model and
    # materially above one for the promoted strict model.
    paired = np.asarray(series("best_fd_cylinder_control", "fd_valve0")) / np.asarray(
        series("best_fd_cylinder_control", "fd"))
    strict = np.asarray(series("best_fd_cylinder_strict", "fd_valve0")) / np.asarray(
        series("best_fd_cylinder_strict", "fd"))
    xpos = np.arange(len(tt))
    width = 0.34
    ax_ratio.bar(xpos - width / 2, paired, width, color=C_ORANGE,
                 label="Paired-control FD")
    ax_ratio.bar(xpos + width / 2, strict, width, color=C_BLUE,
                 label="Strict FD")
    ax_ratio.axhline(1.0, color=C_GRAY, lw=0.8, ls=":")
    ax_ratio.set_xticks(xpos, [f"{t:.2f}" for t in tt])
    ax_ratio.set_xlabel("horizon [s]")
    ax_ratio.set_ylabel("RMSE with valve zeroed / normal RMSE")
    ax_ratio.set_title("(b) Input authority", fontsize=9)
    ax_ratio.grid(True, axis="y")
    ax_ratio.legend(loc="upper left", fontsize=7.2)
    ax_ratio.annotate("strict FD\nresponds to u", xy=(3.0, strict[3]),
                      xytext=(2.75, 1.38),
                      arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0),
                      color=C_BLUE, fontsize=7.7, ha="center")
    fig.savefig(OUT / "ch5_fd_rollout_authority_v2.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "ch5_fd_rollout_authority_v2.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "ch5_fd_rollout_authority_v2.svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_rom_training() -> None:
    hist = json.loads(
        (REPO / "archive/model_training_trees/ROM-cylinders_pipeline/checkpoints/command_run_001/training_history.json").read_text()
    )
    epochs = np.arange(1, len(hist) + 1)
    train = np.array([h["train_loss"] for h in hist])
    val = np.array([h["validation_one_step_loss"] for h in hist])
    roll = np.array([h["validation_rollout_loss"] for h in hist])
    stages = np.array([h["stage"] for h in hist])

    fig, ax = plt.subplots(figsize=(FULLW, 3.2))
    # Stage boundaries and horizon annotations.
    bounds = [0] + [int(np.max(np.where(stages == s)[0])) + 1 for s in sorted(set(stages))]
    horizon_by_stage = {s: next(h["horizon"] for h in hist if h["stage"] == s)
                        for s in sorted(set(stages))}
    for k, s in enumerate(sorted(set(stages))):
        lo, hi = bounds[k], bounds[k + 1]
        if k % 2 == 1:
            ax.axvspan(lo + 0.5, hi + 0.5, color="#000000", alpha=0.04, lw=0)
        ax.text((lo + hi) / 2 + 0.5, 20.0, f"stage {s}\n$H={horizon_by_stage[s]}$",
                ha="center", va="top", fontsize=7.5, color="#444444")
    ax.plot(epochs, train, color=C_BLUE, label="Training loss")
    ax.plot(epochs, val, color=C_ORANGE, label="Validation one-step loss")
    m = roll > 0
    ax.plot(epochs[m], roll[m], color=C_GREEN, label="Validation rollout loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (cumulative over curriculum stages)")
    ax.set_ylabel("Normalized loss")
    ax.set_xlim(1, len(hist))
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "ch5_rom_training_history.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_rom_validation() -> None:
    rep = json.loads(
        (REPO / "archive/model_training_trees/ROM-cylinders_pipeline/reports/command_offline_validation.json").read_text()
    )
    groups = ["Valve", r"$\Delta(\Delta p)$", r"$\Delta \dot{x}$", r"$\Delta x$"]

    def gmean(vals: list[float]) -> list[float]:
        return [float(np.mean(vals[4 * i:4 * i + 4])) for i in range(4)]

    full = gmean(rep["r2_by_output"])
    zero = gmean(rep["zeroed_command_r2_by_output"])
    shuf = gmean(rep["shuffled_command_r2_by_output"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULLW, 3.0),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    xpos = np.arange(4)
    w = 0.26
    ax1.bar(xpos - w, full, w * 0.92, color=C_BLUE, label="Full command")
    ax1.bar(xpos, zero, w * 0.92, color=C_ORANGE, label="Command zeroed")
    ax1.bar(xpos + w, shuf, w * 0.92, color=C_PINK, label="Command shuffled")
    ax1.axhline(0.0, color="#444444", lw=0.7)
    ax1.set_xticks(xpos, groups)
    ax1.set_ylabel("One-step $R^2$ (mean over cylinders)")
    ax1.set_ylim(-0.7, 1.05)
    ax1.grid(True, axis="y")
    ax1.legend(loc="lower left", fontsize=7.5)
    ax1.set_title("(a) Command ablation", fontsize=9)

    horizons = sorted(int(h) for h in rep["rollout_metrics"])
    for i in range(4):
        r2 = [rep["rollout_metrics"][str(h)]["terminal_r2_by_output"][12 + i]
              for h in horizons]
        ax2.plot([0.05 * h for h in horizons], r2, color=C_CYL[i], marker="o",
                 ms=3.5, label=f"Cylinder {i+1}")
    ax2.set_xlabel("Recursive rollout horizon [s]")
    ax2.set_ylabel("Terminal extension $R^2$")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, axis="y")
    ax2.legend(loc="lower right", fontsize=7.5)
    ax2.set_title("(b) Held-out rollout accuracy", fontsize=9)
    fig.savefig(OUT / "ch5_rom_validation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_excitation()
    fig_fd_authority()
    fig_rom_training()
    fig_rom_validation()
    print("wrote figures to", OUT)
