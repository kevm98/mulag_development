#!/usr/bin/env python3
"""Chapter 7 figures for the predictive-utility section.

All numbers are read from validated artifacts under outputs/stage31_rom/;
nothing is synthesized.

Produces
    images/ch7_four_latents.pdf    rollout error vs horizon, four latents
    images/ch7_plant_switch.pdf    latent distance to old/new plant across a switch
    images/ch7_benefit_scaling.pdf latent benefit vs plant deviation
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "outputs/stage31_rom"
DATA = REPO / "AAM_module/datasets/rom_deployment_stage31_v1_frozen"
OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito subset, CVD-safe, matching Chapters 5 and 6.
C_BLUE, C_ORANGE, C_GREEN, C_PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
C_GRAY = "#666666"

mpl.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "grid.color": "#c8c8c8", "grid.linewidth": 0.4, "grid.alpha": 0.7,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "figure.constrained_layout.use": True,
})

FULLW = 6.3
ARMS = [("true", "privileged $z$", C_GRAY, "o", "--"),
        ("zhat", r"estimated $\hat{z}$", C_BLUE, "s", "-"),
        ("mean", "mean, no plant info", C_ORANGE, "^", "-"),
        ("shuffled", "shuffled, wrong plant", C_PINK, "v", "-")]


def boot(d, n=4000, seed=11):
    rng = random.Random(seed)
    m = len(d)
    out = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n)]


def fig_four_latents(per_run):
    runs = sorted(set.intersection(*[set(per_run[a]) for a, *_ in ARMS]))
    hor = [1, 10, 20]
    fig, ax = plt.subplots(1, 2, figsize=(FULLW, 2.5))

    for arm, lab, col, mk, ls in ARMS:
        y = [st.mean(per_run[arm][r]["rollout_rmse"][str(h)]["all"] for r in runs)
             for h in hor]
        ax[0].plot(hor, y, ls, color=col, marker=mk, ms=4, label=lab)
    ax[0].set_xlabel("rollout horizon (ticks)")
    ax[0].set_ylabel("state error, all channels")
    ax[0].set_xticks(hor)
    ax[0].grid(True, axis="y")
    ax[0].legend(loc="upper left")
    ax[0].set_title("(a) rollout error by horizon", loc="left")

    # (b) paired differences against the estimate, H20, per channel
    chans = ["extension", "velocity", "all"]
    labs = ["extension", "velocity", "pooled"]
    width = 0.36
    for j, (arm, lab, col) in enumerate([("mean", "vs mean", C_ORANGE),
                                         ("shuffled", "vs shuffled", C_PINK)]):
        cent, los, his = [], [], []
        for ch in chans:
            d = [(per_run["zhat"][r]["rollout_rmse"]["20"][ch]
                  - per_run[arm][r]["rollout_rmse"]["20"][ch])
                 / abs(st.mean(per_run[arm][rr]["rollout_rmse"]["20"][ch] for rr in runs))
                 for r in runs]
            m = st.mean(d)
            lo, hi = boot(d)
            cent.append(100 * m); los.append(100 * (m - lo)); his.append(100 * (hi - m))
        x = np.arange(len(chans)) + (j - 0.5) * width
        ax[1].barh(x, cent, height=width, color=col, alpha=0.85, label=lab,
                   xerr=[los, his], error_kw={"elinewidth": 0.8, "capsize": 2,
                                              "ecolor": "#333333"})
    ax[1].axvline(0, color="black", lw=0.8)
    ax[1].set_yticks(np.arange(len(chans)))
    ax[1].set_yticklabels(labs)
    ax[1].set_xlabel(r"change in error from using $\hat{z}$ (%)")
    ax[1].grid(True, axis="x")
    ax[1].legend(loc="lower left")
    ax[1].set_title("(b) paired advantage, 1.0 s horizon", loc="left")

    fig.savefig(OUT / "ch7_four_latents.pdf")
    plt.close(fig)
    print("wrote ch7_four_latents.pdf")


def fig_plant_switch(sw):
    tracks = sw["tracks"][:2]
    fig, ax = plt.subplots(1, len(tracks), figsize=(FULLW, 2.4), sharey=True)
    if len(tracks) == 1:
        ax = [ax]
    for k, (tr, a) in enumerate(zip(tracks, ax)):
        t = np.asarray(tr["t"])
        a.axvspan(0, sw["summary"]["window_s"], color=C_GRAY, alpha=0.12, lw=0)
        a.plot(t, tr["dA"], "-", color=C_ORANGE, label=r"$\|\hat{z}-z_A\|$ (old plant)")
        a.plot(t, tr["dB"], "-", color=C_BLUE, label=r"$\|\hat{z}-z_B\|$ (new plant)")
        a.axvline(0, color="black", lw=0.8, ls=":")
        a.set_xlabel("time from switch (s)")
        a.set_xlim(t.min(), t.max())
        a.grid(True, axis="y")
        a.set_title(f"({'ab'[k]}) runs {tr['run_A']} $\\rightarrow$ {tr['run_B']}",
                    loc="left")
        if k == 0:
            a.set_ylabel("latent distance")
            a.legend(loc="center left")
    fig.savefig(OUT / "ch7_plant_switch.pdf")
    plt.close(fig)
    print("wrote ch7_plant_switch.pdf")


def fig_benefit_scaling(per_run):
    info = json.loads((DATA / "splits" / "runs_info.json").read_text())
    runs = sorted(set(per_run["zhat"]) & set(per_run["mean"]))
    dev, ben = [], []
    for r in runs:
        k = info[f"{int(r):05d}"]["key"]
        e = [k[0], k[1], k[2], k[3], k[4] - 1.0, k[5]]
        dev.append(float(np.linalg.norm(e)))
        ben.append(per_run["mean"][r]["rollout_rmse"]["20"]["all"]
                   - per_run["zhat"][r]["rollout_rmse"]["20"]["all"])
    dev, ben = np.asarray(dev), np.asarray(ben) / 1e5

    fig, ax = plt.subplots(figsize=(FULLW * 0.56, 2.4))
    ax.axhline(0, color="black", lw=0.8)
    ax.scatter(dev, ben, s=18, color=C_BLUE, alpha=0.85, edgecolor="none")
    b, a0 = np.polyfit(dev, ben, 1)
    xs = np.linspace(dev.min(), dev.max(), 50)
    ax.plot(xs, b * xs + a0, "--", color=C_GRAY, lw=1.0)
    rho = float(np.corrcoef(dev, ben)[0, 1])
    ax.set_xlabel(r"plant deviation $\|e\|$")
    ax.set_ylabel(r"latent benefit ($\times10^{5}$)")
    ax.set_title(rf"$\rho = {rho:+.3f}$", loc="left")
    ax.grid(True, axis="y")
    fig.savefig(OUT / "ch7_benefit_scaling.pdf")
    plt.close(fig)
    print(f"wrote ch7_benefit_scaling.pdf  (rho={rho:+.3f})")


def main():
    per_run = json.loads((SRC / "latent_utility.json").read_text())["per_run"]
    sw = json.loads((SRC / "plant_switch.json").read_text())
    fig_four_latents(per_run)
    fig_plant_switch(sw)
    fig_benefit_scaling(per_run)


if __name__ == "__main__":
    main()
