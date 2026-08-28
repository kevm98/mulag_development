"""E5: the distribution of per-trajectory performance at each randomisation scale.

Earlier versions plotted the MEAN over trajectory families as a line. That hides
the structure of the result: PPO-ID-FD's apparent spike at scale 1 is a single
trajectory, and a mean carries it while the median does not move at all. Showing
the distribution at each point makes the level, the spread and the outliers
readable at once, and is what the comparison actually rests on.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os

REPO_ROOT = Path(os.environ.get("ROOT", "/home/rrlab/rrlab_mulag"))
OUT = REPO_ROOT / "thesis/thesis/hydraulic_manipulator_thesis/images"
ROOT = REPO_ROOT / "outputs/e5_robustness"
STACKS = [("ppo_rom", "PPO-ROM", "#D31F26"),
          ("ppo_id_fd", "PPO-ID-FD", "#C8860D"),
          ("ppo_aam_id_fd", "PPO-(AAM)-ID-FD", "#0F8A3C")]
SCALES = ["0.0", "0.5", "1.0", "1.5", "2.0"]
MESH = "1337"
mpl.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": .6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": "#c8c8c8", "grid.linewidth": .4, "grid.alpha": .7,
    "legend.frameon": False, "pdf.fonttype": 42})
FULLW = 6.3


def per_family(stack, scale, key):
    p = ROOT / f"{stack}_s{scale}_m{MESH}.json"
    blk = json.loads(p.read_text())["per_mesh"][MESH]
    return np.array([f[key] for f in blk["path_family_results"]], float)


def panel(ax, key, ylabel, logy=True):
    width = 0.24
    for si, (stack, label, col) in enumerate(STACKS):
        pos = np.arange(len(SCALES)) + (si - 1) * width
        data = [per_family(stack, sc, key) for sc in SCALES]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.86,
                        patch_artist=True, showfliers=True, manage_ticks=False,
                        medianprops=dict(color="#111111", lw=1.2),
                        whiskerprops=dict(color=col, lw=1.0),
                        capprops=dict(color=col, lw=1.0),
                        flierprops=dict(marker="o", ms=3.2, mfc="white",
                                        mec=col, mew=1.0, alpha=.95))
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=.55, edgecolor=col, linewidth=1.2)
    ax.set_xticks(np.arange(len(SCALES)))
    ax.set_xticklabels(SCALES)
    ax.set_xlabel("randomisation scale")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y")


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULLW, 2.85))
    panel(a1, "mean_abs_lateral_error_m", "lateral error (m)")
    panel(a2, "mean_abs_cut_height_error_m", "cut-height error (m)")
    a1.set_title("(a) lateral tracking", loc="left")
    a2.set_title("(b) cut height", loc="left")

    # the outlier that a mean would have reported as a breakdown
    y = per_family("ppo_id_fd", "1.0", "mean_abs_lateral_error_m").max()
    a1.annotate("a single trajectory; the median does not move",
                (2.0, y), xytext=(10, 4), textcoords="offset points",
                fontsize=7.2, color="#8A6A10", ha="left",
                arrowprops=dict(arrowstyle="->", color="#8A6A10", lw=.9))

    handles = [plt.Line2D([], [], color=c, lw=7, alpha=.62, label=l)
               for _, l, c in STACKS]
    fig.legend(handles=handles, ncol=3, loc="upper center",
               bbox_to_anchor=(.5, 1.0), fontsize=8, frameon=False)
    fig.subplots_adjust(left=.095, right=.985, top=.845, bottom=.165, wspace=.29)
    fig.savefig(OUT / "ch7_e5_robustness.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_e5_robustness.pdf'}")


if __name__ == "__main__":
    main()
