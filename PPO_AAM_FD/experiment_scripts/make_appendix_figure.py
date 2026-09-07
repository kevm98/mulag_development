#!/usr/bin/env python3
"""Appendix figure: per-environment paired differences against the A/A floor.

Every environment's paired difference is drawn, sorted, rather than summarised.
This shows three things the tables cannot: how many environments improve, how
heterogeneous the effect is, and how it compares with the difference the
instrument reports between two identical arms.

Reads outputs/final. Produces images/appendix/app_paired_differences.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs/final"
AA = REPO / "outputs/aa_floor"
OUT = REPO / "thesis/thesis/hydraulic_manipulator_thesis/images/appendix"
OUT.mkdir(parents=True, exist_ok=True)

C_BLUE, C_ORANGE, C_GREEN, C_PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
C_GRAY = "#666666"
SEEDS = ("17", "23", "101")

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
    "grid.color": "#c8c8c8", "grid.linewidth": 0.4, "grid.alpha": 0.7,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def row(path):
    return list(json.load(open(path))["per_mesh"].values())[0]


def diffs(frozen, arm, key):
    f = [v * 1000 for v in frozen[key]]
    a = [v * 1000 for v in arm[key]]
    return [f[i] - a[i] for i in range(min(len(f), len(a)))]


# This volume is FAT32 and litters AppleDouble "._" siblings next to every
# file; they are not JSON and must not be counted.
aa_files = sorted(p for p in AA.glob("*.json") if not p.name.startswith("._"))
aa = [row(p) for p in aa_files] if len(aa_files) == 2 else None

fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
CH = (("per_env_cut_error_m", "Cut height"), ("per_env_lateral_m", "Lateral"))

for r, (key, name) in enumerate(CH):
    for c, m in enumerate(("4.0", "5.0")):
        ax = axes[r][c]
        fr = row(SRC / f"frozen_p{m}.json")
        pool = []
        for s in SEEDS:
            pool += diffs(fr, row(SRC / f"adaptive_s{s}_p{m}.json"), key)
        pool.sort()
        x = [100.0 * i / (len(pool) - 1) for i in range(len(pool))]
        ax.axhline(0.0, color=C_GRAY, lw=0.8)
        ax.plot(x, pool, color=C_BLUE, label="adaptive vs frozen")
        if aa is not None and m == "4.0":
            ad = sorted(diffs(aa[0], aa[1], key))
            xa = [100.0 * i / (len(ad) - 1) for i in range(len(ad))]
            ax.plot(xa, ad, color=C_ORANGE, lw=1.1, ls="--", zorder=3,
                    label="A/A, two identical arms")
            print(f"   A/A drawn on {name} {m}x, span "
                  f"{ad[0]:.0f} to {ad[-1]:.0f} mm")
        win = 100.0 * sum(1 for v in pool if v > 0) / len(pool)
        import statistics as _st
        ax.set_title(f"{name}, {m}x   ({win:.0f}% of environments improve)")
        ax.grid(True)
        if r == 1:
            ax.set_xlabel("environments, sorted (%)")
        if c == 0:
            ax.set_ylabel("paired difference (mm)")
        if r == 0 and c == 0:
            ax.legend(loc="upper left")

fig.tight_layout(h_pad=0.9, w_pad=1.2)
dest = OUT / "app_paired_differences.pdf"
fig.savefig(dest)
print(f"wrote {dest}")
