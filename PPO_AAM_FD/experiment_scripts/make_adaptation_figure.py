#!/usr/bin/env python3
"""Chapter 7 adaptation figure: paired effect of online payload conditioning.

Numbers are recomputed here from outputs/final/*.json, the same files the
chapter's tables are built from, so the figure and the tables cannot drift
apart. Nothing is hardcoded.

Produces
    images/ch7/ch7_adaptation_forest.pdf
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError:
    mpl = plt = np = None

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs/final"
OUT = REPO / "thesis/thesis/hydraulic_manipulator_thesis/images/ch7"
OUT.mkdir(parents=True, exist_ok=True)

if mpl is None:
    from ch7_svg_fallback import render_adaptation
    dest = OUT / "ch7_adaptation_forest.pdf"
    render_adaptation(REPO, dest)
    print(f"wrote {dest} (SVG fallback)")
    raise SystemExit

B = 20000
SEEDS = ("17", "23", "101")
MASSES = ("3.0", "4.0", "5.0")

# Chapter 7 comparison palette: frozen in blue, adaptive in golden yellow.
C_BLUE, C_GOLD = "#4E79A7", "#C8860D"
C_SEED, C_DARK = "#71808C", "#26343F"
C_GRID, C_SPINE = "#D5DADF", "#7C8791"

mpl.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 9,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "grid.color": C_GRID, "grid.linewidth": 0.5, "grid.alpha": 0.75,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def row(name):
    p = SRC / name
    if not p.exists():
        raise SystemExit(f"missing {p}")
    per_mesh = json.load(open(p))["per_mesh"]
    return list(per_mesh.values())[0]


def paired(frozen, arm, key, unit, higher_better):
    f = [v * unit for v in frozen[key]]
    a = [v * unit for v in arm[key]]
    n = min(len(f), len(a))
    d = [(a[i] - f[i]) if higher_better else (f[i] - a[i]) for i in range(n)]
    random.seed(0)
    boot = sorted(st.median([d[random.randrange(n)] for _ in range(n)])
                  for _ in range(B))
    return st.median(d), boot[int(0.025 * B)], boot[int(0.975 * B)], d


CHANNELS = (
    ("per_env_cut_error_m", "Cut height", "mm", 1000.0, False),
    ("per_env_lateral_m", "Lateral", "mm", 1000.0, False),
    ("per_env_contact_fraction", "Ground contact", "fraction", 1.0, True),
)

fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.75))
x = np.arange(len(MASSES))
width = .34

for ax, (key, title, unit_name, unit, hb) in zip(axes, CHANNELS):
    frozen_vals, adaptive_vals, ci_low, ci_high = [], [], [], []
    for m in MASSES:
        frozen = row(f"frozen_p{m}.json")
        frozen_median = st.median(v * unit for v in frozen[key])
        pool = []
        for s in SEEDS:
            _, _, _, d = paired(
                frozen, row(f"adaptive_s{s}_p{m}.json"), key, unit, hb)
            pool += d
        random.seed(0)
        n = len(pool)
        boot = sorted(st.median([pool[random.randrange(n)] for _ in range(n)])
                      for _ in range(B))
        effect = st.median(pool)
        lo, hi = boot[int(.025 * B)], boot[int(.975 * B)]
        adaptive = frozen_median + effect if hb else frozen_median - effect
        absolute_lo, absolute_hi = ((frozen_median + lo, frozen_median + hi)
                                    if hb else
                                    (frozen_median - hi, frozen_median - lo))
        frozen_vals.append(frozen_median)
        adaptive_vals.append(adaptive)
        ci_low.append(adaptive - absolute_lo)
        ci_high.append(absolute_hi - adaptive)

    frozen_bars = ax.bar(x - width / 2, frozen_vals, width, color=C_BLUE,
                         label="Frozen", zorder=3)
    adaptive_bars = ax.bar(x + width / 2, adaptive_vals, width, color=C_GOLD,
                           label="Adaptive", zorder=3,
                           yerr=np.array([ci_low, ci_high]), capsize=3,
                           error_kw=dict(ecolor=C_DARK, lw=1.0, capthick=1.0))
    fmt = (lambda v: f"{v:.0f}") if unit_name == "mm" else (lambda v: f"{v:.2f}")
    for bars in (frozen_bars, adaptive_bars):
        for bar in bars:
            ax.annotate(fmt(bar.get_height()),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.6, fontweight="bold")
    cue = "lower is better" if unit_name == "mm" else "higher is better"
    ylabel = f"{title.lower()} ({unit_name})" if unit_name == "mm" else "contact fraction"
    ax.set_title(title, fontweight="bold", pad=16)
    ax.text(.5, 1.01, cue, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, color=C_SEED)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"${m}\times$" for m in MASSES])
    ax.set_xlabel("payload")
    ax.set_ylim(0, max(frozen_vals + adaptive_vals +
                       [a + b for a, b in zip(adaptive_vals, ci_high)]) * 1.22)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(C_SPINE)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.0),
           columnspacing=2.0)
fig.subplots_adjust(left=.08, right=.99, top=.79, bottom=.16, wspace=.34)
dest = OUT / "ch7_adaptation_forest.pdf"
fig.savefig(dest)
print(f"wrote {dest}")
