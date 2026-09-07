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

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs/final"
OUT = REPO / "thesis/thesis/hydraulic_manipulator_thesis/images/ch7"
OUT.mkdir(parents=True, exist_ok=True)

B = 20000
SEEDS = ("17", "23", "101")
MASSES = ("3.0", "4.0", "5.0")

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

fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.5))

labels = []
for m in MASSES:
    for s in SEEDS:
        labels.append(rf"${m}\times$  s{s}")
    labels.append(rf"${m}\times$  pooled")

for ax, (key, title, unit_name, unit, hb) in zip(axes, CHANNELS):
    ys, meds, los, his, kinds = [], [], [], [], []
    y = 0
    for m in MASSES:
        frozen = row(f"frozen_p{m}.json")
        pool = []
        for s in SEEDS:
            med, lo, hi, d = paired(frozen, row(f"adaptive_s{s}_p{m}.json"),
                                    key, unit, hb)
            pool += d
            ys.append(y); meds.append(med); los.append(lo); his.append(hi)
            kinds.append("seed")
            y += 1
        random.seed(0)
        n = len(pool)
        boot = sorted(st.median([pool[random.randrange(n)] for _ in range(n)])
                      for _ in range(B))
        ys.append(y); meds.append(st.median(pool))
        los.append(boot[int(0.025 * B)]); his.append(boot[int(0.975 * B)])
        kinds.append("pool")
        y += 1

    ax.axvline(0.0, color=C_GRAY, lw=0.8, zorder=1)
    for yy, md, lo, hi, kind in zip(ys, meds, los, his, kinds):
        sig = lo > 0 or hi < 0
        if kind == "pool":
            col, marker, ms = "black", "D", 4.0
        else:
            col = C_BLUE if md > 0 else C_ORANGE
            marker, ms = "o", 4.5
        ax.plot([lo, hi], [yy, yy], color=col,
                lw=1.9 if kind == "pool" else 1.2, solid_capstyle="round",
                zorder=2)
        ax.plot([md], [yy], marker=marker, ms=ms, color=col, zorder=3,
                markerfacecolor=col if sig else "white",
                markeredgecolor=col, markeredgewidth=1.0)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels if ax is axes[0] else [])
    ax.invert_yaxis()
    ax.set_ylim(len(labels) - 0.4, -0.6)
    for b in (3.5, 7.5):
        ax.axhline(b, color="#c8c8c8", lw=0.5, zorder=0)
    ax.set_title(title)
    ax.set_xlabel(f"paired median difference ({unit_name})")
    ax.grid(axis="x", zorder=0)

axes[0].set_ylabel("payload and seed")
fig.text(0.5, -0.035, r"adaptive better $\rightarrow$", ha="center", fontsize=8)
fig.tight_layout(w_pad=1.4)
dest = OUT / "ch7_adaptation_forest.pdf"
fig.savefig(dest)
print(f"wrote {dest}")
