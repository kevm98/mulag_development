#!/usr/bin/env python3
"""Chapter 7 authority figure: margin at nominal payload and under load.

Left column reads outputs/kappa (nominal payload, unchanged stacks). Right
column reads outputs/kappa_loaded (4.0x payload, frozen vs payload-conditioned
on one checkpoint). The old AFa arm in outputs/kappa was the history residual
and is deliberately not plotted.

Produces
    images/ch7/ch7_kappa_stacks.pdf
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "thesis/thesis/hydraulic_manipulator_thesis/images/ch7"
OUT.mkdir(parents=True, exist_ok=True)

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
    "lines.linewidth": 1.3, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def load(dirname):
    out = {}
    for f in sorted(glob.glob(str(REPO / dirname / "*_m1337.json"))):
        b = os.path.basename(f)
        arm = b.split("_k")[0]
        k = float(b.split("_k")[1].split("_m")[0])
        r = list(json.load(open(f))["per_mesh"].values())[0]
        out.setdefault(arm, {})[k] = (
            st.median([v * 1000 for v in r["per_env_lateral_m"]]),
            r["worst_family_ground_contact_fraction"],
        )
    return out


nom = load("outputs/kappa")
loaded = load("outputs/kappa_loaded")

PANELS = [
    ("Nominal payload", nom,
     [("R", C_GREEN, r"\textsf{R}" if False else "R"),
      ("IF", C_BLUE, "IF"),
      ("AFf", C_ORANGE, "AF frozen")]),
    (r"$4.0\times$ payload", loaded,
     [("AFfL", C_ORANGE, "AF frozen"),
      ("AFaL", C_PINK, "AF adaptive")]),
]

fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)

for col, (title, data, series) in enumerate(PANELS):
    for rowi, (idx, ylabel) in enumerate(((0, "median lateral error (mm)"),
                                          (1, "worst-family ground contact"))):
        ax = axes[rowi][col]
        for arm, colr, label in series:
            d = data.get(arm)
            if not d:
                continue
            ks = sorted(d)
            ax.plot(ks, [d[k][idx] for k in ks], marker="o", ms=3.0,
                    color=colr, label=label)
        ax.set_xscale("log")
        ax.grid(True, which="major")
        if rowi == 0:
            ax.set_title(title)
            ax.legend(loc="upper right")
        else:
            ax.set_xlabel(r"actuator authority $\kappa$")
        if col == 0:
            ax.set_ylabel(ylabel)

# Common y-limits per row so the two columns are directly comparable.
for rowi in range(2):
    lo = min(a.get_ylim()[0] for a in axes[rowi])
    hi = max(a.get_ylim()[1] for a in axes[rowi])
    for a in axes[rowi]:
        a.set_ylim(lo, hi)

fig.tight_layout(h_pad=0.8, w_pad=1.2)
dest = OUT / "ch7_kappa_stacks.pdf"
fig.savefig(dest)
print(f"wrote {dest}")
