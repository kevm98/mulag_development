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

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    mpl = plt = None

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "thesis/thesis/hydraulic_manipulator_thesis/images/ch7"
OUT.mkdir(parents=True, exist_ok=True)

if mpl is None:
    from ch7_svg_fallback import render_kappa
    dest = OUT / "ch7_kappa_stacks.pdf"
    render_kappa(REPO, dest)
    print(f"wrote {dest} (SVG fallback)")
    raise SystemExit

# Chapter 7 stack palette: R is red, IF is amber, and adaptation is green.
# The frozen AF reference stays a quiet slate blue in both columns.
C_RED, C_AMBER, C_GREEN, C_BLUE = "#C1272D", "#C8860D", "#1B843F", "#4E79A7"
C_GRAY, C_GRID, C_SPINE = "#5A6B78", "#D5DADF", "#7C8791"

mpl.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "grid.color": C_GRID, "grid.linewidth": 0.55, "grid.alpha": 0.75,
    "lines.linewidth": 2.0, "legend.frameon": False,
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
     [("R", C_RED, "R (PPO-ROM)", "o", "-"),
      ("IF", C_AMBER, "IF (PPO-ID-FD)", "s", "-"),
      ("AFf", C_BLUE, "AF frozen", "D", "-")]),
    (r"$4.0\times$ payload", loaded,
     [("AFfL", C_BLUE, "AF frozen", "D", "-"),
      ("AFaL", C_GREEN, "AF adaptive", "^", "-")]),
]

fig, axes = plt.subplots(4, 1, figsize=(6.3, 7.4), sharex=True)
specs = [
    ("(a) Nominal payload - lateral error", PANELS[0], 0,
     "median lateral error (mm)", "lower is better"),
    (r"(b) $4.0\times$ payload - lateral error", PANELS[1], 0,
     "median lateral error (mm)", "lower is better"),
    ("(c) Nominal payload - ground contact", PANELS[0], 1,
     "worst-family ground contact", "higher is better"),
    (r"(d) $4.0\times$ payload - ground contact", PANELS[1], 1,
     "worst-family ground contact", "higher is better"),
]

for panel_i, (panel_title, (_, data, series), idx, ylabel, cue) in enumerate(specs):
        ax = axes[panel_i]
        for arm, colr, label, marker, linestyle in series:
            d = data.get(arm)
            if not d:
                continue
            ks = sorted(d)
            ax.plot(ks, [d[k][idx] for k in ks], marker=marker, ms=5.2,
                    color=colr, markeredgecolor="white", markeredgewidth=.65,
                    linestyle=linestyle, linewidth=2.3, label=label, zorder=3)
        ax.set_xscale("log")
        ax.grid(True, which="major")
        ax.set_axisbelow(True)
        ax.set_xticks([.01, .03, .1, .3, 1.0])
        ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1.0"])
        ax.set_xlim(.008, 1.25)
        for spine in ax.spines.values():
            spine.set_color(C_SPINE)
        ax.set_title(panel_title, loc="left", pad=8, fontweight="bold")
        ax.text(.02, .94, cue, transform=ax.transAxes, va="top",
                fontsize=8.2, color=C_GRAY)
        ax.legend(loc="upper right", ncol=len(series), handlelength=2.0,
                  borderaxespad=.45, columnspacing=1.2)
        ax.set_xlabel(r"actuator authority $\kappa$ (fraction of nominal)")
        if idx == 1:
            ax.axhline(.70, color=C_GRAY, lw=1.0, ls=(0, (4, 2)), zorder=1)
            ax.annotate("contact gate 0.70", xy=(.011, .70), xytext=(3, 4),
                        textcoords="offset points", fontsize=7.8, color=C_GRAY,
                        va="bottom")
        ax.set_ylabel(ylabel)

# Common y-limits within each metric pair so payloads remain comparable.
for first, second in ((0, 1), (2, 3)):
    lo = min(axes[first].get_ylim()[0], axes[second].get_ylim()[0])
    hi = max(axes[first].get_ylim()[1], axes[second].get_ylim()[1])
    axes[first].set_ylim(lo, hi)
    axes[second].set_ylim(lo, hi)

fig.subplots_adjust(left=.145, right=.985, top=.97, bottom=.065, hspace=.72)
dest = OUT / "ch7_kappa_stacks.pdf"
fig.savefig(dest)
print(f"wrote {dest}")
