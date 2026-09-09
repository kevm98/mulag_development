"""Chain latency compared across the three stacks, on one time axis."""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(Path.home() / "rrlab_mulag/thesis/thesis/hydraulic_manipulator_thesis/images")
SRC = Path(Path.home() / "rrlab_mulag/experiments/E2/results/chain_delay_by_stack.json")
C_RED, C_AMBER, C_GRN = "#D31F26", "#C8860D", "#0F8A3C"
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
ORDER = [("PPO-ROM", C_RED), ("PPO-ID-FD", C_AMBER), ("PPO-(AAM)-ID-FD", C_GRN)]


def main():
    d = json.loads(SRC.read_text())["stacks"]
    links = list(next(iter(d.values())).keys())
    nice = [l.replace("$\\to$", "→").replace("$", "") for l in links]

    fig, ax = plt.subplots(figsize=(FULLW * 0.92, 3.0))
    y = np.arange(len(links))
    off = np.linspace(-.24, .24, len(ORDER))
    for k, (lab, col) in enumerate(ORDER):
        vals = [d[lab][l]["median_ms"] for l in links]
        ax.barh(y + off[k], vals, height=.22, color=col, label=lab,
                edgecolor="white", linewidth=.6)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.0f}", (v, y[i] + off[k]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7.6,
                        color=col)
    ax.set_yticks(y)
    ax.set_yticklabels(nice, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("cross-correlation lag (ms)")
    ax.set_xlim(0, max(d[l2][l]["median_ms"] for l2 in d for l in links) * 1.16)
    ax.grid(axis="x")
    ax.legend(loc="lower right", fontsize=8)
    ax.axhspan(len(links) - 1.5, len(links) - .5, color="#EEF2F4", zorder=0)
    ax.annotate("control: a static linkage, so this row must read zero",
                (ax.get_xlim()[1] * .99, len(links) - 1), ha="right",
                va="center", fontsize=7.4, color="#6B7780")
    fig.subplots_adjust(left=.30, right=.985, top=.965, bottom=.155)
    fig.savefig(OUT / "ch7_chain_by_stack.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_chain_by_stack.pdf'}")


if __name__ == "__main__":
    main()
