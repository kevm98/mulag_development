#!/usr/bin/env python3
"""Sensor map drawn over the rendered machine.

Every callout marks a quantity the controller actually receives. The set is
taken from the scene construction in verge_task/terrain_env.py and from the
signals the collector records, not from a specification: what is drawn is what
the environment measures.

Each marker carries a soft elliptical shadow so it reads as sitting on the
surface of the machine rather than floating over the picture.

Run: /home/rrlab/miniconda3/envs/thesis_figures/bin/python scripts/make_sensor_overlay.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch
from PIL import Image

BASE = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis")
IMG = BASE / "images"
SRC = IMG / "unimog_mulag_iso.png"

PROP = "#2F5DA8"     # proprioceptive: inside the machine
EXTERO = "#C0392B"   # exteroceptive: looks at the world
INK, MUTED = "#1F2933", "#5A6B78"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
})

# (x, y) in ORIGINAL pixels, label, side the leader leaves from, group
SENSORS = [
    # Positions are read off the render so each marker sits on the component it
    # names, not merely near it.
    (400, 1250, "cutter contact\nand height ray", "left", EXTERO),
    (572,  819, "joint encoders\n(5 joints)", "left", PROP),
    (1143, 1000, "cylinder extension\nand rate", "right", PROP),
    (1330, 1150, "forward height rays\nleft and right of centre", "right", EXTERO),
    (1150, 1270, "wheel contact", "right", EXTERO),
]


def main():
    im = Image.open(SRC)
    W, H = im.size
    fig_w = 6.3
    fig = plt.figure(figsize=(fig_w, fig_w * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    label_x = {"left": 0.055 * W, "right": 0.965 * W}
    lanes = {"left": [], "right": []}
    for x, y, text, side, col in SENSORS:
        lanes[side].append((x, y, text, col))

    for side, items in lanes.items():
        items.sort(key=lambda t: t[1])
        n = len(items)
        for i, (x, y, text, col) in enumerate(items):
            ly = H * (0.16 + 0.68 * (i / max(n - 1, 1)))
            lx = label_x[side]
            # shadow: an ellipse under the marker so it sits ON the surface
            ax.add_patch(Ellipse((x, y + 16), 74, 26, facecolor="black",
                                 alpha=0.22, zorder=3, linewidth=0))
            ax.plot([x], [y], "o", ms=11, mfc=col, mec="white", mew=2.0,
                    zorder=5)
            ax.plot([x], [y], "o", ms=22, mfc="none", mec=col, mew=1.4,
                    alpha=.55, zorder=4)
            ha = "left" if side == "left" else "right"
            xytext = (lx, ly)
            ax.annotate(text, xy=(x, y), xytext=xytext, textcoords="data",
                        ha=ha, va="center", fontsize=7.6, color=INK,
                        zorder=6, linespacing=1.35,
                        bbox=dict(boxstyle="round,pad=0.34", fc="white",
                                  ec=col, lw=1.1, alpha=0.94),
                        arrowprops=dict(arrowstyle="-", color=col, lw=1.3,
                                        shrinkA=2, shrinkB=9,
                                        connectionstyle="arc3,rad=0.08"))

    # legend: the distinction that matters for the thesis
    for i, (col, name) in enumerate(((PROP, "proprioceptive — machine state"),
                                     (EXTERO, "exteroceptive — world contact"))):
        yy = H * (0.955 - 0.038 * i)
        ax.plot([0.055 * W], [yy], "o", ms=9, mfc=col, mec="white", mew=1.6,
                zorder=6)
        ax.annotate(name, (0.055 * W, yy), xytext=(16, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=MUTED, zorder=6)

    fig.savefig(IMG / "ch3_sensor_map.pdf", dpi=300)
    plt.close(fig)
    print(f"  wrote {IMG/'ch3_sensor_map.pdf'}")


if __name__ == "__main__":
    main()
