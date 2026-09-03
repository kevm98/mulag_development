#!/usr/bin/env python3
"""Sensor map drawn over the rendered machine.

Every callout marks a quantity the controller actually receives. The set is
taken from the scene construction in verge_task/terrain_env.py and from the
signals the collector records, not from a specification: what is drawn is what
the environment measures.

Each callout marker carries a soft elliptical shadow so it reads as sitting on
the surface of the machine rather than floating over the picture. The three
exteroceptive sensors additionally get red ground markers showing WHERE they
sense: the tyre contact patches, the strip under the cutter, and the terrain
rays ahead of the cutter. Those ground positions are not placed by eye. They
are built from an affine ground frame fitted to the three visible tyre contact
points, so a marker one metre forward is one metre forward in the render.

Run: python3 scripts/make_sensor_overlay.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
IMG = BASE / "images"
# The clean render, without callouts. Recovered from the published figure after
# the original export was lost, so the plate below is byte-identical to the one
# the previous version of this figure used.
SRC = IMG / "ch3" / "ch3_sensor_base.png"

PROP = "#2F5DA8"     # proprioceptive: inside the machine
EXTERO = "#C0392B"   # exteroceptive: looks at the world
INK, MUTED = "#1F2933", "#5A6B78"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
})

# (x, y) in plate pixels, label, side the leader leaves from, group.
# Positions are read off the render so each marker sits on the component it
# names, not merely near it.
SENSORS = [
    (316, 973, "cutter contact\nand height ray", "left", EXTERO),
    (444, 646, "joint encoders\n(5 joints)", "left", PROP),
    (900, 787, "cylinder extension\nand rate", "right", PROP),
    (1050, 895, "forward height rays\nleft and right of centre", "right", EXTERO),
    (914, 988, "wheel contact", "right", EXTERO),
]

# --- ground frame -----------------------------------------------------------
# The three tyre contact patches visible in the plate, found from the lowest
# dark pixel of each wheel. They fix an affine map from (forward, lateral)
# metres to plate pixels, which is what keeps the sensed-ground markers on the
# ground instead of merely near the wheels.
CONTACT_FRONT_NEAR = np.array([1250.0, 1172.0])
CONTACT_FRONT_FAR = np.array([940.0, 1133.0])
CONTACT_REAR_NEAR = np.array([1570.0, 980.0])

WHEELBASE_M = 3.60          # U 5023 wheelbase
TRACK_M = 2.00              # track width

ORIGIN = CONTACT_FRONT_NEAR
FWD_M = (CONTACT_FRONT_NEAR - CONTACT_REAR_NEAR) / WHEELBASE_M
LAT_M = (CONTACT_FRONT_FAR - CONTACT_FRONT_NEAR) / TRACK_M

# The cutter rides on the far side, ahead of the front axle. These two numbers
# place it and are the only fitted quantities here: they put the ground frame's
# cutter position under the drawn cutter head.
CUTTER_FWD_M, CUTTER_LAT_M = 1.5, 5.27


def ground(fwd_m, lat_m):
    """Plate pixel for a point on the ground, in metres from the front axle."""
    return ORIGIN + FWD_M * fwd_m + LAT_M * lat_m


def sensed_points():
    """Red ground markers, grouped so each group can be styled separately."""
    groups = {}

    # Tyre contact patches: a small rosette at each, so the marker reads as a
    # patch rather than a single point.
    wheel = []
    for centre in (CONTACT_FRONT_NEAR, CONTACT_FRONT_FAR, CONTACT_REAR_NEAR):
        wheel.append(centre)
        for k in range(6):
            th = 2 * np.pi * k / 6
            wheel.append(centre + np.array([26 * np.cos(th), 9 * np.sin(th)]))
    groups["wheel"] = np.array(wheel)

    # The strip the cutter senses, across its working width. It is drawn a
    # little ahead of the drum rather than directly beneath it, so the callout
    # label for the cutter does not cover the markers it refers to.
    strip_fwd = CUTTER_FWD_M + 0.4
    cutter = [ground(strip_fwd + 0.16 * t, CUTTER_LAT_M + 0.42 * s)
              for s in (-1, 0, 1) for t in (-1, 0, 1)]
    groups["cutter"] = np.array(cutter)

    # Terrain rays ahead of the cutter, left and right of its centre line.
    # The forward reach is kept short so the rows stay on visible ground
    # instead of running off the near edge of the plate.
    rays = [ground(strip_fwd + a, CUTTER_LAT_M + b)
            for b in (-0.40, 0.40)
            for a in np.linspace(0.55, 2.10, 6)]
    groups["rays"] = np.array(rays)
    return groups


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

    # --- sensed ground, drawn under the callouts ---------------------------
    groups = sensed_points()
    for name, pts in groups.items():
        size = {"wheel": 4.2, "cutter": 5.0, "rays": 5.4}[name]
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=size + 3.2, mfc="none",
                mec=EXTERO, mew=0.9, alpha=0.30, zorder=2, linestyle="none")
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=size, mfc=EXTERO, mec="none",
                alpha=0.92, zorder=2, linestyle="none")

    label_x = {"left": 0.055 * W, "right": 0.965 * W}
    lanes = {"left": [], "right": []}
    for x, y, text, side, col in SENSORS:
        lanes[side].append((x, y, text, col))

    lane_span = {"left": 0.68, "right": 0.68}
    for side, items in lanes.items():
        items.sort(key=lambda t: t[1])
        n = len(items)
        for i, (x, y, text, col) in enumerate(items):
            ly = H * (0.16 + lane_span[side] * (i / max(n - 1, 1)))
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

    # Legend: the distinction that matters for the thesis. It sits on the empty
    # ground to the right of the machine, clear of the sensed-ground markers
    # that occupy the near corner ahead of the cutter.
    legend_x = 0.50 * W
    entries = ((PROP, "proprioceptive — machine state", 9),
               (EXTERO, "exteroceptive — world contact", 9),
               (EXTERO, "sensed ground (ray hits, contact patches)", 5.4))
    for i, (col, name, ms) in enumerate(entries):
        yy = H * (0.900 + 0.038 * i)
        ax.plot([legend_x], [yy], "o", ms=ms, mfc=col,
                mec="white" if ms > 6 else "none", mew=1.6 if ms > 6 else 0,
                zorder=6)
        ax.annotate(name, (legend_x, yy), xytext=(16, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=MUTED, zorder=6)

    fig.savefig(IMG / "ch3" / "ch3_sensor_map.pdf", dpi=300)
    plt.close(fig)
    print(f"  wrote {IMG/'ch3'/'ch3_sensor_map.pdf'}")


if __name__ == "__main__":
    main()
