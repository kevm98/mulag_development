#!/usr/bin/env python3
"""Matplotlib publication-quality tracking comparison figure."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

t = np.linspace(0, 10, 500)
rng = np.random.default_rng(42)
ref_traj = np.sin(t) + 0.1 * np.sin(3 * t)
meas_traj = ref_traj + rng.normal(0, 0.05, len(t))
error = meas_traj - ref_traj

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "text.usetex": False,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

ax1.plot(t, ref_traj, "k-", linewidth=1.2, label="Reference")
ax1.plot(t, meas_traj, "r--", linewidth=0.8, label="Measured")
ax1.set_ylabel("Position [m]")
ax1.legend(loc="upper right")

ax2.plot(t, error, "b-", linewidth=0.8)
ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Error [m]")

fig.suptitle("Tracking Performance (Verification Data)")
fig.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
for fmt, dpi in [("pdf", None), ("svg", None), ("png", 600)]:
    path = os.path.join(OUTPUT_DIR, f"test_matplotlib.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    size = os.path.getsize(path)
    print(f"  {path}  ({size} bytes)")

plt.close(fig)
print("Matplotlib verification complete.")
