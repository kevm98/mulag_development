#!/usr/bin/env python3
"""Chapter 3 figure: bulk-modulus sensitivity of the Simscape model.

Data are the measured values from the author's master-project report
"Mechanical-Hydraulic Modeling of an Embankment Mower in Simscape"
(Tables I and II), re-plotted in the thesis figure style.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis/images")

C_BLUE, C_ORANGE, C_GREEN, C_PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
C_GRAY = "#666666"

mpl.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "grid.color": "#c8c8c8",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.7,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "figure.constrained_layout.use": True,
})

# Tables I and II of the Simscape report.
beta = [0.5, 1.0, 1.5, 2.0]                    # effective bulk modulus [GPa]
tau_j1 = [847.4, 725.3, 674.0, 649.7]          # valve-to-joint 50% delay [ms]
tau_j3 = [186.3, 168.8, 149.9, 152.4]
tau_j4 = [202.1, 206.4, 215.8, 221.6]
tau_rms = [445.4, 386.4, 361.7, 351.6]         # delay index [ms]
e_rms = [827.9, 775.6, 753.2, 739.4]           # EE tracking RMS error [mm]
e_mean = [211.3, 215.4, 229.1, 228.8]          # EE mean absolute error [mm]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.9))

ax1.plot(beta, tau_j1, color=C_BLUE, marker="o", ms=4, label="Joint 1 (slew)")
ax1.plot(beta, tau_j3, color=C_ORANGE, marker="s", ms=4, label="Joint 3 (boom II)")
ax1.plot(beta, tau_j4, color=C_PINK, marker="^", ms=4, label="Joint 4 (head pivot)")
ax1.plot(beta, tau_rms, color=C_GRAY, marker="d", ms=4, ls="--",
         label=r"Delay index $\tau_{\mathrm{RMS}}$")
ax1.set_xlabel(r"Effective bulk modulus $\beta$ [GPa]")
ax1.set_ylabel("Valve-to-joint 50\\% delay [ms]")
ax1.set_xticks(beta)
ax1.set_ylim(0, 900)
ax1.grid(True, axis="y")
ax1.legend(loc="lower left", fontsize=7.5)
ax1.set_title("(a) Actuation delay", fontsize=9)

ax2.plot(beta, e_rms, color=C_BLUE, marker="o", ms=4, label=r"$E_{\mathrm{RMS}}$")
ax2.plot(beta, e_mean, color=C_ORANGE, marker="s", ms=4, label=r"$E_{\mathrm{mean}}$")
ax2.set_xlabel(r"Effective bulk modulus $\beta$ [GPa]")
ax2.set_ylabel("End-effector tracking error [mm]")
ax2.set_xticks(beta)
ax2.set_ylim(0, 900)
ax2.grid(True, axis="y")
ax2.legend(loc="center right", fontsize=7.5)
ax2.set_title("(b) Task-space tracking error", fontsize=9)

fig.savefig(OUT / "ch3_bulk_modulus_sensitivity.pdf")
print("wrote", OUT / "ch3_bulk_modulus_sensitivity.pdf")
