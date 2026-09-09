"""Control effort against the tracking it buys."""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(Path.home() / "rrlab_mulag/thesis/thesis/hydraulic_manipulator_thesis/images")
SRC = Path(Path.home() / "rrlab_mulag/experiments/E1_tracking_and_control_effort/results/control_effort.json")
TR = Path(Path.home() / "rrlab_mulag/outputs/e2_chain")
C = {"PPO-ROM": "#D31F26", "PPO-ID-FD": "#C8860D", "PPO-(AAM)-ID-FD": "#0F8A3C"}
KEY = {"PPO-ROM": "ppo_rom", "PPO-ID-FD": "ppo_id_fd",
       "PPO-(AAM)-ID-FD": "ppo_aam_id_fd"}
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


def main():
    d = json.loads(SRC.read_text())
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULLW, 2.85),
                                   gridspec_kw=dict(width_ratios=[1, 1.15]))

    # (a) cost against benefit -- effort is meaningless without the error
    for lab, r in d.items():
        axa.scatter(1000 * r["mean_abs_lateral_m"], r["rms_command"], s=95,
                    color=C[lab], zorder=3, edgecolor="white", linewidth=1.0,
                    label=lab)
        axa.annotate(lab.replace("PPO-", ""), (1000 * r["mean_abs_lateral_m"],
                     r["rms_command"]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=7.6,
                     color=C[lab])
    axa.set_xlabel("mean lateral error (mm)")
    axa.set_ylabel("RMS valve command")
    axa.set_xlim(0, 140)
    axa.set_ylim(0.5, 0.92)
    axa.grid(True)
    axa.set_title("(a) what the effort buys", loc="left")
    axa.annotate("better", (8, 0.545), fontsize=7.4, color="#5A6B78")
    axa.annotate("", xy=(4, 0.53), xytext=(30, 0.60),
                 arrowprops=dict(arrowstyle="->", color="#8C99A4", lw=1.0))

    # (b) command reversals: chatter, which the RMS does not reveal
    labs = list(d)
    y = np.arange(len(labs))
    axb.barh(y, [d[l]["reversals_per_s"] for l in labs],
             color=[C[l] for l in labs], height=.55, edgecolor="white")
    for i, l in enumerate(labs):
        axb.annotate(f"{d[l]['reversals_per_s']:.1f}",
                     (d[l]["reversals_per_s"], i), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=8,
                     color=C[l])
    axb.set_yticks(y)
    axb.set_yticklabels([l.replace("PPO-", "") for l in labs], fontsize=8.5)
    axb.invert_yaxis()
    axb.set_xlabel("command reversals per second")
    axb.set_xlim(0, 48)
    axb.grid(axis="x")
    axb.set_title("(b) chatter", loc="left")
    fig.subplots_adjust(left=.10, right=.975, top=.90, bottom=.175, wspace=.42)
    fig.savefig(OUT / "ch7_control_effort.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_control_effort.pdf'}")


if __name__ == "__main__":
    main()
