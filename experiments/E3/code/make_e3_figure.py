"""E3 figure: the sham control, which is what makes the null interpretable.

Both runs are the same policy on the same mesh with the same schedule. In one
the valve effectiveness is scaled by 0.6 at 9 s; in the other the schedule fires
and the plant is left alone. If the perturbation mattered, the two would part
company after 9 s. They do not.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = Path("/home/rrlab/rrlab_mulag")
OUT = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis/images")
V = REPO / "outputs/e3_verify"
C_RED, C_BLUE, C_GRAY = "#D31F26", "#0072B2", "#666666"
mpl.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": "#c8c8c8", "grid.linewidth": 0.4, "grid.alpha": 0.7,
    "legend.frameon": False, "pdf.fonttype": 42,
})
FULLW, STEP_S = 6.3, 9.0


def binned(t, y, w=1.0):
    edges = np.arange(0, t.max() + w, w)
    c, m = [], []
    for lo in edges[:-1]:
        k = (t >= lo) & (t < lo + w)
        if k.sum():
            c.append(lo + w / 2)
            m.append(y[k].mean())
    return np.array(c), np.array(m)


def main():
    A = np.load(V / "verify_m1337.npz")     # plant changes
    B = np.load(V / "sham_m1337.npz")       # plant does not
    ta = A["theta_elapsed_s"][:, :, 0][:, 0]
    tb = B["theta_elapsed_s"][:, :, 0][:, 0]
    la = np.abs(A["lateral_error"][:, :, 0])
    lb = np.abs(B["lateral_error"][:, :, 0])
    aa = A["theta_alpha"][:, :, 0].mean(1)
    ab = B["theta_alpha"][:, :, 0].mean(1)
    za = A["z_hat_raw"][:, :, 0].mean(1)
    zb = B["z_hat_raw"][:, :, 0].mean(1)

    fig, axes = plt.subplots(3, 1, figsize=(FULLW * 0.86, 4.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[.62, 1, .8]))

    ax = axes[0]
    ax.plot(ta, aa, color=C_RED, lw=2.0, label="plant changed")
    ax.plot(tb, ab, color=C_GRAY, lw=2.0, ls="--", label="plant unchanged")
    ax.set_ylabel("valve\neffectiveness")
    ax.legend(fontsize=7.8, loc="lower left", ncol=2)
    ax.set_title("(a) the perturbation really is applied", loc="left")
    ax.grid(True)

    ax = axes[1]
    for t, l, col, ls, lab in ((ta, la, C_RED, "-", "plant changed"),
                               (tb, lb, C_GRAY, "--", "plant unchanged")):
        c, m = binned(t, l.mean(1))
        ax.plot(c, 1000 * m, color=col, lw=2.0, ls=ls, label=lab)
    ax.set_ylabel("lateral error (mm)")
    ax.set_title("(b) and it makes no difference to the task", loc="left")
    ax.grid(True)

    ax = axes[2]
    ax.plot(ta, za, color=C_RED, lw=1.8)
    ax.plot(tb, zb, color=C_GRAY, lw=1.8, ls="--")
    ax.set_ylabel(r"$\hat{z}$  (dim 0)")
    ax.set_xlabel("time within the scored window (s)")
    ax.set_title("(c) the latent moves either way, so its motion is not detection",
                 loc="left")
    ax.grid(True)

    for ax in axes:
        ax.axvline(STEP_S, color="#111111", lw=1.1, ls=":")
    axes[0].annotate("change at 9 s", (STEP_S, aa.max()), xytext=(6, -4),
                     textcoords="offset points", fontsize=8)
    fig.subplots_adjust(left=.135, right=.98, top=.945, bottom=.10, hspace=.42)
    fig.savefig(OUT / "ch7_e3_sham.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_e3_sham.pdf'}")


if __name__ == "__main__":
    main()
