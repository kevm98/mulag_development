"""Step response of the three stacks to a 100 mm step in the commanded line."""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(Path.home() / "rrlab_mulag/thesis/thesis/hydraulic_manipulator_thesis/images")
R = Path(Path.home() / "rrlab_mulag/outputs/e2_refstep")
J = Path(Path.home() / "rrlab_mulag/experiments/E2/results/reference_step_response.json")
C_RED, C_AMBER, C_GRN = "#D31F26", "#C8860D", "#0F8A3C"
STACKS = [("ppo_rom", "PPO-ROM", C_RED), ("ppo_id_fd", "PPO-ID-FD", C_AMBER),
          ("ppo_aam_id_fd", "PPO-(AAM)-ID-FD", C_GRN)]
FS, PLANT_L = 60.0, 117.0
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


def curves(key, pre=30, post=300):
    d = np.load(R / f"{key}_m1337.npz")
    ref, cut = d["reference_y"][:, :, 0], d["cutter_y"][:, :, 0]
    segs = []
    for e in range(ref.shape[1]):
        dr = np.diff(ref[:, e])
        k = int(np.argmax(np.abs(dr))) + 1
        step = float(dr[k - 1])
        if abs(step) < 0.05 or k < pre or k + post >= len(ref):
            continue
        base = ref[:, e].copy()
        base[k:] -= step
        y = (cut[:, e] - base)
        y = y - y[k - pre:k].mean()
        segs.append(y[k - pre:k + post] / step)
    return np.array(segs)


def main():
    stats = json.loads(J.read_text())["stacks"]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(FULLW, 2.9),
                                  gridspec_kw=dict(width_ratios=[1.5, 1]))
    pre, post = 30, 300
    t = (np.arange(-pre, post)) / FS
    ax.axhline(1.0, color="#111111", lw=1.2, ls="--")
    ax.axhline(0.0, color="#999999", lw=.8)
    ax.axvline(0.0, color="#111111", lw=1.0, ls=":")
    ax.axvspan(0, PLANT_L / 1000, color="#8899A6", alpha=.20, lw=0)
    for key, lab, col in STACKS:
        s = curves(key, pre, post)
        ax.plot(t, np.median(s, 0), color=col, lw=2.0, label=lab)
        ax.fill_between(t, np.percentile(s, 25, 0), np.percentile(s, 75, 0),
                        color=col, alpha=.13, lw=0)
    ax.annotate("plant dead time\n117 ms", (PLANT_L / 1000, -0.42), xytext=(6, 0),
                textcoords="offset points", fontsize=7.6, color="#5A6B78")
    ax.set_xlim(-0.4, 4.0)
    ax.set_ylim(-0.55, 2.0)
    ax.set_xlabel("time since the commanded step (s)")
    ax.set_ylabel("response, in units of the step")
    ax.set_title("(a) response to a 100 mm step in the commanded line", loc="left")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=7.8)

    names = ["dead_time_ms", "rise_ms", "settle_ms"]
    nice = ["dead time", "rise", "settled"]
    y = np.arange(len(names))
    off = np.linspace(-.24, .24, len(STACKS))
    for i, (key, lab, col) in enumerate(STACKS):
        vals = [stats[lab][n] for n in names]
        axb.barh(y + off[i], vals, height=.22, color=col, edgecolor="white",
                 linewidth=.6)
        for j, v in enumerate(vals):
            axb.annotate(f"{v:.0f}", (v, y[j] + off[i]), xytext=(3, 0),
                         textcoords="offset points", va="center", fontsize=7.2,
                         color=col)
    axb.axvline(PLANT_L, color="#5A6B78", lw=1.2, ls="--")
    axb.annotate("plant 117 ms", (PLANT_L, 2.42), xytext=(5, 0),
                 textcoords="offset points", fontsize=7.2, color="#5A6B78")
    # The command rate limit alone bounds how fast a 100 mm step can be
    # completed: 0.004 m per control step at 60 Hz is 0.24 m/s, so 417 ms.
    axb.axvline(417, color="#111111", lw=1.0, ls=":")
    axb.annotate("rate limit 417 ms", (417, 2.42), xytext=(5, 0),
                 textcoords="offset points", fontsize=7.2, color="#333333")
    axb.set_yticks(y)
    axb.set_yticklabels(nice, fontsize=8.5)
    axb.invert_yaxis()
    axb.set_xlabel("ms")
    axb.set_xlim(0, 3800)
    axb.grid(axis="x")
    axb.set_title("(b) summary", loc="left")
    fig.subplots_adjust(left=.095, right=.985, top=.90, bottom=.175, wspace=.30)
    fig.savefig(OUT / "ch7_ref_step.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_ref_step.pdf'}")


if __name__ == "__main__":
    main()
