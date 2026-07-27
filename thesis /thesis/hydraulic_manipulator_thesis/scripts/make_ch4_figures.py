#!/usr/bin/env python3
"""Chapter 4 figures, computed from the actual data-collection design.

(a) curriculum composition of the 1000-run closed-loop batch
(b) effective bulk modulus that the hydraulic randomization actually
    produces, evaluated with the same formula and clamp as
    DOMAIN_hydraulic.m
(c) timeline of the 64 s open-loop excitation schedule from
    valve_excitation.m

Run:  python3 scripts/make_ch4_figures.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis/images")

C_BLUE, C_ORANGE, C_GREEN, C_PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
C_GRAY = "#666666"

mpl.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "grid.color": "#c8c8c8", "grid.linewidth": 0.4, "grid.alpha": 0.7,
    "lines.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42,
    "figure.constrained_layout.use": True,
})
FULLW = 6.3


def beta_eff(eps_air, eps_vol):
    """Effective bulk modulus in GPa, exactly as DOMAIN_hydraulic.m computes
    it: series compliance of oil, entrained air and dead volume, clamped."""
    pa = 1.0 / ((1.0 - eps_air) / 0.5e9 + eps_air / (1.4 * 1e7)
                + eps_vol / 0.5e9)
    return np.clip(pa / 1e9, 0.7, 1.8)


def lhs(n, d, rng):
    """Latin hypercube sample, as MATLAB's lhsdesign does."""
    out = np.empty((n, d))
    for j in range(d):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out


def main():
    fig = plt.figure(figsize=(FULLW, 4.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---------------- (a) curriculum composition ----------------
    blocks = [("Nominal", 1, 1, C_GRAY),
              ("Random hydraulics", 2, 251, C_BLUE),
              ("Random mechanics", 252, 501, C_ORANGE),
              ("Fully randomized", 502, 1000, C_GREEN)]
    for name, lo, hi, col in blocks:
        ax_a.barh(0, hi - lo + 1, left=lo - 1, height=0.55,
                  color=col, edgecolor="white", linewidth=1.2)
        if hi - lo > 20:
            ax_a.text((lo + hi) / 2 - 0.5, 0, f"{hi-lo+1}", ha="center",
                      va="center", color="white", fontsize=8.5, weight="bold")
    ax_a.set_xlim(0, 1000)
    ax_a.set_ylim(-0.45, 0.45)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Run index in the closed-loop curriculum")
    ax_a.set_title("(a) Composition of the 1000-run randomized curriculum",
                   fontsize=9)
    ax_a.legend(handles=[Patch(facecolor=c, label=n) for n, _, _, c in blocks],
                ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.55),
                fontsize=7.8)
    for sp in ("left", "right", "top"):
        ax_a.spines[sp].set_visible(False)

    # ---------------- (b) bulk modulus: the clamp swallows the range ---
    rng = np.random.default_rng(0)
    m = lhs(499, 12, rng)                      # fully randomized block size
    eps_air = 0.001 + 0.079 * m[:, 0]
    eps_vol = 0.005 + 0.075 * m[:, 1]

    raw = 1.0 / ((1.0 - eps_air) / 0.5e9 + eps_air / (1.4 * 1e7)
                 + eps_vol / 0.5e9) / 1e9
    ax_b.hist(raw, bins=30, color=C_BLUE, alpha=0.85, edgecolor="white",
              linewidth=0.4, label=r"$\beta$ from formula")
    ax_b.axvline(0.7, color=C_PINK, lw=1.8, label="clamp floor 0.70 GPa")
    ax_b.annotate("whole range sits\nbelow the floor:\n"
                  r"model gets 0.70 GPa",
                  xy=(0.70, ax_b.get_ylim()[1] * 0.60),
                  xytext=(0.24, ax_b.get_ylim()[1] * 0.60),
                  fontsize=7.0, color=C_PINK, va="center",
                  arrowprops=dict(arrowstyle="->", color=C_PINK, lw=1.0))
    ax_b.set_xlim(0.05, 0.85)
    ax_b.set_xlabel(r"$\beta_{\mathrm{eff}}$ before clamping [GPa]")
    ax_b.set_ylabel("runs")
    ax_b.grid(True, axis="y")
    ax_b.legend(loc="upper left", fontsize=7.2)
    ax_b.set_title("(b) Base bulk modulus is clamped flat", fontsize=9)

    # ---------------- (c) excitation schedule ----------------
    T_PRE, T_CYL, T_GAP, T_MIMO, T_SHUT = 1.0, 10.0, 0.5, 20.0, 1.0
    ax_c.barh(4, T_PRE, left=0, height=0.6, color=C_GRAY, alpha=0.55)
    t = T_PRE
    for i in range(4):
        ax_c.barh(3 - i, T_CYL, left=t, height=0.6, color=C_ORANGE)
        ax_c.text(t + T_CYL / 2, 3 - i, f"C{i+1}", ha="center", va="center",
                  color="white", fontsize=8, weight="bold")
        t += T_CYL
        ax_c.barh(3 - i, T_GAP, left=t, height=0.6, color="#DDDDDD")
        t += T_GAP
    for i in range(4):
        ax_c.barh(3 - i, T_MIMO, left=t, height=0.6, color=C_GREEN, alpha=0.9)
    ax_c.text(t + T_MIMO / 2, 1.5, "all four channels", ha="center",
              va="center", color="white", fontsize=8, weight="bold")
    ax_c.barh(4, T_SHUT, left=t + T_MIMO, height=0.6, color=C_GRAY, alpha=0.55)
    ax_c.set_xlim(0, 64)
    ax_c.set_yticks([4, 3, 2, 1, 0])
    ax_c.set_yticklabels(["excl.", "C1", "C2", "C3", "C4"])
    ax_c.set_xlabel("time in the open-loop run [s]")
    ax_c.set_title("(c) 64 s excitation schedule", fontsize=9)
    ax_c.grid(True, axis="x")

    fig.savefig(OUT / "ch4_dataset_design.pdf")
    print("wrote", OUT / "ch4_dataset_design.pdf")

    # =================================================================
    # Second figure: parameters as ACTUALLY DELIVERED to the plant,
    # read from the stored Params of all 1000 curriculum runs.
    # =================================================================
    import json
    pj = Path("/tmp/claude-1000/-home-rrlab-rrlab-mulag/daf43d86-60c4-47af-aabf"
              "-5d25a14e7108/scratchpad/params.json")
    if pj.exists():
        d = json.loads(pj.read_text())
        vd = np.asarray(d["vd"])              # mL
        vd0 = 512.5                            # nominal dead volume, mL
        beta0 = 0.70                           # GPa
        beta_eq = beta0 * vd0 / vd             # GPa
        kp = beta0 / (vd / 1000.0)             # GPa per litre
        cols = {"cd": np.asarray(d["cd"]), "pset": np.asarray(d["pset"]),
                "beq": beta_eq, "kp": kp}
        fig2, axs = plt.subplots(1, 4, figsize=(FULLW, 1.95))
        specs = [
            ("cd", r"discharge coeff. $c_d$", C_BLUE, 0.70),
            ("pset", "relief setting [MPa]", C_ORANGE, 27.6),
            ("beq", r"$\beta_{\mathrm{eq}}$ [GPa]", C_GREEN, beta0),
            ("kp", r"$K_p^{d}=\beta_0/V_d$ [GPa/L]", C_PINK, beta0 / (vd0 / 1e3)),
        ]
        for ax, (key, lab, col, nom) in zip(axs, specs):
            ax.hist(cols[key], bins=26, color=col, alpha=0.85,
                    edgecolor="white", linewidth=0.3)
            ax.axvline(nom, color=C_GRAY, lw=1.3, ls="--")
            ax.set_xlabel(lab, fontsize=8)
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y")
        axs[0].set_ylabel("runs", fontsize=8)
        axs[2].axvspan(0.70, 0.72, color=C_GRAY, alpha=0.12)
        fig2.savefig(OUT / "ch4_delivered_params.pdf")
        print("wrote", OUT / "ch4_delivered_params.pdf")

    print("nominal beta_eff = %.3f GPa" % beta_eff(0.005, 0.02))
    print("pre-clamp beta: min %.3f  median %.3f  max %.3f GPa"
          % (raw.min(), np.median(raw), raw.max()))


SCRATCH = Path("/tmp/claude-1000/-home-rrlab-rrlab-mulag/"
               "daf43d86-60c4-47af-aabf-5d25a14e7108/scratchpad")
C_CYL = [C_BLUE, C_ORANGE, C_GREEN, C_PINK]


def closed_loop_trace():
    """One real closed-loop operating run (fully randomized draw)."""
    import json
    d = json.loads((SCRATCH / "cl.json").read_text())["trace"]
    t = np.asarray(d["t"])
    ext = np.asarray(d["ext"])
    vc = np.asarray(d["vc"])
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(FULLW, 3.2))
    for c in range(4):
        a1.plot(t, ext[c], color=C_CYL[c], lw=1.1, label=f"cyl {c+1}")
        a2.plot(t, vc[c], color=C_CYL[c], lw=0.9)
    a1.set_ylabel("extension [m]")
    a1.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28),
              fontsize=7.5)
    a1.grid(True, axis="y")
    a2.set_ylabel("valve command")
    a2.set_xlabel("time [s]")
    a2.set_xlim(t.min(), t.max())
    a2.grid(True, axis="y")
    fig.savefig(OUT / "ch4_closed_loop_trace.pdf")
    print("wrote", OUT / "ch4_closed_loop_trace.pdf")


def coverage_comparison():
    """Closed-loop (broad, coordinated) vs open-loop (narrow bands)
    extension coverage per cylinder -- the R1-vs-R2 tension made visual."""
    import json
    cl = json.loads((SCRATCH / "cl.json").read_text())["pool"]
    ol = json.loads((SCRATCH / "ol.json").read_text())
    fig, axs = plt.subplots(1, 4, figsize=(FULLW, 2.0), sharey=True)
    for c, ax in enumerate(axs):
        lo = min(min(cl[c]), min(ol[c]))
        hi = max(max(cl[c]), max(ol[c]))
        bins = np.linspace(lo, hi, 34)
        ax.hist(cl[c], bins=bins, density=True, color=C_CYL[c], alpha=0.55,
                label="closed-loop")
        ax.hist(ol[c], bins=bins, density=True, histtype="step",
                color="#222222", lw=1.3, label="open-loop")
        ax.set_title(f"cylinder {c+1}", fontsize=8.5)
        ax.set_xlabel("extension [m]", fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("density", fontsize=8)
    axs[0].legend(fontsize=6.8, loc="upper right")
    fig.savefig(OUT / "ch4_coverage.pdf")
    print("wrote", OUT / "ch4_coverage.pdf")


def control_regimes():
    """Closed-loop data spans saturation / mid / low-control regimes
    (diversity), but the valve command tracks the cylinder velocity
    (the identifiability weakness). Pie of regime occupancy + scatter."""
    import json
    d = json.loads((SCRATCH / "cl.json").read_text())["trace"]
    t = np.asarray(d["t"])
    ext = np.asarray(d["ext"])
    vc = np.asarray(d["vc"])
    dt = np.gradient(t)
    vel = np.vstack([np.gradient(ext[c]) / dt for c in range(4)])
    u = vc.ravel()
    v = vel.ravel()
    au = np.abs(u)
    sat = au >= 0.8
    mid = (au >= 0.3) & (au < 0.8)
    low = au < 0.3
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULLW, 2.6),
                                 gridspec_kw={"width_ratios": [1.0, 1.5]})
    # pie
    sizes = [sat.mean(), mid.mean(), low.mean()]
    cols = [C_PINK, C_ORANGE, C_BLUE]
    labs = [f"saturated\n{sizes[0]*100:.0f}%", f"mid\n{sizes[1]*100:.0f}%",
            f"low control\n{sizes[2]*100:.0f}%"]
    a1.pie(sizes, colors=cols, labels=labs, startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1.5),
           textprops=dict(fontsize=8))
    a1.set_title("(a) time in each control regime", fontsize=9)
    # scatter valve vs velocity
    for mask, col, lab in [(low, C_BLUE, "low"), (mid, C_ORANGE, "mid"),
                           (sat, C_PINK, "saturated")]:
        a2.scatter(v[mask], u[mask], s=3, color=col, alpha=0.35, label=lab,
                   linewidths=0)
    a2.axhline(0.8, color=C_GRAY, lw=0.6, ls=":")
    a2.axhline(-0.8, color=C_GRAY, lw=0.6, ls=":")
    a2.set_xlabel("cylinder velocity [m/s]")
    a2.set_ylabel("valve command")
    a2.set_title("(b) command tracks velocity: valve not identifiable",
                 fontsize=9)
    a2.grid(True)
    a2.legend(fontsize=7.5, loc="lower right", markerscale=2.5)
    fig.savefig(OUT / "ch4_regimes.pdf")
    print("wrote", OUT / "ch4_regimes.pdf")


if __name__ == "__main__":
    main()
    closed_loop_trace()
    coverage_comparison()
    control_regimes()
