#!/usr/bin/env python3
"""Chapter 7 figures: the two open-loop claims and the identifiability envelope.

All numbers are read from the repository's validated artifacts under
outputs/stage31_openloop/; nothing is synthesized.

Produces
    images/ch7_openloop_claims.pdf     paired seeds + latent-mode decomposition
    images/ch7_identifiability.pdf     decode quality and drift vs excitation
    images/ch7_delay_response.pdf      edge-aligned step responses with onsets
    images/ch7_delay_summary.pdf       latency by link and amplitude
    images/ch7_e1_tracking.pdf         per-cell tracking distributions, 3 stacks
    images/ch7_e1_gates.pdf            gate margins, 3 stacks
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/rrlab/rrlab_mulag")
SRC = REPO / "outputs/stage31_openloop"
OUT = Path("/home/rrlab/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis/images")
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito subset, CVD-safe, matching Chapters 5 and 6.
C_BLUE, C_ORANGE, C_GREEN, C_PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
C_GRAY = "#666666"

mpl.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "grid.color": "#c8c8c8", "grid.linewidth": 0.4, "grid.alpha": 0.7,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "figure.constrained_layout.use": True,
})

FULLW = 6.3


def load_seeds():
    seeds, base, e2e, tv, ident = [], [], [], [], []
    for p in sorted((SRC / "seeds").glob("sd_none_s*/metrics.json")):
        s = p.parent.name.split("_s")[-1]
        q = SRC / f"seeds/e2e_s{s}/metrics.json"
        abl = SRC / f"seeds/e2e_s{s}/latent_ablation.json"
        if not (q.is_file() and abl.is_file()):
            continue
        m = json.loads(abl.read_text())["modes"]
        seeds.append(s)
        base.append(json.loads(p.read_text())["best_val_loss"])
        e2e.append(json.loads(q.read_text())["val_rollout_state_mse"])
        tv.append(100 * (m["run_mean"]["val"] / m["online"]["val"] - 1))
        ident.append(100 * (m["shuffled"]["val"] / m["run_mean"]["val"] - 1))
    return (seeds, np.array(base), np.array(e2e), np.array(tv), np.array(ident))


def fig_claims():
    """One ladder, not two panels.

    The four conditions form a single sequence, and the GAPS between them are
    the decomposition: what a latent is worth at all, what being right about the
    plant is worth, and what being allowed to move is worth. Splitting that into
    a slopegraph plus a bar chart of two percentages stated the same thing twice
    and showed the structure once.
    """
    seeds, base, e2e, tv, ident = load_seeds()
    n = len(seeds)
    per = {"online": [], "run_mean": [], "shuffled": []}
    for sd in seeds:
        m = json.loads((SRC / f"seeds/e2e_s{sd}/latent_ablation.json").read_text())["modes"]
        for k in per:
            per[k].append(m[k]["val"])
    per = {k: np.array(v) for k, v in per.items()}

    # Top to bottom is worst to best, so reading down the figure is reading the
    # improvement in order.
    rows = [("no latent at all", base, C_GRAY, True),
            ("constant latent,\nwrong plant", per["shuffled"], C_PINK, False),
            ("constant latent,\ncorrect plant", per["run_mean"], C_ORANGE, False),
            ("online estimate", per["online"], C_BLUE, False)]

    fig, ax = plt.subplots(figsize=(FULLW * 0.9, 3.1))
    ys = np.arange(len(rows))
    means = np.array([r[1].mean() for r in rows])
    ax.plot(means, ys, "-", color="#8C99A4", lw=1.2, zorder=1)
    for i, (label, vals, col, sep) in enumerate(rows):
        ax.scatter(vals, np.full(n, i) + np.linspace(-.10, .10, n), s=30,
                   color=col, zorder=3, edgecolor="white", linewidth=.6)
        ax.plot([vals.mean()] * 2, [i - .22, i + .22], color="#111111", lw=2.0,
                zorder=4)
        ax.annotate(f"{vals.mean():.5f}", (vals.mean(), i), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#333333")

    # the two gaps that are the actual claim
    xr = ax.get_xlim() if False else None
    span = float(means.max() - means.min())
    lane = float(means.max()) + 0.30 * span     # a clear column, clear of the ticks

    def bracket(i, j, text):
        ax.annotate("", xy=(lane, i), xytext=(lane, j),
                    arrowprops=dict(arrowstyle="<->", color="#111111", lw=1.2))
        for k in (i, j):
            ax.plot([lane, means[k]], [k, k], color="#C4CCD2", lw=.8, ls=":",
                    zorder=0)
        ax.annotate(text, (lane, 0.5 * (i + j)), xytext=(8, 0),
                    textcoords="offset points", fontsize=8.4, ha="left",
                    va="center", linespacing=1.35)

    bracket(1, 2, f"being right about\nthe plant is worth\n"
                  f"{100*(per['shuffled'].mean()/per['run_mean'].mean()-1):.2f}%")
    bracket(2, 3, f"being allowed to move\nis worth "
                  f"{100*(per['run_mean'].mean()/per['online'].mean()-1):.1f}%")

    ax.axhline(0.5, color="#C9D1D8", lw=1.0, ls=":")
    ax.annotate("separately trained model,\nnot an ablation of the one below",
                (means[0], 0), xytext=(9, -20), textcoords="offset points",
                fontsize=7.4, color="#6B7780", va="center", linespacing=1.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(-.55, len(rows) - .45)
    ax.invert_yaxis()
    ax.set_xlim(None, lane + 1.15 * span)
    ax.set_xlabel("free-running state error   (points are the five seeds)")
    ax.grid(axis="x")
    fig.subplots_adjust(left=.175, right=.975, top=.965, bottom=.155)
    fig.savefig(OUT / "ch7_openloop_claims.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_openloop_claims.pdf'}  (n={n}, ladder)")


def fig_identifiability():
    """A matrix, not a line plot.

    The quantity is a decode score for six factors at four excitation levels --
    twenty-four numbers on two categorical axes. Drawn as overlaid lines on a
    twin axis it read as clutter and invited the usual dual-axis misreading. As
    a matrix each cell is legible on its own and the pattern is immediate: one
    factor is recoverable, and only at low excitation.
    """
    leg = json.loads((SRC / "legibility_by_amplitude.json").read_text())
    dr = json.loads((SRC / "excitation_drift.json").read_text())["tiers"]
    factors = leg["factors"]
    amps = sorted(float(a) for a in leg["by_amplitude"])
    M = np.array([leg["by_amplitude"][f"{a}"] for a in amps])        # (amp, factor)

    fig = plt.figure(figsize=(FULLW * 0.92, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.0], wspace=.30,
                          left=.14, right=.97, top=.86, bottom=.20)

    ax = fig.add_subplot(gs[0])
    vmax = 0.75
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels([f.replace("_", " ") for f in factors], rotation=28,
                       ha="right", fontsize=8)
    ax.set_yticks(range(len(amps)))
    ax.set_yticklabels([f"{a:g}" for a in amps], fontsize=8.5)
    ax.set_ylabel("excitation amplitude")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                    color="#111111" if abs(v) < 0.45 else "white")
    ax.set_title(r"(a) held-out decode $R^2$ of each plant factor", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    cb.ax.tick_params(labelsize=7.5)

    ax2 = fig.add_subplot(gs[1])
    ratio = [r["drift_over_travel"] for r in dr]
    sat = [r["command_saturation_pct"] / 100 for r in dr]
    y = np.arange(len(amps))
    ax2.barh(y - .19, ratio, height=.36, color="#C8860D", label="drift / stroke")
    ax2.barh(y + .19, sat, height=.36, color="#7A8B99", label="saturation")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax.invert_yaxis()          # row 0 (lowest amplitude) to the bottom
    ax2.set_ylim(ax.get_ylim()[::-1])   # and the bar panel follows it exactly
    ax2.set_xlim(0, 0.62)
    ax2.grid(axis="x")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.set_title("(b) why", loc="left")
    fig.savefig(OUT / "ch7_identifiability.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_identifiability.pdf'}")


# --------------------------------------------------------------------------
# E2 -- transport delay
# --------------------------------------------------------------------------
RAW = REPO / "AAM_module/datasets/rom_openloop_v1"
E1 = REPO / "outputs/e1_tracking"

# The eight-gate contract (PPO_ROM/gates.py). Duplicated here rather than
# imported because that module pulls in gymnasium and the whole IsaacLab stack,
# which the figure environment does not have.
GATES = [
    ("cut height",        "mean_abs_cut_height_error_m",             0.040, "max"),
    ("lateral",           "worst_family_mean_abs_lateral_error_m",   0.300, "max"),
    ("ground contact",    "worst_family_ground_contact_fraction",    0.700, "min"),
    ("head alignment",    "p01_head_alignment",                      0.900, "min"),
    ("hard impacts",      "hard_impact_step_fraction",               0.350, "max"),
    ("tool acceleration", "p95_tool_acceleration_mps2",            120.000, "max"),
]
# Red / amber / green, matching the path figure so a stack keeps one colour
# throughout the chapter.
C_RED, C_AMBER, C_GRN = "#C1272D", "#E0A100", "#1B843F"
STACKS = [("ppo_rom", "PPO-ROM", C_RED),
          ("ppo_id_fd", "PPO-ID-FD", C_AMBER),
          ("ppo_aam_id_fd", "PPO-(AAM)-ID-FD", C_GRN)]
MESHES = ["1337", "17", "909"]


def _edges(cmd, quantile=0.9):
    d = np.abs(np.diff(cmd))
    thr = max(np.quantile(d, quantile), 1e-9)
    idx = np.flatnonzero(d > thr) + 1
    if idx.size == 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] > 8:
            keep.append(i)
    return np.array(keep)


def _tier(meta, amp_want):
    import re
    lo, hi = None, None
    for s in meta["segment_ids"]:
        if s.get("kind") != "probe":
            continue
        m = re.search(r"_a([0-9.]+)_", s["name"])
        if m and abs(float(m.group(1)) - amp_want) < 1e-9:
            lo = s["start"] if lo is None else min(lo, s["start"])
            hi = s["end"] if hi is None else max(hi, s["end"])
    return lo, hi


def fig_delay_response(amp=0.15, n_runs=24):
    """When each stage of the chain actually starts moving, on one time axis.

    A normalised step response answers "what does the pressure do" but not the
    question the section is asking, which is "how long until each thing
    happens". This is that question drawn literally: the command edge is the
    origin, and every downstream stage sits at the delay measured for it, with
    the interquartile spread beside it.
    """
    delays = json.loads((SRC / "transport_delay.json").read_text())["links"]
    key = f"{amp}"
    chain = [("valve command", None, "#111111"),
             ("differential pressure", "command->dP", C_BLUE),
             ("cylinder extension", "command->cyl_ext", C_GREEN),
             ("joint angle", "command->joint_pos", C_PINK),
             ("cylinder velocity", "command->cyl_vel", C_ORANGE)]

    fig, ax = plt.subplots(figsize=(FULLW * 0.86, 2.85))
    ys = np.arange(len(chain))
    xmax = 0.0
    for i, (label, k, col) in enumerate(chain):
        if k is None:
            ax.plot([0], [i], "|", ms=16, color=col, mew=2.4, zorder=4)
            ax.annotate("edge at $t = 0$", (0, i), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=8.4)
            continue
        rec = delays[k][key]
        med = rec["median_ms"]
        q1, q3 = np.array(rec["iqr_steps"]) * 1000.0 / 60.0
        xmax = max(xmax, q3)
        ax.plot([0, med], [i, i], color=col, lw=6.5, alpha=.22,
                solid_capstyle="butt", zorder=1)
        ax.plot([q1, q3], [i, i], color=col, lw=2.0, zorder=3,
                solid_capstyle="round")
        ax.plot([med], [i], "o", ms=8.5, color=col, mec="white", mew=1.1,
                zorder=5)
        ax.annotate(f"{med:.0f} ms", (med, i), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=8.6,
                    color=col)
    ax.axvline(0, color="#111111", lw=1.3, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in chain], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-18, xmax * 1.12)
    ax.set_xlabel("time after the valve command changes (ms)")
    ax.grid(axis="x")
    fig.text(0.985, 0.012, "bar: interquartile range over command edges",
             ha="right", fontsize=7.4, color="#6B7780")
    fig.subplots_adjust(left=.235, right=.98, top=.955, bottom=.235)
    fig.savefig(OUT / "ch7_delay_response.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_delay_response.pdf'}  (propagation timeline)")


def fig_delay_summary():
    d = json.loads((SRC / "transport_delay.json").read_text())["links"]
    order = [("command$\\rightarrow$pressure", "command->dP", C_BLUE),
             ("pressure$\\rightarrow$velocity", "dP->cyl_vel", C_ORANGE),
             ("command$\\rightarrow$extension", "command->cyl_ext", C_GREEN),
             ("command$\\rightarrow$joint", "command->joint_pos", C_PINK)]
    amps = sorted({float(a) for k in d for a in d[k]})
    fig, ax = plt.subplots(figsize=(FULLW * 0.66, 2.5))
    w = 0.2
    x = np.arange(len(amps))
    for i, (label, key, col) in enumerate(order):
        vals = [d[key].get(f"{a}", {}).get("median_ms", np.nan) for a in amps]
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=col, label=label, alpha=.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in amps])
    ax.set_xlabel("excitation amplitude")
    ax.set_ylabel("median onset delay (ms)")
    ax.grid(axis="y")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("(b) latency by link; the last two coincide, which is the control",
                 loc="left", fontsize=8)
    fig.savefig(OUT / "ch7_delay_summary.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_delay_summary.pdf'}")


# --------------------------------------------------------------------------
# E1 -- closed-loop tracking
# --------------------------------------------------------------------------
def _e1_cells(stack):
    """(mesh, family) -> family metric dict."""
    cells = {}
    for m in MESHES:
        p = E1 / f"{stack}_s17_m{m}.json"
        if not p.is_file():
            return None
        blk = json.loads(p.read_text())["per_mesh"][m]
        for fam in blk["path_family_results"]:
            cells[(m, fam["family"])] = fam
    return cells


def _e1_worst(stack):
    """Worst-mesh gate metrics."""
    vals = {}
    for m in MESHES:
        blk = json.loads((E1 / f"{stack}_s17_m{m}.json").read_text())["per_mesh"][m]
        for _, key, _, direction in GATES:
            v = blk.get(key)
            if v is None:
                continue
            vals[key] = v if key not in vals else (
                max(vals[key], v) if direction == "max" else min(vals[key], v))
    return vals


def fig_e1_tracking():
    """A dumbbell chart by trajectory family, sorted by difficulty.

    Box plots over pooled cells hid the thing that matters: WHICH trajectories
    separate the stacks. Here each row is one family, the bar spans the three
    stacks, and the ordering makes the difficulty range explicit -- two families
    are an order of magnitude harder than the rest and are what carry the means.
    """
    data = {s: _e1_cells(s) for s, _, _ in STACKS}
    if any(v is None for v in data.values()):
        print("  skip ch7_e1_tracking: missing reports")
        return
    keys = sorted(set.intersection(*(set(v) for v in data.values())))
    fams = sorted({f for _, f in keys})
    per = {s: {f: np.mean([data[s][(m, f)]["mean_abs_lateral_error_m"]
                           for m in MESHES if (m, f) in data[s]])
               for f in fams} for s, _, _ in STACKS}
    order = sorted(fams, key=lambda f: np.mean([per[s][f] for s, _, _ in STACKS]))

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(FULLW, 3.5), gridspec_kw=dict(width_ratios=[1.9, 1]))

    y = np.arange(len(order))
    for i, f in enumerate(order):
        vals = [per[s][f] for s, _, _ in STACKS]
        axa.plot([min(vals), max(vals)], [i, i], color="#B9C2C9", lw=2.2,
                 zorder=1, solid_capstyle="round")
    for s, label, col in STACKS:
        axa.scatter([per[s][f] for f in order], y, s=42, color=col, zorder=3,
                    edgecolor="white", linewidth=.8, label=label)
    axa.set_yticks(y)
    axa.set_yticklabels([f.replace("_", " ") for f in order], fontsize=8.5)
    axa.invert_yaxis()
    axa.set_xscale("log")
    # Default log minor ticks collide into an unreadable smear at this width.
    axa.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    axa.set_xticklabels(["0.02", "0.05", "0.10", "0.20", "0.50"], fontsize=8.5)
    axa.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axa.set_xlabel("mean lateral error (m)")
    axa.grid(axis="x")
    axa.set_title("(a) by trajectory family, easiest first", loc="left")
    axa.legend(fontsize=7.6, loc="lower right")

    # (b) the paired contrast that isolates adaptation
    d = np.array([per["ppo_aam_id_fd"][f] - per["ppo_id_fd"][f] for f in order])
    col = [C_GRN if v < 0 else C_RED for v in d]
    axb.barh(y, 1000 * d, color=col, height=.62)
    axb.axvline(0, color="#111111", lw=1.0)
    axb.set_yticks(y)
    axb.set_yticklabels([])
    axb.invert_yaxis()
    axb.set_xlabel("AAM $-$ ID-FD (mm)")
    axb.grid(axis="x")
    better = int((d < 0).sum())
    axb.set_title(f"(b) adaptive better in {better} of {len(order)}", loc="left")
    fig.subplots_adjust(left=.20, right=.985, top=.90, bottom=.13, wspace=.10)
    fig.savefig(OUT / "ch7_e1_tracking.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_e1_tracking.pdf'}  ({len(order)} families)")


def fig_e1_gates():
    """Gate margins as a dot chart against a shaded failing region.

    Grouped bars forced the reader to compare bar lengths across a zero line
    that meant pass or fail. One row per gate, one dot per stack and a shaded
    half-plane for failure states the verdict directly.
    """
    fig, ax = plt.subplots(figsize=(FULLW * 0.88, 3.0))
    labels = [g[0] for g in GATES]
    y = np.arange(len(labels))
    marg = {}
    for s, label, col in STACKS:
        w = _e1_worst(s)
        out = []
        for _, key, thr, direction in GATES:
            v = w.get(key, np.nan)
            out.append((thr - v) / abs(thr) if direction == "max"
                       else (v - thr) / abs(thr))
        marg[s] = np.array(out)

    lo = min(float(np.nanmin(m)) for m in marg.values())
    ax.axvspan(min(lo * 1.15, -0.1), 0, color=C_RED, alpha=.07, lw=0)
    for i in y:
        vals = [marg[s][i] for s, _, _ in STACKS]
        ax.plot([min(vals), max(vals)], [i, i], color="#B9C2C9", lw=2.2,
                zorder=1, solid_capstyle="round")
    for s, label, col in STACKS:
        ax.scatter(marg[s], y, s=46, color=col, zorder=3, edgecolor="white",
                   linewidth=.8, label=label)
    ax.axvline(0, color="#111111", lw=1.2)
    ax.annotate("fails", (0, len(labels) - .38), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=8,
                color=C_RED)
    ax.annotate("passes", (0, len(labels) - .38), xytext=(6, 0),
                textcoords="offset points", ha="left", fontsize=8,
                color="#2E7355")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("gate margin, in units of the threshold")
    ax.grid(axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.subplots_adjust(left=.21, right=.985, top=.96, bottom=.145)
    fig.savefig(OUT / "ch7_e1_gates.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'ch7_e1_gates.pdf'}")


if __name__ == "__main__":
    fig_claims()
    fig_identifiability()
    fig_delay_response()
    fig_delay_summary()
    fig_e1_tracking()
    fig_e1_gates()
