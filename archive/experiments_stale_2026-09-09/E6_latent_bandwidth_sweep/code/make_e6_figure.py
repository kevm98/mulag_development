"""E6: task error against the policy-path latent time constant.

Three panels, because the result needs all three to be read honestly. The first
two show the response curve itself, with the unfiltered and latched arms drawn
as the two reference levels the curve sits between. The third shows the paired
per-cell differences for the one contrast that resolves, which is what stops the
curve being read as a uniform improvement: the mean moves because a minority of
cells move a long way, and the panel makes that visible rather than asserted.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(Path.home() / "rrlab_mulag/thesis/thesis/hydraulic_manipulator_thesis/images")
ROOT = Path(Path.home() / "rrlab_mulag/outputs/e6_latent_bandwidth")
MESHES = ["1337", "17", "909"]
LPF = [("bw_lpf_5p0", 0.032), ("bw_lpf_1p0", 0.159), ("bw_lpf_0p64", 0.250),
       ("bw_lpf_0p32", 0.500), ("bw_lpf_0p2", 0.796), ("bw_lpf_0p04", 3.979)]
C_CURVE, C_RAW, C_LATCH = "#0F8A3C", "#D31F26", "#C8860D"

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


def cells(arm):
    out = {}
    for m in MESHES:
        blk = json.loads((ROOT / f"{arm}_m{m}.json").read_text())["per_mesh"][m]
        for f in blk["path_family_results"]:
            out[(m, f["family"])] = f
    return out


DATA = {a: cells(a) for a in
        ["bw_estimated", "bw_frozen"] + [k for k, _ in LPF]}
KEYS = sorted(set.intersection(*(set(v) for v in DATA.values())))


def vals(arm, key):
    return np.array([DATA[arm][c][key] for c in KEYS], float)


def boot_ci(x, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    b = np.array([x[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def panel(ax, key, label, scale, letter):
    taus = np.array([t for _, t in LPF])
    m = np.array([vals(a, key).mean() for a, _ in LPF]) * scale
    lo, hi = np.array([boot_ci(vals(a, key)) for a, _ in LPF]).T * scale
    raw = vals("bw_estimated", key).mean() * scale
    lat = vals("bw_frozen", key).mean() * scale

    ax.axhline(raw, color=C_RAW, lw=1.0, ls="--", zorder=1)
    ax.axhline(lat, color=C_LATCH, lw=1.0, ls=":", zorder=1)
    ax.fill_between(taus, lo, hi, color=C_CURVE, alpha=.13, lw=0, zorder=2)
    ax.plot(taus, m, "-o", color=C_CURVE, lw=1.3, ms=4, zorder=3)

    j = int(np.argmin(m))
    ax.plot([taus[j]], [m[j]], "o", ms=8, mfc="none", mec=C_CURVE, mew=1.4, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel(r"policy-path time constant $\tau$  (s)")
    ax.set_ylabel(label)
    ax.grid(True, which="major", ls="-")
    ax.set_title(f"({letter})", loc="left", fontweight="bold")
    return raw, lat


fig, axes = plt.subplots(1, 3, figsize=(FULLW, 2.35))

raw_l, lat_l = panel(axes[0], "mean_abs_lateral_error_m", "lateral error  (mm)", 1e3, "a")
raw_c, lat_c = panel(axes[1], "mean_abs_cut_height_error_m", "cut-height error  (mm)", 1e3, "b")

axes[0].annotate("unfiltered", xy=(0.045, raw_l), fontsize=7.2, color=C_RAW,
                 va="bottom")
axes[0].annotate("latched", xy=(0.045, lat_l), fontsize=7.2, color=C_LATCH,
                 va="bottom")

# (c) the paired contrast that resolves: tau 0.796 against tau 0.032, cut height
a = vals("bw_lpf_0p2", "mean_abs_cut_height_error_m") * 1e3
b = vals("bw_lpf_5p0", "mean_abs_cut_height_error_m") * 1e3
d = np.sort(a - b)
ax = axes[2]
col = [C_CURVE if v < 0 else C_RAW for v in d]
ax.bar(np.arange(len(d)), d, color=col, width=.86, lw=0)
ax.axhline(0, color="k", lw=.6)
ax.axhline(np.median(d), color="#333333", lw=1.0, ls="--")
ax.set_xlabel("mesh $\\times$ family cell, sorted")
ax.set_ylabel(r"$\Delta$ cut height  (mm)")
ax.set_title("(c)", loc="left", fontweight="bold")
ax.grid(True, axis="y", ls="-")
ax.set_xticks([])

fig.tight_layout(pad=.35, w_pad=1.3)
fig.savefig(OUT / "ch7_e6_bandwidth.pdf", bbox_inches="tight")
print("wrote", OUT / "ch7_e6_bandwidth.pdf")
print(f"  lateral: raw {raw_l:.1f} mm, best {min(vals(a_,'mean_abs_lateral_error_m').mean()*1e3 for a_,_ in LPF):.1f} mm")
print(f"  cut:     raw {raw_c:.1f} mm, best {min(vals(a_,'mean_abs_cut_height_error_m').mean()*1e3 for a_,_ in LPF):.1f} mm")
print(f"  panel c: median {np.median(d):+.3f} mm, {int((d<0).sum())}/{len(d)} cells improved")
