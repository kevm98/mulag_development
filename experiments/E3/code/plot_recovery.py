#!/usr/bin/env python3
"""The mid-episode plant-change time history: the figure the claim rests on.

Four panels on one time axis, with the intervention marked:

    1  lateral tracking error
    2  cut-height error
    3  realised actuator motion (applied / raw), i.e. the perturbation itself
    4  the latent the controller is reading, z_hat

Read top to bottom it is the whole causal chain -- plant changes, response
deteriorates, estimator moves, controller changes, recovery occurs or does not.
An episode mean cannot show any of that, which is why the superseded E3 result
reported online 0.0796 m against frozen 0.0787 m and concluded nothing.

Panel 3 is what makes the figure honest: if the applied/raw trace does not step
at the intervention time, the perturbation never reached the plant and the
other three panels are describing something else.

Usage::

    python experiments/E3/code/plot_recovery.py --trace-dir outputs/e3/traces \\
        --mesh 1337 --change-time 9.0 --out figures/ch7_e3_recovery.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, matching the rest of the chapter's figures.
COLOURS = {"online": "#0072B2", "frozen": "#D55E00", "oracle": "#009E73",
           "mean": "#CC79A7", "shuffled": "#999999", "sham": "#000000",
           "if_ref": "#E69F00"}
ORDER = ("online", "frozen", "oracle", "mean", "shuffled", "sham", "if_ref")
LABELS = {"online": r"online $\hat{z}(t)$", "frozen": "frozen latent",
          "oracle": "oracle $z^\\star$", "mean": "mean latent",
          "shuffled": "shuffled latent", "sham": "sham (no plant change)",
          "if_ref": "IF reference"}


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    out = {}
    with np.load(path, allow_pickle=False) as data:
        for name in ("lateral_error", "cut_height_error", "z_hat",
                     "actuator_delta_raw", "actuator_delta_applied",
                     "theta_elapsed_s", "theta_mass_scale"):
            if name in data.files:
                out[name] = np.asarray(data[name], dtype=np.float64)
    if "lateral_error" not in out:
        return None
    steps = out["lateral_error"].shape[0]
    out["time_s"] = (out["theta_elapsed_s"].reshape(steps, -1)[:, 0]
                     if "theta_elapsed_s" in out
                     else np.arange(steps) / 60.0)
    return out


def across_envs(arr: np.ndarray) -> np.ndarray:
    """Median over environments; collapse a multi-column channel by norm."""
    if arr.ndim == 3:
        arr = np.linalg.norm(arr, axis=2) if arr.shape[2] > 1 else arr[:, :, 0]
    return np.median(arr, axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--change-time", type=float, required=True, dest="change_time")
    p.add_argument("--out", required=True)
    p.add_argument("--window-s", type=float, default=8.0, dest="window_s",
                   help="seconds shown either side of the change")
    a = p.parse_args()

    root = Path(a.trace_dir).expanduser().resolve()
    arms = {}
    for name in ORDER:
        for stem in (f"aif_{name}_m{a.mesh}.npz", f"{name}_m{a.mesh}.npz"):
            trace = load(root / stem)
            if trace is not None:
                arms[name] = trace
                break
    if not arms:
        raise SystemExit(f"no traces found in {root}")

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 1.0]})
    lo, hi = a.change_time - a.window_s, a.change_time + a.window_s

    for name, trace in arms.items():
        t = trace["time_s"]
        keep = (t >= lo) & (t <= hi)
        style = {"color": COLOURS.get(name, "#444444"), "lw": 1.3,
                 "label": LABELS.get(name, name)}
        if name == "sham":
            style |= {"ls": (0, (4, 2)), "lw": 1.0}
        if name == "if_ref":
            style |= {"ls": (0, (1, 1.5)), "lw": 1.0}
        axes[0].plot(t[keep], across_envs(trace["lateral_error"])[keep], **style)
        axes[1].plot(t[keep], across_envs(trace["cut_height_error"])[keep], **style)
        if "actuator_delta_applied" in trace and "actuator_delta_raw" in trace:
            raw = np.abs(trace["actuator_delta_raw"])
            applied = np.abs(trace["actuator_delta_applied"])
            if raw.ndim == 3:
                raw, applied = raw.sum(axis=2), applied.sum(axis=2)
            ratio = np.divide(applied, raw, out=np.ones_like(applied), where=raw > 1e-9)
            axes[2].plot(t[keep], np.median(ratio, axis=1)[keep], **style)
        elif "theta_mass_scale" in trace:
            axes[2].plot(t[keep], across_envs(trace["theta_mass_scale"])[keep], **style)
        if "z_hat" in trace:
            z = trace["z_hat"]
            if z.ndim == 3 and z.shape[2] > 1:
                axes[3].plot(t[keep], np.median(z[:, :, 0], axis=1)[keep], **style)
                axes[3].plot(t[keep], np.median(z[:, :, 1], axis=1)[keep],
                             **(style | {"ls": (0, (2, 2)), "label": None}))
            else:
                axes[3].plot(t[keep], across_envs(z)[keep], **style)

    for ax in axes:
        ax.axvline(a.change_time, color="#666666", lw=1.0, ls="--", zorder=0)
        ax.grid(alpha=0.25, lw=0.5)
        ax.margins(x=0)
    axes[0].set_ylabel("lateral error (m)")
    axes[1].set_ylabel("cut-height error (m)")
    axes[2].set_ylabel("realised / commanded\nactuator motion")
    axes[3].set_ylabel(r"$\hat{z}_1,\ \hat{z}_2$")
    axes[3].set_xlabel("episode time (s)")
    axes[0].annotate("plant change", xy=(a.change_time, 1.02),
                     xycoords=("data", "axes fraction"), ha="center",
                     fontsize=8, color="#666666")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=7.5, ncol=2, frameon=False,
                   loc="upper left")
    fig.suptitle("", y=0.995)
    fig.tight_layout()

    out = Path(a.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  ({len(arms)} arms: {', '.join(sorted(arms))})")
    if "sham" not in arms:
        print("WARNING: no sham arm in the figure; a post-change rise cannot be "
              "distinguished from position along the path")


if __name__ == "__main__":
    main()
