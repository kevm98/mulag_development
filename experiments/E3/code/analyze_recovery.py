#!/usr/bin/env python3
"""Time-domain recovery metrics for the mid-episode plant-change experiment.

Whole-episode means cannot show adaptation.  If the plant changes at ``t_c``
and one arm recovers in two seconds while another never does, an episode mean
over 23 s dilutes that difference into a few per cent -- which is exactly what
the superseded E3 result reported (online 0.0796 m against frozen 0.0787 m).
The claim "the online estimator recovered within 1.7 s while the frozen latent
stayed outside the pre-change band" needs the transient itself.

This reads the per-step npz written by ``--adaptation_trace_path`` and reports,
per arm:

    IAE over [t_c, t_c+1], [t_c+1, t_c+3], [t_c+3, t_c+5]
    peak error within 3 s of the change
    recovery time: first t with E(t) <= tolerance * E_pre, sustained
    pre-change and post-change baselines

Recovery time is None when the band is never re-entered; that is a result, not
a missing value, and it is reported as ``not recovered``.

The traces are paired env-for-env across arms, so the per-env differences are
paired and a sign count over environments is reported alongside the mean.

Usage::

    python experiments/E3/code/analyze_recovery.py \
        --trace online=outputs/e3/online_m1337.npz \
        --trace frozen=outputs/e3/frozen_m1337.npz \
        --trace oracle=outputs/e3/oracle_m1337.npz \
        --trace mean=outputs/e3/mean_m1337.npz \
        --trace sham=outputs/e3/sham_m1337.npz \
        --change-time 9.0 --json experiments/E3/results/recovery.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONTROL_HZ_DEFAULT = 60.0


def load_trace(path: Path, control_hz: float) -> dict:
    """Return per-step arrays as (steps, envs), plus a time axis in seconds."""
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        need = {"lateral_error", "cut_height_error"}
        missing = need - keys
        if missing:
            raise SystemExit(f"{path}: trace lacks {sorted(missing)}")
        out = {name: np.asarray(data[name], dtype=np.float64)
               for name in ("lateral_error", "cut_height_error")}
        for optional in ("z_hat", "z_hat_raw", "theta_alpha", "theta_stepped",
                         "cutter_z", "valve_applied"):
            if optional in keys:
                out[optional] = np.asarray(data[optional], dtype=np.float64)
        steps = out["lateral_error"].shape[0]
        # theta_elapsed_s is the env's own clock and is preferred when present,
        # because a trace can start after the episode does.
        if "theta_elapsed_s" in keys:
            elapsed = np.asarray(data["theta_elapsed_s"], dtype=np.float64)
            out["time_s"] = elapsed.reshape(steps, -1)[:, 0]
        else:
            out["time_s"] = np.arange(steps, dtype=np.float64) / float(control_hz)
    for name, arr in out.items():
        if name != "time_s" and arr.ndim == 3:
            # (steps, envs, width) -> collapse a multi-column channel by norm
            out[name] = np.linalg.norm(arr, axis=2) if arr.shape[2] > 1 else arr[:, :, 0]
    return out


def window_iae(time_s: np.ndarray, error: np.ndarray,
               start: float, stop: float) -> np.ndarray:
    """Integral of |e| over [start, stop) per environment, in metre-seconds."""
    mask = (time_s >= start) & (time_s < stop)
    if not mask.any():
        return np.full(error.shape[1], np.nan)
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.0
    return np.abs(error[mask]).sum(axis=0) * dt


def recovery_time(time_s: np.ndarray, error: np.ndarray, change_time: float,
                  pre_level: np.ndarray, tolerance: float,
                  sustain_s: float) -> np.ndarray:
    """First time after the change at which |e| stays inside the band.

    ``sustain_s`` guards against a single sample dipping into the band during
    an oscillation; the band must hold for that long before recovery counts.
    """
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.0
    sustain_steps = max(int(round(sustain_s / dt)), 1) if dt > 0 else 1
    post = time_s >= change_time
    idx = np.flatnonzero(post)
    out = np.full(error.shape[1], np.nan)
    band = tolerance * pre_level
    for env in range(error.shape[1]):
        inside = np.abs(error[idx, env]) <= band[env]
        if not inside.any():
            continue
        # first index where `sustain_steps` consecutive samples are inside
        run = 0
        for k, ok in enumerate(inside):
            run = run + 1 if ok else 0
            if run >= sustain_steps:
                out[env] = time_s[idx[k - sustain_steps + 1]] - change_time
                break
    return out


def summarise(name: str, trace: dict, a) -> dict:
    time_s = trace["time_s"]
    metrics: dict[str, dict] = {}
    for channel in ("lateral_error", "cut_height_error"):
        error = trace[channel]
        pre = np.abs(error[(time_s >= a.change_time - a.pre_window_s)
                           & (time_s < a.change_time)]).mean(axis=0)
        post = np.abs(error[(time_s >= a.change_time)
                            & (time_s < a.change_time + a.post_window_s)]).mean(axis=0)
        peak_mask = (time_s >= a.change_time) & (time_s < a.change_time + 3.0)
        peak = np.abs(error[peak_mask]).max(axis=0) if peak_mask.any() else np.full_like(pre, np.nan)
        rec = recovery_time(time_s, error, a.change_time, pre, a.tolerance, a.sustain_s)
        metrics[channel] = {
            "pre_mean_m": pre.tolist(),
            "post_mean_m": post.tolist(),
            "post_over_pre": (post / np.where(pre > 0, pre, np.nan)).tolist(),
            "peak_3s_m": peak.tolist(),
            "iae_0_1s": window_iae(time_s, error, a.change_time, a.change_time + 1).tolist(),
            "iae_1_3s": window_iae(time_s, error, a.change_time + 1, a.change_time + 3).tolist(),
            "iae_3_5s": window_iae(time_s, error, a.change_time + 3, a.change_time + 5).tolist(),
            "recovery_time_s": [None if np.isnan(v) else float(v) for v in rec],
            "recovered_envs": int(np.count_nonzero(~np.isnan(rec))),
            "envs": int(error.shape[1]),
        }
    if "z_hat" in trace:
        z = trace["z_hat"]
        pre_z = z[(time_s >= a.change_time - a.pre_window_s) & (time_s < a.change_time)]
        post_z = z[(time_s >= a.change_time) & (time_s < a.change_time + a.post_window_s)]
        metrics["latent"] = {
            "shift_across_change": float(np.abs(post_z.mean(axis=0) - pre_z.mean(axis=0)).mean()),
            "pre_std": float(pre_z.std(axis=0).mean()),
        }
    return {"arm": name, "steps": int(time_s.size), "metrics": metrics}


def paired_report(rows: list[dict], reference: str, channel: str) -> list[str]:
    """Per-arm comparison against one reference arm, paired over environments."""
    by_name = {row["arm"]: row for row in rows}
    if reference not in by_name:
        return [f"(reference arm {reference!r} not among the traces)"]
    ref = np.asarray(by_name[reference]["metrics"][channel]["iae_1_3s"], dtype=float)
    lines = [f"  paired IAE(1-3 s) against {reference}, {channel}:"]
    for row in rows:
        if row["arm"] == reference:
            continue
        arm = np.asarray(row["metrics"][channel]["iae_1_3s"], dtype=float)
        if arm.shape != ref.shape:
            lines.append(f"    {row['arm']:>10}: env counts differ, not paired")
            continue
        delta = arm - ref
        better = int(np.count_nonzero(delta < 0))
        pct = 100.0 * float(np.nanmean(delta)) / float(np.nanmean(ref)) if np.nanmean(ref) else float("nan")
        lines.append(f"    {row['arm']:>10}: {pct:+6.1f}%  better in {better}/{delta.size} envs")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", action="append", required=True, metavar="NAME=PATH",
                   help="one arm's trace; repeat for each arm")
    p.add_argument("--change-time", type=float, required=True, dest="change_time",
                   help="episode time of the plant change, in seconds")
    p.add_argument("--pre-window-s", type=float, default=3.0, dest="pre_window_s")
    p.add_argument("--post-window-s", type=float, default=5.0, dest="post_window_s")
    p.add_argument("--tolerance", type=float, default=1.1,
                   help="recovery band as a multiple of the pre-change level")
    p.add_argument("--sustain-s", type=float, default=0.5, dest="sustain_s",
                   help="how long the band must hold before recovery counts")
    p.add_argument("--control-hz", type=float, default=CONTROL_HZ_DEFAULT,
                   dest="control_hz")
    p.add_argument("--reference-arm", default="frozen", dest="reference_arm")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    rows = []
    for spec in a.trace:
        name, separator, path = spec.partition("=")
        if not separator:
            raise SystemExit(f"--trace expects NAME=PATH, got {spec!r}")
        rows.append(summarise(name, load_trace(Path(path).expanduser().resolve(),
                                               a.control_hz), a))

    print(f"plant change at t = {a.change_time:.2f} s; recovery band = "
          f"{a.tolerance:.2f} x pre-change level, sustained {a.sustain_s:.2f} s\n")
    for channel in ("lateral_error", "cut_height_error"):
        print(f"{channel}:")
        head = (f"  {'arm':>10} {'pre':>9} {'post':>9} {'post/pre':>9} "
                f"{'peak 3s':>9} {'IAE 0-1':>9} {'IAE 1-3':>9} {'recovery':>12}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for row in rows:
            m = row["metrics"][channel]
            rec = [v for v in m["recovery_time_s"] if v is not None]
            rec_text = (f"{np.median(rec):.2f} s ({m['recovered_envs']}/{m['envs']})"
                        if rec else f"not recovered (0/{m['envs']})")
            print(f"  {row['arm']:>10} "
                  f"{np.mean(m['pre_mean_m']):>9.4f} "
                  f"{np.mean(m['post_mean_m']):>9.4f} "
                  f"{np.nanmean(m['post_over_pre']):>9.2f} "
                  f"{np.mean(m['peak_3s_m']):>9.4f} "
                  f"{np.nanmean(m['iae_0_1s']):>9.4f} "
                  f"{np.nanmean(m['iae_1_3s']):>9.4f} "
                  f"{rec_text:>12}")
        for line in paired_report(rows, a.reference_arm, channel):
            print(line)
        print()

    print("A post/pre ratio near 1.0 on every arm means the perturbation did not")
    print("create recoverable error, and no comparison below it is informative.")
    print("Size the perturbation from the Stage A degradation curve (E5) first:")
    print("the useful range is roughly 1.5x to 2.5x the pre-change error.")

    if a.json:
        out = Path(a.json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"change_time_s": a.change_time,
                                   "tolerance": a.tolerance,
                                   "sustain_s": a.sustain_s,
                                   "arms": rows}, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
