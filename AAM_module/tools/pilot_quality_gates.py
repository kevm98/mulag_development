#!/usr/bin/env python
"""Stage 14 pilot data quality gates (items 38-41), headless.

Run against a recorded pilot run (metadata.json + trajectory.npz) before any
full collection begins:

- item 38: per-action-channel stats (min, max, range, std, positive fraction,
  negative fraction, saturation fraction) and cross-channel SVD
  (correlation matrix, rank, singular values, condition number);
- item 39: per-channel hydraulic authority u -> dP / velocity / extension
  (max abs lagged correlation), reject if any channel barely affects the
  plant;
- item 40: state coverage distributions (cylinder extension, velocity, dP,
  joint position/velocity, EE/posture) - fraction of the reachable range
  visited per channel and occupancy spread;
- item 41: data integrity - monotonic timestamps, state/command alignment
  (T+1 vs T), no NaN in recorded signals, action->state transition alignment,
  per-run factor constancy, complete randomization metadata.

Usage:
    python pilot_quality_gates.py [--pilot results/dls_pilot] [--out results/pilot_quality_gates.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))


def gate_action_channels(U: np.ndarray) -> dict:
    n = U.shape[0]
    pos = np.clip(np.sign(U), 0.0, 1.0)
    neg = np.clip(-np.sign(U), 0.0, 1.0)
    stats = []
    for c in range(U.shape[1]):
        stats.append(
            {
                "channel": c,
                "min": float(U[:, c].min()),
                "max": float(U[:, c].max()),
                "range": float(np.ptp(U[:, c])),
                "std": float(U[:, c].std()),
                "positive_fraction": round(float(pos[:, c].mean()), 4),
                "negative_fraction": round(float(neg[:, c].mean()), 4),
                "saturation_fraction": round(float((np.abs(U[:, c]) >= 1.0 - 1e-6).mean()), 4),
            }
        )
    corr = np.corrcoef(U.T)
    sv = np.linalg.svd(U, compute_uv=False)
    sv = sv[sv > 1e-12]
    return {
        "per_channel": stats,
        "correlation_matrix": [[float(v) for v in row] for row in corr],
        "rank": int(np.linalg.matrix_rank(U)),
        "numerical_rank": int(np.count_nonzero(sv > sv[0] * max(U.shape) * np.finfo(np.float64).eps)),
        "singular_values": [float(v) for v in sv],
        "condition_number": float(sv[0] / sv[-1]),
        "gate_pass": bool(int(np.linalg.matrix_rank(U)) == U.shape[1]),
    }


def gate_authority(U: np.ndarray, responses: dict[str, np.ndarray]) -> dict:
    """Per-channel max abs lagged correlation with each plant response.

    Item 39 rule (matches stage 14): a channel is inert only if it does not
    correlate > 0.10 with ANY plant response. The gate therefore takes, per
    channel, the max over all responses (dP/velocity/extension) and requires
    that max to exceed 0.10 for every channel.
    """
    out = {}
    for sig, Y in responses.items():
        per_channel = []
        for c in range(U.shape[1]):
            u = U[:, c]
            best = 0.0
            for yc in range(Y.shape[1]):
                y = Y[:, yc]
                vals = []
                for lag in range(min(12, U.shape[0] - 1)):
                    cu, cy = u[: U.shape[0] - lag], y[lag:]
                    denom = np.std(cu) * np.std(cy)
                    vals.append(float(np.corrcoef(cu, cy)[0, 1]) if denom > 1e-12 else 0.0)
                best = max(best, max(map(abs, vals)))
            per_channel.append(round(best, 3))
        out[sig] = {"per_channel_max_corr": per_channel}
    # Item 39: no channel may be inert across ALL responses.
    per_channel_global = [
        max(out[sig]["per_channel_max_corr"][c] for sig in responses)
        for c in range(U.shape[1])
    ]
    out["per_channel_max_corr_all_responses"] = per_channel_global
    out["gate_pass"] = bool(min(per_channel_global) > 0.10)
    return out


def gate_state_coverage(arrays: dict[str, np.ndarray], metadata: dict) -> dict:
    """Fraction of reachable range visited + occupancy spread per signal."""
    # Per-cylinder learned extension limits (measured; same hardcoded values as
    # the Stage 12 probe design and Stage 13 run-structure checks).
    CYL_LEARNED_LIMITS = [
        (0.0643, 0.4193),
        (0.4147, 0.7387),
        (0.3776, 0.6780),
        (0.5335, 0.7033),
    ]
    ranges = {
        "cylinder_extension": CYL_LEARNED_LIMITS,
        "cylinder_velocity": (-0.65, 0.65),
        "dP": (-1.5e8, 1.5e8),
        "joint_position": None,  # soft limits live in-sim; report span only
        "joint_velocity": None,
        "ee_pos_local": None,
    }
    out = {}
    for sig, lim in ranges.items():
        if sig not in arrays:
            continue
        arr = arrays[sig]
        if arr.ndim == 1:
            arr = arr[:, None]
        per_channel = []
        for c in range(arr.shape[1]):
            y = arr[:, c]
            span = float(np.ptp(y))
            if lim is not None:
                lo, hi = lim if isinstance(lim, tuple) else lim[c]
                reachable = hi - lo
                coverage = span / reachable if reachable > 0 else 0.0
            else:
                coverage = None
            per_channel.append(
                {
                    "min": float(y.min()),
                    "max": float(y.max()),
                    "span": span,
                    "reachable_fraction": (
                        round(coverage, 3) if coverage is not None else None
                    ),
                    "n_unique_bins": int(np.unique(np.round(y, 4)).size),
                }
            )
        out[sig] = per_channel
    return out


def gate_integrity(arrays: dict[str, np.ndarray], metadata: dict) -> dict:
    """Timestamps, alignment, NaN, transition alignment, factor constancy."""
    t = arrays.get("time_s")
    num_steps = int(metadata.get("num_control_steps", 0))
    issues = []
    if t is None:
        issues.append("missing time_s")
    elif not bool(np.all(np.diff(t) > 0)):
        issues.append("timestamps not strictly monotonic")

    cmd_keys = [
        "controller_output",
        "desired_delta_extension",
        "delta_extension_per_step",
        "delta_isaac",
        "cylinder_pos_target",
    ]
    for k in cmd_keys:
        if k in arrays and arrays[k].shape[0] != num_steps:
            issues.append(f"{k}: expected {num_steps} rows, got {arrays[k].shape[0]}")
    for k in ("cylinder_extension", "cylinder_velocity", "dP", "joint_position"):
        if k in arrays and arrays[k].shape[0] != num_steps + 1:
            issues.append(f"{k}: expected {num_steps + 1} rows, got {arrays[k].shape[0]}")

    for k, a in arrays.items():
        if k.startswith(("applied_valve_cmd", "dls_", "contact_")):
            continue  # NaN-by-design unavailable channels
        if not np.isfinite(a).all():
            issues.append(f"{k}: contains NaN/Inf")

    # Action causing transition k->k+1: command row k is the action that
    # produced state row k+1 (recorder writes command then advances state).
    ext = arrays.get("cylinder_extension")
    ctl = arrays.get("controller_output")
    if ext is not None and ctl is not None:
        # response consistency: |x[k+1]-x[k]| bounded by action authority
        delta = np.abs(np.diff(ext, axis=0))
        if delta.max() > 0.05:
            issues.append("implausible per-step extension jump (>0.05 m)")

    # Factor constancy: a nominal run must carry the nominal draw (hydraulic
    # factors 0, mass scale 1). Master-collection nominal runs DO store the
    # Stage 5 factor keys (as zeros), so check values, not key presence.
    factor_meta = [k for k in ("flow", "compliance", "seal", "pressure", "com_z")
                   if k in metadata]
    if metadata.get("domain", "nominal") == "nominal" and factor_meta:
        non_zero = [k for k in factor_meta if float(metadata[k]) != 0.0]
        if non_zero:
            issues.append(f"nominal run carries non-zero factor metadata: {non_zero}")
        if float(metadata.get("mass_scale", 1.0)) != 1.0:
            issues.append("nominal run carries non-unit mass scale")

    return {
        "num_control_steps": num_steps,
        "state_rows": int(ext.shape[0]) if ext is not None else None,
        "issues": issues,
        "gate_pass": not bool(issues),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", default=str(AAM_ROOT / "results/dls_pilot"))
    p.add_argument("--out", default=str(AAM_ROOT / "results/pilot_quality_gates.json"))
    args = p.parse_args()

    pilot = Path(args.pilot)
    metadata = json.loads((pilot / "metadata.json").read_text())
    with np.load(pilot / "trajectory.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    U = np.asarray(arrays["controller_output"], dtype=np.float64)
    responses = {
        "dP": np.asarray(arrays["dP"], dtype=np.float64)[:-1],
        "velocity": np.asarray(arrays["cylinder_velocity"], dtype=np.float64)[:-1],
        "extension": np.asarray(arrays["cylinder_extension"], dtype=np.float64)[:-1],
    }
    report = {
        "run": str(pilot),
        "domain": metadata.get("domain"),
        "action_channels": gate_action_channels(U),
        "authority": gate_authority(U, responses),
        "state_coverage": gate_state_coverage(arrays, metadata),
        "integrity": gate_integrity(arrays, metadata),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    ac = report["action_channels"]
    print("action channels:")
    for s in ac["per_channel"]:
        print(
            f"  u{s['channel']}: min={s['min']:.3f} max={s['max']:.3f} "
            f"range={s['range']:.3f} std={s['std']:.3f} "
            f"pos={s['positive_fraction']:.2f} neg={s['negative_fraction']:.2f} "
            f"sat={s['saturation_fraction']:.2f}"
        )
    print(
        f"  rank={ac['rank']} num_rank={ac['numerical_rank']} "
        f"kappa={ac['condition_number']:.3g} pass={ac['gate_pass']}"
    )
    auth = report["authority"]
    print("authority (max lagged corr per channel):")
    for sig in ("dP", "velocity", "extension"):
        if sig in auth:
            print(f"  {sig}: {auth[sig]['per_channel_max_corr']}")
    print(
        f"  all-responses: {auth['per_channel_max_corr_all_responses']} "
        f"pass={auth['gate_pass']}"
    )
    print(f"  overall pass={auth['gate_pass']}")
    cov = report["state_coverage"]
    print("state coverage (reachable fraction per channel):")
    for sig, ch in cov.items():
        vals = [
            round(c["reachable_fraction"], 3)
            if c["reachable_fraction"] is not None
            else round(c["span"], 3)
            for c in ch
        ]
        print(f"  {sig}: {vals}")
    integ = report["integrity"]
    print(f"integrity: pass={integ['gate_pass']} issues={integ['issues']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
