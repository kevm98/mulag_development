#!/usr/bin/env python3
"""Stage 0 instrumentation check: did the requested stress reach the motion?

Generalises `check_authority_applied.py` from the single authority ratio to the
whole displacement-domain chain. The point of a positive control is that it is
verified numerically rather than inferred from a plot, and the trace records
both sides of the boundary -- `actuator_delta_raw` before the chain and
`actuator_delta_applied` after it -- so the check can be exact.

Method: **replay**. The requested stress is applied to the recorded raw signal
with the same chain the environment used, and the result is compared to the
recorded applied signal. A match to tolerance proves the plant received exactly
the stress that was asked for. Nothing is inferred from the cutter.

Two independent estimates are also reported, so a mismatch can be diagnosed
rather than merely flagged:

  * **delay**, from the cross-correlation peak between raw and applied. The
    first-order lag contributes its own group delay, so this measures TOTAL
    transport lag, not the FIFO depth -- with tau = 0.10 s it reads about one
    step above the requested delay. It is therefore compared against the same
    estimator run on the replayed signal, not against the requested FIFO value:
    both carry the identical lag phase, so any disagreement is a real
    misalignment. Against a bare requested value this check false-flags every
    run that also carries lag.
  * **deadband**, from the fraction of applied steps that are exactly zero while
    the raw step was not.

Failure modes are distinct and both matter. If replay matches but the cutter
still tracks the reference, the stress arrived and the fault is further along.
If replay does not match, nothing downstream is interpretable.

Usage::

    python3.11 experiments/E5/code/check_stress_applied.py \
        --trace-dir outputs/e5_stress_stage0/traces \
        --stress "delay=3,lag=0.10,deadband=0.0005" \
        --report-dir outputs/e5_stress_stage0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mulag_plant.runtime.actuator_stress import ActuatorStress, ActuatorStressChain  # noqa: E402

TOLERANCE = 0.02          # relative, on the replay residual
DT_S = 0.05


def parse_stress(spec: str) -> ActuatorStress:
    """Parse `delay=3,lag=0.10,deadband=0.0005,alpha=0.8,rate=0.001`."""
    fields: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, value = part.partition("=")
        fields[name.strip()] = float(value)
    return ActuatorStress(
        authority=fields.get("alpha", 1.0),
        lag_tau_s=fields.get("lag", 0.0),
        delay_steps=int(fields.get("delay", 0)),
        deadband_m=fields.get("deadband", 0.0),
        rate_limit_m=fields.get("rate", float("inf")),
    )


def replay(raw: np.ndarray, stress: ActuatorStress, *, max_delay: int) -> np.ndarray:
    """Push the recorded raw increments through the same chain the env used."""
    steps, envs, cyl = raw.shape
    chain = ActuatorStressChain(
        stress, dt_s=DT_S, num_envs=envs, num_cylinders=cyl,
        max_delay_steps=max_delay, device="cpu",
    )
    return np.stack([
        chain.step(torch.from_numpy(raw[t].astype(np.float32))).numpy()
        for t in range(steps)
    ])


def estimate_delay(raw: np.ndarray, applied: np.ndarray, max_delay: int) -> int:
    """Delay that best aligns applied with raw, independent of the request."""
    a = applied.reshape(applied.shape[0], -1)
    r = raw.reshape(raw.shape[0], -1)
    best, best_score = 0, -np.inf
    for k in range(max_delay + 1):
        if k >= len(r):
            break
        shifted, target = r[: len(r) - k], a[k:]
        if not np.any(shifted) or not np.any(target):
            continue
        score = float((shifted * target).sum() /
                      (np.linalg.norm(shifted) * np.linalg.norm(target) + 1e-30))
        if score > best_score:
            best, best_score = k, score
    return best


def check_trace(path: Path, stress: ActuatorStress, max_delay: int) -> dict | None:
    with np.load(path, allow_pickle=False) as data:
        if "actuator_delta_raw" not in data.files:
            return None
        raw = np.asarray(data["actuator_delta_raw"], dtype=np.float64)
        applied = np.asarray(data["actuator_delta_applied"], dtype=np.float64)
    if raw.ndim == 2:                      # [steps, cyl] -> [steps, 1, cyl]
        raw, applied = raw[:, None, :], applied[:, None, :]

    expected = replay(raw, stress, max_delay=max_delay).astype(np.float64)
    # Both signals carry the same lag phase, so comparing their estimates
    # isolates the FIFO alignment.
    delay_recorded = estimate_delay(raw, applied, max_delay)
    delay_expected = estimate_delay(raw, expected, max_delay)
    scale = np.abs(raw).max()
    residual = np.abs(expected - applied).max() / (scale + 1e-30)

    moving = np.abs(raw) > 1e-9
    zeroed = float((moving & (np.abs(applied) < 1e-12)).sum() / max(moving.sum(), 1))
    ratio = (float(np.abs(applied)[moving].sum() / np.abs(raw)[moving].sum())
             if moving.any() else 0.0)
    return {
        "trace": path.name,
        "requested": stress.as_dict(),
        "replay_residual": residual,
        "replay_matches": bool(residual <= TOLERANCE),
        "delay_estimated": delay_recorded,
        "delay_expected": delay_expected,
        "delay_requested": int(stress.delay_steps),
        "applied_over_raw": ratio,
        "zeroed_fraction": zeroed,
        "mean_abs_raw": float(np.abs(raw)[moving].mean()) if moving.any() else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--stress", required=True, help="delay=3,lag=0.10,deadband=0.0005")
    p.add_argument("--max-delay", type=int, default=8)
    p.add_argument("--report-dir", default=None)
    args = p.parse_args()

    stress = parse_stress(args.stress)
    traces = sorted(Path(args.trace_dir).glob("*.npz"))
    if not traces:
        print(f"no traces in {args.trace_dir}")
        raise SystemExit(2)

    print(f"requested stress: {stress.label()}")
    print(f"{'trace':<34}{'residual':>10}{'delay exp/est':>15}{'appl/raw':>10}"
          f"{'zeroed':>9}  verdict")
    rows, failures = [], 0
    for path in traces:
        row = check_trace(path, stress, args.max_delay)
        if row is None:
            print(f"{path.name:<34}{'-':>10}{'-':>15}{'-':>10}{'-':>9}  NO FIELDS")
            failures += 1
            continue
        delay_ok = row["delay_estimated"] == row["delay_expected"]
        ok = row["replay_matches"] and delay_ok
        failures += int(not ok)
        rows.append(row)
        verdict = "OK" if ok else ("DELAY MISMATCH" if row["replay_matches"] else "NOT DELIVERED")
        print(f"{row['trace']:<34}{row['replay_residual']:>10.2e}"
              f"{row['delay_expected']:>8d}/{row['delay_estimated']:<6d}"
              f"{row['applied_over_raw']:>10.4f}{row['zeroed_fraction']*100:>8.1f}%  {verdict}")

    print(f"\nreplay residual = max|replayed - recorded| / max|raw|, tolerance {TOLERANCE}")
    print("delay exp/est = cross-correlation peak on the replayed / recorded signal;")
    print("               both carry the same lag phase, so a difference is a real misalignment")
    if args.report_dir:
        out = Path(args.report_dir) / "stress_applied.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"requested": stress.as_dict(), "tolerance": TOLERANCE, "traces": rows}, indent=2) + "\n")
        print(f"-> {out}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
