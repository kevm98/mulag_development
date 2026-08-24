#!/usr/bin/env python3
"""Stage 0 instrumentation check: did the authority factor reach the motion?

The point of a positive control is that it is verified numerically rather than
inferred from a plot.  ``actuator_authority_alpha`` scales the realised cylinder
motion just before it updates the PhysX target, and the trace records both
sides of that multiplication, so the check is exact:

    mean |delta_applied| / mean |delta_raw|  ==  alpha

If that ratio is 1.0 in the alpha=0.01 arm, the factor is not being applied and
nothing downstream is interpretable.  If it is 0.01 and the cutter still tracks
the moving reference, the fault is further along -- the target is being reached
by some other path -- and that is equally a harness failure.  The two are
different diagnoses, which is why both are printed.

Usage::

    python experiments/E5/code/check_authority_applied.py \
        --trace-dir outputs/e5_authority_stage0/traces --mesh 1337 \
        --report-dir outputs/e5_authority_stage0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

TOLERANCE = 0.02          # relative, on the applied/raw ratio


def ratio_from_trace(path: Path) -> tuple[float, float] | None:
    """Return (measured ratio, mean |raw|) or None when the fields are absent."""
    with np.load(path, allow_pickle=False) as data:
        if "actuator_delta_raw" not in data.files:
            return None
        raw = np.abs(np.asarray(data["actuator_delta_raw"], dtype=np.float64))
        applied = np.abs(np.asarray(data["actuator_delta_applied"], dtype=np.float64))
    # Steps where nothing was commanded carry no information about the ratio.
    moving = raw > 1e-9
    if not moving.any():
        return 0.0, 0.0
    return float(applied[moving].sum() / raw[moving].sum()), float(raw[moving].mean())


def lateral_error(report: Path, mesh: str) -> float | None:
    if not report.is_file():
        return None
    block = json.loads(report.read_text()).get("per_mesh", {}).get(mesh)
    if not block:
        return None
    return block.get("worst_family_mean_abs_lateral_error_m")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--report-dir", default=None)
    a = p.parse_args()

    traces = sorted(Path(a.trace_dir).expanduser().resolve().glob(f"*_m{a.mesh}.npz"))
    if not traces:
        raise SystemExit(f"no traces in {a.trace_dir}")

    rows, failures = [], []
    for path in traces:
        # Anchored on the LAST _a<number>_m, since a stack name may itself
        # contain "_a" -- ppo_aam_id_fd does.
        match = re.match(r"(?P<stack>.+)_a(?P<alpha>[0-9.]+)_m", path.name)
        if not match:
            continue
        alpha = float(match.group("alpha"))
        stack = match.group("stack")
        result = ratio_from_trace(path)
        if result is None:
            failures.append(f"{path.name}: trace has no actuator_delta_raw field; "
                            "the runtime did not record it")
            continue
        measured, raw_mean = result
        error = measured - alpha
        ok = abs(error) <= TOLERANCE * max(alpha, 1e-6)
        lat = None
        if a.report_dir:
            lat = lateral_error(
                Path(a.report_dir)
                / f"{stack}_a{match.group('alpha')}_m{a.mesh}.json", a.mesh)
        rows.append((stack, alpha, measured, raw_mean, lat, ok))
        if not ok:
            failures.append(f"{stack} @ alpha={alpha}: applied/raw = {measured:.4f}, "
                            f"expected {alpha:.4f}")

    head = (f"  {'stack':>16} {'alpha':>7} {'applied/raw':>12} "
            f"{'mean|raw| (m)':>14} {'worst lateral':>14}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for stack, alpha, measured, raw_mean, lat, ok in rows:
        mark = "" if ok else "   <-- MISMATCH"
        lat_text = f"{lat:.4f}" if lat is not None else "n/a"
        print(f"  {stack:>16} {alpha:>7.2f} {measured:>12.4f} "
              f"{raw_mean:>14.6f} {lat_text:>14}{mark}")

    # The second half of the control: at alpha ~ 0 the reference must be lost.
    print()
    by_stack: dict[str, dict[float, float]] = {}
    for stack, alpha, _, _, lat, _ in rows:
        if lat is not None:
            by_stack.setdefault(stack, {})[alpha] = lat
    for stack, points in sorted(by_stack.items()):
        if len(points) < 2:
            continue
        nominal = points.get(max(points))
        knocked = points.get(min(points))
        if nominal and knocked:
            growth = knocked / nominal if nominal > 0 else float("inf")
            verdict = ("tracking collapses as required" if growth >= 3.0
                       else "STILL TRACKING -- harness fault, investigate before "
                            "interpreting any sweep")
            print(f"  {stack:>16}: lateral error x{growth:>6.1f} "
                  f"from alpha={max(points):.2f} to {min(points):.2f}  -- {verdict}")

    if failures:
        print("\nFAILED:")
        for line in failures:
            print(f"  {line}")
        raise SystemExit(1)
    print("\nAll arms applied the authority factor as configured.")


if __name__ == "__main__":
    main()
