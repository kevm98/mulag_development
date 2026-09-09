"""Where does the task become infeasible for ANY controller with this action space?

CPU-only, no plant model. `action_separation.py` establishes that every stress
mechanism sits between the commanded and the applied valve, so the command a
plant-aware controller must issue to hold the nominal response is the preimage
`g_p^-1(u)`. When that preimage exceeds the ID's own `command_limits`, no policy
can deliver the nominal response -- not the oracle, not the AAM, not a perfect
controller. That is a *necessary* condition for breakdown and therefore a LOWER
BOUND on the stress at which the real policy fails.

This script locates the crossing that `action_separation.py` only brackets: it
bisects each axis for the severity at which the required-command saturation
reaches the pre-declared `SATURATION_MAX`, instead of reporting the sweep grid.

What this is and is not
-----------------------
It is  : the analytic infeasibility threshold. Beyond it the nominal response is
         unreachable, so the observed policy breaking point cannot be later.
It is not: the observed breaking point. A policy may fail well before this by
         choosing a worse response it *can* reach, and the task thresholds
         (median lateral error, ground contact, safety terminations) still need
         IsaacLab and PhysX. This bounds the search, it does not replace it.

Two thresholds are reported for every axis because the source screen and its own
docstring disagree, and neither is silently overridden here:

  absolute -- `clipped_fraction > SATURATION_MAX`, the rule `action_separation.py`
              froze before it saw any number, kept intact.
  excess   -- the same rule applied to clipping ABOVE the nominal floor, which is
              what that screen's docstring says to use when attributing cost to
              the stress. The recorded valve already clips 5.6% of rows on its
              own, so the two differ near the margin.

Usage:
    python3.11 experiments/RMA_PPO/robustness/analytic_breaking_point.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_separation import (  # noqa: E402
    N_SAMPLES, OUT, SAMPLE_SEED, SATURATION_MAX,
    load_samples, required_command,
)
from robustness_env import ID_CHECKPOINT, REPO  # noqa: E402

# Direction of increasing severity, and the bracket to search inside.
# `nominal` is the unstressed value; `extreme` is the far end of the sweep.
AXES = {
    "authority":  {"nominal": 1.0,  "extreme": 0.01, "continuous": True},
    "deadband":   {"nominal": 0.0,  "extreme": 0.50, "continuous": True},
    "lag":        {"nominal": 1e-4, "extreme": 2.00, "continuous": True},
    "rate_limit": {"nominal": 10.0, "extreme": 0.005, "continuous": True},
    "delay":      {"nominal": 0,    "extreme": 20,   "continuous": False},
}
BISECT_STEPS = 24
MONOTONICITY_GRID = 17


def saturation(axis: str, severity: float, sample, limits) -> dict[str, float]:
    """Fraction of rows whose required command is unreachable, overall and per cylinder."""
    u_req, ok, _ = required_command(axis, float(severity), sample)
    over = (u_req.abs() > limits) | (~ok)
    return {
        "overall": float(over.any(dim=-1).float().mean()),
        "per_cylinder": over.float().mean(dim=0).tolist(),
    }


def bisect(axis: str, target: float, sample, limits, nominal_sat: dict, *, excess: bool
           ) -> float | None:
    """Severity at which saturation first reaches `target`, or None if never."""
    spec = AXES[axis]
    lo, hi = spec["nominal"], spec["extreme"]          # lo = mild, hi = severe

    def value(sev: float) -> float:
        s = saturation(axis, sev, sample, limits)["overall"]
        return s - nominal_sat["overall"] if excess else s

    if value(hi) < target:
        return None                                    # never crosses in range
    if value(lo) >= target:
        return lo                                      # already over at nominal
    if not spec["continuous"]:
        for k in range(int(lo), int(hi) + 1):
            if value(k) >= target:
                return float(k)
        return None
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if value(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def monotone(axis: str, sample, limits) -> bool:
    """Bisection is only valid if saturation rises with severity; check it."""
    spec = AXES[axis]
    a, b = spec["nominal"], spec["extreme"]
    grid = [a + (b - a) * i / (MONOTONICITY_GRID - 1) for i in range(MONOTONICITY_GRID)]
    if not spec["continuous"]:
        grid = list(range(int(a), int(b) + 1))
    vals = [saturation(axis, s, sample, limits)["overall"] for s in grid]
    # allow small non-monotonic wobble from finite sampling
    return all(v2 >= v1 - 0.005 for v1, v2 in zip(vals, vals[1:]))


def main() -> int:
    t0 = time.time()
    torch.manual_seed(SAMPLE_SEED)
    sample = load_samples(N_SAMPLES, SAMPLE_SEED)
    n = sample["s"].shape[0]

    from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime
    limits = CylinderIDRuntime(
        models_root=REPO / "learned_models/id_fd_model/vendor_models", checkpoint=ID_CHECKPOINT,
        num_envs=1, device=torch.device("cpu"),
    ).command_limits.reshape(1, 4)

    nominal_sat = saturation("authority", 1.0, sample, limits)
    print(f"{n} recorded rows | saturation rule: {SATURATION_MAX:.2f}")
    print(f"nominal floor: {nominal_sat['overall']*100:.1f}% of rows, "
          f"per cylinder {[round(v*100, 1) for v in nominal_sat['per_cylinder']]}%")
    binding = max(range(4), key=lambda i: nominal_sat["per_cylinder"][i])
    print(f"binding cylinder at nominal: index {binding} "
          f"({nominal_sat['per_cylinder'][binding]*100:.1f}%)\n")

    report = {
        "screen": "analytic_breaking_point",
        "n_samples": n, "sample_seed": SAMPLE_SEED,
        "saturation_max": SATURATION_MAX,
        "nominal_saturation": nominal_sat,
        "binding_cylinder_index": binding,
        "bound_type": "necessary condition; observed breakdown cannot be later",
        "axes": {},
    }

    print(f"{'axis':<12}{'monotone':>10}{'BP absolute':>14}{'BP excess':>12}"
          f"{'sat at extreme':>16}")
    for axis in AXES:
        mono = monotone(axis, sample, limits)
        bp_abs = bisect(axis, SATURATION_MAX, sample, limits, nominal_sat, excess=False)
        bp_exc = bisect(axis, SATURATION_MAX, sample, limits, nominal_sat, excess=True)
        sat_ext = saturation(axis, AXES[axis]["extreme"], sample, limits)
        report["axes"][axis] = {
            "monotone": mono,
            "breaking_point_absolute": bp_abs,
            "breaking_point_excess": bp_exc,
            "saturation_at_extreme": sat_ext,
            "extreme": AXES[axis]["extreme"],
        }
        fmt = lambda v: "none" if v is None else f"{v:.4g}"  # noqa: E731
        print(f"{axis:<12}{str(mono):>10}{fmt(bp_abs):>14}{fmt(bp_exc):>12}"
              f"{sat_ext['overall']*100:>15.1f}%")

    (OUT / "analytic_breaking_point.json").write_text(json.dumps(report, indent=2))
    print("\nBP = severity at which the required command becomes unreachable for "
          f"{SATURATION_MAX*100:.0f}% of rows.")
    print("'none' = the axis never reaches that rule inside its bracket: it degrades")
    print("        the response without ever making the nominal command unreachable.")
    print(f"\n-> {OUT/'analytic_breaking_point.json'}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
