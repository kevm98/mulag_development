"""Sampling design for the (delay, deadband, lag) stress vector.

CPU-only, no plant model. Turns three independently-derived axis ceilings into
one sampling design with a MEASURED feasible region, rather than a guessed box.

Severity coordinate
-------------------
Each axis is one-sided: its nominal is zero, not one. So the common coordinate
is s in [0, 1] with s = 0 the unstressed plant and s = 1 that axis's own
analytic breaking point from `analytic_breaking_point.py`. Because all three
ceilings come from the same rule -- 30% of rows past the ID command limits --
s is effect-calibrated across axes: s = 0.8 means the same severity on delay as
on lag. A multiplier band like 0.4..1.2 cannot express this, because 1.0 would
have to mean "nominal" and these axes have no nominal at 1.

Why the joint check is needed
-----------------------------
The three ceilings were located ONE AT A TIME. Three axes at s = 0.8 may
jointly exceed the rule even though none does alone. This module composes the
exact preimage of the real chain and measures that.

Exact chained preimage
----------------------
`stress.py` applies deadband -> authority -> lag -> delay -> rate_limit, so the
command delivering a target applied valve u* is B^-1(L^-1(D^-1(u*))). On a
recorded sequence every intermediate is available:

    D^-1 : y(t)  = u*(t + k)                      (shift, non-causal)
    L^-1 : w(t)  = y(t-1) + (y(t) - y(t-1)) / b,  b = 1 - exp(-dt/tau)
                   where y(t-1) = u*(t + k - 1), also recorded
    B^-1 : cmd(t) = sign(w) * (|w| + d)

so the composition is exact, not a small-stress approximation.

Usage:
    python3.11 experiments/RMA_PPO/robustness/stress_vector.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_separation import (  # noqa: E402
    N_SAMPLES, OUT, SAMPLE_SEED, SATURATION_MAX, load_samples,
)
from robustness_env import DT_S, ID_CHECKPOINT, REPO  # noqa: E402

# s = 1 is each axis's analytic breaking point (results/analytic_breaking_point.json).
CEILING = {"delay_ticks": 8, "deadband": 0.07, "lag_s": 0.10}
DELAY_LEVELS = CEILING["delay_ticks"] + 1        # 0..8 inclusive
AXES = ("delay", "deadband", "lag")
DESIGN_SEED = 20260827


def to_physical(s: dict[str, float]) -> dict[str, float]:
    """Map severity in [0,1] per axis to physical units.

    Delay is integer with only 9 levels, so a continuous s cannot explore it
    smoothly. `floor(9s)` clipped to 8 gives every level equal probability
    under uniform s; `round(8s)` would give the two end levels half weight.
    """
    k = min(int(np.floor(s["delay"] * DELAY_LEVELS)), CEILING["delay_ticks"])
    return {
        "delay_steps": int(k),
        "deadband": float(s["deadband"] * CEILING["deadband"]),
        "lag_tau_s": float(s["lag"] * CEILING["lag_s"]),
    }


def required_command(phys: dict[str, float], sample) -> torch.Tensor:
    """Exact preimage of the deadband -> lag -> delay chain on recorded data."""
    k = int(phys["delay_steps"])
    lead = sample["u_lead"]
    y = lead[:, k, :]
    y_prev = lead[:, k - 1, :] if k >= 1 else sample["u_prev"]

    tau = float(phys["lag_tau_s"])
    if tau > 0.0:
        beta = 1.0 - float(np.exp(-DT_S / tau))
        w = y_prev + (y - y_prev) / beta
    else:
        w = y

    d = float(phys["deadband"])
    return torch.sign(w) * (w.abs() + d) if d > 0.0 else w


def saturation(phys, sample, limits) -> float:
    """Fraction of rows whose required command is unreachable on any cylinder."""
    over = required_command(phys, sample).abs() > limits
    return float(over.any(dim=-1).float().mean())


def frontier_on_ray(direction: dict[str, float], sample, limits, floor: float,
                    steps: int = 20) -> float:
    """Largest t with all axes at t*direction still inside the rule."""
    def sat(t: float) -> float:
        return saturation(to_physical({a: t * direction[a] for a in AXES}), sample, limits)

    if sat(1.0) - floor < SATURATION_MAX:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if sat(mid) - floor >= SATURATION_MAX:
            hi = mid
        else:
            lo = mid
    return lo


def stratified_design(n: int, rng: np.random.Generator) -> list[dict[str, float]]:
    """Latin hypercube on s, with delay stratified over its integer levels.

    Pure uniform draws leave holes at the episode counts this protocol can
    afford; LHS guarantees each axis's range is covered exactly once per bin.
    """
    design = []
    cols = {a: (rng.permutation(n) + rng.random(n)) / n for a in AXES}
    for i in range(n):
        design.append({a: float(cols[a][i]) for a in AXES})
    return design


def main() -> int:
    t0 = time.time()
    torch.manual_seed(SAMPLE_SEED)
    rng = np.random.default_rng(DESIGN_SEED)
    sample = load_samples(N_SAMPLES, SAMPLE_SEED)

    from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime
    limits = CylinderIDRuntime(
        models_root=REPO / "id_fd_model/vendor_models", checkpoint=ID_CHECKPOINT,
        num_envs=1, device=torch.device("cpu"),
    ).command_limits.reshape(1, 4)

    floor = saturation({"delay_steps": 0, "deadband": 0.0, "lag_tau_s": 0.0}, sample, limits)
    print(f"{sample['s'].shape[0]} rows | rule: excess saturation < {SATURATION_MAX:.2f}")
    print(f"nominal floor: {floor*100:.1f}% of rows\n")

    # --- mains: each axis alone, to confirm s=1 really is its own ceiling ----
    print("MAINS -- one axis at a time (excess saturation over the nominal floor)")
    print(f"{'s':>6}{'delay':>10}{'deadband':>11}{'lag':>9}")
    mains = {}
    for s in (0.25, 0.5, 0.75, 1.0):
        row = {}
        for axis in AXES:
            v = {a: (s if a == axis else 0.0) for a in AXES}
            row[axis] = saturation(to_physical(v), sample, limits) - floor
        mains[s] = row
        print(f"{s:>6.2f}{row['delay']*100:>9.1f}%{row['deadband']*100:>10.1f}%{row['lag']*100:>8.1f}%")

    # --- the diagonal: do the axes compound? --------------------------------
    print("\nDIAGONAL -- all three at the same severity t")
    print(f"{'t':>6}{'excess sat':>12}{'sum of mains':>14}{'interaction':>13}")
    diag = []
    for t in (0.25, 0.5, 0.75, 1.0):
        joint = saturation(to_physical({a: t for a in AXES}), sample, limits) - floor
        additive = sum(mains[t][a] for a in AXES)
        diag.append({"t": t, "excess": joint, "additive": additive})
        print(f"{t:>6.2f}{joint*100:>11.1f}%{additive*100:>13.1f}%{(joint-additive)*100:>12.1f}pp")

    t_max = frontier_on_ray({a: 1.0 for a in AXES}, sample, limits, floor)
    print(f"\nfeasible frontier on the equal-severity diagonal: t = {t_max:.3f}")

    # --- the design ---------------------------------------------------------
    n = 16
    raw = stratified_design(n, rng)
    rows = []
    for s in raw:
        phys = to_physical(s)
        excess = saturation(phys, sample, limits) - floor
        rows.append({"s": s, "physical": phys, "excess_saturation": excess,
                     "feasible": bool(excess < SATURATION_MAX)})
    kept = sum(r["feasible"] for r in rows)

    print(f"\nSTRATIFIED DESIGN, n={n} (Latin hypercube on s, delay on its 9 levels)")
    print(f"{'#':>3}{'s_delay':>9}{'s_dead':>8}{'s_lag':>7}"
          f"{'ticks':>7}{'deadband':>10}{'lag s':>8}{'excess':>9}  ok")
    for i, r in enumerate(rows):
        p, s = r["physical"], r["s"]
        print(f"{i:>3}{s['delay']:>9.3f}{s['deadband']:>8.3f}{s['lag']:>7.3f}"
              f"{p['delay_steps']:>7d}{p['deadband']:>10.4f}{p['lag_tau_s']:>8.4f}"
              f"{r['excess_saturation']*100:>8.1f}%  {'y' if r['feasible'] else 'NO'}")
    print(f"\n{kept}/{n} draws inside the rule; delay levels covered: "
          f"{sorted({r['physical']['delay_steps'] for r in rows})}")

    report = {
        "screen": "stress_vector_design", "n_samples": sample["s"].shape[0],
        "ceilings": CEILING, "saturation_max": SATURATION_MAX,
        "nominal_floor": floor, "mains": mains, "diagonal": diag,
        "diagonal_frontier_t": t_max, "design_seed": DESIGN_SEED, "design": rows,
    }
    (OUT / "stress_vector_design.json").write_text(json.dumps(report, indent=2))
    print(f"\n-> {OUT/'stress_vector_design.json'}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
