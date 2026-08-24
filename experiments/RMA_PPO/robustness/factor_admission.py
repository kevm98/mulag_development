"""Stage 0, factor-admission screens F1 and F4, on the CPU surrogate.

WHAT THIS CAN AND CANNOT DECIDE
-------------------------------
The frozen SD surrogate reproduces the MAGNITUDE of recorded cylinder motion
well (excursion ratio 0.97, correlation 0.92 over a 1 s window) but its
per-step direction is only correlated 0.75, so integrated position random-walks
roughly 4.7 mm per second against ~9.6 mm of demanded motion.  An ABSOLUTE
tracking-error metric is therefore not measurable here, and the degradation
gate G1 -- "baseline error >= 25% worse than nominal" -- must be run in
IsaacLab against PhysX.  See results/stage0_gate.json.

What survives is every DIFFERENTIAL question.  Both arms of a paired point use
the same frozen policy, the same episodes and the same initial states, and the
stress chain is an exact analytic transform applied to the valve BEFORE the
surrogate, so the two rollouts share their drift and it largely cancels:

  F1 response authority -- does moving one factor alone change the cylinder
     response at all?  A factor that does not move the plant cannot be
     adapted to, and sweeping it on the GPU is wasted time.
  F4 observability -- does the deployable observation history change when the
     factor changes?  A factor invisible in history cannot be estimated by
     RLA no matter how large the network.

A factor must pass BOTH to be worth carrying into the IsaacLab sweep.  Passing
here is necessary, not sufficient: control relevance (F2) and headroom (F3)
remain GPU questions.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import protocol as P
import surrogate_validity as SV
from frozen_ppo import SOURCE, load_frozen_actor
from robustness_env import DT_S, RobustnessEnv
from stress import Stress

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# 1.0 s keeps the surrogate's integrated drift near 4.7 mm while still leaving
# ~9.6 mm of demanded motion.  Frozen before any factor is screened.
HORIZON = 20
MIN_MOTION_M = 0.005
N_EPISODES = 64

# A factor's effect must clear the surrogate's own credibility floor -- the
# median open-loop replay drift over the same horizon -- or the surrogate
# cannot resolve it and the question moves to the GPU.
F1_FLOOR_MULTIPLE = 1.0

SWEEPS = {
    "authority": [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1],
    "delay": [1, 2, 4, 6, 8, 12, 20],
    "lag": [0.05, 0.10, 0.20, 0.40, 1.00],
    "rate_limit": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
    "deadband": [0.01, 0.02, 0.05, 0.10, 0.20],
}


def build(axis: str, severity: float) -> Stress:
    if axis == "authority":
        return Stress(authority=severity)
    if axis == "delay":
        return Stress(delay_steps=int(severity))
    if axis == "lag":
        return Stress(lag_tau_s=severity)
    if axis == "rate_limit":
        return Stress(rate_limit=severity)
    if axis == "deadband":
        return Stress(deadband=severity)
    raise ValueError(axis)


@torch.no_grad()
def trace(env: RobustnessEnv, actor, stress: Stress) -> dict[str, torch.Tensor]:
    """One deterministic closed-loop rollout of the frozen policy."""
    obs = env.reset(stress)
    ext, obs_z, valve, sat = [], [], [], []
    for _ in range(HORIZON):
        a = torch.clamp(actor.mean(obs), -1.0, 1.0)
        obs_z.append(actor.z(obs).clone())
        obs, r, done, info = env.step(a)
        ext.append(info["extension"].clone())
        valve.append(info["applied_valve"].clone())
        sat.append(info["saturated"].clone())
    return {"ext": torch.stack(ext), "obs_z": torch.stack(obs_z),
            "valve": torch.stack(valve), "sat": torch.stack(sat)}


def main() -> int:
    t0 = time.time()
    actor = load_frozen_actor()

    candidates = P.frozen_episode_set(actor.obs_mean, actor.obs_std,
                                      min_motion_m=MIN_MOTION_M, n_episodes=4 * N_EPISODES)
    P.HORIZON = HORIZON          # the episode screen above already used it
    env = RobustnessEnv(episodes=candidates, horizon=HORIZON, obs_mean=actor.obs_mean)

    drift = SV.replay_drift(env, candidates, HORIZON)
    floor_m = float(np.median(drift.drift_m))
    motion_m = float(np.median(drift.motion_m))
    print(f"surrogate credibility floor over {HORIZON*DT_S:.2f}s: "
          f"drift={floor_m*1000:.3f} mm against motion={motion_m*1000:.3f} mm")

    episodes = candidates[:N_EPISODES]
    env = RobustnessEnv(episodes=episodes, horizon=HORIZON, obs_mean=actor.obs_mean)
    base = trace(env, actor, Stress())

    report = {
        "screen": "stage0_factor_admission",
        "checkpoint": str(SOURCE),
        "horizon_steps": HORIZON,
        "horizon_s": HORIZON * DT_S,
        "n_episodes": N_EPISODES,
        "min_motion_m": MIN_MOTION_M,
        "surrogate_drift_floor_m": floor_m,
        "median_demanded_motion_m": motion_m,
        "f1_floor_multiple": F1_FLOOR_MULTIPLE,
        "axes": {},
    }

    for axis, grid in SWEEPS.items():
        print(f"\n=== {axis} ===")
        print(f"{'severity':>10}{'dExt mm':>10}{'/floor':>8}{'dValve':>9}"
              f"{'dObs z':>9}{'sat':>7}  F1     F4")
        rows = []
        for sev in grid:
            st = build(axis, sev)
            cur = trace(env, actor, st)
            d_ext = (cur["ext"] - base["ext"]).abs().amax(dim=-1).amax(dim=0)   # per episode [m]
            d_obs = (cur["obs_z"] - base["obs_z"]).abs().amax(dim=-1).amax(dim=0)
            d_val = (cur["valve"] - base["valve"]).abs().amax(dim=-1).amax(dim=0)
            ext_med = float(d_ext.median())
            obs_med = float(d_obs.median())
            f1 = ext_med >= F1_FLOOR_MULTIPLE * floor_m
            f4 = obs_med >= 0.10          # a tenth of a training standard deviation
            row = {
                "severity": sev,
                "stress_label": st.label(),
                "response_authority_m": ext_med,
                "response_authority_over_floor": ext_med / max(floor_m, 1e-12),
                "valve_divergence": float(d_val.median()),
                "observation_divergence_z": obs_med,
                "saturation": float(cur["sat"].mean()),
                "F1_response_authority": bool(f1),
                "F4_observability": bool(f4),
            }
            rows.append(row)
            print(f"{sev:>10}{ext_med*1000:>10.3f}{row['response_authority_over_floor']:>8.2f}"
                  f"{row['valve_divergence']:>9.4f}{obs_med:>9.3f}{row['saturation']:>7.3f}"
                  f"  {'PASS' if f1 else 'fail':<6}{'PASS' if f4 else 'fail'}")
        admitted = [r for r in rows if r["F1_response_authority"] and r["F4_observability"]]
        report["axes"][axis] = {
            "points": rows,
            "admitted_severities": [r["severity"] for r in admitted],
            "admitted": bool(admitted),
            "min_admitted_severity": admitted[0]["severity"] if admitted else None,
        }

    (OUT / "stage0_factor_admission.json").write_text(json.dumps(report, indent=2))
    print("\n=== admission summary ===")
    for axis, a in report["axes"].items():
        verdict = "ADMIT" if a["admitted"] else "REJECT"
        print(f"  {axis:<12}{verdict:<8}{a['admitted_severities']}")
    print(f"\n-> {OUT/'stage0_factor_admission.json'}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
