"""Frozen-PPO Robustness Envelope Under Plant Degradation -- one-factor sweeps.

The controller is frozen at every point: weights, observation normalizer,
reward, episode set, horizon and seeds are identical.  Only plant-side stress
varies.  A do-nothing null controller is swept alongside as a reference arm:
it isolates how much of any measured change is the SURROGATE responding to
actuator perturbation rather than the CONTROLLER failing.
"""
from __future__ import annotations

import csv, json, math, time
from pathlib import Path

import numpy as np
import torch

from frozen_ppo import load_frozen_actor, SOURCE
from robustness_env import RobustnessEnv, MAX_DELAY_STEPS, DT_S
from stress import Stress
import protocol as P

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

actor = load_frozen_actor()
episodes = P.frozen_episode_set(actor.obs_mean, actor.obs_std)
env = RobustnessEnv(episodes=episodes, horizon=P.HORIZON, obs_mean=actor.obs_mean)
envelope = P.measure_validity_envelope(env.metrics)
EXT_MIN = torch.tensor(envelope.extension_min).reshape(1, 4)
EXT_MAX = torch.tensor(envelope.extension_max).reshape(1, 4)


@torch.no_grad()
def evaluate(stress: Stress, policy: str = "ppo") -> dict:
    """One sweep point.  Returns aggregate metrics plus per-episode rows."""
    obs = env.reset(stress)
    n = env.num_envs
    err = torch.zeros(P.HORIZON, n)
    sat = torch.zeros(P.HORIZON, n)
    eff = torch.zeros(P.HORIZON, n)
    s_z = torch.zeros(P.HORIZON, n)
    u_z = torch.zeros(P.HORIZON, n)
    ret = torch.zeros(n)
    out_of_range = torch.zeros(n, dtype=torch.bool)
    nonfinite = torch.zeros(n, dtype=torch.bool)
    over_p = torch.zeros(n, dtype=torch.bool)
    obs_z_max = 0.0
    delivered_valve_in = 0.0
    delivered_valve_out = 0.0

    for t in range(P.HORIZON):
        a = torch.clamp(actor.mean(obs), -1.0, 1.0) if policy == "ppo" else torch.zeros(n, 4)
        obs_z_max = max(obs_z_max, float(actor.z(obs).abs().amax()))
        obs, r, done, info = env.step(a)
        ret += r
        err[t] = info["extension_rmse"]; sat[t] = info["saturated"]; eff[t] = info["control_effort"]
        s_z[t] = info["sd_state_z"]; u_z[t] = info["sd_action_z"]
        nonfinite |= info["nonfinite"]; over_p |= info["over_pressure"]
        x = info["extension"]
        out_of_range |= ((x < EXT_MIN) | (x > EXT_MAX)).any(dim=-1)
        delivered_valve_in += float(info["commanded_valve"].norm()) ** 2
        delivered_valve_out += float(info["applied_valve"].norm()) ** 2

    ep_err = err.median(dim=0).values                    # per-episode median error
    ep_state_z = s_z.max(dim=0).values
    ep_action_z = u_z.max(dim=0).values
    ep_valid = (~nonfinite) & (~over_p) & (~out_of_range) \
        & (ep_state_z <= envelope.state_z_max) & (ep_action_z <= envelope.action_z_max)

    rows = [{
        "run_id": episodes[i].run_id, "start_row": episodes[i].start,
        "episode_median_extension_error_m": float(ep_err[i]),
        "episode_return": float(ret[i]),
        "episode_saturation": float(sat[:, i].mean()),
        "episode_control_effort": float(eff[:, i].mean()),
        "episode_max_sd_state_z": float(ep_state_z[i]),
        "episode_max_sd_action_z": float(ep_action_z[i]),
        "episode_nonfinite": bool(nonfinite[i]),
        "episode_over_pressure": bool(over_p[i]),
        "episode_extension_out_of_recorded_range": bool(out_of_range[i]),
        "episode_surrogate_valid": bool(ep_valid[i]),
    } for i in range(n)]

    return {
        "stress": stress.as_dict(),
        "stress_label": stress.label(),
        "policy": policy,
        "median_extension_error_m": float(ep_err.median()),
        "mean_extension_error_m": float(err.mean()),
        "return_mean": float(ret.mean()),
        "saturation": float(sat.mean()),
        "control_effort": float(eff.mean()),
        "max_sd_state_z": float(s_z.max()),
        "max_sd_action_z": float(u_z.max()),
        "max_policy_obs_z": obs_z_max,
        "valid_fraction": float(ep_valid.float().mean()),
        "surrogate_valid": bool(ep_valid.all()),
        # positive control: realized actuator gain actually delivered
        "delivered_valve_gain": math.sqrt(delivered_valve_out / max(delivered_valve_in, 1e-30)),
        "delivered_rate_clipped_fraction": (env.actuation.delivered_rate_clipped
                                            / max(env.actuation.delivered_steps, 1)),
        "delivered_deadband_zeroed_fraction": (env.actuation.delivered_deadband_zeroed
                                               / max(env.actuation.delivered_steps, 1)),
        "episodes": rows,
    }


# ---------------------------------------------------------------------------
# Sweep grids -- deliberately pushed far past the previous experiment's range
# ---------------------------------------------------------------------------
SWEEPS = {
    "authority": [Stress(authority=a) for a in
                  (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.0)],
    "delay": [Stress(delay_steps=d) for d in
              (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40)],
    "lag": [Stress(lag_tau_s=t) for t in
            (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00, 2.00, 5.00)],
    "rate_limit": [Stress(rate_limit=r) for r in
                   (float("inf"), 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)],
    "deadband": [Stress(deadband=d) for d in
                 (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)],
}

t0 = time.time()
results = {"metadata": {
    "experiment": "Frozen-PPO Robustness Envelope Under Plant Degradation",
    "checkpoint": str(SOURCE),
    "control_dt_s": DT_S,
    "horizon_steps": P.HORIZON,
    "horizon_s": P.HORIZON * DT_S,
    "n_episodes": len(episodes),
    "episode_seed": P.EPISODE_SEED,
    "support_z": P.SUPPORT_Z,
    "max_delay_steps": MAX_DELAY_STEPS,
    "criteria": {
        "degradation_ratio": P.DEGRADATION_RATIO,
        "return_drop": P.RETURN_DROP,
        "breakdown_ratio": P.BREAKDOWN_RATIO,
        "breakdown_saturation": P.BREAKDOWN_SATURATION,
    },
    "validity_envelope": envelope.to_dict(),
    "episode_set": [{"run_id": e.run_id, "start_row": e.start} for e in episodes],
}, "sweeps": {}}

for axis, grid in SWEEPS.items():
    print(f"\n=== {axis} ===")
    print(f"{'severity':>12}{'medE mm':>10}{'D_E':>7}{'return':>10}{'D_ret':>7}"
          f"{'sat':>7}{'gain':>7}{'sdZ':>7}{'valid':>7}  region     nullE mm")
    axis_rows = []
    base = base_null = None
    for st in grid:
        r = evaluate(st, "ppo")
        rn = evaluate(st, "zero")
        if base is None:
            base, base_null = r, rn
        ratio = r["median_extension_error_m"] / base["median_extension_error_m"]
        ret_ratio = r["return_mean"] / base["return_mean"] if base["return_mean"] != 0 else float("nan")
        region = P.classify(error_ratio=ratio, saturation=r["saturation"],
                            surrogate_valid=r["surrogate_valid"], is_nominal=st.is_nominal)
        r["normalized_error"] = ratio
        r["normalized_return"] = ret_ratio
        r["region"] = region
        r["null_median_extension_error_m"] = rn["median_extension_error_m"]
        r["null_normalized_error"] = rn["median_extension_error_m"] / base_null["median_extension_error_m"]
        sev = {"authority": st.authority, "delay": st.delay_steps, "lag": st.lag_tau_s,
               "rate_limit": st.rate_limit, "deadband": st.deadband}[axis]
        r["severity"] = None if math.isinf(sev) else float(sev)
        axis_rows.append(r)
        print(f"{str(sev):>12}{r['median_extension_error_m']*1000:>10.3f}{ratio:>7.2f}"
              f"{r['return_mean']:>10.2f}{ret_ratio:>7.2f}{r['saturation']:>7.3f}"
              f"{r['delivered_valve_gain']:>7.3f}{r['max_sd_state_z']:>7.1f}"
              f"{r['valid_fraction']:>7.2f}  {region:<11}{rn['median_extension_error_m']*1000:>7.3f}")
    results["sweeps"][axis] = axis_rows

(OUT / "robustness_sweeps.json").write_text(json.dumps(results, indent=2))

with (OUT / "robustness_sweeps.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["axis", "severity", "stress_label", "run_id", "start_row",
                "episode_median_extension_error_m", "episode_return", "episode_saturation",
                "episode_control_effort", "episode_max_sd_state_z", "episode_max_sd_action_z",
                "episode_nonfinite", "episode_over_pressure",
                "episode_extension_out_of_recorded_range", "episode_surrogate_valid",
                "point_normalized_error", "point_region", "point_delivered_valve_gain"])
    for axis, rows in results["sweeps"].items():
        for r in rows:
            for e in r["episodes"]:
                w.writerow([axis, r["severity"], r["stress_label"], e["run_id"], e["start_row"],
                            e["episode_median_extension_error_m"], e["episode_return"],
                            e["episode_saturation"], e["episode_control_effort"],
                            e["episode_max_sd_state_z"], e["episode_max_sd_action_z"],
                            e["episode_nonfinite"], e["episode_over_pressure"],
                            e["episode_extension_out_of_recorded_range"],
                            e["episode_surrogate_valid"], r["normalized_error"],
                            r["region"], r["delivered_valve_gain"]])

print(f"\ntotal {time.time()-t0:.1f}s -> {OUT}")
