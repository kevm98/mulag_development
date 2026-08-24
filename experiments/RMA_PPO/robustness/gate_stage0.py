"""Stage 0 preflight: is this surrogate able to host a degradation experiment?

Two gates, both run before any stress sweep, in this order:

  S0-A  surrogate credibility -- open-loop replay drift must be small compared
        with the motion the episode actually demands (no policy involved);
  S0-B  controller functionality -- the frozen PPO must beat a do-nothing null
        controller on the NOMINAL plant.

If S0-A fails there is no valid plant to stress.  If S0-B fails there is no
nominal performance envelope to leave, so no robustness boundary exists and no
degradation ratio can be interpreted.  Neither gate looks at a stressed point,
so passing them cannot bias the sweep that follows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import protocol as P
import surrogate_validity as SV
from frozen_ppo import SOURCE, load_frozen_actor
from robustness_env import RobustnessEnv
from stress import Stress

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

OVERSAMPLE = 4               # draw this many times N_EPISODES before screening


@torch.no_grad()
def rollout(env: RobustnessEnv, actor, policy: str, stress: Stress = Stress()) -> dict:
    obs = env.reset(stress)
    err, sat, z = [], [], []
    ret = torch.zeros(env.num_envs)
    for _ in range(P.HORIZON):
        a = torch.clamp(actor.mean(obs), -1.0, 1.0) if policy == "ppo" else torch.zeros(env.num_envs, 4)
        obs, r, done, info = env.step(a)
        ret += r
        err.append(info["extension_rmse"])
        sat.append(info["saturated"])
        z.append(info["sd_state_z"])
    err = torch.stack(err)
    return {
        "policy": policy,
        "median_extension_error_m": float(err.median(dim=0).values.median()),
        "mean_extension_error_m": float(err.mean()),
        "return_mean": float(ret.mean()),
        "saturation": float(torch.stack(sat).mean()),
        "max_sd_state_z": float(torch.stack(z).max()),
    }


def main() -> int:
    t0 = time.time()
    actor = load_frozen_actor()

    candidates = P.frozen_episode_set(actor.obs_mean, actor.obs_std,
                                      n_episodes=OVERSAMPLE * P.N_EPISODES)
    env = RobustnessEnv(episodes=candidates, horizon=P.HORIZON, obs_mean=actor.obs_mean)

    kept, drift = SV.screen(env, candidates, horizon=P.HORIZON,
                            max_fraction=P.MAX_DRIFT_FRACTION)
    s0a = drift.summary(P.MAX_DRIFT_FRACTION)
    s0a["required_episodes"] = P.N_EPISODES
    s0a["passed"] = len(kept) >= P.N_EPISODES
    print("S0-A surrogate credibility:", json.dumps(s0a, indent=None))

    report = {
        "gate": "stage0_preflight",
        "checkpoint": str(SOURCE),
        "horizon_steps": P.HORIZON,
        "min_motion_m": P.MIN_MOTION_M,
        "max_drift_fraction": P.MAX_DRIFT_FRACTION,
        "candidates": len(candidates),
        "S0A_surrogate_credibility": s0a,
    }

    if not s0a["passed"]:
        report["verdict"] = "STOP: surrogate drift dominates the task signal"
        (OUT / "stage0_gate.json").write_text(json.dumps(report, indent=2))
        print(f"\nVERDICT: {report['verdict']}  ({time.time()-t0:.1f}s)")
        return 1

    episodes = kept[:P.N_EPISODES]
    env = RobustnessEnv(episodes=episodes, horizon=P.HORIZON, obs_mean=actor.obs_mean)
    ppo = rollout(env, actor, "ppo")
    null = rollout(env, actor, "zero")
    margin = null["median_extension_error_m"] / max(ppo["median_extension_error_m"], 1.0e-12)
    s0b = {"ppo": ppo, "null": null, "null_over_ppo_error": margin,
           "passed": margin > 1.0}
    print("S0-B ppo :", json.dumps(ppo, indent=None))
    print("S0-B null:", json.dumps(null, indent=None))
    print(f"S0-B null/ppo error ratio = {margin:.2f} (must exceed 1.0)")

    report["episode_set"] = [{"run_id": e.run_id, "start_row": e.start} for e in episodes]
    report["S0B_controller_functional"] = s0b
    report["verdict"] = ("PROCEED: sweep the stress axes" if s0b["passed"] else
                         "STOP: frozen PPO does not control the nominal surrogate")
    (OUT / "stage0_gate.json").write_text(json.dumps(report, indent=2))
    print(f"\nVERDICT: {report['verdict']}  ({time.time()-t0:.1f}s)")
    return 0 if s0b["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
