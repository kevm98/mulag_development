"""Gate: is the frozen PPO a functional closed-loop controller at nominal?

If the frozen policy cannot beat an open-loop null controller on its own
nominal plant, there is no nominal performance envelope to leave and no
robustness boundary can be defined.  This gate runs BEFORE any stress sweep.
"""
import json, time
import torch

from frozen_ppo import load_frozen_actor
from robustness_env import RobustnessEnv
from stress import Stress
import protocol as P

actor = load_frozen_actor()
episodes = P.frozen_episode_set(actor.obs_mean, actor.obs_std)
print(f"episodes: {len(episodes)} across {len({e.run_id for e in episodes})} runs")

env = RobustnessEnv(episodes=episodes, horizon=P.HORIZON, obs_mean=actor.obs_mean)
envelope = P.measure_validity_envelope(env.metrics)
print("validity envelope:", json.dumps(envelope.to_dict(), indent=None)[:220])


def rollout(policy: str, stress=Stress()):
    obs = env.reset(stress)
    err, sat, ret, z = [], [], torch.zeros(env.num_envs), []
    obs_z = []
    for _ in range(P.HORIZON):
        if policy == "ppo":
            a = torch.clamp(actor.mean(obs), -1.0, 1.0)
        elif policy == "zero":
            a = torch.zeros(env.num_envs, 4)
        obs_z.append(actor.z(obs).abs().amax())
        obs, r, d, info = env.step(a)
        ret += r
        err.append(info["extension_rmse"])
        sat.append(info["saturated"])
        z.append(info["sd_state_z"])
    err = torch.stack(err)
    return {
        "median_extension_rmse_m": float(err.median()),
        "mean_extension_rmse_m": float(err.mean()),
        "return_mean": float(ret.mean()),
        "saturation": float(torch.stack(sat).mean()),
        "max_sd_state_z": float(torch.stack(z).max()),
        "max_obs_z": float(torch.stack(obs_z).max()),
    }

t0 = time.time()
for name in ("ppo", "zero"):
    print(f"{name:5s}", json.dumps(rollout(name)))
print(f"({time.time()-t0:.1f}s)")
