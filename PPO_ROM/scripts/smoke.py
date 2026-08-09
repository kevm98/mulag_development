#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Build and step the merged PPO3_1 task once, before committing GPU hours.

Checks the four things that would silently invalidate a long run:
  * the registered task really is the MERGED env (c4 authority + new reward),
  * the c4 per-axis authority is active, not a stale PPO3 default,
  * every reward term is finite and inside its published bound,
  * the reward is not accidentally constant (a dead objective).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--seed", type=int, default=17)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import PPO_ROM  # noqa: E402,F401
from PPO_ROM import TERRAIN_TASK_ID  # noqa: E402
from PPO_ROM.env import Ppo31SoleEnv  # noqa: E402
from PPO_ROM.rewards import RewardWeights  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"[{'ok  ' if condition else 'FAIL'}] {message}", flush=True)
    if not condition:
        failures.append(message)


env_cfg = parse_env_cfg(TERRAIN_TASK_ID, device=a.device, num_envs=a.num_envs)
env = gym.make(TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None)
raw = env.unwrapped
raw.reset(seed=int(a.seed))

check(isinstance(raw, Ppo31SoleEnv), f"task builds the merged env ({type(raw).__name__})")

lead = tuple(getattr(raw.cfg, "cylinder_target_max_lead_per_axis_m", ()))
scale = tuple(getattr(raw.cfg, "cylinder_action_scale_per_axis_m", ()))
check(len(lead) == 4 and len(scale) == 4, f"c4 per-axis authority present lead={lead} scale={scale}")
check(len(lead) == 4 and lead[3] > lead[0], f"c4 lead {lead[3] if lead else '?'} exceeds c1-c3 {lead[0] if lead else '?'}")
check(len(scale) == 4 and scale[3] > scale[0], f"c4 action scale {scale[3] if scale else '?'} exceeds c1-c3 {scale[0] if scale else '?'}")

weights = RewardWeights()
rewards = []
torch.manual_seed(int(a.seed))
for _ in range(int(a.steps)):
    action = torch.clamp(
        torch.randn((raw.num_envs, int(raw.cfg.action_space)), device=raw.device) * 0.3,
        -1.0, 1.0,
    )
    raw.step(action)
    terms = getattr(raw, "_ppo31_last_terms", None)
    if terms is None:
        continue
    for name, value in terms.as_dict().items():
        limit = abs(getattr(weights, name)) + 1e-4
        if not torch.isfinite(value).all():
            failures.append(f"{name} produced non-finite values")
        elif float(value.abs().max()) > limit:
            failures.append(f"{name} exceeded its bound {limit}: {float(value.abs().max())}")
    rewards.append(float(terms.total.mean()))

check(bool(rewards), "reward terms were produced")
if rewards:
    spread = max(rewards) - min(rewards)
    print(f"       reward mean range over {len(rewards)} steps: "
          f"{min(rewards):.4f} .. {max(rewards):.4f}", flush=True)
    check(spread > 1e-6, "reward responds to actions (not a dead objective)")
    check(all(abs(r) < 40.0 for r in rewards), "reward stays in a sane range")

env.close()
print(("[SMOKE PASSED]" if not failures else f"[SMOKE FAILED] {failures}"), flush=True)
app.close()
sys.exit(1 if failures else 0)
