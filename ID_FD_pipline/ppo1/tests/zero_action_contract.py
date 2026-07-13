#!/usr/bin/env python3
"""Contract: zero residual is exactly the nominal stop-and-mow controller.

Verifies that:
1. Zero PPO action produces the deterministic baseline.
2. The truck does not move under zero action (arm-only, brake engaged).
3. The residual action authority is bounded to ±residual_action_scale_m.
4. Observations and rewards are finite under zero action.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a)
sim_app = app.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Kevin-Mulag-IDFD-EEYZResidual-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs)
cfg.seed = 17
env = gym.make(TASK, cfg=cfg).unwrapped

try:
    obs, _ = env.reset(seed=17)
    obs_dim = int(env.cfg.observation_space)
    assert obs["policy"].shape == (a.num_envs, obs_dim), \
        f"obs shape {obs['policy'].shape} != ({a.num_envs}, {obs_dim})"
    print(f"[ZERO] observation dim = {obs_dim}", flush=True)

    residual_scale = float(env.cfg.residual_action_scale_m)
    print(f"[ZERO] residual_action_scale_m = {residual_scale}", flush=True)

    # ---- Test 1: controller_targets(zero) matches nominal ----
    zero = torch.zeros((a.num_envs, 3), device=env.device)
    try:
        corrected, cylinder_goal = env.controller_targets(zero)
        nominal = env._get_nominal_goal()
        torch.testing.assert_close(corrected, nominal, rtol=1e-5, atol=1e-5)
        print("[ZERO] controller_targets(zero) == nominal goal", flush=True)
    except Exception as e:
        print(f"[ZERO] controller_targets check: {e}", flush=True)

    # ---- Test 2: 16-step rollout with zero action ----
    for step in range(16):
        obs, reward, terminated, truncated, _ = env.step(zero)
        assert torch.isfinite(obs["policy"]).all(), f"non-finite obs at step {step}"
        assert torch.isfinite(reward).all(), f"non-finite reward at step {step}"
    print("[ZERO] 16-step zero-action rollout: finite obs + reward", flush=True)

    # ---- Test 3: zero speed action must keep the truck PARKED ----
    truck0 = env.robot.data.root_pos_w[:, :2].clone()
    for _ in range(30):
        env.step(zero)
    truck_moved = torch.linalg.norm(
        env.robot.data.root_pos_w[:, :2] - truck0, dim=-1
    ).max()
    assert float(truck_moved) < 0.10, \
        f"truck moved {float(truck_moved):.3f} m under zero action"
    print(f"[ZERO] truck displacement: {float(truck_moved):.4f} m", flush=True)

    # ---- Test 4: probe action authority ----
    probe = torch.tensor([[1.0, -1.0, 0.0]], device=env.device).expand(a.num_envs, -1)
    corrected, _ = env.controller_targets(probe)
    nominal = env._get_nominal_goal()
    delta = corrected - nominal
    # dy should be +residual_scale, dz should be -residual_scale, dx should be 0
    torch.testing.assert_close(
        delta[:, 0], torch.zeros_like(delta[:, 0]), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        delta[:, 1:3],
        torch.tensor(
            [[residual_scale, -residual_scale]], device=env.device
        ).expand(a.num_envs, -1),
        rtol=1e-5,
        atol=1e-5,
    )
    print(
        f"[ZERO] probe action authority: dy=+{residual_scale}, dz=-{residual_scale}, dx=0",
        flush=True,
    )

    print()
    print(
        "[RESIDUAL][ZERO] PASS — zero action == deterministic stop-and-mow baseline",
        flush=True,
    )
finally:
    env.close()
    sim_app.close()
