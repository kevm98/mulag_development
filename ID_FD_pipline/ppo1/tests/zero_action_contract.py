#!/usr/bin/env python3
"""Contract: zero residual is exactly the nominal cascaded controller."""

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
    assert obs["policy"].shape == (a.num_envs, 21)
    zero = torch.zeros((a.num_envs, 3), device=env.device)
    corrected, cylinder_goal = env.controller_targets(zero)
    nominal_cylinder_goal = env._ee_cascade.cylinder_goal_seeded(
        env._ee_goal_local,
        env._nominal_seed,
        correction_gain=float(env.cfg.residual_ee_correction_gain),
    )
    torch.testing.assert_close(corrected, env._ee_goal_local, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cylinder_goal, nominal_cylinder_goal, rtol=0.0, atol=0.0)

    for _ in range(16):
        obs, reward, terminated, truncated, _ = env.step(zero)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(reward).all()
        torch.testing.assert_close(
            env._corrected_ee_goal_local,
            env._ee_goal_local,
            rtol=0.0,
            atol=0.0,
        )

    # zero speed action must keep the truck PARKED
    truck0 = env.robot.data.root_pos_w[:, :2].clone()
    for _ in range(30):
        env.step(zero)
    truck_moved = torch.linalg.norm(env.robot.data.root_pos_w[:, :2] - truck0, dim=-1).max()
    assert float(truck_moved) < 0.05, f"truck moved {float(truck_moved):.3f} m under zero action"

    probe = torch.tensor([[1.0, -1.0, 0.0]], device=env.device).expand(a.num_envs, -1)
    corrected, _ = env.controller_targets(probe)
    delta = corrected - env._ee_goal_local
    torch.testing.assert_close(delta[:, 0], torch.zeros_like(delta[:, 0]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        delta[:, 1:3],
        torch.tensor([0.05, -0.05], device=env.device).expand(a.num_envs, -1),
        rtol=0.0,
        atol=1.0e-7,
    )
    print(
        "[RESIDUAL][ZERO] PASS zero action == parked-truck mapped-arc controller exactly; "
        "residual authority is Y/Z (0.05 m) + truck speed",
        flush=True,
    )
finally:
    env.close()
    sim_app.close()
