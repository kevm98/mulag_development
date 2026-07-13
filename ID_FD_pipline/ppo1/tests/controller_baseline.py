#!/usr/bin/env python3
"""Deterministic zero-residual baseline on all three moving-truck paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--warmup_steps", type=int, default=100)
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
cfg.enable_scripted_vehicle_motion = True
cfg.episode_length_s = 300.0
env = gym.make(TASK, cfg=cfg).unwrapped

try:
    zero = torch.zeros((a.num_envs, 3), device=env.device)
    results = {}
    for mode in ("mapped-arc",):
        env.reset(seed=17)
        errors = []
        for step in range(int(env.cfg.residual_settle_steps) + int(a.steps)):
            env.step(zero)
            if step >= int(env.cfg.residual_settle_steps) + int(a.warmup_steps):
                ee = env._compute_ee_pos_local()
                error = torch.linalg.norm(env._ee_goal_local - ee, dim=-1)
                errors.extend(error.detach().cpu().tolist())
        values = np.asarray(errors, dtype=np.float64)
        results[mode] = float(values.mean())
        print(
            f"[RESIDUAL][BASELINE] mode={mode} mean={values.mean():.5f} "
            f"p95={np.percentile(values, 95):.5f} final={values[-50:].mean():.5f}",
            flush=True,
        )
    print(f"[RESIDUAL][BASELINE] {results}", flush=True)
finally:
    env.close()
    sim_app.close()
