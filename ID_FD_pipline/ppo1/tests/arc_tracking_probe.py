#!/usr/bin/env python3
"""Probe: can the deterministic controller track the arc?"""

import argparse, sys, numpy as np
from pathlib import Path
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--steps", type=int, default=1200)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch
import ID_FD_pipline.tasks
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Kevin-Mulag-IDFD-EEYZResidual-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs)
cfg.seed = 17
cfg.residual_mow_shape = "arc"
cfg.residual_min_accumulated_path_m = 2.0
cfg.residual_path_speed_mps = 0.05
cfg.episode_length_s = 120.0
env = gym.make(TASK, cfg=cfg).unwrapped

try:
    zero = torch.zeros((a.num_envs, 3), device=env.device)
    obs, _ = env.reset(seed=17)
    
    errors = []
    heights = []
    phases = []
    goals = []
    ees = []
    path_accum = []
    
    # Debug: initial state
    ee0 = env._compute_ee_pos_local()
    nominal0 = env._get_nominal_goal()
    print(f'[ARC-PROBE] Reset EE: ({ee0[0,0]:.3f}, {ee0[0,1]:.3f}, {ee0[0,2]:.3f})', flush=True)
    print(f'[ARC-PROBE] Arc start: ({nominal0[0,0]:.3f}, {nominal0[0,1]:.3f}, {nominal0[0,2]:.3f})', flush=True)
    print(f'[ARC-PROBE] Initial error: {float(torch.linalg.norm(nominal0-ee0)):.3f}m', flush=True)
    
    for step in range(int(a.steps)):
        obs, reward, term, trunc, _ = env.step(zero)
        ee = env._compute_ee_pos_local()
        nominal = env._get_nominal_goal()
        err = float(torch.linalg.norm(nominal - ee, dim=-1)[0].item())
        z = float(env._compute_cutter_world_z()[0].item())
        phase = int(env._state_machine.state.phase[0].item())
        acc = float(env._state_machine.state.accumulated_path[0].item())
        errors.append(err)
        heights.append(z)
        phases.append(phase)
        goals.append(nominal[0].detach().cpu().numpy().copy())
        ees.append(ee[0].detach().cpu().numpy().copy())
        path_accum.append(acc)

    errs = np.array(errors)
    mowing_mask = np.isin(phases, [2, 3])  # MOW_OUT=2, MOW_RETURN=3
    mow_errs = errs[mowing_mask] if mowing_mask.any() else errs
    
    print(f"[ARC-PROBE] Steps: {len(errors)}, Mowing steps: {mowing_mask.sum()}")
    print(f"[ARC-PROBE] All steps: mean={errs.mean():.4f}m p95={np.percentile(errs,95):.4f}m max={errs.max():.4f}m")
    print(f"[ARC-PROBE] Mow only:  mean={mow_errs.mean():.4f}m p95={np.percentile(mow_errs,95):.4f}m")
    print(f"[ARC-PROBE] Cutter Z: mean={np.mean(heights):.4f}m range=[{np.min(heights):.4f},{np.max(heights):.4f}]")
    print(f"[ARC-PROBE] Accumulated path: {path_accum[-1]:.3f}m")
    print(f"[ARC-PROBE] Final phase: {phases[-1]}")
    
    # ARC tracking check
    passed = mow_errs.size > 0 and np.percentile(mow_errs, 95) < 0.10
    print(f"[ARC-PROBE] {'PASS' if passed else 'FAIL'} — p95 < 0.10m", flush=True)
finally:
    env.close(); sim_app.close()
