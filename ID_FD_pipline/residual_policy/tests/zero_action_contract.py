"""Isaac Sim contract: zero residual is the deterministic controller exactly."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=2)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--active_steps", type=int, default=0, help="0 runs one complete A-B-A cycle")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch

import ID_FD_pipline.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task = "Kevin-Mulag-IDFD-Residual-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    cfg.force_zero_policy = True
    cfg.randomize_start_phase = False
    cfg.episode_length_s = 200.0
    cfg.enable_scripted_vehicle_motion = False
    env = gym.make(task, cfg=cfg).unwrapped
    errors: list[float] = []
    max_goal_delta = 0.0
    max_c4_delta = 0.0
    terminated_early = False
    try:
        print("[ZERO-ACTION] resetting environment", flush=True)
        env.reset(seed=int(a.seed))
        print("[ZERO-ACTION] reset complete", flush=True)
        zero = torch.zeros((env.num_envs, 3), device=env.device)
        active_steps = 0
        while True:
            _, _, terminated, _, _ = env.step(zero)
            active = (env._preposition_counter <= 0) & (env._settle_counter <= 0)
            if active.any():
                delta = env._corrected_cylinder_goal - env._baseline_cylinder_goal
                max_goal_delta = max(max_goal_delta, float(delta[active].abs().max().item()))
                max_c4_delta = max(max_c4_delta, float(delta[active, 3].abs().max().item()))
                tool_error = torch.linalg.norm(
                    env._original_reference_tool - env._compute_ee_pos_local(), dim=-1
                )
                errors.extend(tool_error[active].detach().cpu().tolist())
                active_steps += 1
            if terminated.any():
                terminated_early = True
                break
            if int(a.active_steps) > 0 and active_steps >= int(a.active_steps):
                break
            if int(a.active_steps) == 0 and active_steps > 0 and bool(
                (env._scored_elapsed >= env._complete_cycle_time).all()
            ):
                break

        scored = np.asarray(errors)
        warmup = min(120 * env.num_envs, max(len(scored) // 20, 1))
        scored = scored[warmup:]
        result = {
            "passed": bool(
                not terminated_early
                and max_goal_delta <= 1.0e-7
                and max_c4_delta <= 1.0e-7
                and float(env._residual_filtered.abs().max().item()) <= 1.0e-7
                and float(env._max_truck_drift.max().item()) < 0.02
                and len(scored) > 0
                and float(np.percentile(scored, 95)) < 0.05
            ),
            "terminated_early": terminated_early,
            "active_steps": active_steps,
            "max_corrected_vs_baseline_m": max_goal_delta,
            "max_cylinder4_delta_m": max_c4_delta,
            "max_filtered_residual": float(env._residual_filtered.abs().max().item()),
            "mean_tool_error_m": float(scored.mean()) if len(scored) else float("inf"),
            "p95_tool_error_m": float(np.percentile(scored, 95)) if len(scored) else float("inf"),
            "max_tool_error_m": float(scored.max()) if len(scored) else float("inf"),
            "max_truck_drift_m": float(env._max_truck_drift.max().item()),
        }
        print(f"[ZERO-ACTION] {'PASS' if result['passed'] else 'FAIL'} {json.dumps(result)}", flush=True)
        if not result["passed"]:
            raise SystemExit(2)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close()
