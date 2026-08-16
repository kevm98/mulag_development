#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Visualize a trained constrained four-cylinder Policy."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", required=True)
p.add_argument("--trajectory_shape", choices=("line", "arc", "sine", "circle", "square", "sine_chain", "rounded_square"), default="circle")
p.add_argument("--trajectory_2d", action="store_true", help="Load the standalone 45-D trajectory-v2 task and library.")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--cylinder4_residual_scale_m", type=float, default=0.0005)
p.add_argument("--realtime", action="store_true")
p.add_argument("--seed", type=int, default=17)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]
if not 0.0 <= a.cylinder4_residual_scale_m <= 0.0005:
    p.error("cylinder-4 residual scale must be in [0, 0.0005] m")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import mulag_plant.tasks  # noqa: F401,E402
from mulag_plant.policy.env_cfg import apply_residual_curriculum_stage
from mulag_plant.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyPPORunnerCfg,
    FourCylinderTrajectory2DPPORunnerCfg,
)
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task = (
        "RRLAB-Mulag-PPO2-A-Trajectory2D-PLAY-v0"
        if a.trajectory_2d
        else "RRLAB-Mulag-PPO2-A-Policy-V3-PLAY-v0"
    )
    cfg = parse_env_cfg(task, device=a.device, num_envs=1)
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 45 if a.trajectory_2d else 42
    cfg.trajectory_geometry_observation = bool(a.trajectory_2d)
    if a.trajectory_2d:
        cfg.trajectory_library_path = str(ROOT / "mulag_plant/models/trajectory_2d_library_v2.json")
    cfg.trajectory_shape = str(a.trajectory_shape)
    cfg.randomize_trajectory_shape = False
    cfg.randomize_start_phase = False
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.cylinder4_residual_scale_m = float(a.cylinder4_residual_scale_m)
    cfg.head_down_termination_alignment = 0.98
    cfg.clearance_relative_to_reference = True
    if a.trajectory_2d:
        cfg.clearance_frame_consistent = True
    cfg.cutter_clearance_hard_band_m = 0.20
    cfg.episode_length_s = 200.0

    gym_env = gym.make(task, cfg=cfg)
    raw = gym_env.unwrapped
    agent_cfg = FourCylinderTrajectory2DPPORunnerCfg() if a.trajectory_2d else FourCylinderResidualPolicyPPORunnerCfg()
    agent_cfg.device = a.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=a.device)
    checkpoint = str(Path(a.checkpoint).expanduser().resolve())
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=raw.device)
    print(f"[4CYL PLAY] loaded {checkpoint}", flush=True)

    obs, _ = env.reset()
    step = 0
    try:
        while simulation_app.is_running():
            started = time.perf_counter()
            with torch.no_grad():
                obs, _, dones, _ = env.step(policy(obs))
            if step % 30 == 0:
                scales_mm = raw._residual_scale_vector() * 1000.0
                correction_mm = raw._residual_filtered[0] * scales_mm
                print(
                    f"step={step:05d} error="
                    f"{torch.linalg.norm(raw._original_reference_tool[0] - raw._compute_ee_pos_local()[0]).item():.4f}m "
                    f"residual_mm={correction_mm.detach().cpu().numpy().round(5)} "
                    f"head={raw._measure_head_down_alignment()[0].item():.5f} "
                    f"drift={raw._truck_post_constraint_drift_xy[0].item():.6f}m",
                    flush=True,
                )
            if dones.any():
                print("[4CYL PLAY] episode reset", flush=True)
            step += 1
            if a.realtime:
                time.sleep(max(0.0, raw.step_dt - (time.perf_counter() - started)))
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
