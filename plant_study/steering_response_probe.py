#!/usr/bin/env python3
"""Measure MULAG front-steering sign conventions in four parallel environments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 4:
    raise ValueError("Steering probe requires four environments")
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Kevin-Mulag-IDFD-EETracking-PLAY-v0", device=args.device, num_envs=4)
cfg.enable_scripted_vehicle_motion = True
cfg.vehicle_speed_min_kmh = 4.5
cfg.vehicle_speed_max_kmh = 4.5
cfg.vehicle_wheel_speed_gain = 1.0
cfg.episode_length_s = 1000.0
env = gym.make("Kevin-Mulag-IDFD-EETracking-PLAY-v0", cfg=cfg).unwrapped

try:
    env.reset(seed=17)
    env.robot.write_joint_stiffness_to_sim(1.0e7, joint_ids=env._steering_joint_ids)
    env.robot.write_joint_damping_to_sim(1.0e6, joint_ids=env._steering_joint_ids)
    targets = torch.tensor(
        [[0.0, 0.0], [0.05, 0.05], [0.05, -0.05], [-0.05, 0.05]],
        device=env.device,
    )
    env._scripted_steering_target = targets
    base_id = env._truck_base_body_id()
    start = env.robot.data.body_link_state_w[:, base_id].detach().clone()
    q0 = start[:, 3:7] / start[:, 3:7].norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
    ex = torch.tensor([[1.0, 0.0, 0.0]], device=env.device).expand(4, -1)
    ey = torch.tensor([[0.0, 1.0, 0.0]], device=env.device).expand(4, -1)
    forward0 = quat_apply(q0, ex); lateral0 = quat_apply(q0, ey)
    zero = torch.zeros((4, 4), device=env.device)
    for _ in range(int(args.steps)):
        env.step(zero)
    state = env.robot.data.body_link_state_w[:, base_id]
    delta = state[:, :3] - start[:, :3]
    q = state[:, 3:7] / state[:, 3:7].norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
    forward = quat_apply(q, ex)
    for index in range(4):
        longitudinal = float(torch.dot(delta[index], forward0[index]).item())
        lateral = float(torch.dot(delta[index], lateral0[index]).item())
        heading = torch.atan2(
            torch.dot(forward[index], lateral0[index]),
            torch.dot(forward[index], forward0[index]),
        )
        measured = env.robot.data.joint_pos[index, env._steering_joint_ids]
        print(
            f"[STEER] env={index} target={targets[index].cpu().tolist()} "
            f"measured={measured.cpu().tolist()} forward={longitudinal:.3f}m "
            f"lateral={lateral:.3f}m heading_deg={float(torch.rad2deg(heading).item()):+.2f}",
            flush=True,
        )
finally:
    env.close(); simulation_app.close()
