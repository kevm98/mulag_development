#!/usr/bin/env python3
"""Measure the live base-frame tool Jacobian with respect to cylinder extension."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--num_poses", type=int, default=10)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--pose_delta_m", type=float, default=0.012)
p.add_argument("--cylinder_eps_m", type=float, default=0.008)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.num_envs != 1 or not 10 <= a.num_poses <= 20:
    raise ValueError("Use --num_envs 1 and --num_poses in [10, 20]")

random.seed(a.seed)
np.random.seed(a.seed)
app = AppLauncher(a)
sim_app = app.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(a.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(a.seed)

import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=1)
cfg.seed = int(a.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 300.0
cfg.ee_success_hold_steps = 100000
env = gym.make(TASK, cfg=cfg).unwrapped


def pose_patterns(device: torch.device) -> torch.Tensor:
    # Center plus deterministic, well-spread corners of the local 4-D workspace.
    patterns = [
        [0, 0, 0, 0], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1],
        [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, -1, 1, 1],
        [1, 1, 1, -1], [-1, 1, 1, 1], [1, -1, 1, 1],
        [1, 1, -1, 1], [-1, -1, -1, 1], [1, -1, -1, -1],
        [-1, 1, -1, -1], [-1, -1, 1, -1], [1, 1, 1, 1],
        [-1, -1, -1, -1], [1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
    ]
    return torch.tensor(patterns[: a.num_poses], dtype=torch.float32, device=device)


@torch.no_grad()
def settle_neutral(steps: int = 30) -> None:
    zero = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
    for _ in range(steps):
        env.step(zero)


@torch.no_grad()
def direct_measure(goal_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Set exact articulation coordinates and update live link transforms."""
    scaling = env._cylinder_adapter.scaling
    if scaling is None:
        raise RuntimeError("EE probe requires the configured learned/USD cylinder scaling")
    q_goal = scaling.learned_x_to_usd_q(goal_x.reshape(1, 4))
    q_goal = scaling.clamp_usd_q(q_goal)
    joint_pos = env.robot.data.joint_pos.detach().clone()
    joint_vel = env.robot.data.joint_vel.detach().clone()
    joint_pos[:, env._cylinder_joint_ids] = q_goal
    joint_vel[:, env._cylinder_joint_ids] = 0.0
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env.sim.forward()
    x, _ = env._refresh_cylinder_state()
    ee = env._compute_ee_pos_local()
    return x[0].detach().clone(), ee[0].detach().clone()


def axis_description(vector: torch.Tensor) -> str:
    names = ("base-x/forward", "base-y/lateral", "base-z/height")
    index = int(torch.argmax(vector.abs()).item())
    sign = "+" if float(vector[index]) >= 0.0 else "-"
    return f"dominant={sign}{names[index]} xyz={[round(float(v), 4) for v in vector.cpu()]}"


try:
    patterns = pose_patterns(env.device)
    worst_cond = 0.0
    minimum_rank = 3
    minimum_effective_rank = 3
    pose_results = []
    for pose_index, pattern in enumerate(patterns):
        env.reset(seed=int(a.seed))
        settle_neutral()
        start_x, _ = env._refresh_cylinder_state()
        pose_goal = start_x[0] + float(a.pose_delta_m) * pattern
        base_x, _ = direct_measure(pose_goal)

        dx_columns, dee_columns = [], []
        for cylinder in range(4):
            direction = torch.zeros(4, dtype=torch.float32, device=env.device)
            direction[cylinder] = float(a.cylinder_eps_m)
            x_plus, ee_plus = direct_measure(base_x + direction)
            x_minus, ee_minus = direct_measure(base_x - direction)
            dx_columns.append(x_plus - x_minus)
            dee_columns.append(ee_plus - ee_minus)
        dx_matrix = torch.stack(dx_columns, dim=1)
        dee_matrix = torch.stack(dee_columns, dim=1)
        jacobian = dee_matrix @ torch.linalg.pinv(dx_matrix)
        u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=True)
        rank = int(torch.linalg.matrix_rank(jacobian).item())
        effective_rank = int((singular_values > 0.01 * singular_values[0]).sum().item())
        controllable_cond = float(
            (singular_values[0] / singular_values[max(effective_rank - 1, 0)].clamp(min=1.0e-12)).item()
        )
        cond = float((singular_values[0] / singular_values[-1].clamp(min=1.0e-12)).item())
        minimum_rank = min(minimum_rank, rank)
        minimum_effective_rank = min(minimum_effective_rank, effective_rank)
        worst_cond = max(worst_cond, cond)
        pose_results.append((jacobian.detach().cpu(), u.detach().cpu(), singular_values.detach().cpu(), vh.detach().cpu()))
        print(
            f"[EE-JAC] pose={pose_index:02d} numerical_rank={rank}/3 "
            f"physical_rank={effective_rank}/3 full_cond={cond:.2f} "
            f"controllable_cond={controllable_cond:.2f} "
            f"SVs={[f'{float(v):.4e}' for v in singular_values]} "
            f"base_x={[round(float(v), 5) for v in base_x.cpu()]}",
            flush=True,
        )
        for direction_index in range(3):
            print(
                f"[EE-DIR] pose={pose_index:02d} sigma={float(singular_values[direction_index]):.4e} "
                f"{axis_description(u[:, direction_index])} "
                f"cylinder_v={[round(float(v), 4) for v in vh[direction_index].cpu()]}",
                flush=True,
            )
        print(
            f"[EE-NULL] pose={pose_index:02d} cylinder_v="
            f"{[round(float(v), 4) for v in vh[3].cpu()]}",
            flush=True,
        )

    mean_jacobian = torch.stack([result[0] for result in pose_results]).mean(dim=0)
    mean_u, mean_s, mean_vh = torch.linalg.svd(mean_jacobian, full_matrices=True)
    print("\n========== EE JACOBIAN SUMMARY ==========", flush=True)
    print(
        f"[EE-WORST] min_numerical_rank={minimum_rank}/3 "
        f"min_physical_rank={minimum_effective_rank}/3 max_full_cond={worst_cond:.2f}",
        flush=True,
    )
    print(f"[EE-MEAN] J={mean_jacobian.numpy().tolist()}", flush=True)
    for index in range(3):
        print(
            f"[EE-MEAN-DIR] sigma={float(mean_s[index]):.4e} "
            f"{axis_description(mean_u[:, index])} "
            f"cylinder_v={[round(float(v), 4) for v in mean_vh[index]]}",
            flush=True,
        )
    branch = "FULL_XYZ" if minimum_effective_rank == 3 else "BASE_YZ_PLUS_TRUCK_X"
    print(f"[EE-GATE] {branch}", flush=True)
finally:
    env.close()
    sim_app.close()
