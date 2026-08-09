#!/usr/bin/env python3
"""Measure a dense four-cylinder head-down workspace in batched PhysX clones."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--batches", type=int, default=4)
p.add_argument("--deploy_steps", type=int, default=300)
p.add_argument("--settle_steps", type=int, default=180)
p.add_argument("--hold_steps", type=int, default=60)
p.add_argument("--head_min_alignment", type=float, default=0.98)
p.add_argument("--out", default="mulag_plant/models/head_down_workspace_dense_v3.json")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.num_envs < 16 or a.batches < 1:
    p.error("--num_envs must be >=16 and --batches must be positive")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply

import mulag_plant.tasks  # noqa: F401,E402
from mulag_plant.dls_solver.direct_drive import DirectCylinderDriver
from isaaclab_tasks.utils import parse_env_cfg


@torch.no_grad()
def main() -> None:
    cfg = parse_env_cfg(
        "RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0",
        device=a.device,
        num_envs=int(a.num_envs),
    )
    cfg.enable_scripted_vehicle_motion = False
    cfg.episode_length_s = 100000.0
    env = gym.make(
        "RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0", cfg=cfg
    ).unwrapped
    rows: list[dict[str, object]] = []
    # Safely inside the trained ROM limits but materially wider than v1.
    lower = torch.tensor([0.13, 0.54, 0.41, 0.52], device=env.device)
    upper = torch.tensor([0.43, 0.62, 0.54, 0.64], device=env.device)
    try:
        env.reset(seed=17)
        driver = DirectCylinderDriver(env)
        wheels = env._wheel_joint_ids
        wheel_target = env.robot.data.joint_pos[:, wheels].detach().clone()
        truck_id = env._truck_base_body_id()
        anchor_xy = env.robot.data.body_link_state_w[:, truck_id, :2].detach().clone()
        sobol = torch.quasirandom.SobolEngine(4, scramble=True, seed=17)

        def hold_truck() -> None:
            env.robot.write_joint_stiffness_to_sim(2.0e9, joint_ids=wheels)
            env.robot.write_joint_damping_to_sim(1.0e7, joint_ids=wheels)
            env.robot.set_joint_position_target(wheel_target, joint_ids=wheels)
            env.robot.set_joint_velocity_target(
                torch.zeros_like(wheel_target), joint_ids=wheels
            )

        def constrain_truck() -> None:
            xy = env.robot.data.body_link_state_w[:, truck_id, :2]
            mask = torch.linalg.norm(anchor_xy - xy, dim=1) > 0.001
            if mask.any():
                pose = env.robot.data.root_state_w[:, :7].detach().clone()
                pose[mask, :2] += anchor_xy[mask] - xy[mask]
                env.robot.write_root_pose_to_sim(pose)

        for batch in range(int(a.batches)):
            target = lower + sobol.draw(int(a.num_envs)).to(env.device) * (upper - lower)
            start = driver.measured_x().clone()
            for step in range(int(a.deploy_steps)):
                u = float(step + 1) / int(a.deploy_steps)
                blend = 0.5 * (1.0 - math.cos(math.pi * u))
                hold_truck()
                driver.step(start + blend * (target - start), substeps=2)
                constrain_truck()
            for _ in range(int(a.settle_steps)):
                hold_truck(); driver.step(target, substeps=2); constrain_truck()

            x1 = driver.measured_x().clone()
            tool = env._compute_ee_pos_local().clone()
            world_z = env._compute_cutter_world_z().clone()
            q = env.robot.data.body_link_state_w[:, env._ee_body_id, 3:7]
            q = q / q.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
            axis = quat_apply(
                q,
                torch.tensor([[0.0, 0.0, 1.0]], device=env.device).expand(a.num_envs, -1),
            )
            alignment = -axis[:, 2]
            before = env.robot.data.body_link_state_w[:, truck_id, :2].clone()
            for _ in range(int(a.hold_steps)):
                hold_truck(); driver.step(target, substeps=2); constrain_truck()
            x2 = driver.measured_x().clone()
            drift = torch.linalg.norm(
                env.robot.data.body_link_state_w[:, truck_id, :2] - before, dim=1
            )
            reach = (x1 - target).abs().amax(dim=1)
            hold = (x2 - x1).abs().amax(dim=1)
            ids = env._cylinder_joint_ids
            effort = env.robot.data.joint_effort_limits[:, ids].abs().clamp(min=1.0)
            saturated = (
                env.robot.data.computed_torque[:, ids].abs() >= 0.999 * effort
            ).any(dim=1)
            valid = (
                (reach <= 0.010)
                & (hold <= 0.003)
                & (world_z >= -0.15)
                & (world_z <= 0.28)
                & (alignment >= float(a.head_min_alignment))
                & (drift <= 0.02)
                & (~saturated)
            )
            for i in range(int(a.num_envs)):
                rows.append({
                    "valid": bool(valid[i]),
                    "cylinder_cmd": target[i].cpu().tolist(),
                    "cylinder_measured": x1[i].cpu().tolist(),
                    "cutter_truck": tool[i].cpu().tolist(),
                    "cutter_world_z": float(world_z[i]),
                    "head_alignment": float(alignment[i]),
                    "reach_error_m": float(reach[i]),
                    "hold_drift_m": float(hold[i]),
                    "truck_drift_m": float(drift[i]),
                    "saturated": bool(saturated[i]),
                })
            print(f"[DENSE WORKSPACE] batch={batch + 1}/{a.batches} valid={int(valid.sum())}/{a.num_envs}", flush=True)

        valid_rows = [row for row in rows if row["valid"]]
        if len(valid_rows) < 48:
            raise RuntimeError(f"dense workspace has only {len(valid_rows)} valid poses")
        tool_np = np.asarray([row["cutter_truck"] for row in valid_rows])
        cmd_np = np.asarray([row["cylinder_cmd"] for row in valid_rows])
        payload = {
            "schema_version": 3,
            "note": "Dense Sobol direct-PhysX head-down workspace for large v3 shapes.",
            "sampling_bounds_m": [lower.cpu().tolist(), upper.cpu().tolist()],
            "num_valid": len(valid_rows),
            "tool_span_m": np.ptp(tool_np, axis=0).tolist(),
            "cylinder_span_m": np.ptp(cmd_np, axis=0).tolist(),
            "minimum_head_alignment": min(float(row["head_alignment"]) for row in valid_rows),
            "samples": rows,
        }
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[DENSE WORKSPACE] valid={len(valid_rows)}/{len(rows)} tool_span={np.ptp(tool_np, axis=0).round(3).tolist()} wrote={out}", flush=True)
    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close()
