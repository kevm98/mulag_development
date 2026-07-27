#!/usr/bin/env python3
"""Build a verified head-down 3-D workspace with explicit cylinder-3 excitation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--deploy_steps", type=int, default=300)
p.add_argument("--settle_steps", type=int, default=180)
p.add_argument("--hold_steps", type=int, default=60)
p.add_argument("--reach_tol_m", type=float, default=0.01)
p.add_argument("--hold_tol_m", type=float, default=0.003)
p.add_argument("--height_min_m", type=float, default=-0.12)
p.add_argument("--height_max_m", type=float, default=0.25)
p.add_argument("--head_min_alignment", type=float, default=0.995)
p.add_argument("--truck_drift_tol_m", type=float, default=0.02)
p.add_argument("--c4", type=float, default=0.588991)
p.add_argument("--out", default="PPO2_ROM_PhysX/models/head_down_workspace_3d.json")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply

import PPO2_ROM_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_PhysX.dls_solver.direct_drive import DirectCylinderDriver
from isaaclab_tasks.utils import parse_env_cfg


@torch.no_grad()
def main() -> None:
    if a.num_envs != 64:
        raise ValueError("the maintained workspace grid contains exactly 64 samples")
    cfg = parse_env_cfg("RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0", device=a.device, num_envs=64)
    cfg.enable_scripted_vehicle_motion = False
    cfg.episode_length_s = 100000.0
    env = gym.make("RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0", cfg=cfg).unwrapped
    try:
        env.reset(seed=17)
        driver = DirectCylinderDriver(env)
        wheel_ids = env._wheel_joint_ids
        wheel_target = env.robot.data.joint_pos[:, wheel_ids].detach().clone()
        truck_id = env._truck_base_body_id()
        anchor_xy = env.robot.data.body_link_state_w[:, truck_id, :2].detach().clone()

        teacher = json.loads((ROOT / "PPO2_ROM_PhysX/models/mowing_arc_head_down.json").read_text())
        teacher_seed = np.asarray([r["cylinder_cmd"] for r in teacher["cylinder_teacher"]])
        c1_values = np.linspace(teacher_seed[:, 0].min(), teacher_seed[:, 0].max(), 4)
        # Cylinder 2 has dominant height authority; cylinder 3 changes boom reach.
        c2_values = np.asarray([0.547, 0.567, 0.587, 0.607], dtype=np.float32)
        c3_values = np.asarray([0.423, 0.453, 0.483, 0.513], dtype=np.float32)
        grid = np.asarray(
            [[c1, c2, c3, float(a.c4)] for c1 in c1_values for c2 in c2_values for c3 in c3_values],
            dtype=np.float32,
        )
        target = torch.as_tensor(grid, device=env.device)
        start = driver.measured_x().clone()

        def hold_truck() -> None:
            env.robot.write_joint_stiffness_to_sim(2.0e9, joint_ids=wheel_ids)
            env.robot.write_joint_damping_to_sim(1.0e7, joint_ids=wheel_ids)
            env.robot.set_joint_position_target(wheel_target, joint_ids=wheel_ids)
            env.robot.set_joint_velocity_target(torch.zeros_like(wheel_target), joint_ids=wheel_ids)

        def constrain_truck() -> None:
            xy = env.robot.data.body_link_state_w[:, truck_id, :2]
            correction = anchor_xy - xy
            mask = torch.linalg.norm(correction, dim=1) > 0.001
            if mask.any():
                pose = env.robot.data.root_state_w[:, :7].detach().clone()
                pose[mask, :2] += correction[mask]
                env.robot.write_root_pose_to_sim(pose)

        for step in range(int(a.deploy_steps)):
            u = float(step + 1) / max(int(a.deploy_steps), 1)
            blend = 0.5 * (1.0 - math.cos(math.pi * u))
            hold_truck()
            driver.step(start + blend * (target - start), substeps=2)
            constrain_truck()
        for _ in range(int(a.settle_steps)):
            hold_truck()
            driver.step(target, substeps=2)
            constrain_truck()

        x1 = driver.measured_x().clone()
        tool = env._compute_ee_pos_local().clone()
        world_z = env._compute_cutter_world_z().clone()
        q = env.robot.data.body_link_state_w[:, env._ee_body_id, 3:7]
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
        axis = quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=env.device).expand(64, -1))
        head_alignment = -axis[:, 2]
        truck_before_hold = env.robot.data.body_link_state_w[:, truck_id, :2].detach().clone()

        for _ in range(int(a.hold_steps)):
            hold_truck()
            driver.step(target, substeps=2)
            constrain_truck()
        x2 = driver.measured_x().clone()
        truck_after_hold = env.robot.data.body_link_state_w[:, truck_id, :2]

        reach = (x1 - target).abs().amax(dim=1)
        hold_drift = (x2 - x1).abs().amax(dim=1)
        truck_drift = torch.linalg.norm(truck_after_hold - truck_before_hold, dim=1)
        ids = env._cylinder_joint_ids
        torque = env.robot.data.computed_torque[:, ids].abs()
        effort = env.robot.data.joint_effort_limits[:, ids].abs().clamp(min=1.0)
        saturated = (torque >= 0.999 * effort).any(dim=1)
        valid = (
            (reach <= float(a.reach_tol_m))
            & (hold_drift <= float(a.hold_tol_m))
            & (world_z >= float(a.height_min_m))
            & (world_z <= float(a.height_max_m))
            & (head_alignment >= float(a.head_min_alignment))
            & (truck_drift <= float(a.truck_drift_tol_m))
            & (~saturated)
        )

        rows = []
        for i in range(64):
            rows.append({
                "valid": bool(valid[i].item()),
                "cylinder_cmd": target[i].cpu().tolist(),
                "cylinder_measured": x1[i].cpu().tolist(),
                "cutter_truck": tool[i].cpu().tolist(),
                "cutter_world_z": float(world_z[i].item()),
                "head_alignment": float(head_alignment[i].item()),
                "reach_error_m": float(reach[i].item()),
                "hold_drift_m": float(hold_drift[i].item()),
                "truck_drift_m": float(truck_drift[i].item()),
                "saturated": bool(saturated[i].item()),
            })
        valid_np = valid.cpu().numpy()
        tool_np = tool.cpu().numpy()[valid_np]
        cmd_np = grid[valid_np]
        payload = {
            "schema_version": 1,
            "note": "Direct-PhysX verified head-down workspace. Cylinder 2 is swept for height; "
                    "cylinder 3 is explicitly swept for extension/contraction.",
            "grid": {"c1": c1_values.tolist(), "c2": c2_values.tolist(), "c3": c3_values.tolist(), "c4": float(a.c4)},
            "filters": {
                "reach_tol_m": a.reach_tol_m, "hold_tol_m": a.hold_tol_m,
                "height_min_m": a.height_min_m, "height_max_m": a.height_max_m,
                "head_min_alignment": a.head_min_alignment,
                "truck_drift_tol_m": a.truck_drift_tol_m,
            },
            "num_valid": int(valid.sum().item()),
            "valid_tool_span_m": np.ptp(tool_np, axis=0).tolist() if len(tool_np) else [0, 0, 0],
            "valid_cylinder_span_m": np.ptp(cmd_np, axis=0).tolist() if len(cmd_np) else [0, 0, 0, 0],
            "samples": rows,
        }
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(
            f"[WORKSPACE] valid={payload['num_valid']}/64 "
            f"tool_span={np.round(payload['valid_tool_span_m'], 3).tolist()} "
            f"cylinder_span={np.round(payload['valid_cylinder_span_m'], 3).tolist()} "
            f"wrote={out}", flush=True
        )
        if payload["num_valid"] < 16 or payload["valid_cylinder_span_m"][2] < 0.05:
            raise SystemExit(2)
    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close()
