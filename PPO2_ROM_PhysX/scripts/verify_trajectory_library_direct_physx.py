#!/usr/bin/env python3
"""Verify trajectory cylinder seeds against the direct PhysX plant in parallel."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("library", type=Path)
p.add_argument("--shapes", default="line,arc,sine,circle,square")
p.add_argument("--samples_per_shape", type=int, default=61)
p.add_argument("--deploy_steps", type=int, default=240)
p.add_argument("--settle_steps", type=int, default=120)
p.add_argument("--max_mean_tool_error_m", type=float, default=0.030)
p.add_argument("--max_p95_tool_error_m", type=float, default=0.060)
p.add_argument("--max_tool_error_m", type=float, default=0.090)
p.add_argument("--max_cylinder_error_m", type=float, default=0.010)
p.add_argument("--min_head_alignment", type=float, default=0.98)
p.add_argument(
    "--out",
    default="outputs/ppo2_rom_residual/direct_plant_reachability.json",
)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
selected = tuple(value.strip() for value in a.shapes.split(",") if value.strip())
if a.samples_per_shape < 9:
    p.error("--samples_per_shape must be at least 9")

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
    library_path = a.library.expanduser().resolve()
    payload = json.loads(library_path.read_text())
    unavailable = sorted(set(selected) - set(payload.get("shapes", {})))
    if unavailable:
        raise ValueError(f"library does not contain shapes {unavailable}")

    num_envs = int(a.samples_per_shape)
    cfg = parse_env_cfg(
        "RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0",
        device=a.device,
        num_envs=num_envs,
    )
    cfg.enable_scripted_vehicle_motion = False
    cfg.episode_length_s = 100000.0
    env = gym.make("RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0", cfg=cfg).unwrapped
    report: dict[str, object] = {}
    try:
        env.reset(seed=17)
        driver = DirectCylinderDriver(env)
        wheel_ids = env._wheel_joint_ids
        wheel_target = env.robot.data.joint_pos[:, wheel_ids].detach().clone()
        truck_id = env._truck_base_body_id()
        anchor_xy = env.robot.data.body_link_state_w[:, truck_id, :2].detach().clone()

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

        for shape in selected:
            rows = payload["shapes"][shape]["samples"]
            indices = np.linspace(0, len(rows) - 1, num_envs).round().astype(np.int64)
            target_np = np.asarray([rows[index]["cylinder_cmd"] for index in indices])
            reference_np = np.asarray([rows[index]["cutter_truck"] for index in indices])
            target = torch.as_tensor(target_np, device=env.device, dtype=torch.float32)
            start = driver.measured_x().clone()
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

            measured_x = driver.measured_x()
            measured_tool = env._compute_ee_pos_local()
            measured_clearance = env._compute_cutter_world_z()
            cylinder_error = torch.max(torch.abs(measured_x - target), dim=1).values
            reference = torch.as_tensor(reference_np, device=env.device, dtype=torch.float32)
            tool_error = torch.linalg.norm(measured_tool - reference, dim=1)
            q = env.robot.data.body_link_state_w[:, env._ee_body_id, 3:7]
            q = q / q.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
            axis = quat_apply(
                q,
                torch.tensor([[0.0, 0.0, 1.0]], device=env.device).expand(num_envs, -1),
            )
            head_alignment = -axis[:, 2]
            drift = torch.linalg.norm(
                env.robot.data.body_link_state_w[:, truck_id, :2] - anchor_xy,
                dim=1,
            )
            errors = tool_error.cpu().numpy()
            result = {
                "samples": num_envs,
                "mean_tool_error_m": float(np.mean(errors)),
                "p95_tool_error_m": float(np.percentile(errors, 95)),
                "max_tool_error_m": float(np.max(errors)),
                "max_cylinder_error_m": float(cylinder_error.max().item()),
                "minimum_head_alignment": float(head_alignment.min().item()),
                "max_truck_drift_m": float(drift.max().item()),
                "worst_sample_index": int(indices[int(np.argmax(errors))]),
                "worst_reference_tool": reference_np[int(np.argmax(errors))].tolist(),
                "worst_measured_tool": measured_tool[int(np.argmax(errors))].cpu().tolist(),
                "records": [
                    {
                        "library_sample_index": int(indices[index]),
                        "requested_cylinder": target_np[index].tolist(),
                        "measured_cylinder": measured_x[index].cpu().tolist(),
                        "reference_tool": reference_np[index].tolist(),
                        "measured_tool": measured_tool[index].cpu().tolist(),
                        "measured_clearance": float(measured_clearance[index].item()),
                        "tool_error_m": float(errors[index]),
                        "head_alignment": float(head_alignment[index].item()),
                    }
                    for index in range(num_envs)
                ],
            }
            result["passed"] = bool(
                result["mean_tool_error_m"] <= float(a.max_mean_tool_error_m)
                and result["p95_tool_error_m"] <= float(a.max_p95_tool_error_m)
                and result["max_tool_error_m"] <= float(a.max_tool_error_m)
                and result["max_cylinder_error_m"] <= float(a.max_cylinder_error_m)
                and result["minimum_head_alignment"] >= float(a.min_head_alignment)
                and result["max_truck_drift_m"] <= 0.0011
            )
            report[shape] = result
            compact = {key: value for key, value in result.items() if key != "records"}
            print(f"[DIRECT PLANT] {shape} {json.dumps(compact)}", flush=True)

        passed = all(bool(value["passed"]) for value in report.values())
        output = Path(a.out).expanduser()
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"passed": passed, "shapes": report}, indent=2) + "\n")
        print(f"[DIRECT PLANT] {'PASS' if passed else 'FAIL'} wrote {output}", flush=True)
        if not passed:
            raise SystemExit(2)
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
