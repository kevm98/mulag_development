#!/usr/bin/env python3
"""Verify an ordered candidate arc pose-by-pose in one reset environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--map", default=str(ROOT / "ID_FD_pipline/models/ground_band_map.json"))
p.add_argument("--poses", type=int, default=25)
p.add_argument("--settle_steps", type=int, default=40)
p.add_argument("--target_world_z", type=float, default=-0.05)
p.add_argument("--height_iterations", type=int, default=4)
p.add_argument("--seed", type=int, default=17)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver.ground_paths import GroundPathPlanner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402

path = Path(a.map).expanduser().resolve()
payload = json.loads(path.read_text())
candidate = GroundPathPlanner(path, use_standalone=False).arc()
indices = np.unique(np.linspace(0, len(candidate.goals) - 1, int(a.poses), dtype=np.int64))

cfg = parse_env_cfg("Kevin-Mulag-IDFD-EETracking-PLAY-v0", device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 100000.0
env = gym.make("Kevin-Mulag-IDFD-EETracking-PLAY-v0", cfg=cfg).unwrapped

try:
    driver = DirectCylinderDriver(env)
    verified: list[dict[str, object]] = []
    for output_index, candidate_index in enumerate(indices):
        target = torch.as_tensor(
            candidate.direct_targets[candidate_index], device=env.device, dtype=torch.float32
        ).reshape(1, 4)
        # Cylinder 1 is the measured height-dominant direction (cached stepped
        # dZ/dx ~= +9.89). Correct the coarse grid's clearance error, then reset
        # and verify the adjusted target from the same initial state.
        for _iteration in range(int(a.height_iterations)):
            env.reset(seed=int(a.seed))
            wheel_ids = env._wheel_joint_ids
            env.robot.write_joint_stiffness_to_sim(6.0e8, joint_ids=wheel_ids)
            env.robot.write_joint_damping_to_sim(1.0e6, joint_ids=wheel_ids)
            current = env.robot.data.joint_pos[:, wheel_ids].detach().clone()
            env.robot.set_joint_position_target(current, joint_ids=wheel_ids)
            driver.settle(int(a.settle_steps))
            for _ in range(int(a.settle_steps)):
                driver.step(target)
            measured_z = float(env._compute_cutter_world_z()[0].item())
            error_z = float(a.target_world_z) - measured_z
            if abs(error_z) <= 0.01:
                break
            target[:, 1] = torch.clamp(
                target[:, 1] + error_z / 9.89,
                driver.lo[:, 1],
                driver.hi[:, 1],
            )
        measured = driver.measured_x()[0].detach().cpu().tolist()
        cutter = driver.ee_local()[0].detach().cpu().tolist()
        world_z = float(env._compute_cutter_world_z()[0].item())
        verified.append(
            {
                "target_x": target[0].detach().cpu().tolist(),
                "cylinder_x": measured,
                "cutter_truck": cutter,
                "cutter_world_z": world_z,
                "ground": abs(world_z) <= 0.10,
                "candidate_distance_m": float(candidate.cumulative_m[candidate_index]),
            }
        )
        print(
            f"[ARC-VERIFY] {output_index + 1}/{len(indices)} "
            f"candidate_s={candidate.cumulative_m[candidate_index]:.3f}m world_z={world_z:+.3f}m",
            flush=True,
        )
    ground = [sample for sample in verified if bool(sample["ground"])]
    payload["standalone_arc_samples"] = verified
    payload["standalone_arc_ground"] = ground
    payload["standalone_arc_reset_each_pose"] = True
    path.write_text(json.dumps(payload, indent=2))
    if len(ground) < 4:
        raise RuntimeError(f"Only {len(ground)}/{len(verified)} standalone arc poses reach ground")
    points = np.asarray([sample["cutter_truck"] for sample in ground], dtype=np.float64)
    ordered_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    print(
        f"[ARC-VERIFY][DONE] ground={len(ground)}/{len(verified)} ordered_length={ordered_length:.3f}m map={path}",
        flush=True,
    )
finally:
    env.close()
    simulation_app.close()
