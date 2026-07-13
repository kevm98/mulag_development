#!/usr/bin/env python3
"""One-shot measurement of the cutter Jacobian at a settled mowing pose.

The arm is first driven to the ground with the normal ID/FD cylinder cascade.
Only then is the 3x4 truck-frame cutter Jacobian measured by central-difference
position targets with stepped PhysX.  The process exits immediately afterwards;
the cached matrix is the only object used by online DLS control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=17)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 1:
    raise ValueError("Ground Jacobian measurement requires --num_envs 1")

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController  # noqa: E402
from ID_FD_pipline.dls_solver.ee_cascade import EECascadeController  # noqa: E402
from ID_FD_pipline.dls_solver.kinematics import ee_xyz_jacobian_stepped  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
cfg.seed = int(args.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 300.0
env = gym.make(TASK, cfg=cfg).unwrapped

try:
    env.reset(seed=int(args.seed))
    zero = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
    for _ in range(60):
        env.step(zero)

    reset_cache = ROOT / "ID_FD_pipline/models/ee_jacobian_truck3x4.json"
    reset_jacobian = torch.tensor(
        json.loads(reset_cache.read_text())["jacobian"],
        dtype=torch.float32,
        device=env.device,
    )
    cylinder = CylinderDLSController(
        env,
        CylinderDLSConfig(
            invert_learned_pipeline=False,
            error_gain=0.10,
            max_desired_delta_m=0.01,
            max_action=1.0,
            action_smoothing=0.35,
        ),
    )
    cascade = EECascadeController(
        env,
        cylinder_controller=cylinder,
        jacobian=reset_jacobian,
        ee_feedback_gain=0.30,
        max_cylinder_lead_m=0.01,
    )
    cascade.reset()

    ground_goal = env._compute_ee_pos_local().detach().clone()
    cutter_z = float(env._compute_cutter_world_z()[0].item())
    ground_goal[:, 2] -= max(cutter_z - 0.05, 0.0)
    for _ in range(300):
        cylinder.step(cascade.cylinder_goal(ground_goal))

    measured_ee = env._compute_ee_pos_local()[0].detach().cpu()
    measured_x = env._refresh_cylinder_state()[0][0].detach().cpu()
    settled_z = float(env._compute_cutter_world_z()[0].item())
    print(
        f"[GROUND-JAC] settled cutter_world_z={settled_z:+.4f}m "
        f"cutter_truck={measured_ee.tolist()} cylinder_x={measured_x.tolist()}",
        flush=True,
    )

    jacobian = ee_xyz_jacobian_stepped(env)[0].detach().cpu()
    singular_values = torch.linalg.svdvals(jacobian)
    rank = int((singular_values > 0.01 * singular_values[0]).sum().item())
    output = ROOT / "ID_FD_pipline/models/ee_jacobian_ground_truck3x4.json"
    output.write_text(
        json.dumps(
            {
                "jacobian": jacobian.tolist(),
                "svs": singular_values.tolist(),
                "rank": rank,
                "frame": "truck base_link, cutter tip, learned cylinder x",
                "measurement": "ID/FD descent followed by central-difference position targets and stepped PhysX",
                "cutter_world_z": settled_z,
                "cutter_truck": measured_ee.tolist(),
                "cylinder_x": measured_x.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[GROUND-JAC] saved {output} rank={rank}/3 "
        f"SVs={[round(float(value), 6) for value in singular_values]}",
        flush=True,
    )
finally:
    env.close()
    simulation_app.close()
