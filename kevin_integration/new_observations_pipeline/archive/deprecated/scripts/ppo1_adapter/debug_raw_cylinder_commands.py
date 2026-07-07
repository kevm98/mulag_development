"""
Debug: Check what cylinder commands the analytical adapter produces
and what motion results from applying them directly to the robot.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--adapter_checkpoint", type=str, default="outputs/ppo1_adapter/analytical_adapter.pt")
parser.add_argument("--num_steps", type=int, default=10)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
from pathlib import Path
ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from kevin_integration.new_observations_pipeline.tasks.flat_ee_target_task_no_adapter import FlatEETargetTask
from kevin_integration.new_observations_pipeline.ppo1_adapter.analytical_adapter import AnalyticalAdapter

def test_raw_cylinder_commands():
    """Test if raw cylinder commands produce expected motion."""

    print("="*80)
    print("RAW CYLINDER COMMAND TEST")
    print("="*80)

    # Create basic task environment
    cfg = FlatEETargetTask.Config()
    cfg.num_envs = args_cli.num_envs
    cfg.env_spacing = 10.0
    cfg.episode_length_s = 10.0
    cfg.sim.dt = 1/120
    cfg.sim.render_interval = 2
    cfg.sim.device = "cuda:0"

    env = FlatEETargetTask(cfg)

    print(f"\n[1/3] Environment created: {args_cli.num_envs} envs")

    # Load analytical adapter
    adapter = AnalyticalAdapter.load(args_cli.adapter_checkpoint, device="cuda:0")
    adapter.eval()

    print(f"[2/3] Adapter loaded: {args_cli.adapter_checkpoint}")

    # Reset env and let it settle
    print(f"\n[3/3] Testing cylinder commands...")
    env.reset()
    for _ in range(50):  # settle
        action = torch.zeros((args_cli.num_envs, 5), device="cuda:0")
        env.step(action)

    # Get initial state
    ee_data = env.robot.data.body_pos_w[:, env.ee_body_idx, :2]  # Y,Z position
    ee_init = ee_data.clone()

    cyl_data = env.robot.data.joint_pos[:, env.cylinder_indices]
    cyl_init = cyl_data.clone()

    print(f"\nInitial state:")
    print(f"  EE (Y,Z): {ee_init[0].cpu().numpy()}")
    print(f"  Cylinders: {cyl_init[0].cpu().numpy()}")

    # Test different desired motions
    test_cases = [
        ("Zero", torch.tensor([[0.0, 0.0]], device="cuda:0")),
        ("+Y only", torch.tensor([[0.005, 0.0]], device="cuda:0")),  # +5mm Y
        ("-Y only", torch.tensor([[-0.005, 0.0]], device="cuda:0")),
        ("+Z only", torch.tensor([[0.0, 0.005]], device="cuda:0")),
        ("-Z only", torch.tensor([[0.0, -0.005]], device="cuda:0")),
    ]

    for test_name, desired_ee_delta in test_cases:
        # Reset
        env.reset()
        for _ in range(50):
            action = torch.zeros((args_cli.num_envs, 5), device="cuda:0")
            env.step(action)

        ee_start = env.robot.data.body_pos_w[:, env.ee_body_idx, :2]
        cyl_start = env.robot.data.joint_pos[:, env.cylinder_indices]
        cyl_vel = env.robot.data.joint_vel[:, env.cylinder_indices]

        # Get adapter output
        prev_delta = torch.zeros((args_cli.num_envs, 4), device="cuda:0")
        cylinder_delta = adapter(
            desired_ee_delta,
            ee_start,
            cyl_start,
            cyl_vel,
            prev_delta,
            enforce_limits=True,
        )

        print(f"\n[{test_name}]")
        print(f"  Desired EE delta: {desired_ee_delta[0].cpu().numpy() * 1000} mm")
        print(f"  Adapter output: {cylinder_delta[0].cpu().numpy() * 1000} mm")

        # Apply cylinder commands for N steps
        for step in range(args_cli.num_steps):
            # Construct 5D action (4 cylinders + 1 tool joint)
            action = torch.zeros((args_cli.num_envs, 5), device="cuda:0")

            # Compute cylinder targets as: current + delta
            current_cyl = env.robot.data.joint_pos[:, env.cylinder_indices]
            target_cyl = current_cyl + cylinder_delta
            action[:, :4] = target_cyl
            action[:, 4] = 0.0  # tool joint neutral

            # Step
            obs, rewards, terminated, truncated, info = env.step(action)

        # Measure result
        ee_end = env.robot.data.body_pos_w[:, env.ee_body_idx, :2]
        cyl_end = env.robot.data.joint_pos[:, env.cylinder_indices]

        ee_motion = (ee_end - ee_start) * 1000  # mm
        cyl_motion = (cyl_end - cyl_start) * 1000  # mm

        print(f"  Measured EE motion: {ee_motion[0].cpu().numpy()} mm")
        print(f"  Measured cylinder motion: {cyl_motion[0].cpu().numpy()} mm")

        # Check sign matching
        if desired_ee_delta[0, 0].abs() > 1e-6:  # Y component
            y_match = (desired_ee_delta[0, 0] * ee_motion[0, 0]) > 0
            print(f"  Y sign match: {'✓' if y_match else '✗'}")
        if desired_ee_delta[0, 1].abs() > 1e-6:  # Z component
            z_match = (desired_ee_delta[0, 1] * ee_motion[0, 1]) > 0
            print(f"  Z sign match: {'✓' if z_match else '✗'}")

    env.close()
    print("\n" + "="*80)

if __name__ == "__main__":
    test_raw_cylinder_commands()
    simulation_app.close()
