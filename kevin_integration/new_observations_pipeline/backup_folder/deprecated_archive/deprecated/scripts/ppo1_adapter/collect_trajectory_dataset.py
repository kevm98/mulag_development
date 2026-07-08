"""
Collect adapter training data using trajectory-based approach.

Instead of random single-step commands, we:
1. Pick random EE target deltas
2. Apply proportional cylinder control for multiple steps
3. Record successful trajectories
4. Extract (desired_delta, state, avg_cylinder_cmd) tuples

This creates training data with actual cause-effect relationship.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch
import numpy as np
from datetime import datetime

# Parse args BEFORE importing IsaacLab
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--num_trajectories", type=int, default=2000, help="Number of trajectories per env")
parser.add_argument("--trajectory_steps", type=int, default=20, help="Steps per trajectory")
parser.add_argument("--max_ee_target", type=float, default=0.05, help="Max EE target magnitude (m)")
parser.add_argument("--cylinder_gain", type=float, default=0.5, help="Proportional gain for cylinder control")
parser.add_argument("--max_delta_cylinder", type=float, default=0.0025, help="Max cylinder command per step")
parser.add_argument("--settle_steps", type=int, default=100)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0)
parser.add_argument("--output_dir", type=str, default="outputs/ppo1_adapter")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")

# Add IsaacLab launcher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Create launcher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import IsaacLab modules
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.math import quat_apply_inverse

print("="*80)
print("[collect_trajectory_dataset] Multi-Step Trajectory Data Collection")
print("="*80)


def create_env(args):
    """Create environment for data collection."""
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_config import (
        MulagUpperLevelEnvCfg,
    )

    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.device = args.device
    cfg.sim.device = args.device
    cfg.lowlevel_actuator_backend = args.backend
    cfg.explicit_pid_kp_scale = args.kp_scale
    cfg.explicit_pid_kd_scale = args.kd_scale
    cfg.bias_calibration_time_s = args.bias_calibration_time_s
    cfg.allow_debug_manual_cylinder_targets = True

    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import (
        MulagUpperLevelEnv,
    )
    from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
    apply_kevin_sim_memory_optimizations(cfg.sim)

    return MulagUpperLevelEnv(cfg=cfg)


def get_ee_yz(env):
    """Extract end-effector YZ position in root frame."""
    robot = env.unwrapped.scene["robot"]
    ee_id = env.unwrapped._ee_body_id

    ee_pos_w = robot.data.body_state_w[:, ee_id, :3]
    root_body = robot.data.root_state_w
    root_pos_w = root_body[:, :3]
    root_quat_w = root_body[:, 3:7]

    ee_pos_rel = quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)
    return ee_pos_rel[:, [1, 2]]  # Y, Z


def simple_jacobian_approximation(cyl_delta, max_delta=0.0025):
    """
    Simple approximation: assume roughly equal contribution from all cylinders.
    EE_Y ≈ (cyl1 - cyl2) / 2 * scale
    EE_Z ≈ (cyl3 - cyl4) / 2 * scale

    Returns: approx cylinder deltas for desired EE delta
    """
    # This is a rough approximation - the learned adapter should do better
    scale = 2.0  # Rough guess at kinematic ratio
    desired_ee_y, desired_ee_z = cyl_delta[0], cyl_delta[1]

    # Differential cylinders
    delta_diff_y = desired_ee_y * scale
    delta_diff_z = desired_ee_z * scale

    # Distribute to cylinders
    cyl_cmd = torch.zeros(4, device=cyl_delta.device)
    cyl_cmd[0] = delta_diff_y / 2  # +Y contribution
    cyl_cmd[1] = -delta_diff_y / 2  # -Y contribution
    cyl_cmd[2] = delta_diff_z / 2  # +Z contribution
    cyl_cmd[3] = -delta_diff_z / 2  # -Z contribution

    # Clip to safe range
    cyl_cmd = torch.clamp(cyl_cmd, -max_delta, max_delta)
    return cyl_cmd


def collect_trajectory(env, target_ee_delta, args):
    """
    Execute one trajectory trying to achieve target_ee_delta.

    Returns:
        dict with trajectory data or None if failed
    """
    device = args.device
    B = env.num_envs
    cyl_ids = env.unwrapped._cylinder_joint_ids
    robot = env.unwrapped.scene["robot"]

    # Initial state
    initial_ee = get_ee_yz(env)
    initial_cyl_pos = robot.data.joint_pos[:, cyl_ids]

    # Track trajectory
    cylinder_commands = []
    ee_positions = [initial_ee.clone()]

    # Execute trajectory with proportional control
    for step in range(args.trajectory_steps):
        current_ee = get_ee_yz(env)
        current_cyl_pos = robot.data.joint_pos[:, cyl_ids]
        current_cyl_vel = robot.data.joint_vel[:, cyl_ids]

        # Compute error
        ee_error = target_ee_delta - (current_ee - initial_ee)

        # Simple proportional control using approximation
        # In reality, the adapter should learn the true mapping
        desired_step_delta = ee_error * args.cylinder_gain

        # Approximate cylinder command
        cylinder_delta_batch = []
        for i in range(B):
            cyl_delta = simple_jacobian_approximation(
                desired_step_delta[i],
                args.max_delta_cylinder
            )
            cylinder_delta_batch.append(cyl_delta)

        cylinder_delta = torch.stack(cylinder_delta_batch)

        # Execute command
        cylinder_target = current_cyl_pos + cylinder_delta
        tool_target = torch.zeros((B, 1), device=device)
        action = torch.cat([cylinder_target, tool_target], dim=-1)

        obs, rewards, terminated, truncated, info = env.step(action)

        # Record
        cylinder_commands.append(cylinder_delta.clone())
        ee_positions.append(get_ee_yz(env))

    # Final measurement
    final_ee = ee_positions[-1]
    achieved_delta = final_ee - initial_ee

    # Compute average cylinder command over trajectory
    avg_cylinder_cmd = torch.stack(cylinder_commands).mean(dim=0)  # (B, 4)

    # Success criterion: achieved at least 50% of target
    success = (achieved_delta - target_ee_delta).norm(dim=-1) < target_ee_delta.norm(dim=-1) * 0.5

    return {
        "initial_ee": initial_ee,  # (B, 2)
        "initial_cyl_pos": initial_cyl_pos,  # (B, 4)
        "target_delta": target_ee_delta,  # (B, 2)
        "achieved_delta": achieved_delta,  # (B, 2)
        "avg_cylinder_cmd": avg_cylinder_cmd,  # (B, 4)
        "success": success,  # (B,)
    }


def main(args):
    print(f"\n[1/4] Creating environment with {args.num_envs} parallel envs...")
    env = create_env(args)
    env.reset()

    # Settle
    device = args.device
    B = args.num_envs
    print(f"\n[2/4] Settling for {args.settle_steps} steps...")
    zero_action = torch.zeros((B, 5), device=device)
    for _ in range(args.settle_steps):
        env.step(zero_action)

    print(f"\n[3/4] Collecting {args.num_trajectories} trajectories...")

    collected_data = {
        "initial_ee": [],
        "initial_cyl_pos": [],
        "target_delta": [],
        "achieved_delta": [],
        "avg_cylinder_cmd": [],
    }

    num_collected = 0
    num_attempted = 0

    for traj_idx in range(args.num_trajectories):
        # Random target EE delta
        target_ee_delta = (torch.rand((B, 2), device=device) * 2 - 1) * args.max_ee_target

        # Collect trajectory
        traj_data = collect_trajectory(env, target_ee_delta, args)
        num_attempted += B

        # Filter successful trajectories
        success_mask = traj_data["success"]
        num_success = success_mask.sum().item()

        if num_success > 0:
            for key in collected_data.keys():
                collected_data[key].append(traj_data[key][success_mask])
            num_collected += num_success

        if (traj_idx + 1) % 100 == 0:
            print(f"  Trajectory {traj_idx+1}/{args.num_trajectories}: {num_collected}/{num_attempted} successful ({num_collected/num_attempted*100:.1f}%)")

        # Reset periodically
        if (traj_idx + 1) % 50 == 0:
            env.reset()
            for _ in range(args.settle_steps):
                env.step(zero_action)

    print(f"\n[4/4] Saving dataset...")

    # Concatenate all data
    for key in collected_data.keys():
        collected_data[key] = torch.cat(collected_data[key], dim=0)

    print(f"  Collected {num_collected} successful samples ({num_collected/num_attempted*100:.1f}% success rate)")
    print(f"  Target delta range: Y[{collected_data['target_delta'][:,0].min()*1000:.1f}, {collected_data['target_delta'][:,0].max()*1000:.1f}]mm, Z[{collected_data['target_delta'][:,1].min()*1000:.1f}, {collected_data['target_delta'][:,1].max()*1000:.1f}]mm")
    print(f"  Achieved delta range: Y[{collected_data['achieved_delta'][:,0].min()*1000:.1f}, {collected_data['achieved_delta'][:,0].max()*1000:.1f}]mm, Z[{collected_data['achieved_delta'][:,1].min()*1000:.1f}, {collected_data['achieved_delta'][:,1].max()*1000:.1f}]mm")

    # Save
    output_path = Path(args.output_dir) / "trajectory_adapter_dataset.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "dataset": collected_data,
        "metadata": {
            "num_samples": num_collected,
            "success_rate": num_collected / num_attempted,
            "num_envs": args.num_envs,
            "num_trajectories": args.num_trajectories,
            "trajectory_steps": args.trajectory_steps,
            "max_ee_target": args.max_ee_target,
            "timestamp": datetime.now().isoformat(),
        }
    }, output_path)

    print(f"  Saved to: {output_path}")
    print("="*80)

    env.close()


if __name__ == "__main__":
    main(args)
    simulation_app.close()
