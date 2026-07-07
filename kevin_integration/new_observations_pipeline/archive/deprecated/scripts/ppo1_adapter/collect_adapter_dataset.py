#!/usr/bin/env python3
"""
Collect adapter dataset in IsaacLab.

Collects paired samples of:
    - State: EE position, cylinder positions/velocities
    - Command: Random smooth cylinder delta
    - Result: Measured EE motion

Dataset is used to train:
    1. Forward model: (state, command) → EE motion
    2. Adapter: (desired EE motion, state) → cylinder command
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch
import numpy as np
from tqdm import tqdm

# Parse args before importing IsaacLab
parser = argparse.ArgumentParser(description="Collect adapter training dataset")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of steps per environment")
parser.add_argument("--max_delta_cylinder", type=float, default=0.0025, help="Max cylinder command (m)")
parser.add_argument("--action_smoothing_beta", type=float, default=0.85, help="Action smoothing factor")
parser.add_argument("--action_rate_limit", type=float, default=0.08, help="Max action change per step")
parser.add_argument("--settle_steps", type=int, default=30, help="Post-reset settle steps")
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias", help="Lower-level controller")
parser.add_argument("--kp_scale", type=float, default=0.1, help="PID proportional gain scale")
parser.add_argument("--kd_scale", type=float, default=0.5, help="PID derivative gain scale")
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0, help="Bias calibration time")
parser.add_argument("--output_dir", type=str, default="outputs/ppo1_adapter", help="Output directory")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Add IsaacLab launcher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args.headless = True  # Force headless
args.device = args.device if hasattr(args, 'device') else "cuda:0"

# Launch IsaacLab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app



# Now import IsaacLab components
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

def create_env(args):
    """Create Mulag environment for data collection."""
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_scripted_vehicle_motion = False
    cfg.enable_startup_summary = False
    cfg.log_target_delta_xdot_clipping = False
    cfg.allow_debug_manual_cylinder_targets = True
    cfg.lowlevel_actuator_backend = args.backend
    cfg.explicit_pid_kp_scale = args.kp_scale
    cfg.explicit_pid_kd_scale = args.kd_scale
    cfg.explicit_pid_effort_scale = 1.0
    cfg.explicit_force_rate_limit_fraction = 0.01
    cfg.explicit_pd_velocity_filter_beta = 0.8
    cfg.bias_calibration_time_s = args.bias_calibration_time_s
    cfg.bias_average_window_s = 0.5
    cfg.bias_calibration_ki_scale = 5.0
    cfg.bias_calibration_requires_static = False
    cfg.bias_runtime_ki_scale = 0.0
    cfg.bias_online_update_rate = 0.0
    cfg.debug_bias_shapes = False
    cfg.device = args.device
    cfg.sim.device = args.device

    apply_kevin_sim_memory_optimizations(cfg.sim)

    env = MulagUpperLevelEnv(cfg=cfg)
    return env


def collect_dataset(args):
    """Collect adapter dataset with random smooth cylinder commands."""
    print("[collect_adapter_dataset] Starting data collection")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  max_delta_cylinder: {args.max_delta_cylinder}")
    print(f"  action_smoothing_beta: {args.action_smoothing_beta}")
    print(f"  backend: {args.backend}")

    # Create environment
    env = create_env(args)
    uw = env.unwrapped if hasattr(env, 'unwrapped') else env

    # Get robot references
    robot = uw.scene["robot"]
    cyl_ids = uw._cylinder_joint_ids
    ee_id = uw._ee_body_id
    device = args.device
    B = args.num_envs

    # Storage buffers
    dataset = {
        # State at time k
        "ee_yz_k": [],
        "cylinder_pos_k": [],
        "cylinder_vel_k": [],
        "previous_cylinder_delta_k": [],
        # Command at time k
        "command_cylinder_delta_k": [],
        # Result at time k+1
        "measured_delta_ee_yz": [],
        "measured_delta_cylinder": [],
    }

    # Rejection statistics
    total_samples = 0
    accepted_samples = 0
    rejection_reasons = {
        "ee_jump_too_large": 0,
        "cylinder_near_limit": 0,
        "nan_or_inf": 0,
        "root_motion_too_large": 0,
        "ee_below_ground": 0,
        "command_saturation": 0,
    }

    # Safety thresholds
    max_ee_jump = 0.12  # 120mm (relaxed from 80mm)
    cylinder_limit_margin = 0.02  # 20mm
    cylinder_min = torch.tensor([-0.2, -0.2, -0.2, -0.2], device=device)
    cylinder_max = torch.tensor([0.2, 0.2, 0.2, 0.2], device=device)
    max_root_motion = 0.10  # 100mm (relaxed from 50mm)
    min_ee_z_clearance = 0.05  # 50mm above ground (relaxed from 200mm)
    command_saturation_threshold = 0.002  # 2mm (relaxed from 1mm)

    # Reset environment
    print("[collect_adapter_dataset] Resetting environment...")
    env.reset()

    # Settle for bias calibration
    print(f"[collect_adapter_dataset] Settling for {args.settle_steps} steps...")
    zero_action = torch.zeros((B, 5), device=device)
    for _ in range(args.settle_steps):
        env.step(zero_action)

    # Get initial EE position
    from isaaclab.utils.math import quat_apply_inverse

    def get_ee_yz():
        """Get EE position in root frame."""
        ee_body = robot.data.body_state_w[:, ee_id, :]
        ee_pos_w = ee_body[:, :3]
        root_state = robot.data.root_state_w
        root_pos_w = root_state[:, :3]
        root_quat_w = root_state[:, 3:7]
        ee_pos_rel = quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)
        return ee_pos_rel[:, [1, 2]]  # Y, Z

    # Check initial state
    initial_ee_yz = get_ee_yz()
    initial_cyl_pos = robot.data.joint_pos[:, cyl_ids]
    print(f"[collect_adapter_dataset] Initial EE Y: {initial_ee_yz[:, 0].mean().item():.3f} ± {initial_ee_yz[:, 0].std().item():.3f}")
    print(f"[collect_adapter_dataset] Initial EE Z: {initial_ee_yz[:, 1].mean().item():.3f} ± {initial_ee_yz[:, 1].std().item():.3f}")
    print(f"[collect_adapter_dataset] Initial Cyl: {initial_cyl_pos.mean(dim=0).cpu().numpy()}")

    prev_ee_yz = get_ee_yz()
    prev_root_pos = robot.data.root_state_w[:, :3].clone()
    prev_cylinder_delta = torch.zeros((B, 4), device=device)

    # Generate random smooth commands
    print("[collect_adapter_dataset] Collecting data...")
    for step in tqdm(range(args.num_steps), desc="Data collection"):
        # Get current state
        cylinder_pos = robot.data.joint_pos[:, cyl_ids]
        cylinder_vel = robot.data.joint_vel[:, cyl_ids]
        current_ee_yz = get_ee_yz()
        current_root_pos = robot.data.root_state_w[:, :3]

        # Periodic progress report
        if step > 0 and step % 200 == 0:
            current_rate = accepted_samples / total_samples if total_samples > 0 else 0
            print(f"\n[Step {step}] Acceptance: {accepted_samples}/{total_samples} ({100*current_rate:.1f}%)", flush=True)

        # Generate random smooth cylinder command
        if step == 0:
            # First step: random command
            random_command = torch.randn((B, 4), device=device) * args.max_delta_cylinder
        else:
            # Subsequent steps: smooth from previous command
            random_noise = torch.randn((B, 4), device=device) * args.max_delta_cylinder
            random_command = (
                args.action_smoothing_beta * prev_cylinder_delta
                + (1 - args.action_smoothing_beta) * random_noise
            )

        # Apply rate limit
        if args.action_rate_limit > 0 and step > 0:
            delta_command = random_command - prev_cylinder_delta
            delta_command = torch.clamp(
                delta_command,
                -args.action_rate_limit,
                args.action_rate_limit,
            )
            random_command = prev_cylinder_delta + delta_command

        # Clamp to max magnitude
        random_command = torch.clamp(
            random_command,
            -args.max_delta_cylinder,
            args.max_delta_cylinder,
        )

        # Convert to cylinder target
        cylinder_target = cylinder_pos + random_command

        # Package as 5D action (4 cylinders + 1 tool)
        tool_target = torch.zeros((B, 1), device=device)
        action = torch.cat([cylinder_target, tool_target], dim=-1)

        # Execute step (DirectRLEnv returns 5 values)
        obs, rewards, terminated, truncated, info = env.step(action)
        dones = terminated | truncated

        # Get new state
        new_cylinder_pos = robot.data.joint_pos[:, cyl_ids]
        new_ee_yz = get_ee_yz()
        new_root_pos = robot.data.root_state_w[:, :3]

        # Compute measurements
        measured_delta_ee_yz = new_ee_yz - current_ee_yz
        measured_delta_cylinder = new_cylinder_pos - cylinder_pos
        ee_jump = measured_delta_ee_yz.norm(dim=-1)
        root_motion = (new_root_pos - current_root_pos).norm(dim=-1)

        # Check for valid samples
        valid_mask = torch.ones(B, dtype=torch.bool, device=device)

        # Reject: EE jump too large (indicates sim instability)
        ee_jump_too_large = ee_jump > max_ee_jump
        valid_mask &= ~ee_jump_too_large
        rejection_reasons["ee_jump_too_large"] += ee_jump_too_large.sum().item()

        # Reject: Cylinder near limit
        near_min = (new_cylinder_pos < cylinder_min + cylinder_limit_margin).any(dim=1)
        near_max = (new_cylinder_pos > cylinder_max - cylinder_limit_margin).any(dim=1)
        cylinder_near_limit = near_min | near_max
        valid_mask &= ~cylinder_near_limit
        rejection_reasons["cylinder_near_limit"] += cylinder_near_limit.sum().item()

        # Reject: NaN or inf
        has_nan_inf = (
            torch.isnan(measured_delta_ee_yz).any(dim=1)
            | torch.isinf(measured_delta_ee_yz).any(dim=1)
            | torch.isnan(measured_delta_cylinder).any(dim=1)
            | torch.isinf(measured_delta_cylinder).any(dim=1)
        )
        valid_mask &= ~has_nan_inf
        rejection_reasons["nan_or_inf"] += has_nan_inf.sum().item()

        # NOTE: Removed root_motion, ee_below_ground, command_saturation filters
        # These were rejecting valid operational data due to coordinate frame assumptions

        # Store valid samples
        total_samples += B
        n_valid = valid_mask.sum().item()
        accepted_samples += n_valid

        if n_valid > 0:
            # Store state at k
            dataset["ee_yz_k"].append(current_ee_yz[valid_mask].cpu())
            dataset["cylinder_pos_k"].append(cylinder_pos[valid_mask].cpu())
            dataset["cylinder_vel_k"].append(cylinder_vel[valid_mask].cpu())
            dataset["previous_cylinder_delta_k"].append(prev_cylinder_delta[valid_mask].cpu())

            # Store command at k
            dataset["command_cylinder_delta_k"].append(random_command[valid_mask].cpu())

            # Store result at k+1
            dataset["measured_delta_ee_yz"].append(measured_delta_ee_yz[valid_mask].cpu())
            dataset["measured_delta_cylinder"].append(measured_delta_cylinder[valid_mask].cpu())

        # Update for next step
        prev_ee_yz = new_ee_yz
        prev_root_pos = new_root_pos
        prev_cylinder_delta = random_command

    # Concatenate all samples
    print("[collect_adapter_dataset] Concatenating samples...")
    for key in dataset.keys():
        if len(dataset[key]) > 0:
            dataset[key] = torch.cat(dataset[key], dim=0)
        else:
            dataset[key] = torch.tensor([])

    # Compute statistics
    print("\n" + "="*80)
    print("[collect_adapter_dataset] Dataset Statistics")
    print("="*80)
    print(f"Total samples: {total_samples}")
    print(f"Accepted samples: {accepted_samples}")
    print(f"Acceptance rate: {100 * accepted_samples / total_samples:.1f}%")
    print(f"\nRejection reasons:")
    for reason, count in rejection_reasons.items():
        if total_samples > 0:
            print(f"  {reason}: {count} ({100 * count / total_samples:.1f}%)")

    if accepted_samples > 0:
        print(f"\nDataset shape: {dataset['ee_yz_k'].shape[0]} samples")
        print(f"\nDelta EE ranges:")
        print(f"  delta_y: [{dataset['measured_delta_ee_yz'][:, 0].min():.4f}, {dataset['measured_delta_ee_yz'][:, 0].max():.4f}] m")
        print(f"  delta_z: [{dataset['measured_delta_ee_yz'][:, 1].min():.4f}, {dataset['measured_delta_ee_yz'][:, 1].max():.4f}] m")
        print(f"\nDelta cylinder ranges:")
        for i in range(4):
            print(f"  delta_cyl{i+1}: [{dataset['command_cylinder_delta_k'][:, i].min():.4f}, {dataset['command_cylinder_delta_k'][:, i].max():.4f}] m")

        # Sign balance
        delta_ee = dataset['measured_delta_ee_yz']
        pos_y = (delta_ee[:, 0] > 0.001).sum().item()
        neg_y = (delta_ee[:, 0] < -0.001).sum().item()
        pos_z = (delta_ee[:, 1] > 0.001).sum().item()
        neg_z = (delta_ee[:, 1] < -0.001).sum().item()
        print(f"\nSign balance:")
        print(f"  +Y: {pos_y}, -Y: {neg_y}")
        print(f"  +Z: {pos_z}, -Z: {neg_z}")

    # Save dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "adapter_dataset.pt"

    torch.save({
        "dataset": dataset,
        "metadata": {
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "total_samples": total_samples,
            "accepted_samples": accepted_samples,
            "acceptance_rate": accepted_samples / total_samples if total_samples > 0 else 0,
            "rejection_reasons": rejection_reasons,
            "max_delta_cylinder": args.max_delta_cylinder,
            "action_smoothing_beta": args.action_smoothing_beta,
            "action_rate_limit": args.action_rate_limit,
            "backend": args.backend,
            "kp_scale": args.kp_scale,
            "kd_scale": args.kd_scale,
        },
    }, output_path)

    print(f"\n[collect_adapter_dataset] Saved dataset to {output_path}")
    print("="*80)

    # Close environment
    env.close()


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    collect_dataset(args)
    simulation_app.close()
