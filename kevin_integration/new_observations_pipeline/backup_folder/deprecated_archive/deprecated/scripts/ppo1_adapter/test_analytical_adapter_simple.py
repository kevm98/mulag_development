"""
Simplified test: Check what cylinder commands the analytical adapter produces.
This is a simpler version that just creates one environment and tests cylinder outputs.
"""

import argparse

parser = argparse.ArgumentParser()
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

# Import task config
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1_adapter.stage1_adapter_env_wrapper import Stage1AdapterEnvWrapper
from kevin_integration.new_observations_pipeline.ppo1_adapter.analytical_adapter import AnalyticalAdapter

def test_adapter_commands():
    """Test what cylinder commands the analytical adapter produces."""

    print("="*80)
    print("ANALYTICAL ADAPTER COMMAND TEST")
    print("="*80)

    # Create wrapper environment (1 env for simplicity)
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = 1

    print(f"\n[1/3] Creating environment...")

    env = Stage1AdapterEnvWrapper(
        cfg=cfg,
        adapter_checkpoint_path=args_cli.adapter_checkpoint,
        horizon_steps=50,
        ppo_action_scale=0.02,
        post_reset_settle_steps=50,
    )

    print(f"[2/3] Environment created")

    # Load analytical adapter directly to test outputs
    adapter = AnalyticalAdapter.load(args_cli.adapter_checkpoint, device="cuda:0")
    adapter.eval()

    print(f"\n[3/3] Testing adapter outputs...")

    # Reset env and let it settle
    env.reset()

    # Get state after settle
    ee_yz = torch.zeros((1, 2), device="cuda:0")  # assume centered
    cyl_pos = torch.zeros((1, 4), device="cuda:0")
    cyl_vel = torch.zeros((1, 4), device="cuda:0")
    prev_delta = torch.zeros((1, 4), device="cuda:0")

    # Test different desired motions
    test_cases = [
        ("Zero", torch.tensor([[0.0, 0.0]], device="cuda:0")),
        ("+Y only", torch.tensor([[0.005, 0.0]], device="cuda:0")),  # +5mm Y
        ("-Y only", torch.tensor([[-0.005, 0.0]], device="cuda:0")),
        ("+Z only", torch.tensor([[0.0, 0.005]], device="cuda:0")),
        ("-Z only", torch.tensor([[0.0, -0.005]], device="cuda:0")),
        ("+Y+Z", torch.tensor([[0.005, 0.005]], device="cuda:0")),
        ("+Y-Z", torch.tensor([[0.005, -0.005]], device="cuda:0")),
        ("-Y+Z", torch.tensor([[-0.005, 0.005]], device="cuda:0")),
        ("-Y-Z", torch.tensor([[-0.005, -0.005]], device="cuda:0")),
    ]

    print("\nAdapter Outputs:")
    print("-" * 80)

    for test_name, desired_ee_delta in test_cases:
        # Get adapter output
        cylinder_delta = adapter(
            desired_ee_delta,
            ee_yz,
            cyl_pos,
            cyl_vel,
            prev_delta,
            enforce_limits=True,
        )

        print(f"{test_name:12s} → Desired: [{desired_ee_delta[0,0].item()*1000:+6.1f}, {desired_ee_delta[0,1].item()*1000:+6.1f}] mm" +
              f" → Cylinders: [{cylinder_delta[0,0].item()*1000:+5.1f}, {cylinder_delta[0,1].item()*1000:+5.1f}, {cylinder_delta[0,2].item()*1000:+5.1f}, {cylinder_delta[0,3].item()*1000:+5.1f}] mm")

    print("-" * 80)

    # Now test actual motion in the environment
    print("\n[Testing actual robot motion]")
    print("Applying +Y command (+5mm Y, 0mm Z) for 10 steps...")

    env.reset()

    # Get initial EE position
    ee_start = env._get_observations()[0, :2]  # Y, Z

    # Apply +Y motion via PPO action (scaled to ±1)
    ppo_action = torch.tensor([[+1.0, 0.0]], device="cuda:0")  # +1 Y, 0 Z

    for _ in range(args_cli.num_steps):
        env.step(ppo_action)

    # Get final EE position
    ee_end = env._get_observations()[0, :2]
    ee_motion = (ee_end - ee_start) * 1000  # convert to mm

    print(f"Measured EE motion: [{ee_motion[0].item():+6.1f}, {ee_motion[1].item():+6.1f}] mm")
    print(f"Expected: +Y motion (positive Y, near-zero Z)")

    if ee_motion[0] > 0:
        print("✓ Y sign correct")
    else:
        print("✗ Y sign WRONG")

    if abs(ee_motion[1]) < abs(ee_motion[0]) / 2:
        print("✓ Z coupling acceptable")
    else:
        print("✗ Z coupling too large")

    env.close()
    print("\n" + "="*80)

if __name__ == "__main__":
    test_adapter_commands()
    simulation_app.close()
