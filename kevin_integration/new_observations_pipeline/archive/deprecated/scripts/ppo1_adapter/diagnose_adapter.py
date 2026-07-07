"""Diagnose adapter behavior with simple test inputs."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from kevin_integration.new_observations_pipeline.ppo1_adapter.task_to_cylinder_adapter import TaskToCylinderAdapter

def test_adapter(checkpoint_path: str):
    """Test adapter with simple inputs."""
    print("="*80)
    print("ADAPTER DIAGNOSIS")
    print("="*80)

    # Load adapter
    device = "cuda:0"
    adapter = TaskToCylinderAdapter.load(checkpoint_path, device=device)
    adapter.eval()

    print(f"\nLoaded adapter from: {checkpoint_path}")
    print(f"  max_delta_cylinder: {adapter.max_delta_cylinder*1000:.2f} mm")
    print(f"  Input mean: {adapter.input_mean[:2].cpu().numpy()}")
    print(f"  Input std: {adapter.input_std[:2].cpu().numpy()}")
    print(f"  Cylinder limits: [{adapter.cylinder_min[0].item():.3f}, {adapter.cylinder_max[0].item():.3f}]")

    # Create test inputs (batch size 1)
    # Neutral state: EE at origin, cylinders at zero, zero velocities, zero prev_delta
    current_ee_yz = torch.zeros((1, 2), device=device)
    cylinder_pos = torch.zeros((1, 4), device=device)
    cylinder_vel = torch.zeros((1, 4), device=device)
    prev_delta = torch.zeros((1, 4), device=device)

    # Test different desired EE motions
    test_cases = [
        ("Zero", torch.tensor([[0.0, 0.0]], device=device)),
        ("+Y only", torch.tensor([[0.005, 0.0]], device=device)),
        ("-Y only", torch.tensor([[-0.005, 0.0]], device=device)),
        ("+Z only", torch.tensor([[0.0, 0.005]], device=device)),
        ("-Z only", torch.tensor([[0.0, -0.005]], device=device)),
        ("+Y+Z", torch.tensor([[0.005, 0.005]], device=device)),
        ("-Y-Z", torch.tensor([[-0.005, -0.005]], device=device)),
    ]

    print("\n" + "="*80)
    print("TEST RESULTS (5mm desired motion)")
    print("="*80)
    print(f"{'Direction':<10} | {'Desired (mm)':<15} | {'Output (mm)':<40}")
    print("-"*80)

    with torch.no_grad():
        for name, desired_ee in test_cases:
            output = adapter(
                desired_ee,
                current_ee_yz,
                cylinder_pos,
                cylinder_vel,
                prev_delta,
                enforce_limits=False,
            )

            desired_str = f"[{desired_ee[0,0].item()*1000:+6.1f}, {desired_ee[0,1].item()*1000:+6.1f}]"
            output_str = f"[{output[0,0].item()*1000:+6.2f}, {output[0,1].item()*1000:+6.2f}, {output[0,2].item()*1000:+6.2f}, {output[0,3].item()*1000:+6.2f}]"

            print(f"{name:<10} | {desired_str:<15} | {output_str:<40}")

    print("="*80)

    # Test with non-zero cylinder state
    print("\nTesting with non-zero cylinder state:")
    current_ee_yz = torch.tensor([[0.5, 1.0]], device=device)  # 50cm Y, 100cm Z
    cylinder_pos = torch.tensor([[0.05, 0.05, 0.05, 0.05]], device=device)  # 5cm extended

    desired_ee = torch.tensor([[0.005, 0.0]], device=device)  # +Y
    output = adapter(desired_ee, current_ee_yz, cylinder_pos, cylinder_vel, prev_delta, enforce_limits=False)

    print(f"  Desired: +Y 5mm at EE=[500mm, 1000mm], cyl=[50mm, 50mm, 50mm, 50mm]")
    print(f"  Output: [{output[0,0].item()*1000:+.2f}, {output[0,1].item()*1000:+.2f}, {output[0,2].item()*1000:+.2f}, {output[0,3].item()*1000:+.2f}] mm")

    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/ppo1_adapter/task_to_cylinder_adapter_best.pt")
    args = parser.parse_args()

    test_adapter(args.checkpoint)
