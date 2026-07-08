#!/usr/bin/env python3
"""
Validate adapter offline before running in Isaac.

Validation Gate 1: Offline Metrics

Pass criteria:
    - Forward model: MAE < 5-10mm, sign accuracy > 85%, R² > 0.7
    - Adapter: Forward-consistency MAE < 10mm, sign accuracy > 90%
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

def parse_args():
    parser = argparse.ArgumentParser(description="Validate adapter offline")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset .pt file")
    parser.add_argument("--forward_checkpoint", type=str, required=True, help="Path to trained forward model")
    parser.add_argument("--adapter_checkpoint", type=str, required=True, help="Path to trained adapter")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("[validate_adapter_offline] Validation Gate 1: Offline Metrics")
    print("="*80)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    data = torch.load(args.dataset_path)
    dataset = data["dataset"]
    metadata = data["metadata"]

    # Use test split (last 20%)
    N = dataset["ee_yz_k"].shape[0]
    test_start = int(0.8 * N)

    print(f"  Total samples: {N}")
    print(f"  Test samples: {N - test_start}")

    # Load models
    print("\n[2/5] Loading models...")
    from kevin_integration.new_observations_pipeline.ppo1_adapter.forward_response_model import (
        ForwardResponseModel, compute_forward_model_metrics,
    )
    from kevin_integration.new_observations_pipeline.ppo1_adapter.task_to_cylinder_adapter import (
        TaskToCylinderAdapter,
    )

    forward_model = ForwardResponseModel.load(args.forward_checkpoint, device=args.device)
    adapter = TaskToCylinderAdapter.load(args.adapter_checkpoint, device=args.device)

    # Prepare test data
    ee_yz = dataset["ee_yz_k"][test_start:].to(args.device)
    cyl_pos = dataset["cylinder_pos_k"][test_start:].to(args.device)
    cyl_vel = dataset["cylinder_vel_k"][test_start:].to(args.device)
    prev_delta = dataset["previous_cylinder_delta_k"][test_start:].to(args.device)
    cmd_delta = dataset["command_cylinder_delta_k"][test_start:].to(args.device)
    measured_ee_delta = dataset["measured_delta_ee_yz"][test_start:].to(args.device)

    # Forward model validation
    print("\n[3/5] Validating forward model...")
    forward_metrics = compute_forward_model_metrics(
        forward_model,
        ee_yz,
        cyl_pos,
        cyl_vel,
        prev_delta,
        cmd_delta,
        measured_ee_delta,
    )

    print(f"  MAE Y: {forward_metrics['mae_y']*1000:.2f} mm")
    print(f"  MAE Z: {forward_metrics['mae_z']*1000:.2f} mm")
    print(f"  RMSE Y: {forward_metrics['rmse_y']*1000:.2f} mm")
    print(f"  RMSE Z: {forward_metrics['rmse_z']*1000:.2f} mm")
    print(f"  Sign Accuracy Y: {forward_metrics['sign_accuracy_y']*100:.1f}%")
    print(f"  Sign Accuracy Z: {forward_metrics['sign_accuracy_z']*100:.1f}%")
    print(f"  R² Y: {forward_metrics['r2_y']:.3f}")
    print(f"  R² Z: {forward_metrics['r2_z']:.3f}")

    # Check pass criteria
    forward_pass = (
        forward_metrics['mae_y'] < 0.010  # 10mm
        and forward_metrics['mae_z'] < 0.010
        and forward_metrics['sign_accuracy_y'] > 0.85
        and forward_metrics['sign_accuracy_z'] > 0.85
        and forward_metrics['r2_y'] > 0.7
        and forward_metrics['r2_z'] > 0.7
    )

    print(f"\n  Forward model: {'PASS' if forward_pass else 'FAIL'}")

    # Adapter validation
    print("\n[4/5] Validating adapter...")

    with torch.no_grad():
        # Run adapter
        adapter_output = adapter(
            measured_ee_delta,  # desired = what actually happened
            ee_yz,
            cyl_pos,
            cyl_vel,
            prev_delta,
            enforce_limits=True,
        )

        # Command metrics
        command_error = (adapter_output - cmd_delta).abs()
        command_mae = command_error.mean().item()
        command_norm_mean = adapter_output.abs().mean().item()

        # Saturation
        command_saturation = (adapter_output.abs() > adapter.max_delta_cylinder * 0.95).any(dim=1)
        saturation_frac = command_saturation.float().mean().item()

        # Limit violations
        next_cyl_pos = cyl_pos + adapter_output
        margin = 0.01
        limit_violation = (
            (next_cyl_pos < adapter.cylinder_min + margin).any(dim=1)
            | (next_cyl_pos > adapter.cylinder_max - margin).any(dim=1)
        )
        limit_violation_frac = limit_violation.float().mean().item()

        # Forward-consistency
        predicted_ee_delta = forward_model(
            ee_yz,
            cyl_pos,
            cyl_vel,
            prev_delta,
            adapter_output,
        )

        consistency_error = (predicted_ee_delta - measured_ee_delta).abs()
        consistency_mae_y = consistency_error[:, 0].mean().item()
        consistency_mae_z = consistency_error[:, 1].mean().item()
        consistency_mae = consistency_error.mean().item()

        # Sign accuracy
        pred_sign = torch.sign(predicted_ee_delta)
        true_sign = torch.sign(measured_ee_delta)
        sign_acc_y = (pred_sign[:, 0] == true_sign[:, 0]).float().mean().item()
        sign_acc_z = (pred_sign[:, 1] == true_sign[:, 1]).float().mean().item()

    print(f"  Command MAE: {command_mae*1000:.2f} mm")
    print(f"  Command Norm Mean: {command_norm_mean*1000:.2f} mm")
    print(f"  Command Saturation: {saturation_frac*100:.1f}%")
    print(f"  Limit Violations: {limit_violation_frac*100:.1f}%")
    print(f"  Forward-Consistency MAE Y: {consistency_mae_y*1000:.2f} mm")
    print(f"  Forward-Consistency MAE Z: {consistency_mae_z*1000:.2f} mm")
    print(f"  Forward-Consistency Sign Accuracy Y: {sign_acc_y*100:.1f}%")
    print(f"  Forward-Consistency Sign Accuracy Z: {sign_acc_z*100:.1f}%")

    # Check pass criteria
    adapter_pass = (
        consistency_mae < 0.010  # 10mm
        and sign_acc_y > 0.90
        and sign_acc_z > 0.90
        and saturation_frac < 0.05
        and limit_violation_frac < 0.01
    )

    print(f"\n  Adapter: {'PASS' if adapter_pass else 'FAIL'}")

    # Overall result
    print("\n[5/5] Overall Result")
    print("="*80)

    if forward_pass and adapter_pass:
        print("✓ PASS: Proceed to Gate 2 (Direction Tests)")
    else:
        print("✗ FAIL: Improve dataset/model before Isaac validation")
        if not forward_pass:
            print("  - Forward model does not meet criteria")
        if not adapter_pass:
            print("  - Adapter does not meet criteria")

    print("="*80)


if __name__ == "__main__":
    main()
