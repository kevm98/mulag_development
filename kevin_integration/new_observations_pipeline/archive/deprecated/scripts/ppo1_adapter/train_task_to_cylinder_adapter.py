#!/usr/bin/env python3
"""Train task-to-cylinder adapter with supervised + forward-consistency losses."""

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
    parser = argparse.ArgumentParser(description="Train task-to-cylinder adapter")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset .pt file")
    parser.add_argument("--forward_checkpoint", type=str, required=True, help="Path to trained forward model")
    parser.add_argument("--output_dir", type=str, default="outputs/ppo1_adapter", help="Output directory")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_consistency", type=float, default=1.0, help="Forward-consistency loss weight")
    parser.add_argument("--lambda_command", type=float, default=0.01, help="Command regularization weight")
    parser.add_argument("--lambda_limit", type=float, default=0.1, help="Limit penalty weight")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("[train_adapter] Loading dataset...")
    data = torch.load(args.dataset_path)
    dataset = data["dataset"]
    metadata = data["metadata"]

    print(f"  Accepted samples: {metadata['accepted_samples']}")

    # Prepare adapter dataset
    # Input: desired_ee_delta, ee_yz, cyl_pos, cyl_vel, prev_delta (16D)
    # Output: command_cylinder_delta (4D)

    # For inverse training, we use measured EE delta as "desired"
    X = torch.cat([
        dataset["measured_delta_ee_yz"],  # desired (what happened)
        dataset["ee_yz_k"],
        dataset["cylinder_pos_k"],
        dataset["cylinder_vel_k"],
        dataset["previous_cylinder_delta_k"],
    ], dim=-1)

    Y = dataset["command_cylinder_delta_k"]  # cylinder command that caused it

    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")

    # Split train/val
    N = X.shape[0]
    perm = torch.randperm(N)
    split_idx = int(N * (1 - args.val_split))

    X_train, Y_train = X[perm[:split_idx]], Y[perm[:split_idx]]
    X_val, Y_val = X[perm[split_idx:]], Y[perm[split_idx:]]

    # Also split state variables for forward-consistency validation
    state_ee_train = dataset["ee_yz_k"][perm[:split_idx]]
    state_cyl_pos_train = dataset["cylinder_pos_k"][perm[:split_idx]]
    state_cyl_vel_train = dataset["cylinder_vel_k"][perm[:split_idx]]
    state_prev_delta_train = dataset["previous_cylinder_delta_k"][perm[:split_idx]]

    state_ee_val = dataset["ee_yz_k"][perm[split_idx:]]
    state_cyl_pos_val = dataset["cylinder_pos_k"][perm[split_idx:]]
    state_cyl_vel_val = dataset["cylinder_vel_k"][perm[split_idx:]]
    state_prev_delta_val = dataset["previous_cylinder_delta_k"][perm[split_idx:]]

    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # Compute normalization stats
    input_mean = X_train.mean(dim=0)
    input_std = X_train.std(dim=0)
    output_mean = Y_train.mean(dim=0)
    output_std = Y_train.std(dim=0)

    # Load forward model
    from kevin_integration.new_observations_pipeline.ppo1_adapter.forward_response_model import (
        ForwardResponseModel,
    )
    forward_model = ForwardResponseModel.load(args.forward_checkpoint, device=args.device)
    forward_model.eval()
    for param in forward_model.parameters():
        param.requires_grad = False

    # Create adapter
    from kevin_integration.new_observations_pipeline.ppo1_adapter.task_to_cylinder_adapter import (
        TaskToCylinderAdapter, compute_adapter_total_loss,
    )

    max_delta_cylinder = metadata.get("max_delta_cylinder", 0.0025)
    adapter = TaskToCylinderAdapter(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_layer_norm=True,
        max_delta_cylinder=max_delta_cylinder,
    ).to(args.device)

    adapter.set_normalization_stats(
        input_mean.to(args.device),
        input_std.to(args.device),
        output_mean.to(args.device),
        output_std.to(args.device),
    )

    # Set cylinder limits
    cylinder_min = torch.tensor([-0.2, -0.2, -0.2, -0.2], device=args.device)
    cylinder_max = torch.tensor([0.2, 0.2, 0.2, 0.2], device=args.device)
    adapter.set_cylinder_limits(cylinder_min, cylinder_max)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    # Training loop
    print("[train_adapter] Training...")
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        adapter.train()
        train_losses = {"total": 0.0, "supervised": 0.0, "consistency": 0.0, "command": 0.0, "limit": 0.0}

        # Shuffle train data
        perm = torch.randperm(X_train.shape[0])
        X_train_shuffled = X_train[perm]
        Y_train_shuffled = Y_train[perm]
        state_ee_shuffled = state_ee_train[perm]
        state_cyl_pos_shuffled = state_cyl_pos_train[perm]
        state_cyl_vel_shuffled = state_cyl_vel_train[perm]
        state_prev_delta_shuffled = state_prev_delta_train[perm]

        n_batches = 0
        for i in range(0, X_train.shape[0], args.batch_size):
            batch_X = X_train_shuffled[i:i+args.batch_size].to(args.device)
            batch_Y = Y_train_shuffled[i:i+args.batch_size].to(args.device)
            batch_ee = state_ee_shuffled[i:i+args.batch_size].to(args.device)
            batch_cyl_pos = state_cyl_pos_shuffled[i:i+args.batch_size].to(args.device)
            batch_cyl_vel = state_cyl_vel_shuffled[i:i+args.batch_size].to(args.device)
            batch_prev_delta = state_prev_delta_shuffled[i:i+args.batch_size].to(args.device)

            # Split input
            desired_ee_delta = batch_X[:, :2]

            # Compute loss
            losses = compute_adapter_total_loss(
                adapter,
                forward_model,
                desired_ee_delta,
                batch_ee,
                batch_cyl_pos,
                batch_cyl_vel,
                batch_prev_delta,
                batch_Y,
                lambda_consistency=args.lambda_consistency,
                lambda_command=args.lambda_command,
                lambda_limit=args.lambda_limit,
            )

            # Backward
            optimizer.zero_grad()
            losses["loss_total"].backward()
            optimizer.step()

            # Accumulate losses
            for key in train_losses.keys():
                train_losses[key] += losses[f"loss_{key}"].item()
            n_batches += 1

        for key in train_losses.keys():
            train_losses[key] /= n_batches

        # Validation
        adapter.eval()
        val_losses = {"total": 0.0, "supervised": 0.0, "consistency": 0.0, "command": 0.0, "limit": 0.0}

        with torch.no_grad():
            X_val_gpu = X_val.to(args.device)
            Y_val_gpu = Y_val.to(args.device)
            state_ee_val_gpu = state_ee_val.to(args.device)
            state_cyl_pos_val_gpu = state_cyl_pos_val.to(args.device)
            state_cyl_vel_val_gpu = state_cyl_vel_val.to(args.device)
            state_prev_delta_val_gpu = state_prev_delta_val.to(args.device)

            desired_ee_delta = X_val_gpu[:, :2]

            losses = compute_adapter_total_loss(
                adapter,
                forward_model,
                desired_ee_delta,
                state_ee_val_gpu,
                state_cyl_pos_val_gpu,
                state_cyl_vel_val_gpu,
                state_prev_delta_val_gpu,
                Y_val_gpu,
                lambda_consistency=args.lambda_consistency,
                lambda_command=args.lambda_command,
                lambda_limit=args.lambda_limit,
            )

            for key in val_losses.keys():
                val_losses[key] = losses[f"loss_{key}"].item()

        print(f"Epoch {epoch+1}/{args.num_epochs} | "
              f"Train: {train_losses['total']:.6f} | "
              f"Val: {val_losses['total']:.6f} | "
              f"Sup: {val_losses['supervised']:.6f} | "
              f"Cons: {val_losses['consistency']:.6f} | "
              f"Cmd: {val_losses['command']:.6f} | "
              f"Lim: {val_losses['limit']:.6f}")

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            adapter.save(output_dir / "task_to_cylinder_adapter_best.pt")
            print(f"  → Saved best model (loss: {val_losses['total']:.6f})")

    # Save final model
    adapter.save(Path(args.output_dir) / "task_to_cylinder_adapter_final.pt")

    print("\n[train_adapter] Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
