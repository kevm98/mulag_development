#!/usr/bin/env python3
"""Train forward response model: (state, command) → EE motion."""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train forward response model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset .pt file")
    parser.add_argument("--output_dir", type=str, default="outputs/ppo1_adapter", help="Output directory")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("[train_forward_model] Loading dataset...")
    data = torch.load(args.dataset_path)
    dataset = data["dataset"]
    metadata = data["metadata"]

    print(f"  Accepted samples: {metadata['accepted_samples']}")

    # Prepare forward model dataset
    # Input: ee_yz, cyl_pos, cyl_vel, prev_delta, cmd_delta (18D)
    # Output: measured_delta_ee (2D)
    X = torch.cat([
        dataset["ee_yz_k"],
        dataset["cylinder_pos_k"],
        dataset["cylinder_vel_k"],
        dataset["previous_cylinder_delta_k"],
        dataset["command_cylinder_delta_k"],
    ], dim=-1)

    Y = dataset["measured_delta_ee_yz"]

    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")

    # Split train/val
    N = X.shape[0]
    perm = torch.randperm(N)
    split_idx = int(N * (1 - args.val_split))

    X_train, Y_train = X[perm[:split_idx]], Y[perm[:split_idx]]
    X_val, Y_val = X[perm[split_idx:]], Y[perm[split_idx:]]

    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # Compute normalization stats
    input_mean = X_train.mean(dim=0)
    input_std = X_train.std(dim=0)
    output_mean = Y_train.mean(dim=0)
    output_std = Y_train.std(dim=0)

    # Create model
    from kevin_integration.new_observations_pipeline.ppo1_adapter.forward_response_model import (
        ForwardResponseModel, compute_forward_model_metrics,
    )

    model = ForwardResponseModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_layer_norm=True,
        max_delta_ee=0.05,
    ).to(args.device)

    model.set_normalization_stats(
        input_mean.to(args.device),
        input_std.to(args.device),
        output_mean.to(args.device),
        output_std.to(args.device),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("[train_forward_model] Training...")
    best_val_mae = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0

        # Shuffle train data
        perm = torch.randperm(X_train.shape[0])
        X_train_shuffled = X_train[perm]
        Y_train_shuffled = Y_train[perm]

        for i in range(0, X_train.shape[0], args.batch_size):
            batch_X = X_train_shuffled[i:i+args.batch_size].to(args.device)
            batch_Y = Y_train_shuffled[i:i+args.batch_size].to(args.device)

            # Split input
            ee_yz = batch_X[:, :2]
            cyl_pos = batch_X[:, 2:6]
            cyl_vel = batch_X[:, 6:10]
            prev_delta = batch_X[:, 10:14]
            cmd_delta = batch_X[:, 14:18]

            # Forward
            pred = model(ee_yz, cyl_pos, cyl_vel, prev_delta, cmd_delta)
            loss = nn.functional.mse_loss(pred, batch_Y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= (X_train.shape[0] / args.batch_size)

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_gpu = X_val.to(args.device)
            Y_val_gpu = Y_val.to(args.device)

            ee_yz = X_val_gpu[:, :2]
            cyl_pos = X_val_gpu[:, 2:6]
            cyl_vel = X_val_gpu[:, 6:10]
            prev_delta = X_val_gpu[:, 10:14]
            cmd_delta = X_val_gpu[:, 14:18]

            metrics = compute_forward_model_metrics(
                model, ee_yz, cyl_pos, cyl_vel, prev_delta, cmd_delta, Y_val_gpu
            )

        val_mae = metrics["mae_mean"]

        print(f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.6f} | "
              f"Val MAE: {val_mae:.5f} | R2: Y={metrics['r2_y']:.3f} Z={metrics['r2_z']:.3f} | "
              f"Sign Acc: Y={metrics['sign_accuracy_y']:.3f} Z={metrics['sign_accuracy_z']:.3f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save(output_dir / "forward_response_model_best.pt")
            print(f"  → Saved best model (MAE: {val_mae:.5f})")

    # Save final model
    model.save(Path(args.output_dir) / "forward_response_model_final.pt")

    print("\n[train_forward_model] Training complete!")
    print(f"  Best val MAE: {best_val_mae:.5f}")

    # Final validation metrics
    print("\n[train_forward_model] Final validation metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.5f}")


if __name__ == "__main__":
    main()
