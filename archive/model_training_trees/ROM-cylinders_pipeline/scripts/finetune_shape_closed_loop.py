#!/usr/bin/env python3
"""Fine-tune ROM velocity/extension heads on stationary shape transitions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data import load_dataset_bundle  # noqa: E402
from ROM_pipeline.models import load_checkpoint, save_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transitions", type=Path)
    parser.add_argument("base_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--replay_dataset",
        type=Path,
        default=ROOT / "data/command_rom_v2",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--replay_weight", type=float, default=0.25)
    parser.add_argument(
        "--target_mode",
        choices=("physical_transition", "causal_command"),
        default="physical_transition",
        help=(
            "physical_transition identifies the observed plant; causal_command "
            "calibrates the ROM output used by the runtime target integrator"
        ),
    )
    parser.add_argument(
        "--command_horizon_steps",
        type=float,
        default=3.0,
        help="Control steps represented by one 50 ms ROM tick in causal_command mode.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.base_checkpoint, device=device)
    reference, _ = load_checkpoint(args.base_checkpoint, device=device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    # Preserve the complete base ROM. The additive correction is explicitly
    # command-gated and therefore exactly zero at stationary/hold commands.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.command_extension_correction_head.parameters():
        parameter.requires_grad_(True)

    data = np.load(args.transitions, allow_pickle=False)
    history = torch.from_numpy(data["history"]).float()
    physical_dx = torch.from_numpy(data["delta_extension"]).float()
    if args.target_mode == "causal_command":
        if args.command_horizon_steps <= 0.0:
            raise ValueError("--command_horizon_steps must be positive")
        # The runtime invokes the 50 ms ROM once every three 60 Hz controller
        # steps.  Its output is integrated into a PhysX position target, so the
        # correct control-specific label is the requested causal increment over
        # that horizon, not the already-lagged motion subsequently observed in
        # PhysX.  Learning the latter reproduces wrong-direction drift.
        target_dx = (
            history[:, -1, 16:20] * float(args.command_horizon_steps)
        ).clone()
    else:
        target_dx = physical_dx
    target_v = torch.from_numpy(data["next_velocity"]).float()
    shape_id = torch.from_numpy(data["shape_id"]).long()
    if history.ndim != 3 or history.shape[1:] != (6, 24):
        raise ValueError(f"expected history [N,6,24], got {tuple(history.shape)}")
    if len(history) < 500:
        raise ValueError(f"at least 500 closed-loop transitions are required, found {len(history)}")

    # Deterministic within-shape split prevents a long shape segment from
    # appearing only in train or validation.
    indices = torch.arange(len(history))
    validation_mask = torch.zeros(len(history), dtype=torch.bool)
    for sid in torch.unique(shape_id):
        rows = indices[shape_id == sid]
        validation_mask[rows[::5]] = True
    train_indices = indices[~validation_mask]
    validation_indices = indices[validation_mask]

    replay = load_dataset_bundle(args.replay_dataset).train
    replay_history = torch.from_numpy(replay.history).float()
    replay_target = torch.from_numpy(replay.target).float()
    target_std = torch.as_tensor(checkpoint["normalization"]["target_std"]).float().to(device)
    dx_scale = torch.clamp(target_dx[train_indices].std(dim=0), min=5.0e-4).to(device)
    velocity_scale = torch.clamp(target_v[train_indices].std(dim=0), min=2.0e-2).to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1.0e-5,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    stale = 0
    report: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = train_indices[torch.randperm(len(train_indices))]
        train_total = 0.0
        train_seen = 0
        for start in range(0, len(permutation), args.batch_size):
            rows = permutation[start : start + args.batch_size]
            batch_h = history[rows].to(device)
            batch_dx = target_dx[rows].to(device)
            batch_v = target_v[rows].to(device)
            output = model(batch_h)
            shape_loss = ((output.delta_extension - batch_dx) / dx_scale).square().mean()
            shape_loss = shape_loss + 0.25 * (
                (output.next_velocity - batch_v) / velocity_scale
            ).square().mean()

            replay_rows = torch.randint(len(replay_history), (len(rows),))
            replay_h = replay_history[replay_rows].to(device)
            replay_y = replay_target[replay_rows].to(device)
            replay_output = model(replay_h)
            replay_loss = (
                (replay_output.delta_velocity - replay_y[:, 8:12])
                / target_std[8:12].clamp_min(1.0e-8)
            ).square().mean()
            replay_loss = replay_loss + (
                (replay_output.delta_extension - replay_y[:, 12:16])
                / target_std[12:16].clamp_min(1.0e-8)
            ).square().mean()

            # A small teacher penalty prevents large off-manifold movement on
            # the collected histories while still allowing response correction.
            with torch.no_grad():
                teacher = reference(batch_h)
            retention = (output.delta_extension - teacher.delta_extension).square().mean() / 1.0e-5
            loss = shape_loss + float(args.replay_weight) * replay_loss + 0.02 * retention
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            train_total += float(loss.detach()) * len(rows)
            train_seen += len(rows)

        model.eval()
        with torch.no_grad():
            val_h = history[validation_indices].to(device)
            val_dx = target_dx[validation_indices].to(device)
            val_v = target_v[validation_indices].to(device)
            val_output = model(val_h)
            validation = ((val_output.delta_extension - val_dx) / dx_scale).square().mean()
            validation = validation + 0.25 * (
                (val_output.next_velocity - val_v) / velocity_scale
            ).square().mean()
        row = {
            "epoch": epoch,
            "train_loss": train_total / max(train_seen, 1),
            "validation_loss": float(validation),
        }
        report.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row), flush=True)
        if float(validation) < best_validation:
            best_validation = float(validation)
            stale = 0
            metadata = dict(checkpoint.get("training_metadata", {}))
            metadata.update(
                {
                    "fine_tune_kind": "stationary_shape_closed_loop",
                    "fine_tune_target_mode": args.target_mode,
                    "command_horizon_steps": float(args.command_horizon_steps),
                    "base_checkpoint": str(args.base_checkpoint.resolve()),
                    "transition_source": str(args.transitions.resolve()),
                    "transition_count": len(history),
                    "best_epoch": epoch,
                    "best_validation_loss": best_validation,
                    "trainable_modules": ["command_extension_correction_head"],
                }
            )
            save_checkpoint(
                args.output_dir / "best_rom.pt",
                model,
                target_mean=torch.as_tensor(checkpoint["normalization"]["target_mean"]),
                target_std=torch.as_tensor(checkpoint["normalization"]["target_std"]),
                cylinder_limits=torch.as_tensor(checkpoint["cylinder_limits"]),
                training_metadata=metadata,
            )
        else:
            stale += 1
        (args.output_dir / "training_history.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if stale >= 15:
            print(f"early stopping at epoch {epoch}; best validation={best_validation:.6f}")
            break


if __name__ == "__main__":
    main()
