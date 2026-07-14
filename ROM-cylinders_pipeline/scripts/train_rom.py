#!/usr/bin/env python3
"""Train the hydraulic-aware ROM with staged one-step and recursive rollout losses."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data import DEFAULT_SCHEMA, load_dataset_bundle  # noqa: E402
from ROM_pipeline.models import HydraulicROM, HydraulicROMConfig, save_checkpoint  # noqa: E402
from ROM_pipeline.training import TrainingConfig, compute_one_step_loss, rollout  # noqa: E402
from ROM_pipeline.training.rollout_loss import output_tensor  # noqa: E402


def batch_rollout_loss(model, dataset, starts, horizon, target_std, weights, device, teacher_probability, state_only=False):
    indices = dataset.sequence_indices(starts.detach().cpu().numpy(), horizon)
    if len(indices) == 0:
        return torch.zeros((), device=device), 0
    history = torch.from_numpy(dataset.history[indices[:, 0]]).to(device)
    commands = torch.from_numpy(dataset.history[indices, -1, DEFAULT_SCHEMA.command_slice]).to(device)
    targets = torch.from_numpy(dataset.target[indices]).to(device)
    teacher_rows = torch.from_numpy(dataset.history[indices, -1]).to(device)
    if state_only:
        history[:, :, DEFAULT_SCHEMA.command_slice] = 0.0
        commands.zero_()
        teacher_rows[:, :, DEFAULT_SCHEMA.command_slice] = 0.0
    outputs = rollout(model, history, commands, teacher_rows=teacher_rows, teacher_probability=teacher_probability)
    total = torch.zeros((), device=device)
    for step, output in enumerate(outputs):
        error = (output_tensor(output) - targets[:, step]) / target_std.reshape(1, -1).clamp_min(1.0e-8)
        total = total + (0.98 ** step) * error.square().mean()
    return weights.rollout * total / horizon, len(indices)


def matched_command_loss(model, dataset, starts, target_std, weights, device, state_only=False):
    if state_only:
        return torch.zeros((), device=device), 0
    pairs = dataset.command_pair_indices(starts.detach().cpu().numpy())
    if len(pairs) == 0:
        return torch.zeros((), device=device), 0
    left_history = torch.from_numpy(dataset.history[pairs[:, 0]]).to(device)
    right_history = torch.from_numpy(dataset.history[pairs[:, 1]]).to(device)
    predicted_difference = model(left_history).delta_extension - model(right_history).delta_extension
    true_difference = torch.from_numpy(
        dataset.target[pairs[:, 0], 8:12] - dataset.target[pairs[:, 1], 8:12]
    ).to(device)
    scale = target_std[8:12].reshape(1, 4).clamp_min(1.0e-8)
    return weights.command * ((predicted_difference - true_difference) / scale).square().mean(), len(pairs)


@torch.no_grad()
def validate(model, loader, dataset, target_std, weights, device, horizon, state_only=False):
    model.eval()
    total = 0.0
    seen = 0
    for batch in loader:
        history = batch["history"].to(device)
        if state_only:
            history[:, :, DEFAULT_SCHEMA.command_slice] = 0.0
        target = batch["target"].to(device)
        valve = batch["valve_label"].to(device)
        loss, _ = compute_one_step_loss(model(history), target, target_std, valve_label=valve, weights=weights)
        total += float(loss) * len(history)
        seen += len(history)
    one_step = total / max(seen, 1)
    rollout_score = 0.0
    rollout_count = 0
    if horizon > 1:
        starts = torch.arange(min(len(dataset), 256), dtype=torch.long)
        recursive, rollout_count = batch_rollout_loss(
            model, dataset, starts, horizon, target_std, weights, device, teacher_probability=0.0,
            state_only=state_only,
        )
        rollout_score = float(recursive)
    return one_step, rollout_score, rollout_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", help="Override config device, e.g. cpu or cuda:0")
    parser.add_argument("--state-only", action="store_true", help="Train the required command-free ablation model.")
    args = parser.parse_args()
    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
    device = torch.device(args.device or config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    bundle = load_dataset_bundle(args.dataset)
    stats = bundle.stats.torch()
    model = HydraulicROM(
        HydraulicROMConfig(), input_mean=stats["input_mean"], input_std=stats["input_std"]
    ).to(device)
    target_std = stats["target_std"].to(device)
    train_loader = DataLoader(bundle.train, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(bundle.validation, batch_size=config.validation_batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(args.output_dir / "resolved_config.json")
    best = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    global_epoch = 0
    report = []
    total_epochs = sum(config.stage_epochs)
    for stage, (epochs, horizon) in enumerate(zip(config.stage_epochs, config.stage_rollout_horizons, strict=True), start=1):
        for _ in range(epochs):
            global_epoch += 1
            progress = (global_epoch - 1) / max(total_epochs - 1, 1)
            predicted_probability = config.scheduled_sampling_start + progress * (
                config.scheduled_sampling_end - config.scheduled_sampling_start
            )
            teacher_probability = 1.0 - predicted_probability
            model.train()
            running = 0.0
            seen = 0
            for batch in train_loader:
                history = batch["history"].to(device)
                if args.state_only:
                    history[:, :, DEFAULT_SCHEMA.command_slice] = 0.0
                target = batch["target"].to(device)
                valve = batch["valve_label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                one_step, _ = compute_one_step_loss(
                    model(history), target, target_std, valve_label=valve, weights=config.loss_weights
                )
                recursive, recursive_count = batch_rollout_loss(
                    model, bundle.train, batch["index"][:config.rollout_batch_size], horizon,
                    target_std, config.loss_weights, device,
                    teacher_probability,
                    state_only=args.state_only,
                ) if horizon > 1 else (torch.zeros((), device=device), 0)
                command_loss, command_pair_count = matched_command_loss(
                    model, bundle.train, batch["index"], target_std, config.loss_weights, device,
                    state_only=args.state_only,
                )
                loss = one_step + recursive + command_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                running += float(loss.detach()) * len(history)
                seen += len(history)
            val, val_rollout, val_rollout_count = validate(
                model, val_loader, bundle.validation, target_std, config.loss_weights, device, horizon,
                state_only=args.state_only,
            )
            selection_score = val + val_rollout
            row = {"stage": stage, "epoch": global_epoch, "horizon": horizon,
                   "train_loss": running / max(seen, 1), "validation_one_step_loss": val,
                   "validation_rollout_loss": val_rollout, "validation_rollout_sequences": val_rollout_count,
                   "selection_score": selection_score}
            report.append(row)
            print(json.dumps(row), flush=True)
            if selection_score < best:
                best = selection_score
                best_epoch = global_epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    args.output_dir / "best_rom.pt", model,
                    target_mean=stats["target_mean"], target_std=stats["target_std"],
                    cylinder_limits=torch.tensor(
                        bundle.manifest["cylinder_limits_from_training_extension"], dtype=torch.float32
                    ),
                    training_metadata={"best_epoch": best_epoch, "best_validation_loss": best,
                                       "dataset_manifest": bundle.manifest, "state_only_ablation": args.state_only},
                )
            else:
                epochs_without_improvement += 1
            (args.output_dir / "training_history.json").write_text(json.dumps(report, indent=2) + "\n")
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping at epoch {global_epoch}; best epoch was {best_epoch}.")
                return


if __name__ == "__main__":
    main()
