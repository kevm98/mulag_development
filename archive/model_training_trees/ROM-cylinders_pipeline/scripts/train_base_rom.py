#!/usr/bin/env python3
"""Train the valve-conditioned base hydraulic ROM on Simscape transitions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data import BASE_ROM_SCHEMA, load_dataset_bundle  # noqa: E402
from ROM_pipeline.models import BaseHydraulicROM, BaseHydraulicROMConfig, save_base_checkpoint  # noqa: E402
from ROM_pipeline.training import TrainingConfig, compute_one_step_loss, rollout  # noqa: E402
from ROM_pipeline.training.rollout_loss import output_tensor  # noqa: E402


def recursive_loss(model, dataset, starts, horizon, target_std, weights, device, teacher_probability):
    indices = dataset.sequence_indices(starts.detach().cpu().numpy(), horizon)
    if len(indices) == 0:
        return torch.zeros((), device=device), 0
    schema = BASE_ROM_SCHEMA
    history = torch.from_numpy(dataset.history[indices[:, 0]]).to(device)
    valves = torch.from_numpy(dataset.history[indices, -1, schema.valve_slice]).to(device)
    targets = torch.from_numpy(dataset.target[indices]).to(device)
    teacher_rows = torch.from_numpy(dataset.history[indices, -1]).to(device)
    outputs = rollout(model, history, valves, teacher_rows=teacher_rows, teacher_probability=teacher_probability)
    total = torch.zeros((), device=device)
    for step, output in enumerate(outputs):
        error = (output_tensor(output) - targets[:, step]) / target_std.reshape(1, -1).clamp_min(1.0e-8)
        total = total + (0.98 ** step) * error.square().mean()
    return weights.rollout * total / horizon, len(indices)


@torch.no_grad()
def validate(model, loader, dataset, horizon, target_std, weights, device):
    model.eval()
    total = 0.0
    seen = 0
    for batch in loader:
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        loss, _ = compute_one_step_loss(
            model(history), target, target_std, valve_label=batch["valve_label"].to(device), weights=weights
        )
        total += float(loss) * len(history)
        seen += len(history)
    one_step = total / max(seen, 1)
    rollout_value = 0.0
    rollout_count = 0
    if horizon > 1:
        starts = torch.arange(min(len(dataset), 256), dtype=torch.long)
        value, rollout_count = recursive_loss(
            model, dataset, starts, horizon, target_std, weights, device, teacher_probability=0.0
        )
        rollout_value = float(value)
    return one_step, rollout_value, rollout_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", help="Override the training configuration device.")
    args = parser.parse_args()
    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
    device = torch.device(args.device or config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    bundle = load_dataset_bundle(args.dataset, schema=BASE_ROM_SCHEMA)
    stats = bundle.stats.torch()
    model = BaseHydraulicROM(
        BaseHydraulicROMConfig(), input_mean=stats["input_mean"], input_std=stats["input_std"]
    ).to(device)
    target_std = stats["target_std"].to(device)
    strict_source_ids = {
        index for index, source in enumerate(bundle.manifest.get("sources", []))
        if "open_loop" in str(source.get("kind", ""))
    }
    sample_weights = np.ones(len(bundle.train), dtype=np.float64)
    if strict_source_ids and config.excitation_source_weight > 1.0:
        sample_weights[np.isin(bundle.train.source_id, list(strict_source_ids))] = config.excitation_source_weight
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(bundle.train, batch_size=config.batch_size, sampler=sampler,
                              num_workers=config.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(bundle.validation, batch_size=config.validation_batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(args.output_dir / "resolved_config.json")
    history_report = []
    global_epoch = 0
    total_epochs = sum(config.stage_epochs)
    for stage, (epochs, horizon) in enumerate(zip(config.stage_epochs, config.stage_rollout_horizons, strict=True), 1):
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        stage_best = float("inf")
        stage_best_epoch = -1
        stale = 0
        for _ in range(epochs):
            global_epoch += 1
            progress = (global_epoch - 1) / max(total_epochs - 1, 1)
            teacher_probability = 1.0 - (
                config.scheduled_sampling_start
                + progress * (config.scheduled_sampling_end - config.scheduled_sampling_start)
            )
            model.train()
            running = 0.0
            seen = 0
            for batch in train_loader:
                history = batch["history"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad(set_to_none=True)
                one_step, _ = compute_one_step_loss(
                    model(history), target, target_std,
                    valve_label=batch["valve_label"].to(device), weights=config.loss_weights,
                )
                recursive, _ = recursive_loss(
                    model, bundle.train, batch["index"][:config.rollout_batch_size], horizon,
                    target_std, config.loss_weights,
                    device, teacher_probability,
                ) if horizon > 1 else (torch.zeros((), device=device), 0)
                loss = one_step + recursive
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                running += float(loss.detach()) * len(history)
                seen += len(history)
            val, val_rollout, count = validate(
                model, val_loader, bundle.validation, horizon, target_std, config.loss_weights, device
            )
            score = val + val_rollout
            row = {"stage": stage, "epoch": global_epoch, "horizon": horizon,
                   "train_loss": running / max(seen, 1), "validation_one_step_loss": val,
                   "validation_rollout_loss": val_rollout, "validation_rollout_sequences": count,
                   "selection_score": score}
            history_report.append(row)
            print(json.dumps(row), flush=True)
            if score < stage_best:
                stage_best, stage_best_epoch, stale = score, global_epoch, 0
                metadata = {"stage": stage, "rollout_horizon": horizon, "best_epoch": stage_best_epoch,
                            "best_selection_score": stage_best, "dataset_manifest": bundle.manifest}
                limits = torch.tensor(bundle.manifest["cylinder_limits_from_training_extension"], dtype=torch.float32)
                for filename in (f"best_stage_{stage}.pt", "best_base_rom.pt"):
                    save_base_checkpoint(
                        args.output_dir / filename, model, target_mean=stats["target_mean"],
                        target_std=stats["target_std"], cylinder_limits=limits, training_metadata=metadata,
                    )
            else:
                stale += 1
            (args.output_dir / "training_history.json").write_text(json.dumps(history_report, indent=2) + "\n")
            if stale >= config.early_stopping_patience:
                print(f"Early stopping stage {stage} at epoch {global_epoch}; best epoch was {stage_best_epoch}.")
                break
        best_payload = torch.load(args.output_dir / f"best_stage_{stage}.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_payload["state_dict"], strict=True)


if __name__ == "__main__":
    main()
