#!/usr/bin/env python3
"""Offline one-step, command-ablation and command-Jacobian validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data import DEFAULT_SCHEMA, load_dataset_bundle  # noqa: E402
from ROM_pipeline.models import load_checkpoint  # noqa: E402
from ROM_pipeline.training.rollout_loss import output_tensor  # noqa: E402
from ROM_pipeline.training.rollout_loss import rollout  # noqa: E402


def r2(true: torch.Tensor, prediction: torch.Tensor) -> list[float]:
    residual = (true - prediction).square().sum(dim=0)
    total = (true - true.mean(dim=0)).square().sum(dim=0).clamp_min(1.0e-12)
    return (1.0 - residual / total).detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    bundle = load_dataset_bundle(args.dataset)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    count = min(args.max_samples, len(bundle.test))
    history = torch.from_numpy(bundle.test.history[:count]).to(device)
    target = torch.from_numpy(bundle.test.target[:count]).to(device)
    with torch.no_grad():
        start = time.perf_counter()
        prediction = output_tensor(model(history))
        elapsed = time.perf_counter() - start
        zeroed = history.clone()
        zeroed[:, :, DEFAULT_SCHEMA.command_slice] = 0.0
        zero_prediction = output_tensor(model(zeroed))
        shuffled = history.clone()
        permutation = torch.randperm(count, device=device)
        shuffled[:, :, DEFAULT_SCHEMA.command_slice] = shuffled[permutation, :, DEFAULT_SCHEMA.command_slice]
        shuffled_prediction = output_tensor(model(shuffled))
    jac_history = history[: min(128, count)].clone()
    command = jac_history[:, -1, DEFAULT_SCHEMA.command_slice].clone().requires_grad_(True)
    delta = model(jac_history, command).delta_extension
    rows = []
    for output_axis in range(4):
        gradient = torch.autograd.grad(delta[:, output_axis].sum(), command, retain_graph=True)[0]
        rows.append(gradient[:, :4].mean(dim=0))
    jacobian = torch.stack(rows)
    singular_values = torch.linalg.svdvals(jacobian)
    rollout_metrics = {}
    with torch.no_grad():
        for horizon in (5, 10, 20, 50, 100):
            starts = np.arange(min(len(bundle.test), 2048), dtype=np.int64)
            indices = bundle.test.sequence_indices(starts, horizon)[:256]
            if not len(indices):
                rollout_metrics[str(horizon)] = {"available_sequences": 0}
                continue
            initial = torch.from_numpy(bundle.test.history[indices[:, 0]]).to(device)
            commands = torch.from_numpy(bundle.test.history[indices, -1, DEFAULT_SCHEMA.command_slice]).to(device)
            outputs = rollout(model, initial, commands, teacher_probability=0.0)
            terminal_prediction = output_tensor(outputs[-1])
            terminal_target = torch.from_numpy(bundle.test.target[indices[:, -1]]).to(device)
            terminal_error = terminal_prediction - terminal_target
            rollout_metrics[str(horizon)] = {
                "available_sequences": int(len(indices)),
                "terminal_rmse_by_output": terminal_error.square().mean(dim=0).sqrt().cpu().tolist(),
                "terminal_r2_by_output": r2(terminal_target, terminal_prediction),
            }
    normal_r2 = r2(target, prediction)
    zero_r2 = r2(target, zero_prediction)
    shuffled_r2 = r2(target, shuffled_prediction)
    gates = {
        "valve_mean_r2_at_least_0p9": float(np.mean(normal_r2[0:4])) >= 0.9,
        "pressure_mean_r2_at_least_0p05": float(np.mean(normal_r2[4:8])) >= 0.05,
        "velocity_mean_r2_at_least_0p3": float(np.mean(normal_r2[8:12])) >= 0.3,
        "extension_mean_r2_at_least_0p9": float(np.mean(normal_r2[12:16])) >= 0.9,
        "zeroed_command_degradation_at_least_0p05": float(np.mean(np.asarray(normal_r2) - np.asarray(zero_r2))) >= 0.05,
        "shuffled_command_degradation_at_least_0p05": float(np.mean(np.asarray(normal_r2) - np.asarray(shuffled_r2))) >= 0.05,
        "finite_full_rank_command_jacobian": bool(torch.isfinite(singular_values).all() and singular_values.min() > 1.0e-5),
    }
    report = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "acceptance_gates": gates,
        "samples": count,
        "r2_by_output": normal_r2,
        "zeroed_command_r2_by_output": zero_r2,
        "shuffled_command_r2_by_output": shuffled_r2,
        "command_jacobian_delta_extension_wrt_delta_extension_command": jacobian.detach().cpu().tolist(),
        "command_jacobian_singular_values": singular_values.detach().cpu().tolist(),
        "command_jacobian_condition": float((singular_values.max() / singular_values.min().clamp_min(1.0e-12)).detach()),
        "mean_step_time_seconds": elapsed / max(count, 1),
        "rollout_metrics": rollout_metrics,
        "note": "Pressure-velocity phase-plane plots and IsaacLab step tests remain separate acceptance gates.",
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if args.require_pass and report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
