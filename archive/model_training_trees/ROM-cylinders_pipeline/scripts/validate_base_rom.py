#!/usr/bin/env python3
"""Offline authority, rollout, phase-plane and amplitude audit for a base ROM."""

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

from ROM_pipeline.data import BASE_ROM_SCHEMA, load_dataset_bundle  # noqa: E402
from ROM_pipeline.models import load_base_checkpoint  # noqa: E402
from ROM_pipeline.training import rollout  # noqa: E402
from ROM_pipeline.training.rollout_loss import output_tensor  # noqa: E402


def r2(true: torch.Tensor, prediction: torch.Tensor) -> list[float]:
    residual = (true - prediction).square().sum(0)
    total = (true - true.mean(0)).square().sum(0).clamp_min(1.0e-12)
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
    bundle = load_dataset_bundle(args.dataset, schema=BASE_ROM_SCHEMA)
    model, _ = load_base_checkpoint(args.checkpoint, device=device)
    count = min(args.max_samples, len(bundle.test))
    strict_source_ids = {
        index for index, source in enumerate(bundle.manifest.get("sources", []))
        if "open_loop" in str(source.get("kind", ""))
    }
    strict_indices = np.flatnonzero(np.isin(bundle.test.source_id, list(strict_source_ids)))
    base_indices = np.flatnonzero(~np.isin(bundle.test.source_id, list(strict_source_ids)))
    strict_count = min(len(strict_indices), count // 2)
    selection = np.concatenate((base_indices[:count - strict_count], strict_indices[:strict_count]))
    count = len(selection)
    history = torch.from_numpy(bundle.test.history[selection]).to(device)
    target = torch.from_numpy(bundle.test.target[selection]).to(device)
    schema = BASE_ROM_SCHEMA
    with torch.no_grad():
        started = time.perf_counter()
        prediction = output_tensor(model(history))
        elapsed = time.perf_counter() - started
        zero = history.clone()
        zero[:, :, schema.valve_slice] = 0.0
        zero_prediction = output_tensor(model(zero))
        shuffled = history.clone()
        shuffled[:, :, schema.valve_slice] = shuffled[torch.randperm(count, device=device), :, schema.valve_slice]
        shuffled_prediction = output_tensor(model(shuffled))
    jac_history = history[:min(128, count)].clone()
    valve = jac_history[:, -1, schema.valve_slice].clone().requires_grad_(True)
    delta = model(jac_history, valve).delta_extension
    jacobian = torch.stack([
        torch.autograd.grad(delta[:, axis].sum(), valve, retain_graph=True)[0].mean(0) for axis in range(4)
    ])
    singular_values = torch.linalg.svdvals(jacobian)
    rollout_metrics = {}
    with torch.no_grad():
        for horizon in (5, 10, 20):
            indices = bundle.test.sequence_indices(selection, horizon)[:256]
            if not len(indices):
                rollout_metrics[str(horizon)] = {"available_sequences": 0}
                continue
            initial = torch.from_numpy(bundle.test.history[indices[:, 0]]).to(device)
            valves = torch.from_numpy(bundle.test.history[indices, -1, schema.valve_slice]).to(device)
            teacher = torch.from_numpy(bundle.test.history[indices, -1]).to(device)
            outputs = rollout(model, initial, valves, teacher_rows=teacher, teacher_probability=0.0)
            terminal_prediction = output_tensor(outputs[-1])
            terminal_target = torch.from_numpy(bundle.test.target[indices[:, -1]]).to(device)
            rollout_metrics[str(horizon)] = {
                "available_sequences": int(len(indices)),
                "terminal_rmse_by_output": (terminal_prediction - terminal_target).square().mean(0).sqrt().cpu().tolist(),
                "terminal_r2_by_output": r2(terminal_target, terminal_prediction),
            }
        amplitude = {}
        reference = history[:min(256, count)]
        for level in (-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0):
            command = torch.full((len(reference), 4), level, device=device)
            response = model(reference, command)
            amplitude[str(level)] = {
                "mean_delta_pressure": response.delta_pressure.mean(0).cpu().tolist(),
                "mean_delta_velocity": response.delta_velocity.mean(0).cpu().tolist(),
                "mean_delta_extension": response.delta_extension.mean(0).cpu().tolist(),
            }
    current_pressure = history[:, -1, schema.pressure_slice]
    current_velocity = history[:, -1, schema.velocity_slice]
    phase_true = torch.stack((current_pressure + target[:, 0:4], current_velocity + target[:, 4:8]), dim=-1)
    phase_pred = torch.stack((current_pressure + prediction[:, 0:4], current_velocity + prediction[:, 4:8]), dim=-1)
    r2_normal = r2(target, prediction)
    r2_zero = r2(target, zero_prediction)
    r2_shuffled = r2(target, shuffled_prediction)
    pressure_r2 = float(np.mean(r2_normal[0:4]))
    velocity_r2 = float(np.mean(r2_normal[4:8]))
    extension_r2 = float(np.mean(r2_normal[8:12]))
    zero_degradation = float(np.mean(np.asarray(r2_normal[:12]) - np.asarray(r2_zero[:12])))
    shuffled_degradation = float(np.mean(np.asarray(r2_normal[:12]) - np.asarray(r2_shuffled[:12])))
    gates = {
        "pressure_mean_r2_at_least_0p2": pressure_r2 >= 0.2,
        "velocity_mean_r2_at_least_0p1": velocity_r2 >= 0.1,
        "extension_mean_r2_at_least_0p8": extension_r2 >= 0.8,
        "zeroed_valve_degradation_at_least_0p02": zero_degradation >= 0.02,
        "shuffled_valve_degradation_at_least_0p02": shuffled_degradation >= 0.02,
        "finite_full_rank_valve_jacobian": bool(torch.isfinite(singular_values).all() and singular_values.min() > 1.0e-8),
    }
    report = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "acceptance_gates": gates,
        "samples": count,
        "selected_source_counts": {str(int(source)): int(np.sum(bundle.test.source_id[selection] == source))
                                   for source in np.unique(bundle.test.source_id[selection])},
        "r2_by_output": r2_normal,
        "zeroed_valve_r2_by_output": r2_zero,
        "shuffled_valve_r2_by_output": r2_shuffled,
        "valve_to_delta_extension_jacobian": jacobian.detach().cpu().tolist(),
        "jacobian_singular_values": singular_values.detach().cpu().tolist(),
        "jacobian_condition": float((singular_values.max() / singular_values.min().clamp_min(1.0e-12)).detach()),
        "phase_plane_rmse_dP_xdot": (phase_pred - phase_true).square().mean((0, 1)).sqrt().cpu().tolist(),
        "rollout_metrics": rollout_metrics,
        "amplitude_response": amplitude,
        "mean_step_time_seconds": elapsed / max(count, 1),
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
