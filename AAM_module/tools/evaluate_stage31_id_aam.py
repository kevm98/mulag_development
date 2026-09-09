#!/usr/bin/env python3
"""Held-out valve-command gate for the Stage 31 AAM-conditioned ID layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
AAM_ROOT = ROOT / "AAM_module"
for path in (ROOT, AAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aam_data import load_runs_info, raw_dir, read_raw_run  # noqa: E402
from aam_data.rom_stage31 import rom_ticks  # noqa: E402
from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime  # noqa: E402
from tools.train_stage31_id_aam import build_run, load_zhat, runtime_mean  # noqa: E402
from tools.train_stage29_latent_adapter import ResidualAdapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--zhat", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--models-root", default=str(ROOT / "learned_models/id_fd_model/vendor_models"))
    parser.add_argument("--id-checkpoint", default=str(ROOT / "learned_models/id_fd_model/best_id_cylinder_control.pt"))
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument("--smoothing-window", type=int, default=None,
                        help="override the adapter checkpoint's causal latent window")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    payload = torch.load(Path(args.adapter).expanduser().resolve(), map_location="cpu", weights_only=False)
    config = payload["model_config"]
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    smoothing_window = int(payload.get("smoothing_window_predictions", 20))
    if args.smoothing_window is not None:
        smoothing_window = int(args.smoothing_window)
    if smoothing_window < 1:
        raise SystemExit("--smoothing-window must be positive")
    adapter = ResidualAdapter(int(config["in_dim"]), int(config["hidden"]), int(config["depth"]), int(config["out_dim"])).to(device)
    adapter.load_state_dict(payload["model_state"])
    adapter.eval()
    feature_mean = payload["feature_mean"].to(device=device, dtype=torch.float32)
    feature_std = payload["feature_std"].to(device=device, dtype=torch.float32).clamp_min(1.0e-8)
    bound = payload["residual_bound"].to(device=device, dtype=torch.float32)
    limits = payload["command_limits"].to(device=device, dtype=torch.float32)
    zhat = load_zhat(Path(args.zhat).expanduser().resolve())
    info = load_runs_info(root)
    id_runtime = CylinderIDRuntime(models_root=args.models_root, checkpoint=args.id_checkpoint, num_envs=1, device=device)
    dP_mean = runtime_mean(id_runtime, [f"dP_{i}" for i in range(1, 5)])
    force_mean = runtime_mean(id_runtime, [f"Force_{i}" for i in range(1, 5)])
    records = []
    sums = {"val": {"n": 0, "base": 0.0, "adapted": 0.0}, "test": {"n": 0, "base": 0.0, "adapted": 0.0}}
    for run_dir in sorted((raw_dir(root) / "runs").iterdir()):
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        run_id = int(run_dir.name)
        split = str(info[f"{run_id:05d}"]["frozen_split"])
        if split not in sums:
            continue
        loaded = read_raw_run(root, run_id, verify=True)
        features, target, base = build_run(
            loaded["arrays"], run_id, args.lag, zhat.get(run_id, {}),
            id_runtime, dP_mean, force_mean, smoothing_window,
        )
        X = torch.as_tensor(features, dtype=torch.float32, device=device)
        B = torch.as_tensor(base, dtype=torch.float32, device=device)
        Y = torch.as_tensor(target, dtype=torch.float32, device=device)
        with torch.no_grad():
            residual = bound * torch.tanh(adapter((X - feature_mean) / feature_std))
            final = torch.clamp(B + residual, -limits, limits)
            base_mse = float(torch.mean((B - Y) ** 2).cpu())
            adapted_mse = float(torch.mean((final - Y) ** 2).cpu())
        n = int(len(features))
        sums[split]["n"] += n
        sums[split]["base"] += base_mse * n
        sums[split]["adapted"] += adapted_mse * n
        records.append({"run_id": run_id, "split": split, "n": n, "base_valve_rmse": base_mse ** 0.5, "adapted_valve_rmse": adapted_mse ** 0.5})
    summary = {}
    for split, values in sums.items():
        if values["n"]:
            base_mse = values["base"] / values["n"]
            adapted_mse = values["adapted"] / values["n"]
            summary[split] = {"n": values["n"], "base_valve_rmse": base_mse ** 0.5, "adapted_valve_rmse": adapted_mse ** 0.5, "relative_mse_change": adapted_mse / max(base_mse, 1.0e-12) - 1.0}
    passed = bool(summary.get("val", {}).get("adapted_valve_rmse", float("inf")) <= summary.get("val", {}).get("base_valve_rmse", float("-inf")))
    report = {
        "format": "stage31_id_aam_valve_gate_v1",
        "passed": passed,
        "target": "ROM command-interface predicted valve",
        "smoothing_window_predictions": smoothing_window,
        "summary": summary,
        "per_run": records,
    }
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "summary": summary}, indent=2))
    if not passed:
        raise SystemExit("Stage 31 AAM-conditioned ID valve gate failed on validation")


if __name__ == "__main__":
    main()
