#!/usr/bin/env python3
"""Gate the Stage 31 causal AAM against a constant-latent baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_split  # noqa: E402
from train_stage31_aam import AAM, load_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--aam-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    aam_root = Path(args.aam_root).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    metrics = json.loads((model_dir / "metrics.json").read_text())
    if int(metrics.get("feature_dim", -1)) != 18 or int(metrics.get("lag_ticks", -1)) != args.lag:
        raise SystemExit("AAM checkpoint does not match the Stage 31 feature/lag contract")
    norm = metrics["input_normalization"]
    x_mean = np.asarray(norm["per_position_mean"], dtype=np.float32)
    x_std = np.asarray(norm["per_position_std"], dtype=np.float32)
    device = torch.device("cpu")
    model = AAM(
        int(metrics["in_dim"]), int(metrics["latent_dim"]),
        hidden=int(metrics.get("hidden", 256)),
        depth=int(metrics.get("depth", 3)),
        dropout=float(metrics.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "aam.pt", map_location=device, weights_only=False))
    model.eval()

    split_ids: dict[str, list[int]] = {}
    for split in ("train", "val", "test"):
        split_ids[split] = [int(value) for value in load_split(root, split)]
    records: dict[str, dict[str, float | int]] = {}
    for split in ("val", "test"):
        windows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        run_ids: list[int] = []
        for run_id in split_ids[split]:
            loaded = load_windows(aam_root, run_id, args.lag)
            if loaded is None or not len(loaded["window"]):
                continue
            windows.append(loaded["window"])
            targets.append(loaded["z"])
            run_ids.extend([run_id] * len(loaded["z"]))
        if not windows:
            records[split] = {"n": 0, "aam_rmse": float("nan"), "constant_baseline_rmse": float("nan")}
            continue
        x = np.concatenate(windows).astype(np.float32)
        y = np.concatenate(targets).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.as_tensor((x - x_mean) / x_std, device=device)).numpy()
        baseline = np.stack([targets[index][0] for index in range(len(targets)) for _ in range(len(targets[index]))])
        # The target is constant within each deployment run; use the train-only
        # mean latent as the no-history control to avoid a per-run oracle.
        train_targets: list[np.ndarray] = []
        for run_id in split_ids["train"]:
            loaded = load_windows(aam_root, run_id, args.lag)
            if loaded is not None and len(loaded["z"]):
                train_targets.append(loaded["z"])
        train_mean = np.concatenate(train_targets).mean(axis=0)
        baseline[:] = train_mean
        aam_mse = float(np.mean((pred - y) ** 2))
        baseline_mse = float(np.mean((baseline - y) ** 2))
        records[split] = {
            "n": int(len(y)),
            "aam_rmse": float(np.sqrt(aam_mse)),
            "constant_baseline_rmse": float(np.sqrt(baseline_mse)),
            "relative_mse_change": float(aam_mse / max(baseline_mse, 1.0e-12) - 1.0),
        }

    failures = []
    for split in ("val", "test"):
        row = records[split]
        if int(row["n"]) == 0:
            failures.append(f"{split} AAM split is empty")
        elif not np.isfinite(float(row["aam_rmse"])) or float(row["aam_rmse"]) > float(row["constant_baseline_rmse"]):
            failures.append(f"AAM does not beat constant-latent baseline on {split}")
    report = {
        "format": "stage31_aam_gate_v1",
        "strict_pass": not failures,
        "feature_contract": "joint_q + joint_qdot + dP + previous_valve",
        "lag_ticks": args.lag,
        "splits": records,
        "failures": failures,
        "model_dir": str(model_dir),
    }
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("Stage 31 AAM gate failed")


if __name__ == "__main__":
    main()
