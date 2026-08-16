#!/usr/bin/env python3
"""Export causal Stage 31 AAM predictions keyed by ROM tick."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_split  # noqa: E402
from tools.train_stage31_aam import AAM, load_windows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--aam-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    aam_root = Path(args.aam_root).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    metrics = json.loads((model_dir / "metrics.json").read_text())
    lag = int(metrics["lag_ticks"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = AAM(
        int(metrics["in_dim"]), int(metrics["latent_dim"]),
        int(metrics.get("hidden", 256)), int(metrics.get("depth", 3)), float(metrics.get("dropout", 0.05)),
        architecture=str(metrics.get("architecture", "mlp")),
        feature_dim=int(metrics.get("feature_dim", 18)),
        lag=lag,
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "aam.pt", map_location=device, weights_only=False))
    model.eval()
    mean = np.asarray(metrics["input_normalization"]["per_position_mean"], dtype=np.float64)
    std = np.asarray(metrics["input_normalization"]["per_position_std"], dtype=np.float64).clip(min=1.0e-8)
    run_ids = sorted({int(run_id) for split in args.splits for run_id in load_split(root, split)})
    by_run: dict[str, dict[str, list[float]]] = {}
    with torch.no_grad():
        for run_id in run_ids:
            data = load_windows(aam_root, run_id, lag)
            if data is None:
                raise SystemExit(f"missing Stage 31 AAM windows for run {run_id:05d}")
            X = torch.as_tensor(((data["window"] - mean) / std).astype(np.float32), device=device)
            predictions = []
            for start in range(0, len(X), args.batch_size):
                predictions.append(model(X[start:start + args.batch_size]).cpu().numpy())
            pred = np.concatenate(predictions, axis=0)
            if not np.isfinite(pred).all():
                raise SystemExit(f"run {run_id:05d}: non-finite AAM prediction")
            by_run[str(run_id)] = {
                str(int(tick)): [float(value) for value in row]
                for tick, row in zip(data["ticks"], pred)
            }
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "format": "stage31_aam_zhat_by_window_end_tick_v1",
                "lag_ticks": lag,
                "tick_timebase": "control_60hz",
                "rom_steps_per_tick": 3,
                "model_dir": str(model_dir),
                "feature_contract": metrics["feature_contract"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "by_run": by_run,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"exported causal Stage 31 z_hat for {len(by_run)} runs -> {output}")


if __name__ == "__main__":
    main()
