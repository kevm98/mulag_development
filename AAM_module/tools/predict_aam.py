#!/usr/bin/env python
"""Causal, inference-only AAM prediction export (Stages 25–27).

The training script historically exported only the mean prediction over an
entire held-out run. That is useful as a summary, but it is not a deployable
online estimate because it includes future windows. This tool exports the
prediction for each valid window, keyed by its causal window-end tick, without
retraining or changing the checkpoint.

Output format::

    {"run_id": {"window_end_tick": [z0, z1], ...}, ...}
"""

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
from tools.train_aam import AAM, load_windows  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="frozen master dataset root")
    p.add_argument("--aam-root", default=None, help="derived AAM dataset root")
    p.add_argument("--model-dir", required=True, help="trained AAM directory")
    p.add_argument("--out", required=True, help="causal z_hat JSON output")
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=("val", "test"),
        help="frozen dataset splits to export (default: val test)",
    )
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    aam_root = Path(args.aam_root or root / "aam" / "dataset").expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    metrics = json.loads((model_dir / "metrics.json").read_text())
    lag = int(metrics["lag_ticks"])
    latent_dim = int(metrics["latent_dim"])
    in_dim = int(metrics["in_dim"])
    norm = metrics["input_normalization"]
    x_mean = np.asarray(norm["per_position_mean"], dtype=np.float64)
    x_std = np.asarray(norm["per_position_std"], dtype=np.float64)
    if x_mean.shape != (in_dim,) or x_std.shape != (in_dim,):
        raise SystemExit("AAM input normalization shape does not match checkpoint")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    model = AAM(
        in_dim,
        latent_dim,
        hidden=int(metrics.get("hidden", 256) if args.hidden is None else args.hidden),
        depth=int(metrics.get("depth", 3) if args.depth is None else args.depth),
        dropout=float(metrics.get("dropout", 0.05) if args.dropout is None else args.dropout),
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "aam.pt", map_location=device))
    model.eval()

    ids = sorted({run_id for split in args.splits for run_id in load_split(root, split)})
    by_run: dict[str, dict[str, list[float]]] = {}
    with torch.no_grad():
        for rid in ids:
            loaded = load_windows(aam_root, rid, lag)
            if loaded is None:
                raise SystemExit(f"missing L{lag} windows for run {rid:05d}")
            W = loaded["window"]
            ticks = loaded["window_end_tick"]
            if W.shape[0] != ticks.shape[0]:
                raise SystemExit(f"run {rid:05d}: window/tick count mismatch")
            Wn = ((W - x_mean) / x_std).astype(np.float32)
            preds = []
            Xt = torch.as_tensor(Wn, device=device)
            for start in range(0, Xt.shape[0], args.batch_size):
                preds.append(model(Xt[start:start + args.batch_size]).cpu().numpy())
            pred = np.concatenate(preds, axis=0)
            if not np.isfinite(pred).all():
                raise SystemExit(f"run {rid:05d}: non-finite AAM predictions")
            by_run[str(rid)] = {
                str(int(tick)): [float(v) for v in row]
                for tick, row in zip(ticks, pred)
            }

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "aam_zhat_by_window_end_tick_v1",
        "lag_ticks": lag,
        "model_dir": str(model_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "by_run": by_run,
    }, indent=2) + "\n")
    print(f"exported causal z_hat for {len(by_run)} held-out runs -> {out}")


if __name__ == "__main__":
    main()
