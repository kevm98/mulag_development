#!/usr/bin/env python
"""Apply a strictly past-only smoother to causal AAM predictions.

The input must be the output of ``predict_aam.py``. At window-end tick ``t``
the output uses only predictions at ticks ``<= t``; no held-out future window
is read. This is an inference-only improvement and does not retrain the AAM.

Methods:

``raw``
    The original per-window prediction.
``cumulative``
    Mean of all predictions observed so far in the run.
``rolling``
    Mean of the most recent ``--window`` observed predictions, including the
    current tick.
``ewma``
    Exponentially weighted mean with current-sample weight ``--alpha``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def smooth(values: np.ndarray, method: str, window: int, alpha: float) -> np.ndarray:
    if method == "raw":
        return values.copy()
    out = np.empty_like(values)
    if method == "cumulative":
        np.cumsum(values, axis=0, out=out)
        out /= np.arange(1, values.shape[0] + 1, dtype=np.float64)[:, None]
        return out
    if method == "rolling":
        if window < 1:
            raise ValueError("rolling window must be positive")
        for i in range(values.shape[0]):
            out[i] = values[max(0, i - window + 1):i + 1].mean(axis=0)
        return out
    if method == "ewma":
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EWMA alpha must be in (0, 1]")
        out[0] = values[0]
        for i in range(1, values.shape[0]):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out
    raise ValueError(f"unknown smoothing method: {method}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="predict_aam.py JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=("raw", "cumulative", "rolling", "ewma"),
                   default="cumulative")
    p.add_argument("--window", type=int, default=20,
                   help="past prediction count for rolling")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="current-sample weight for EWMA")
    args = p.parse_args()

    src = json.loads(Path(args.input).read_text())
    if src.get("format") != "aam_zhat_by_window_end_tick_v1":
        raise SystemExit("input is not a causal AAM prediction file")
    output_runs: dict[str, dict[str, list[float]]] = {}
    for rid, tick_map in src["by_run"].items():
        ticks = sorted((int(t), v) for t, v in tick_map.items())
        if not ticks:
            raise SystemExit(f"run {rid}: no causal predictions")
        tick_ids = [t for t, _ in ticks]
        values = np.asarray([v for _, v in ticks], dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise SystemExit(f"run {rid}: invalid prediction matrix")
        filtered = smooth(values, args.method, args.window, args.alpha)
        output_runs[rid] = {
            str(t): [float(v) for v in row]
            for t, row in zip(tick_ids, filtered)
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "aam_zhat_by_window_end_tick_v1",
        "lag_ticks": int(src["lag_ticks"]),
        "model_dir": src.get("model_dir"),
        "causal": True,
        "smoother": {
            "method": args.method,
            "window": args.window if args.method == "rolling" else None,
            "alpha": args.alpha if args.method == "ewma" else None,
            "uses_only_past_predictions": True,
        },
        "by_run": output_runs,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote past-only {args.method} z_hat -> {out}")


if __name__ == "__main__":
    main()
