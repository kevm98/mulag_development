#!/usr/bin/env python
"""Assemble the Stage 26 AAM evaluation report.

Reads the trained AAM metrics (latent RMSE + temporal stability from
per-window predictions), the paired downstream ``SD_z(s, u, *)`` rollout
report (true/zhat/zero/shuffled/mean) and the zhat-vs-no-latent comparison,
and writes a single machine-readable summary used as the Stage 26 record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CHANNELS = ("extension", "velocity", "dP", "all")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aam-metrics", required=True)
    p.add_argument("--zhat-stats", required=True)
    p.add_argument("--rollout", required=True)
    p.add_argument("--vs-none", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    metrics = json.loads(Path(args.aam_metrics).read_text())
    zstats = json.loads(Path(args.zhat_stats).read_text())
    rollout = json.loads(Path(args.rollout).read_text())
    vs_none = json.loads(Path(args.vs_none).read_text())

    # Temporal stability: mean per-run std of per-window z_hat predictions.
    stds = np.stack([np.asarray(v["std"]) for v in zstats.values()], axis=0)
    temporal = {
        "n_runs": len(zstats),
        "mean_per_run_std_per_dim": [float(v) for v in stds.mean(0)],
        "mean_overall_std": float(np.sqrt((stds ** 2).mean())),
        "note": "std of AAM per-window z_hat within each held-out run",
    }

    horizon = "20"
    h20 = rollout["rollout_rmse"]
    modes = list(h20)
    downstream_h20 = {
        mode: {
            c: float(h20[mode][horizon][c])
            for c in CHANNELS
        }
        for mode in modes
    }
    paired = rollout.get("paired_vs_true", {})
    paired_h20 = {
        mode: {
            c: paired[mode][horizon][c]
            for c in CHANNELS
        }
        for mode in paired
    }
    splits_h20 = rollout.get("splits", {})

    report = {
        "stage": "26",
        "history_lag_ticks": metrics["lag_ticks"],
        "history_seconds": metrics["lag_seconds"],
        "latent": {
            "architecture": metrics["architecture"],
            "best_epoch": metrics["best_epoch"],
            "train_runs": metrics["train_runs"],
            "val_latent_rmse": metrics["val"],
            "test_latent_rmse": metrics["test"],
        },
        "temporal_stability": temporal,
        "downstream_h20_rmse": downstream_h20,
        "paired_vs_true_h20": paired_h20,
        "splits_h20_all": {
            sp: {m: splits_h20[sp][m][horizon]["all"] for m in splits_h20[sp]}
            for sp in splits_h20
        },
        "zhat_vs_no_latent_h20": vs_none,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("latent RMSE:", {
        "rmse_all": round(metrics["test"]["rmse_all"], 5),
        "rmse_per_dim": [round(v, 5) for v in metrics["test"]["rmse_per_dim"]],
    })
    print("temporal stability (mean per-run std):",
          [round(v, 5) for v in temporal["mean_per_run_std_per_dim"]])
    print("downstream H20 rollout RMSE (ext/vel/dP/all):")
    for mode in modes:
        row = downstream_h20[mode]
        print(f"  {mode:>9}: ext={row['extension']:.6g} "
              f"vel={row['velocity']:.6g} dP={row['dP']:.4g} all={row['all']:.4g}")
    print("paired zhat vs no-latent H20:",
          {c: [round(v, 5) for v in vs_none["pooled"][c]["ci95"]]
           for c in ("extension", "velocity")})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
