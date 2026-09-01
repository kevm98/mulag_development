#!/usr/bin/env python
"""Build the SD dataset from frozen raw ROM transitions (Stage 18, items 47-49).

Per raw run this tool builds one SD sample per aligned 50 ms ROM transition
(Stage 17 grid):

    s_t^H         12-D hydraulic state  [Extension 4, Velocity 4, dP 4]
    target        12-D  Delta s_{t+1}^H = s_{t+1}^H - s_t^H
    q / qdot      10-D joint context (context only; joints are never targets)
    u              8-D ROM command row
    z              6-D per-run privileged factor vector
    history        causal [s_{t-k}^H; u_{t-k}] stacks for lags 3 and 6

Samples are exact 50 ms transitions; nothing is interpolated. History
variants drop the first L-1 ticks of each run (reported, never padded).

Item 49 diagnostics are written to ``<root>/analysis/sd_diagnostics.json``:
measured input->response delays (ROM ticks) and residual autocorrelation of an
instantaneous linear predictor, i.e. the evidence used to choose the minimum
useful history before training.

Usage:
    python build_sd_dataset.py [--root datasets/master_v1_frozen]
        [--lags 3 6] [--force] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


AAM_ROOT = Path(__file__).resolve().parents[2] / "AAM_module"  # SD tools live in sd_model/; aam_data and datasets stay under AAM_module
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import (  # noqa: E402
    CONTROL_DT,
    ROM_DT,
    iter_raw_runs,
    load_runs_info,
    raw_dir,
    read_raw_run,
    sha256_file,
)
from aam_data.frozen import CHECKSUM_FILE  # noqa: E402
from aam_data.sd import (  # noqa: E402
    DEFAULT_HISTORY_LAGS,
    FACTOR_KEYS,
    build_sd_samples,
)


def measured_delays(all_u: np.ndarray, all_s: np.ndarray, max_lag: int = 6) -> dict:
    """Pooled u -> response lagged correlation; argmax lag per response block."""
    n = min(all_u.shape[0], all_s.shape[0])
    u = all_u[:n]
    s = all_s[:n]
    blocks = {"extension": (0, 4), "velocity": (4, 8), "dP": (8, 12)}
    report = {}
    for name, (lo, hi) in blocks.items():
        best_lag = {}
        best_corr = {}
        for lag in range(max_lag):
            if n - lag < 32:
                break
            uu = u[: n - lag]
            ss = s[lag:]
            for i in range(u.shape[1]):
                for j in range(lo, hi):
                    denom = np.std(uu[:, i]) * np.std(ss[:, j])
                    if denom <= 1e-12:
                        continue
                    c = abs(float(np.corrcoef(uu[:, i], ss[:, j])[0, 1]))
                    key = (i, j)
                    if key not in best_corr or c > best_corr[key]:
                        best_corr[key] = c
                        best_lag[key] = lag
        # Aggregate over input channels: report the dominant lag per response channel.
        per_channel = [
            {"response_channel": j, "argmax_lag_ticks": best_lag[(i, j)]}
            for j in range(lo, hi)
            for i in range(u.shape[1])
            if (i, j) in best_lag
        ]
        # Most common argmax lag across input x response pairs.
        lags = [p["argmax_lag_ticks"] for p in per_channel]
        report[name] = {
            "argmax_lag_ticks_mode": int(np.bincount(lags).argmax()) if lags else None,
            "lags": sorted(set(lags)),
            "n_pairs": len(per_channel),
        }
    return report


def residual_autocorr(
    samples: list[dict], max_lag: int = 6
) -> dict:
    """Per-run residual autocorrelation of an instantaneous linear predictor.

    Fits Delta s = A . [s_t^H; q; u; 1] per run and reports the median
    lag-1..max_lag autocorrelation of the residuals across runs/channels.
    Persistent autocorrelation means the instantaneous model is not Markovian
    and history is required (item 49).
    """
    n_out = 12
    by_run: dict[int, list[dict]] = {}
    for s in samples:
        by_run.setdefault(int(s["_run_id"]), []).append(s)
    agg = {lag: [] for lag in range(1, max_lag + 1)}
    n_runs_used = 0
    for run_id, run_samples in sorted(by_run.items()):
        if len(run_samples) < 64:
            continue
        X = np.stack([s["input_instantaneous"] for s in run_samples])
        Xc = np.column_stack([X, np.ones(X.shape[0])])
        Y = np.stack([s["target"] for s in run_samples])
        beta, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
        resid = Y - Xc @ beta
        n_runs_used += 1
        for ch in range(n_out):
            r = resid[:, ch]
            r = r - r.mean()
            var = float(r @ r)
            if var <= 1e-12:
                continue
            for lag in range(1, max_lag + 1):
                if len(r) <= lag:
                    continue
                ac = float((r[lag:] @ r[:-lag]) / var)
                agg[lag].append(ac)
    medians = [abs(float(np.median(v))) for v in agg.values() if v]
    return {
        "median_autocorr_by_lag": {
            str(lag): float(np.median(agg[lag])) if agg[lag] else None
            for lag in range(1, max_lag + 1)
        },
        "n_runs_used": n_runs_used,
        "max_abs_median": float(max(medians)) if medians else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=str(AAM_ROOT / "datasets/master_v1_frozen"),
        help="frozen master dataset root",
    )
    p.add_argument(
        "--out",
        default=None,
        help="derived SD dataset dir (default: <root>/sd/dataset)",
    )
    p.add_argument(
        "--report",
        default=None,
        help="diagnostics json (default: <root>/analysis/sd_diagnostics.json)",
    )
    p.add_argument("--lags", type=int, nargs="+", default=list(DEFAULT_HISTORY_LAGS))
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    raw = raw_dir(root)
    if not (raw / "manifest.csv").is_file():
        raise SystemExit(f"not a frozen dataset: {root}")
    out = Path(args.out or str(root / "sd" / "dataset")).expanduser().resolve()
    report_path = Path(
        args.report or str(root / "analysis" / "sd_diagnostics.json")
    ).expanduser().resolve()
    if out == raw or raw in out.parents:
        raise SystemExit(f"refusing to write SD data into raw/: {out}")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output exists and is not empty: {out} (use --force)")
    if out.exists():
        shutil.rmtree(out)
    lags = tuple(sorted(set(args.lags)))

    try:
        runs_info = load_runs_info(root)
    except FileNotFoundError:
        runs_info = {}
    checksum_digest = sha256_file(raw / CHECKSUM_FILE)

    all_u: list[np.ndarray] = []
    all_s: list[np.ndarray] = []
    all_samples: list[dict] = []
    per_run: list[dict] = []
    n_total = 0
    for run_id, _run_path in iter_raw_runs(root):
        if args.limit is not None and run_id > args.limit - 1:
            break
        data = read_raw_run(root, run_id, verify=True)
        meta, arrays = data["metadata"], data["arrays"]
        built = build_sd_samples(arrays, meta, history_lags=lags)
        samples = built["samples"]
        n = len(samples)
        n_total += n
        run_out = out / f"{run_id:05d}"
        run_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_out / "samples.npz",
            tick_control_step=np.asarray(
                [s["tick_control_step"] for s in samples], dtype=np.int64
            ),
            s_h=np.stack([s["s_h"] for s in samples]),
            target=np.stack([s["target"] for s in samples]),
            q=np.stack([s["q"] for s in samples]),
            u=np.stack([s["u"] for s in samples]),
            z=np.asarray([s["z"] for s in samples]),
            input_instantaneous=np.stack([s["input_instantaneous"] for s in samples]),
            cylinder_limit_flag=np.stack(
                [s["flags"]["cylinder_limit_flag"] for s in samples]
            ),
            command_saturation_flag=np.stack(
                [s["flags"]["command_saturation_flag"] for s in samples]
            ),
            valve_saturation_flag=np.stack(
                [s["flags"]["valve_saturation_flag"] for s in samples]
            ),
            invalid_physics_flag=np.stack(
                [s["flags"]["invalid_physics_flag"] for s in samples]
            ),
        )
        for lag in lags:
            hist = built["histories"][lag]
            np.savez_compressed(
                run_out / f"history_L{lag}.npz",
                history_input=hist["stack"],
                valid_mask=hist["valid_mask"],
            )
        (run_out / "sample_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "domain": meta["domain"],
                    "frozen_split": runs_info.get(f"{run_id:05d}", {}).get("frozen_split"),
                    "n_samples": n,
                    "z": {k: float(meta[k]) for k in FACTOR_KEYS},
                    "history_lags": {
                        str(lag): {
                            "n_valid": built["histories"][lag]["n_valid"],
                            "n_dropped": built["histories"][lag]["n_dropped"],
                        }
                        for lag in lags
                    },
                    "grid_uniform": built["grid"]["uniform"],
                    "target": "Delta s_{t+1}^H = s_{t+1}^H - s_t^H (12-D); joints context only",
                    "no_interpolation": True,
                    "built_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        per_run.append(
            {
                "run_id": run_id,
                "domain": meta["domain"],
                "frozen_split": runs_info.get(f"{run_id:05d}", {}).get("frozen_split"),
                "n_samples": n,
                "history_dropped": {
                    str(lag): built["histories"][lag]["n_dropped"] for lag in lags
                },
            }
        )
        all_u.append(np.stack([s["u"] for s in samples]))
        all_s.append(np.stack([s["s_h"] for s in samples]))
        for s in samples:
            s["_run_id"] = run_id
        all_samples.extend(samples)
        print(
            f"run {run_id:05d} [{meta['domain']}]: {n} SD samples "
            f"(hist drops { {str(lag): built['histories'][lag]['n_dropped'] for lag in lags} })",
            flush=True,
        )

    U = np.concatenate(all_u) if all_u else np.zeros((0, 8))
    S = np.concatenate(all_s) if all_s else np.zeros((0, 12))
    delays = measured_delays(U, S)
    resid = residual_autocorr(all_samples)

    (out / "index.json").write_text(
        json.dumps(
            {
                "frozen_root": str(root),
                "raw_checksums_digest": checksum_digest,
                "timebase": {"control_dt": CONTROL_DT, "rom_dt": ROM_DT},
                "s_h": "Extension 4 + Velocity 4 + dP 4 (12)",
                "target": "Delta s_{t+1}^H (12); joints context only",
                "history_lags": list(lags),
                "history_channels_per_lag": 20,
                "n_runs": len(per_run),
                "n_samples_total": n_total,
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "measured_delays_ticks": delays,
                "residual_autocorrelation": resid,
                "note": (
                    "item 49: history is required while residual autocorrelation "
                    "is non-negligible and beyond the measured delay; final L is "
                    "confirmed by validation rollout performance at training time."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{len(per_run)} runs, {n_total} SD samples -> {out}")
    print("measured delays (ticks):", json.dumps(delays))
    print("residual autocorr:", json.dumps(resid))
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
