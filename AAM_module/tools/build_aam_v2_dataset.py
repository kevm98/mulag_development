#!/usr/bin/env python3
"""Build a direct, six-factor AAM_v2 dataset from frozen Stage-31 ROM rows.

This deliberately does not load the legacy 2-D privileged encoder.  The
Stage-31 collection holds one verified, constant physical plant per run, so
each valid causal window receives that run's applied factor vector.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import FACTOR_KEYS, load_runs_info, load_split  # noqa: E402
from aam_data.rom_stage31 import AAM_FEATURE_DIM, AAM_FEATURE_GROUPS, causal_stack  # noqa: E402


def constant_applied_factors(root: Path, run_id: int) -> tuple[np.ndarray, dict]:
    """Return the verified fixed plant identity, rejecting dynamic ambiguity."""
    run_dir = root / "raw" / "runs" / f"{run_id:05d}"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if (run_dir / "factor_track.npz").exists() or metadata.get("factor_schedule"):
        raise SystemExit(
            f"run {run_id:05d} is time-varying; this constant-target v2 builder "
            "must not reconstruct it from run metadata"
        )
    target = np.asarray([metadata[k] for k in FACTOR_KEYS], dtype=np.float64)
    if not np.isfinite(target).all():
        raise SystemExit(f"run {run_id:05d} has non-finite factor metadata")
    if not np.isclose(target[4], metadata["applied_mass_scale"]):
        raise SystemExit(f"run {run_id:05d}: mass_scale differs from applied metadata")
    if not np.isclose(target[5], metadata["applied_com_z"]):
        raise SystemExit(f"run {run_id:05d}: com_z differs from applied metadata")
    return target.astype(np.float32), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="frozen Stage-31 root")
    parser.add_argument("--rows", default=None, help="existing aam/rom_rows directory")
    parser.add_argument("--out", default=None, help="new aam_v2/dataset directory")
    parser.add_argument("--lags", type=int, nargs="+", default=[20])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    rows_root = Path(args.rows or root / "aam" / "rom_rows").expanduser().resolve()
    out = Path(args.out or root / "aam_v2" / "dataset").expanduser().resolve()
    if not rows_root.is_dir():
        raise SystemExit(f"missing ROM rows: {rows_root}")
    if out.exists():
        if not args.force:
            raise SystemExit(f"output exists: {out} (use --force only for this v2 output)")
        shutil.rmtree(out)
    lags = tuple(sorted(set(args.lags)))
    if not lags or any(lag < 1 for lag in lags):
        raise SystemExit("--lags must be positive")

    info = load_runs_info(root)
    run_ids = sorted(int(path.name) for path in rows_root.iterdir() if path.is_dir() and path.name.isdigit())
    if not run_ids:
        raise SystemExit("no numeric run directories found")
    targets: dict[int, np.ndarray] = {}
    metadatas: dict[int, dict] = {}
    for run_id in run_ids:
        targets[run_id], metadatas[run_id] = constant_applied_factors(root, run_id)

    train_ids = set(load_split(root, "train"))
    if not set(run_ids).issubset(set().union(train_ids, set(load_split(root, "val")), set(load_split(root, "test")))):
        raise SystemExit("ROM rows contain a run absent from the frozen split")
    train_target = np.stack([targets[rid] for rid in run_ids if rid in train_ids])
    mean = train_target.mean(axis=0)
    std = train_target.std(axis=0)
    if (std < 1.0e-8).any():
        bad = [FACTOR_KEYS[i] for i in np.flatnonzero(std < 1.0e-8)]
        raise SystemExit(f"zero training target variation: {bad}")

    out.mkdir(parents=True)
    per_run = []
    for run_id in run_ids:
        with np.load(rows_root / f"{run_id:05d}" / "rom_rows.npz", allow_pickle=False) as rows:
            features = np.asarray(rows["features"], dtype=np.float32)
            ticks = np.asarray(rows["tick_control_step"], dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != AAM_FEATURE_DIM:
            raise SystemExit(f"run {run_id:05d}: invalid feature shape {features.shape}")
        if ticks.shape != (len(features),):
            raise SystemExit(f"run {run_id:05d}: tick/feature length mismatch")
        run_out = out / f"{run_id:05d}"
        run_out.mkdir()
        target = targets[run_id]
        for lag in lags:
            windows, valid = causal_stack(features, lag)
            valid_idx = np.flatnonzero(valid)
            e_target = np.repeat(target[None, :], len(valid_idx), axis=0)
            e_normalized = ((e_target - mean) / std).astype(np.float32)
            # Windows are already large; avoiding per-run DEFLATE keeps this
            # deterministic CPU-only build practical and does not alter NPZ
            # contents or the downstream loading contract.
            np.savez(
                run_out / f"history_L{lag}.npz",
                window_input=windows.astype(np.float32),
                e_target=e_target,
                e_target_normalized=e_normalized,
                valid_mask=valid,
                window_end_tick=ticks[valid_idx],
            )
        split = info[f"{run_id:05d}"]["frozen_split"]
        (run_out / "sample_meta.json").write_text(json.dumps({
            "run_id": run_id,
            "frozen_split": split,
            "factor_keys": list(FACTOR_KEYS),
            "factor_target_source": "verified constant raw metadata/applied fields",
            "factors_constant_within_run": True,
            "physical_factors": target.tolist(),
        }, indent=2) + "\n")
        per_run.append({"run_id": run_id, "split": split, "n_rows": int(len(features))})

    normalization = {
        "kind": "zscore_train_runs_only",
        "factor_keys": list(FACTOR_KEYS),
        "mean": mean.tolist(), "std": std.tolist(),
        "fit_run_ids": sorted(train_ids),
        "inverse": "e_physical = e_normalized * std + mean",
    }
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    (out / "index.json").write_text(json.dumps({
        "format": "aam_v2_direct_factor_dataset_v1",
        "source_rows": str(rows_root), "factor_keys": list(FACTOR_KEYS),
        "feature_dim": AAM_FEATURE_DIM, "feature_groups": AAM_FEATURE_GROUPS,
        "lags": list(lags), "target": "constant verified applied factors per run",
        "normalization": "normalization.json", "runs": per_run,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
