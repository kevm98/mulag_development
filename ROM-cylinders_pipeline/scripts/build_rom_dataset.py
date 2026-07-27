#!/usr/bin/env python3
"""Build leakage-safe command-conditioned ROM histories from a flat CSV export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data.rom_schema import DEFAULT_SCHEMA  # noqa: E402


META = ("run_id", "time_index", "time", "parameter_draw_id")


def assign_parameter_draws(draw_ids: np.ndarray, seed: int) -> dict[int, str]:
    unique = np.unique(draw_ids.astype(np.int64))
    if len(unique) < 3:
        raise SystemExit("At least three complete parameter draws are required for train/validation/test isolation.")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_test = max(1, int(round(0.10 * len(shuffled))))
    n_validation = max(1, int(round(0.20 * len(shuffled))))
    if n_test + n_validation >= len(shuffled):
        n_validation = 1
        n_test = 1
    mapping = {int(draw): "train" for draw in shuffled}
    for draw in shuffled[-(n_validation + n_test):-n_test]:
        mapping[int(draw)] = "validation"
    for draw in shuffled[-n_test:]:
        mapping[int(draw)] = "test"
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--history-length", type=int, default=DEFAULT_SCHEMA.history_length)
    parser.add_argument("--sample-time", type=float, default=DEFAULT_SCHEMA.sample_time_s)
    args = parser.parse_args()
    schema = DEFAULT_SCHEMA
    if args.history_length != schema.history_length or args.sample_time != schema.sample_time_s:
        raise SystemExit("The v2 runtime contract requires history=6 and dt=0.05 s.")
    frame = pd.read_csv(args.source_csv)
    required = set(META + schema.input_feature_names + schema.auxiliary_feature_names)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"Missing required source columns: {missing}")
    if "intervention_id" not in frame:
        frame["intervention_id"] = -1
    if frame[list(required)].isna().any().any():
        raise SystemExit("Required source columns contain NaN.")
    frame = frame.sort_values(["run_id", "time_index"], kind="stable").reset_index(drop=True)
    draw_splits = assign_parameter_draws(frame["parameter_draw_id"].to_numpy(), args.seed)

    packed: dict[str, dict[str, list[np.ndarray]]] = {
        name: {key: [] for key in ("history", "target", "valve_label", "run_id", "time_index", "intervention_id")}
        for name in ("train", "validation", "test")
    }
    run_splits: dict[str, list[int]] = {name: [] for name in packed}
    for run_id, run in frame.groupby("run_id", sort=True):
        run = run.reset_index(drop=True)
        dt = np.diff(run["time"].to_numpy(dtype=np.float64))
        if len(dt) and not np.allclose(dt, args.sample_time, rtol=1.0e-4, atol=1.0e-8):
            raise SystemExit(f"Run {run_id} does not have the required {args.sample_time}s timestep.")
        time_index = run["time_index"].to_numpy(dtype=np.int64)
        if len(time_index) > 1 and not np.all(np.diff(time_index) == 1):
            raise SystemExit(f"Run {run_id} has non-contiguous time_index values.")
        draw = int(run["parameter_draw_id"].iloc[0])
        if not np.all(run["parameter_draw_id"].to_numpy() == draw):
            raise SystemExit(f"Run {run_id} contains multiple parameter draws.")
        split = draw_splits[draw]
        run_splits[split].append(int(run_id))
        values = run[list(schema.input_feature_names)].to_numpy(dtype=np.float32)
        valves = run[list(schema.auxiliary_feature_names)].to_numpy(dtype=np.float32)
        ext = values[:, schema.extension_slice]
        vel = values[:, schema.velocity_slice]
        pressure = values[:, schema.pressure_slice]
        for end in range(schema.history_length - 1, len(run) - 1):
            target = np.concatenate((
                valves[end],
                pressure[end + 1] - pressure[end],
                vel[end + 1] - vel[end],
                ext[end + 1] - ext[end],
            )).astype(np.float32)
            destination = packed[split]
            destination["history"].append(values[end - schema.history_length + 1:end + 1])
            destination["target"].append(target)
            destination["valve_label"].append(valves[end])
            destination["run_id"].append(np.int64(run_id))
            destination["time_index"].append(np.int64(time_index[end]))
            destination["intervention_id"].append(np.int64(run["intervention_id"].iloc[end]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, arrays in packed.items():
        if not arrays["history"]:
            raise SystemExit(f"Split {split} has no samples; collect more complete runs/parameter draws.")
        np.savez_compressed(args.output_dir / f"{split}.npz", **{
            key: np.asarray(value, dtype=np.float32 if key in ("history", "target", "valve_label") else np.int64)
            for key, value in arrays.items()
        })
    train_history = np.asarray(packed["train"]["history"], dtype=np.float32)
    train_target = np.asarray(packed["train"]["target"], dtype=np.float32)
    input_mean = train_history.reshape(-1, schema.input_dim).mean(axis=0)
    input_std = train_history.reshape(-1, schema.input_dim).std(axis=0).clip(1.0e-8)
    target_mean = train_target.mean(axis=0)
    target_std = train_target.std(axis=0).clip(1.0e-8)
    np.savez(args.output_dir / "normalization.npz", input_mean=input_mean, input_std=input_std,
             target_mean=target_mean, target_std=target_std)
    manifest = {
        **schema.to_dict(),
        "source": str(args.source_csv.resolve()),
        "split_seed": args.seed,
        "split_rule": "complete parameter draws (and therefore complete runs); seeded approximately 70/20/10",
        "parameter_draw_splits": {str(key): value for key, value in sorted(draw_splits.items())},
        "runs": run_splits,
        "samples": {name: len(values["history"]) for name, values in packed.items()},
        "cylinder_limits_from_training_extension": np.stack((
            train_history[:, :, schema.extension_slice].min(axis=(0, 1)),
            train_history[:, :, schema.extension_slice].max(axis=(0, 1)),
        ), axis=-1).tolist(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "samples": manifest["samples"], "runs": run_splits}, indent=2))


if __name__ == "__main__":
    main()
