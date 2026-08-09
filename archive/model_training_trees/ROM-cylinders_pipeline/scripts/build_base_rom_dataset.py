#!/usr/bin/env python3
"""Build the valve-conditioned base ROM dataset from original Simscape curricula."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data import BASE_ROM_SCHEMA  # noqa: E402


DEFAULT_SOURCES = (
    "/home/rrlab/Desktop/hydraulic_manipulator/Hydraulic_Physics_Curriculum_10_2.mat",
    "/home/rrlab/Desktop/hydraulic_manipulator/Hydraulic_Physics_Curriculum_10.mat",
    "/home/rrlab/Desktop/hydraulic_manipulator/Hydraulic_Physics_Curriculum_20.mat",
    "/home/rrlab/Desktop/hydraulic_manipulator/Hydraulic_Mechanic_Dataset_Curriculum_1000.mat",
)
DEFAULT_STRICT_CSV = ROOT.parent / "PPO1_ID_FD_PhysX" / "training_data" / "Excitation_FD_Strict.csv"


def rows(dataset: h5py.Dataset, width: int, sample_count: int) -> np.ndarray:
    value = np.asarray(dataset, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"Expected a matrix, got {value.shape} at {dataset.name}")
    if value.shape[0] == sample_count:
        pass
    elif value.shape[1] == sample_count:
        value = value.T
    else:
        raise ValueError(f"Neither axis matches {sample_count} samples at {dataset.name}: {value.shape}")
    if value.shape[1] < width:
        raise ValueError(f"Expected at least {width} columns, got {value.shape} at {dataset.name}")
    return value[:, :width]


def split_runs(run_ids: np.ndarray, seed: int) -> dict[int, str]:
    shuffled = np.random.default_rng(seed).permutation(np.unique(run_ids))
    if len(shuffled) < 3:
        raise ValueError("At least three complete Simscape runs are required.")
    n_test = max(1, int(round(0.10 * len(shuffled))))
    n_validation = max(1, int(round(0.10 * len(shuffled))))
    mapping = {int(run): "train" for run in shuffled}
    for run in shuffled[-(n_validation + n_test):-n_test]:
        mapping[int(run)] = "validation"
    for run in shuffled[-n_test:]:
        mapping[int(run)] = "test"
    return mapping


def extract_run(group: h5py.Group, run_id: int, source_id: int) -> dict[str, np.ndarray] | None:
    schema = BASE_ROM_SCHEMA
    time = np.asarray(group["Time"], dtype=np.float64).reshape(-1)
    if len(time) < schema.history_length + 1:
        return None
    if not np.all(np.diff(time) > 0.0):
        return None
    dt = float(np.median(np.diff(time)))
    if not np.isclose(dt, schema.sample_time_s, rtol=1.0e-3, atol=1.0e-7):
        raise ValueError(f"Run {run_id} has dt={dt}, expected {schema.sample_time_s}.")
    valve = rows(group["ValveCmd"], 4, len(time))
    extension = rows(group["Extension"], 4, len(time))
    pressure = rows(group["dP"], 4, len(time))
    force = rows(group["Force"], 4, len(time))
    lengths = {len(value) for value in (time, valve, extension, pressure, force)}
    if len(lengths) != 1:
        raise ValueError(f"Run {run_id} contains misaligned signal lengths {sorted(lengths)}.")
    velocity = np.gradient(extension, time, axis=0, edge_order=2)
    features = np.concatenate((valve, extension, velocity, pressure, force), axis=1)
    count = len(time) - schema.history_length
    windows = np.lib.stride_tricks.sliding_window_view(features, schema.history_length, axis=0)
    history = windows[:count].transpose(0, 2, 1).astype(np.float32, copy=True)
    end = np.arange(schema.history_length - 1, len(time) - 1)
    target = np.concatenate((
        pressure[end + 1] - pressure[end],
        velocity[end + 1] - velocity[end],
        extension[end + 1] - extension[end],
        force[end + 1],
    ), axis=1).astype(np.float32)
    finite = np.isfinite(history).all(axis=(1, 2)) & np.isfinite(target).all(axis=1)
    return {
        "history": history[finite],
        "target": target[finite],
        "valve_label": features[end, schema.valve_slice].astype(np.float32)[finite],
        "run_id": np.full(finite.sum(), run_id, dtype=np.int64),
        "time_index": end.astype(np.int64)[finite],
        "intervention_id": np.full(finite.sum(), -1, dtype=np.int64),
        "source_id": np.full(finite.sum(), source_id, dtype=np.int64),
    }


def extract_strict_runs(path: Path, first_run_id: int, source_id: int) -> list[dict[str, np.ndarray]]:
    """Convert verified 0.01 s MIMO PRBS runs to native 0.05 s ROM transitions."""
    frame = pd.read_csv(path)
    schema = BASE_ROM_SCHEMA
    columns = {
        "valve": [f"ValveCmd_{i}" for i in range(1, 5)],
        "extension": [f"Extension_{i}" for i in range(1, 5)],
        "velocity": [f"ExtensionVelocity_{i}" for i in range(1, 5)],
        "pressure": [f"dP_{i}" for i in range(1, 5)],
        "force": [f"Force_{i}" for i in range(1, 5)],
    }
    missing = sorted({name for names in columns.values() for name in names}.difference(frame.columns))
    if missing:
        raise ValueError(f"Strict excitation CSV is missing {missing}")
    records = []
    split_order = ("train", "train", "validation", "test")
    for order, (source_run, run) in enumerate(frame.groupby("run_id", sort=True)):
        run = run.sort_values("t", kind="stable").reset_index(drop=True)
        raw_time = run["t"].to_numpy(np.float64)
        if not np.allclose(np.diff(raw_time), 0.01, rtol=1.0e-4, atol=1.0e-8):
            raise ValueError(f"Strict run {source_run} is not sampled at 0.01 s.")
        index = np.arange(0, len(run), 5, dtype=np.int64)
        if len(index) < schema.history_length + 1:
            continue
        valve_raw = run[columns["valve"]].to_numpy(np.float64)
        valve = np.stack([valve_raw[i:min(i + 5, len(run))].mean(0) for i in index])
        extension = run[columns["extension"]].to_numpy(np.float64)[index]
        velocity = run[columns["velocity"]].to_numpy(np.float64)[index]
        pressure = run[columns["pressure"]].to_numpy(np.float64)[index]
        force = run[columns["force"]].to_numpy(np.float64)[index]
        features = np.concatenate((valve, extension, velocity, pressure, force), 1)
        count = len(index) - schema.history_length
        history = np.lib.stride_tricks.sliding_window_view(
            features, schema.history_length, axis=0
        )[:count].transpose(0, 2, 1).astype(np.float32, copy=True)
        end = np.arange(schema.history_length - 1, len(index) - 1)
        target = np.concatenate((
            pressure[end + 1] - pressure[end], velocity[end + 1] - velocity[end],
            extension[end + 1] - extension[end], force[end + 1],
        ), 1).astype(np.float32)
        finite = np.isfinite(history).all((1, 2)) & np.isfinite(target).all(1)
        global_run = first_run_id + order
        record = {
            "history": history[finite], "target": target[finite],
            "valve_label": valve[end].astype(np.float32)[finite],
            "run_id": np.full(finite.sum(), global_run, np.int64),
            "time_index": end.astype(np.int64)[finite],
            "intervention_id": np.full(finite.sum(), -1, np.int64),
            "source_id": np.full(finite.sum(), source_id, np.int64),
            "_forced_split": split_order[min(order, len(split_order) - 1)],
        }
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sources", nargs="*", type=Path, default=[Path(value) for value in DEFAULT_SOURCES])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-runs-per-file", type=int, default=0, help="Nonzero is intended only for smoke tests.")
    parser.add_argument("--strict-csv", type=Path, default=DEFAULT_STRICT_CSV)
    args = parser.parse_args()
    records = []
    source_manifest = []
    global_run_id = 0
    for source_id, source in enumerate(args.sources):
        if not source.is_file():
            raise SystemExit(f"Missing Simscape source: {source}")
        accepted = 0
        with h5py.File(source, "r") as handle:
            references = np.asarray(handle["Dataset/Data"]).reshape(-1)
            limit = len(references) if args.max_runs_per_file <= 0 else min(len(references), args.max_runs_per_file)
            for local_run_id, reference in enumerate(references[:limit]):
                record = extract_run(handle[reference], global_run_id, source_id)
                global_run_id += 1
                if record is not None and len(record["history"]):
                    records.append(record)
                    accepted += 1
        source_manifest.append({"path": str(source.resolve()), "accepted_runs": accepted})
    if args.strict_csv.is_file():
        strict_records = extract_strict_runs(args.strict_csv, global_run_id, len(source_manifest))
        records.extend(strict_records)
        source_manifest.append({"path": str(args.strict_csv.resolve()), "accepted_runs": len(strict_records),
                                "kind": "rank4_open_loop_prbs_0p01_to_0p05"})
    if len(records) < 3:
        raise SystemExit("Fewer than three valid runs were extracted.")
    run_ids = np.asarray([int(record["run_id"][0]) for record in records])
    assignment = split_runs(run_ids, args.seed)
    split_records = {name: [] for name in ("train", "validation", "test")}
    for record in records:
        split = record.get("_forced_split", assignment[int(record["run_id"][0])])
        split_records[split].append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packed = {}
    for split, selected in split_records.items():
        packed[split] = {
            key: np.concatenate([record[key] for record in selected], axis=0)
            for key in ("history", "target", "valve_label", "run_id", "time_index", "intervention_id", "source_id")
        }
        np.savez_compressed(args.output_dir / f"{split}.npz", **packed[split])
    schema = BASE_ROM_SCHEMA
    train_history = packed["train"]["history"]
    train_target = packed["train"]["target"]
    flattened = train_history.reshape(-1, schema.input_dim)
    np.savez(
        args.output_dir / "normalization.npz",
        input_mean=flattened.mean(0), input_std=flattened.std(0).clip(1.0e-8),
        target_mean=train_target.mean(0), target_std=train_target.std(0).clip(1.0e-8),
    )
    manifest = {
        **schema.to_dict(),
        "purpose": "valve-conditioned base hydraulic ROM; not desired-command PPO2 ROM",
        "sources": source_manifest,
        "split_seed": args.seed,
        "split_rule": "complete Simscape parameter-draw runs, seeded 80/10/10",
        "runs": {name: sorted({int(value) for value in arrays["run_id"]}) for name, arrays in packed.items()},
        "samples": {name: len(arrays["history"]) for name, arrays in packed.items()},
        "cylinder_limits_from_training_extension": np.stack((
            train_history[:, :, schema.extension_slice].min((0, 1)),
            train_history[:, :, schema.extension_slice].max((0, 1)),
        ), axis=-1).tolist(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "samples": manifest["samples"], "sources": source_manifest}, indent=2))


if __name__ == "__main__":
    main()
