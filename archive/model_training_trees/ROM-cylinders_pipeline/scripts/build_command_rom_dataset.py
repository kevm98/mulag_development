#!/usr/bin/env python3
"""Build the cylinder-only desired-command ROM dataset from aligned ID/FD exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data import DEFAULT_SCHEMA  # noqa: E402


DEFAULT_ID = REPO_ROOT / "PPO1_ID_FD_PhysX/models/id/ID_cylinder_data.mat"
DEFAULT_FD = REPO_ROOT / "PPO1_ID_FD_PhysX/models/fd/FD_cylinder_data.mat"


def matrix(handle: h5py.File, name: str, width: int) -> np.ndarray:
    value = np.asarray(handle[name], dtype=np.float32)
    if value.shape[0] == width:
        value = value.T
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must contain {width} features, got {value.shape}.")
    return value


def vector(handle: h5py.File, name: str, dtype) -> np.ndarray:
    return np.asarray(handle[name], dtype=dtype).reshape(-1)


def load_split(id_handle: h5py.File, fd_handle: h5py.File, split: str) -> dict[str, np.ndarray]:
    prefix = "Train" if split == "train" else "Val"
    x_id = matrix(id_handle, f"X_{split}_id", 24)
    valve = matrix(id_handle, f"Y_{split}_id", 4)
    x_fd = matrix(fd_handle, f"X_{split}_fd", 20)
    y_fd = matrix(fd_handle, f"Y_{split}_fd", 12)
    run_id = vector(id_handle, f"{prefix}RunID", np.int64)
    time_index = vector(id_handle, f"{prefix}TimeIndex", np.int64)
    sample_time = vector(id_handle, f"{prefix}SampleTime", np.float32)
    source_class = vector(id_handle, f"{prefix}Class", np.int64)

    count = len(x_id)
    if not all(len(value) == count for value in (
        valve, x_fd, y_fd, run_id, time_index, sample_time, source_class
    )):
        raise ValueError(f"Aligned {split} arrays do not have equal length.")
    if not np.array_equal(run_id, vector(fd_handle, f"{prefix}RunID", np.int64)):
        raise ValueError(f"ID/FD {split} run IDs are not aligned.")
    if not np.array_equal(time_index, vector(fd_handle, f"{prefix}TimeIndex", np.int64)):
        raise ValueError(f"ID/FD {split} time indices are not aligned.")
    if not np.allclose(x_id[:, :16], x_fd[:, 4:20], rtol=0.0, atol=0.0):
        raise ValueError(f"ID/FD {split} cylinder states are not exactly aligned.")
    if not np.allclose(valve, x_fd[:, :4], rtol=0.0, atol=1.0e-6):
        raise ValueError(f"ID output and FD valve input differ in {split}.")
    if not np.allclose(sample_time, DEFAULT_SCHEMA.sample_time_s, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError(f"{split} sample time does not match the ROM schema.")
    return {
        "cylinder_state": x_id[:, :12],
        "command": x_id[:, 16:24],
        "valve": valve,
        "fd_target": y_fd,
        "run_id": run_id,
        "time_index": time_index,
        "source_class": source_class,
    }


def windows(raw: dict[str, np.ndarray], selected_runs: set[int]) -> dict[str, np.ndarray]:
    schema = DEFAULT_SCHEMA
    output = {name: [] for name in (
        "history", "target", "valve_label", "run_id", "time_index", "intervention_id", "source_id"
    )}
    for run in sorted(selected_runs):
        rows = np.flatnonzero(raw["run_id"] == run)
        rows = rows[np.argsort(raw["time_index"][rows], kind="stable")]
        times = raw["time_index"][rows]
        if len(rows) <= schema.history_length or not np.all(np.diff(times) == 1):
            continue
        valve = raw["valve"][rows]
        previous_valve = np.concatenate((np.zeros((1, 4), dtype=np.float32), valve[:-1]), axis=0)
        features = np.concatenate(
            (raw["cylinder_state"][rows], previous_valve, raw["command"][rows]), axis=1
        )
        history = np.lib.stride_tricks.sliding_window_view(
            features, schema.history_length, axis=0
        ).transpose(0, 2, 1)
        end_local = np.arange(schema.history_length - 1, len(rows))
        history = history[:len(end_local)].astype(np.float32, copy=True)
        fd = raw["fd_target"][rows[end_local]]
        target = np.concatenate(
            (valve[end_local], fd[:, 8:12], fd[:, 4:8], fd[:, 0:4]), axis=1
        )
        finite = np.isfinite(history).all((1, 2)) & np.isfinite(target).all(1)
        source = int(raw["source_class"][rows[0]])
        output["history"].append(history[finite])
        output["target"].append(target.astype(np.float32)[finite])
        output["valve_label"].append(valve[end_local].astype(np.float32)[finite])
        output["run_id"].append(np.full(finite.sum(), run, dtype=np.int64))
        output["time_index"].append(times[end_local].astype(np.int64)[finite])
        output["intervention_id"].append(np.full(finite.sum(), -1, dtype=np.int64))
        output["source_id"].append(np.full(finite.sum(), source, dtype=np.int64))
    if not output["history"]:
        raise ValueError("No complete command-conditioned histories were produced.")
    return {name: np.concatenate(values, axis=0) for name, values in output.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--id-mat", type=Path, default=DEFAULT_ID)
    parser.add_argument("--fd-mat", type=Path, default=DEFAULT_FD)
    args = parser.parse_args()
    with h5py.File(args.id_mat, "r") as id_handle, h5py.File(args.fd_mat, "r") as fd_handle:
        train_raw = load_split(id_handle, fd_handle, "train")
        held_raw = load_split(id_handle, fd_handle, "val")
    # The exported Train/Val MAT partitions number their runs independently.
    # Offset held-out IDs so the persisted metadata remains globally disjoint.
    held_raw["run_id"] = held_raw["run_id"] + int(train_raw["run_id"].max())
    held_runs = np.unique(held_raw["run_id"])
    validation_runs = set(int(value) for value in held_runs[::2])
    test_runs = set(int(value) for value in held_runs[1::2])
    packed = {
        "train": windows(train_raw, set(int(value) for value in np.unique(train_raw["run_id"]))),
        "validation": windows(held_raw, validation_runs),
        "test": windows(held_raw, test_runs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, arrays in packed.items():
        np.savez_compressed(args.output_dir / f"{split}.npz", **arrays)
    train_history = packed["train"]["history"]
    train_target = packed["train"]["target"]
    flattened = train_history.reshape(-1, DEFAULT_SCHEMA.input_dim)
    np.savez(
        args.output_dir / "normalization.npz",
        input_mean=flattened.mean(0), input_std=flattened.std(0).clip(1.0e-8),
        target_mean=train_target.mean(0), target_std=train_target.std(0).clip(1.0e-8),
    )
    manifest = {
        **DEFAULT_SCHEMA.to_dict(),
        "purpose": "final desired-command hydraulic ROM; cylinder-space only; no runtime valve input",
        "sources": {"id": str(args.id_mat.resolve()), "fd": str(args.fd_mat.resolve())},
        "alignment": "exact RunID, TimeIndex, SampleTime and 16D cylinder-state equality",
        "held_out_run_id_offset": int(train_raw["run_id"].max()),
        "splits": {name: sorted({int(v) for v in arrays["run_id"]}) for name, arrays in packed.items()},
        "samples": {name: len(arrays["history"]) for name, arrays in packed.items()},
        "cylinder_limits_from_training_extension": np.stack((
            train_history[:, :, DEFAULT_SCHEMA.extension_slice].min((0, 1)),
            train_history[:, :, DEFAULT_SCHEMA.extension_slice].max((0, 1)),
        ), axis=-1).tolist(),
        "valve_role": (
            "current actual valve is supervision only; runtime predicts it internally and feeds the "
            "previous predicted valve back as recurrent cylinder state"
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "samples": manifest["samples"]}, indent=2))


if __name__ == "__main__":
    main()
