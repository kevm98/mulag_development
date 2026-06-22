from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_NAME = "pytorch_hydraulic_rom_dataset_causal.mat"
DEFAULT_OUTPUT = ROOT / "kevin_integration/config/simscape_velocity_statistics.json"
METADATA_PATH = (
    ROOT
    / "kevin_integration/rom_cylinders/exported_valve_conditioned_causal_rom/"
    / "best_valve_conditioned_causal_rom_metadata.json"
)


def _candidate_paths() -> list[Path]:
    candidates = [
        ROOT / DEFAULT_DATASET_NAME,
        ROOT / "kevin_integration" / DEFAULT_DATASET_NAME,
        ROOT / "kevin_integration/rom_cylinders" / DEFAULT_DATASET_NAME,
    ]
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        dataset = metadata.get("dataset_identity") or metadata.get("audit", {}).get("dataset")
        if dataset:
            candidates.extend(
                [
                    ROOT / str(dataset),
                    ROOT / "kevin_integration" / str(dataset),
                    ROOT / "kevin_integration/rom_cylinders" / str(dataset),
                ]
            )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _load_mat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat

        data = loadmat(path, simplify_cells=True)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        pass
    try:
        import h5py

        out: dict[str, Any] = {}
        with h5py.File(path, "r") as h5:
            for key in h5.keys():
                out[key] = np.asarray(h5[key])
        return out
    except ImportError as exc:
        raise RuntimeError("Reading MATLAB v7.3 files requires h5py.") from exc


def _field(data: dict[str, Any], name: str) -> np.ndarray:
    if name in data:
        return np.asarray(data[name], dtype=np.float64)
    raise KeyError(f"Required Simscape field {name!r} is missing.")


def _stats(values: np.ndarray) -> dict[str, float | int]:
    flat = values.reshape(-1)
    abs_flat = np.abs(flat)
    return {
        "count": int(flat.size),
        "minimum": float(np.min(flat)),
        "maximum": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "standard_deviation": float(np.std(flat)),
        "absolute_maximum": float(np.max(abs_flat)),
        "absolute_p90": float(np.percentile(abs_flat, 90.0)),
        "absolute_p95": float(np.percentile(abs_flat, 95.0)),
        "absolute_p99": float(np.percentile(abs_flat, 99.0)),
        "absolute_p995": float(np.percentile(abs_flat, 99.5)),
        "absolute_p999": float(np.percentile(abs_flat, 99.9)),
    }


def _channel_stats(values: np.ndarray) -> list[dict[str, float | int]]:
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"Expected a 2D array with at least 4 channels, got shape {values.shape}.")
    return [_stats(values[:, i]) for i in range(4)]


def analyze(dataset: Path, output: Path) -> dict[str, Any]:
    data = _load_mat(dataset)
    time = _field(data, "Time").reshape(-1)
    extension = _field(data, "Extension")[:, :4]
    extension_velocity = _field(data, "ExtensionVelocity")[:, :4]
    joint_vel = _field(data, "JointVel")[:, :5]

    if not np.isfinite(time).all() or not np.isfinite(extension).all():
        raise ValueError("Time and Extension must be finite.")
    if not np.isfinite(extension_velocity).all() or not np.isfinite(joint_vel).all():
        raise ValueError("ExtensionVelocity and JointVel must be finite.")
    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise ValueError("Time contains duplicated or non-increasing timestamps.")

    extension_velocity_fd = np.diff(extension, axis=0) / dt.reshape(-1, 1)
    stored_for_comparison = extension_velocity[1:, :]
    comparison_error = stored_for_comparison - extension_velocity_fd

    summary = {
        "dataset": str(dataset),
        "sample_period_seconds": {
            "minimum": float(np.min(dt)),
            "maximum": float(np.max(dt)),
            "median": float(np.median(dt)),
        },
        "units": {
            "Time": "s",
            "Extension": "m",
            "ExtensionVelocity": "m/s",
            "JointVel": "mixed rad/s for revolute coordinates and m/s for prismatic coordinates",
        },
        "fields": {
            "Time": "Time",
            "Extension": "Extension[:, 0:4]",
            "ExtensionVelocity": "ExtensionVelocity[:, 0:4]",
            "JointVel": "JointVel[:, 0:5]",
        },
        "extension_velocity_stored_stats": _channel_stats(extension_velocity),
        "extension_velocity_fd_stats": _channel_stats(extension_velocity_fd),
        "extension_velocity_comparison_abs_error": _channel_stats(np.abs(comparison_error)),
        "joint_velocity_stats": [_stats(joint_vel[:, i]) for i in range(5)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = args.dataset
    if dataset is None:
        candidates = _candidate_paths()
        dataset = next((path for path in candidates if path.exists()), None)
        if dataset is None:
            joined = "\n  ".join(str(path) for path in candidates)
            raise SystemExit(
                "blocked: Simscape dataset is unavailable. Expected "
                f"{DEFAULT_DATASET_NAME!r}; checked:\n  {joined}"
            )
    analyze(dataset.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
