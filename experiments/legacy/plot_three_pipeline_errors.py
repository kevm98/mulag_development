#!/usr/bin/env python3
"""Plot aligned EE and per-joint tracking errors for the three PPO pipelines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = REPO_ROOT / "experiments" / "three_pipeline_tracking" / "traces"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "three_pipeline_tracking" / "plots"

PIPELINES = {
    "rom": ("PPO-ROM", "#0072B2"),
    "idfd": ("PPO-ID-FD", "#D55E00"),
    "aam": ("PPO-ID(AAM)-FD", "#009E73"),
}
JOINT_NAMES = (
    "Drehzapfen",
    "Ausleger I",
    "Ausleger II",
    "Messerkopf Schwenk",
    "Messerkopf",
)
EE_COMPONENTS = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-step end-effector and individual arm-joint tracking "
            "errors from the aligned ROM, ID-FD and ID(AAM)-FD traces."
        )
    )
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shape", default="double_s_curve")
    parser.add_argument("--mesh", type=int, default=909)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def scalar(data: np.lib.npyio.NpzFile, key: str) -> object | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    return value.item() if value.ndim == 0 else value.tolist()


def load_trace(path: Path, expected_shape: str, expected_mesh: int) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing trace: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "time_s",
            "ee_actual_m",
            "ee_reference_m",
            "ee_error_m",
            "joint_actual_rad",
            "joint_reference_rad",
            "joint_error_rad",
            "travel_m",
            "lateral_error_m",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing trace fields: {', '.join(missing)}")
        trace = {key: np.asarray(data[key], dtype=float) for key in required}
        trace["mesh"] = np.asarray(data["mesh"])
        trace["tracking_amplitude"] = np.asarray(data["tracking_amplitude"])
        trace["pipeline"] = np.asarray(data["pipeline"])
        trace["trajectory_shape"] = np.asarray(data["trajectory_shape"])

    length = trace["time_s"].shape[0]
    if trace["ee_actual_m"].shape != (length, 3):
        raise ValueError(f"{path}: expected EE arrays with shape (T, 3)")
    if trace["joint_actual_rad"].shape != (length, 5):
        raise ValueError(f"{path}: expected joint arrays with shape (T, 5)")
    if trace["ee_reference_m"].shape != trace["ee_actual_m"].shape:
        raise ValueError(f"{path}: shape mismatch for ee_reference_m")
    for key in ("joint_reference_rad", "joint_error_rad"):
        if trace[key].shape != trace["joint_actual_rad"].shape:
            raise ValueError(f"{path}: shape mismatch for {key}")
    for key in ("ee_error_m", "travel_m", "lateral_error_m"):
        if trace[key].shape != (length,):
            raise ValueError(f"{path}: expected {key} to have shape (T,)")

    shape = str(np.asarray(trace["trajectory_shape"]).item())
    mesh = int(np.asarray(trace["mesh"]).item())
    if shape != expected_shape:
        raise ValueError(f"{path}: shape is {shape!r}, expected {expected_shape!r}")
    if mesh != expected_mesh:
        raise ValueError(f"{path}: mesh is {mesh}, expected {expected_mesh}")
    return trace


def validate_alignment(traces: dict[str, dict[str, np.ndarray]]) -> None:
    reference = traces["rom"]
    for key, trace in traces.items():
        if trace["time_s"].shape != reference["time_s"].shape:
            raise ValueError(f"{key}: rollout lengths do not match ROM")
        if not np.allclose(trace["time_s"], reference["time_s"], atol=1.0e-9):
            raise ValueError(f"{key}: time grids do not match ROM")


def error_metrics(trace: dict[str, np.ndarray]) -> dict[str, float]:
    ee_mm = trace["ee_error_m"] * 1000.0
    joint_deg = np.rad2deg(trace["joint_error_rad"])
    result = {
        "ee_mae_mm": float(np.mean(np.abs(ee_mm))),
        "ee_rms_mm": float(np.sqrt(np.mean(ee_mm**2))),
        "ee_p95_mm": float(np.percentile(np.abs(ee_mm), 95)),
        "ee_max_mm": float(np.max(np.abs(ee_mm))),
        "lateral_mae_mm": float(np.mean(np.abs(trace["lateral_error_m"])) * 1000.0),
    }
    for index, name in enumerate(JOINT_NAMES, start=1):
        values = joint_deg[:, index - 1]
        prefix = f"joint_{index}_{name.lower().replace(' ', '_')}_"
        result[f"{prefix}mae_deg"] = float(np.mean(np.abs(values)))
        result[f"{prefix}rms_deg"] = float(np.sqrt(np.mean(values**2)))
        result[f"{prefix}p95_deg"] = float(np.percentile(np.abs(values), 95))
        result[f"{prefix}max_deg"] = float(np.max(np.abs(values)))
    return result


def save_summary(
    output_dir: Path,
    traces: dict[str, dict[str, np.ndarray]],
) -> None:
    rows = []
    for key, trace in traces.items():
        row = {"pipeline": PIPELINES[key][0], **error_metrics(trace)}
        rows.append(row)
    fields = list(rows[0])
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_ee_errors(output_dir: Path, traces: dict[str, dict[str, np.ndarray]]) -> None:
    time_s = traces["rom"]["time_s"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()
    for key, trace in traces.items():
        label, color = PIPELINES[key]
        axes[0].plot(time_s, trace["ee_error_m"] * 1000.0, label=label, color=color)
        signed_mm = (trace["ee_reference_m"] - trace["ee_actual_m"]) * 1000.0
        for axis, component in zip(axes[1:], range(3)):
            axis.plot(time_s, signed_mm[:, component], label=label, color=color)
    axes[0].set_title("EE position error norm")
    axes[0].set_ylabel("error [mm]")
    axes[0].legend(frameon=False)
    for axis, component_name in zip(axes[1:], EE_COMPONENTS):
        axis.set_title(f"EE {component_name} signed error")
        axis.set_ylabel("reference − actual [mm]")
    for axis in axes[2:]:
        axis.set_xlabel("rollout time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("End-effector tracking on double_s_curve, mesh 909", y=1.01)
    figure.tight_layout()
    figure.savefig(output_dir / "ee_tracking_errors.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_joint_errors(output_dir: Path, traces: dict[str, dict[str, np.ndarray]]) -> None:
    time_s = traces["rom"]["time_s"]
    figure, axes = plt.subplots(3, 2, figsize=(14, 11))
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        axis = axes.ravel()[joint_index]
        for key, trace in traces.items():
            label, color = PIPELINES[key]
            values = np.rad2deg(trace["joint_error_rad"][:, joint_index])
            axis.plot(time_s, values, label=label, color=color)
        axis.set_title(f"Joint {joint_index + 1}: {joint_name}")
        axis.set_ylabel("reference − actual [deg]")
        axis.grid(alpha=0.25)
    summary_axis = axes.ravel()[-1]
    x = np.arange(len(JOINT_NAMES))
    width = 0.24
    for offset, (key, (label, color)) in enumerate(PIPELINES.items()):
        rms = [
            float(np.sqrt(np.mean(np.rad2deg(traces[key]["joint_error_rad"][:, i]) ** 2)))
            for i in range(len(JOINT_NAMES))
        ]
        summary_axis.bar(x + (offset - 1) * width, rms, width, label=label, color=color)
    summary_axis.set_xticks(x, [f"J{i}" for i in range(1, 6)])
    summary_axis.set_title("Per-joint RMS error")
    summary_axis.set_ylabel("RMS [deg]")
    summary_axis.legend(frameon=False, fontsize=8)
    summary_axis.grid(axis="y", alpha=0.25)
    for axis in axes.ravel()[2:5]:
        axis.set_xlabel("rollout time [s]")
    summary_axis.set_xlabel("joint")
    figure.suptitle("Individual joint-space tracking errors", y=0.995)
    figure.tight_layout()
    figure.savefig(output_dir / "joint_tracking_errors.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_ee_paths(output_dir: Path, traces: dict[str, dict[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    reference = traces["rom"]["ee_reference_m"]
    axes[0].plot(reference[:, 0], reference[:, 1], "k--", linewidth=2, label="reference")
    for key, trace in traces.items():
        label, color = PIPELINES[key]
        axes[0].plot(trace["ee_actual_m"][:, 0], trace["ee_actual_m"][:, 1], color=color, label=label)
    axes[0].set_title("EE path in local x-y plane")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].legend(frameon=False)
    axes[1].plot(reference[:, 0], reference[:, 2], "k--", linewidth=2, label="reference")
    for key, trace in traces.items():
        label, color = PIPELINES[key]
        axes[1].plot(trace["ee_actual_m"][:, 0], trace["ee_actual_m"][:, 2], color=color, label=label)
    axes[1].set_title("EE path in local x-z plane")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Actual versus reference EE trajectory", y=1.02)
    figure.tight_layout()
    figure.savefig(output_dir / "ee_paths.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    trace_dir = resolve(args.trace_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = {
        key: load_trace(trace_dir / f"{key}.npz", args.shape, args.mesh)
        for key in PIPELINES
    }
    validate_alignment(traces)
    save_summary(output_dir, traces)
    plot_ee_errors(output_dir, traces)
    plot_joint_errors(output_dir, traces)
    plot_ee_paths(output_dir, traces)
    metadata = {
        "shape": args.shape,
        "mesh": args.mesh,
        "pipelines": {key: label for key, (label, _) in PIPELINES.items()},
        "trace_dir": str(trace_dir),
        "plots": [
            "ee_tracking_errors.png",
            "joint_tracking_errors.png",
            "ee_paths.png",
            "summary.csv",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote comparison plots and summary to {output_dir}")
    if not args.no_show:
        print("Plots were saved with the non-interactive Agg backend; use an image viewer to inspect them.")


if __name__ == "__main__":
    main()
