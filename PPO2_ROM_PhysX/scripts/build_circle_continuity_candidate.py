#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Build a measured-model-gated cyclic circle command candidate.

The tool geometry is never changed. Candidate cylinder commands are obtained by
periodically smoothing the existing inverse path, then projecting every point
back through the measured cylinder-to-tool model. The script searches a small
declared grid and writes a new library; it never edits the source library.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[2]


def _roughness(commands: np.ndarray, path_s: np.ndarray) -> tuple[float, float]:
    first = np.gradient(commands, path_s, axis=0)
    second = np.gradient(first, path_s, axis=0)
    magnitude = np.linalg.norm(second, axis=1)
    return float(np.percentile(magnitude, 95)), float(np.max(magnitude))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "ROM-cylinders_pipeline/data/shape_calibration_v2/transitions.npz",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / "PPO2_ROM_PhysX/models/head_down_workspace_dense_v3.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_circle_c2.json",
    )
    parser.add_argument("--neighbors", type=int, default=192)
    parser.add_argument("--cylinder_margin_m", type=float, default=0.001)
    args = parser.parse_args()

    source = json.loads(args.source.expanduser().read_text())
    rows = source["shapes"]["circle"]["samples"]
    goals = np.asarray([row["cutter_truck"] for row in rows], dtype=np.float64)
    original = np.asarray([row["cylinder_cmd"] for row in rows], dtype=np.float64)
    original_clearance = np.asarray(
        [row["cutter_world_z"] for row in rows], dtype=np.float64
    )
    path_s = np.asarray([row["path_s_m"] for row in rows], dtype=np.float64)
    if not np.allclose(goals[0], goals[-1]) or not np.allclose(original[0], original[-1]):
        raise ValueError("circle source must be closed in tool and cylinder space")

    data = np.load(args.calibration.expanduser(), allow_pickle=False)
    measured_x = np.asarray(data["measured_extension"], dtype=np.float64)
    measured_tool = np.asarray(data["measured_tool"], dtype=np.float64)
    measured_clearance = np.asarray(data["measured_clearance"], dtype=np.float64)
    finite = (
        np.isfinite(measured_x).all(axis=1)
        & np.isfinite(measured_tool).all(axis=1)
        & np.isfinite(measured_clearance)
    )
    measured_x, measured_tool, measured_clearance = (
        measured_x[finite], measured_tool[finite], measured_clearance[finite]
    )
    workspace = json.loads(args.workspace.expanduser().read_text())
    valid = [row for row in workspace["samples"] if bool(row.get("valid", False))]
    workspace_cmd = np.asarray([row["cylinder_cmd"] for row in valid], dtype=np.float64)
    workspace_x = np.asarray(
        [row.get("cylinder_measured", row["cylinder_cmd"]) for row in valid],
        dtype=np.float64,
    )
    workspace_tool = np.asarray([row["cutter_truck"] for row in valid], dtype=np.float64)
    workspace_clearance = np.asarray(
        [row["cutter_world_z"] for row in valid], dtype=np.float64
    )
    workspace_head = np.asarray([row["head_alignment"] for row in valid], dtype=np.float64)
    fit_x = np.concatenate((measured_x, workspace_x))
    fit_tool = np.concatenate((measured_tool, workspace_tool))
    fit_clearance = np.concatenate((measured_clearance, workspace_clearance))
    x_center = np.mean(fit_x, axis=0)
    x_scale = np.maximum(np.ptp(fit_x, axis=0), 1.0e-4)
    neighbors = min(int(args.neighbors), len(fit_x))
    forward = RBFInterpolator(
        (fit_x - x_center) / x_scale,
        np.column_stack((fit_tool, fit_clearance)),
        kernel="thin_plate_spline",
        smoothing=1.0e-5,
        neighbors=neighbors,
    )
    workspace_center = np.mean(workspace_x, axis=0)
    workspace_scale = np.maximum(np.ptp(workspace_x, axis=0), 1.0e-4)
    forward_head = RBFInterpolator(
        (workspace_x - workspace_center) / workspace_scale,
        workspace_head[:, None],
        kernel="thin_plate_spline",
        smoothing=1.0e-4,
    )
    lower = np.min(workspace_cmd, axis=0) + float(args.cylinder_margin_m)
    upper = np.max(workspace_cmd, axis=0) - float(args.cylinder_margin_m)

    # Exclude the duplicate endpoint while applying a periodic filter.
    unique_original = original[:-1]
    unique_goals = goals[:-1]
    unique_clearance = original_clearance[:-1]
    configs = [
        (window, prior)
        for window in (15, 21)
        # Strong priors prevent pointwise projection from jumping back to a
        # different inverse branch and recreating the curvature defect.
        for prior in (0.75, 1.00, 2.00)
    ]
    candidate_rows: list[dict[str, object]] = []
    candidates: list[tuple[dict[str, object], np.ndarray, np.ndarray]] = []
    source_p95, source_max = _roughness(original, path_s)
    for window, prior in configs:
        smooth = savgol_filter(
            unique_original, window_length=window, polyorder=3,
            axis=0, mode="wrap",
        )
        solved = np.empty_like(smooth)
        for index, (goal, target_clearance, initial) in enumerate(
            zip(unique_goals, unique_clearance, smooth, strict=True)
        ):
            def residual(command: np.ndarray) -> np.ndarray:
                plant = forward(((command - x_center) / x_scale).reshape(1, -1))[0]
                return np.concatenate((
                    plant[:3] - goal,
                    np.asarray([0.5 * (plant[3] - target_clearance)]),
                    float(prior) * (command - initial) / x_scale,
                ))

            fit = least_squares(
                residual, np.clip(initial, lower, upper), bounds=(lower, upper),
                max_nfev=80, xtol=1.0e-9, ftol=1.0e-9, gtol=1.0e-9,
            )
            solved[index] = fit.x
        closed = np.vstack((solved, solved[0]))
        predicted = forward((closed - x_center) / x_scale)
        head = np.clip(
            forward_head((closed - workspace_center) / workspace_scale).reshape(-1),
            -1.0, 1.0,
        )
        tool_error = np.linalg.norm(predicted[:, :3] - goals, axis=1)
        clearance_shift = np.abs(predicted[:, 3] - original_clearance)
        rough_p95, rough_max = _roughness(closed, path_s)
        normalized_jump = np.linalg.norm(np.diff(closed, axis=0) / x_scale, axis=1)
        row = {
            "window": window,
            "projection_prior": prior,
            "tool_mean_error_m": float(np.mean(tool_error)),
            "tool_p95_error_m": float(np.percentile(tool_error, 95)),
            "tool_max_error_m": float(np.max(tool_error)),
            "clearance_shift_p95_m": float(np.percentile(clearance_shift, 95)),
            "clearance_shift_max_m": float(np.max(clearance_shift)),
            "minimum_predicted_head_alignment": float(np.min(head)),
            "roughness_p95": rough_p95,
            "roughness_max": rough_max,
            "roughness_p95_ratio": rough_p95 / source_p95,
            "roughness_max_ratio": rough_max / source_max,
            "maximum_normalized_seed_jump": float(np.max(normalized_jump)),
        }
        row["passed_offline"] = bool(
            row["tool_mean_error_m"] <= 0.015
            and row["tool_p95_error_m"] <= 0.045
            and row["tool_max_error_m"] <= 0.12
            and row["clearance_shift_p95_m"] <= 0.010
            and row["minimum_predicted_head_alignment"] >= 0.98
            and row["roughness_p95_ratio"] < 0.75
            and row["roughness_max_ratio"] < 0.75
        )
        candidate_rows.append(row)
        if row["passed_offline"]:
            candidates.append((row, closed, predicted))
        print(json.dumps(row), flush=True)

    if not candidates:
        raise RuntimeError("no cyclic circle candidate passed the offline gates")
    selected, commands, predicted = min(
        candidates,
        key=lambda item: float(item[0]["roughness_p95_ratio"])
        + 0.25 * float(item[0]["roughness_max_ratio"]),
    )
    head = np.clip(
        forward_head((commands - workspace_center) / workspace_scale).reshape(-1),
        -1.0, 1.0,
    )
    output = copy.deepcopy(source)
    output["generation_contract"]["circle_continuity_repair"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(args.source.expanduser().resolve()),
        "method": "periodic Savitzky-Golay seed plus measured-model projection",
        "tool_geometry_changed": False,
        "selected": selected,
    }
    circle = output["shapes"]["circle"]
    for index, row in enumerate(circle["samples"]):
        row["cylinder_cmd"] = commands[index].tolist()
        row["plant_projected_cutter_truck"] = predicted[index, :3].tolist()
        row["cutter_world_z"] = float(predicted[index, 3])
        row["head_alignment"] = float(head[index])
    circle["plant_reachability"].update({
        "plant_projection_mean_error_m": selected["tool_mean_error_m"],
        "plant_projection_p95_error_m": selected["tool_p95_error_m"],
        "plant_projection_max_error_m": selected["tool_max_error_m"],
        "minimum_predicted_head_alignment": selected["minimum_predicted_head_alignment"],
        "maximum_normalized_seed_jump": selected["maximum_normalized_seed_jump"],
        "circle_command_roughness_p95_ratio": selected["roughness_p95_ratio"],
        "circle_command_roughness_max_ratio": selected["roughness_max_ratio"],
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    report = {
        "source": str(args.source.expanduser().resolve()),
        "output": str(args.out.expanduser().resolve()),
        "source_roughness_p95": source_p95,
        "source_roughness_max": source_max,
        "selected": selected,
        "candidates": candidate_rows,
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"selected={json.dumps(selected)} output={args.out}", flush=True)


if __name__ == "__main__":
    main()
