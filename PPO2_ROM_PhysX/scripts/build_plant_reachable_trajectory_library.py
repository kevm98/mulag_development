#!/usr/bin/env python3
"""Build exact reference shapes by inverting measured PhysX plant data.

Unlike the legacy workspace generator, this script does not call a reference
"reachable" merely because its cylinder seed is inside a rectangular limit.
It fits a measured cylinder->tool map, solves every reference point inside an
interior cylinder envelope, and rejects the library when the plant projection
does not reproduce the requested geometry within the configured error gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
EXACT_SHAPES = ("line", "arc", "sine", "circle", "square")
# Conservative interior variations used only in the last PPO curriculum phase.
# Each is re-solved and gated by the same measured PhysX forward model as the
# exact references; the policy never receives the variant name.
PROCEDURAL_VARIANTS = {
    "proc_line_compact_low": ("line", 0.90, (-0.03, -0.03), -0.010),
    "proc_line_offset_high": ("line", 0.95, (0.03, 0.02), 0.010),
    "proc_arc_compact": ("arc", 0.90, (-0.03, -0.02), -0.010),
    "proc_arc_offset": ("arc", 0.95, (0.03, -0.03), 0.010),
    "proc_sine_compact": ("sine", 0.88, (-0.02, -0.02), -0.010),
    "proc_sine_offset": ("sine", 0.94, (0.03, 0.01), 0.010),
    "proc_circle_inner": ("circle", 0.85, (-0.03, -0.02), -0.010),
    "proc_circle_offset": ("circle", 0.92, (0.03, -0.03), 0.010),
    "proc_square_inner": ("square", 0.87, (-0.03, 0.00), -0.010),
    "proc_square_offset": ("square", 0.93, (0.03, -0.02), 0.010),
}
ALL_SHAPES = (*EXACT_SHAPES, *PROCEDURAL_VARIANTS)
HEIGHT_M = {
    "circle": 1.48,
    "line": 1.55,
    "arc": 1.58,
    "sine": 1.62,
    "square": 1.67,
}
SPEED_FACTOR = {
    "line": 1.00,
    "arc": 0.85,
    "sine": 0.75,
    "circle": 0.70,
    # A geometric square has discontinuous tangent at its corners.  Slow the
    # existing constant-speed supervisor until corner-aware time scaling lands.
    "square": 0.50,
}

DEFAULT_GEOMETRY = {
    "line": {
        "x_start_m": 3.80,
        "length_m": 2.50,
        "center_y_m": -2.75,
        "height_m": 1.55,
    },
    "arc": {
        "center_x_m": 5.30,
        "center_y_m": -2.95,
        "radius_x_m": 0.90,
        "radius_y_m": 0.40,
        "height_m": 1.58,
    },
    "sine": {
        "x_start_m": 4.10,
        "length_m": 2.20,
        "center_y_m": -2.75,
        "amplitude_m": 0.25,
        "cycles": 2.0,
        "height_m": 1.62,
    },
    "circle": {
        "center_x_m": 5.40,
        "center_y_m": -2.90,
        "radius_m": 0.40,
        "height_m": 1.48,
    },
    "square": {
        "center_x_m": 5.50,
        "center_y_m": -2.85,
        "half_side_m": 0.30,
        "height_m": 1.67,
    },
}


def _exact_shape(
    name: str,
    points: int,
    geometry: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    geometry = DEFAULT_GEOMETRY if geometry is None else geometry
    u = np.linspace(0.0, 1.0, points)
    z = np.full(points, float(geometry[name]["height_m"]), dtype=np.float64)
    if name == "line":
        spec = geometry[name]
        return np.column_stack((
            float(spec["x_start_m"]) + float(spec["length_m"]) * u,
            np.full(points, float(spec["center_y_m"])),
            z,
        ))
    if name == "arc":
        spec = geometry[name]
        angle = np.pi * (1.0 - u)
        return np.column_stack((
            float(spec["center_x_m"]) + float(spec["radius_x_m"]) * np.cos(angle),
            float(spec["center_y_m"]) + float(spec["radius_y_m"]) * np.sin(angle),
            z,
        ))
    if name == "sine":
        spec = geometry[name]
        return np.column_stack((
            float(spec["x_start_m"]) + float(spec["length_m"]) * u,
            float(spec["center_y_m"]) + float(spec["amplitude_m"])
            * np.sin(2.0 * np.pi * float(spec["cycles"]) * u),
            z,
        ))
    if name == "circle":
        spec = geometry[name]
        angle = 2.0 * np.pi * u
        return np.column_stack((
            float(spec["center_x_m"]) + float(spec["radius_m"]) * np.cos(angle),
            float(spec["center_y_m"]) + float(spec["radius_m"]) * np.sin(angle),
            z,
        ))
    if name == "square":
        spec = geometry[name]
        # 240 intervals gives exactly 60 intervals per side.  Every sample is
        # on one of the four sharp boundaries; no fillet or rounded corner.
        phase = 4.0 * u
        segment = np.minimum(np.floor(phase).astype(np.int64), 3)
        local = phase - segment
        cx = float(spec["center_x_m"])
        cy = float(spec["center_y_m"])
        half = float(spec["half_side_m"])
        x = np.where(
            segment == 0,
            cx + half - 2.0 * half * local,
            np.where(
                segment == 1,
                cx - half,
                np.where(segment == 2, cx - half + 2.0 * half * local, cx + half),
            ),
        )
        y = np.where(
            segment == 0,
            cy - half,
            np.where(
                segment == 1,
                cy - half + 2.0 * half * local,
                np.where(segment == 2, cy + half, cy + half - 2.0 * half * local),
            ),
        )
        return np.column_stack((x, y, z))
    raise ValueError(f"unknown shape {name!r}")


def desired_shape(
    name: str,
    points: int,
    geometry: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    if name in EXACT_SHAPES:
        return _exact_shape(name, points, geometry)
    base, scale, shift, height_offset = PROCEDURAL_VARIANTS[name]
    goals = _exact_shape(base, points, geometry).copy()
    center = 0.5 * (goals[:, :2].min(axis=0) + goals[:, :2].max(axis=0))
    goals[:, :2] = center + float(scale) * (goals[:, :2] - center)
    goals[:, 0] += float(shift[0])
    goals[:, 1] += float(shift[1])
    goals[:, 2] += float(height_offset)
    return goals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "ROM-cylinders_pipeline/data/shape_calibration_v2/transitions.npz",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / "PPO2_ROM_PhysX/models/head_down_workspace_3d.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v1.json",
    )
    parser.add_argument(
        "--direct_audit",
        type=Path,
        action="append",
        default=[],
        help="Append requested-cylinder/direct-PhysX tool records from the verifier.",
    )
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--neighbors", type=int, default=192)
    parser.add_argument("--cylinder_margin_m", type=float, default=0.001)
    parser.add_argument("--max_mean_error_m", type=float, default=0.015)
    parser.add_argument("--max_p95_error_m", type=float, default=0.030)
    parser.add_argument("--max_error_m", type=float, default=0.040)
    parser.add_argument("--min_head_alignment", type=float, default=0.98)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=None,
        help="Optional JSON geometry overrides; omitted fields retain v1 defaults.",
    )
    args = parser.parse_args()
    if args.points < 81 or (args.points - 1) % 4 != 0:
        raise ValueError("--points minus one must be divisible by four and at least 80")

    geometry = json.loads(json.dumps(DEFAULT_GEOMETRY))
    if args.geometry is not None:
        overrides = json.loads(args.geometry.expanduser().read_text())
        unknown = sorted(set(overrides) - set(EXACT_SHAPES))
        if unknown:
            raise ValueError(f"unknown geometry shapes: {unknown}")
        for shape, values in overrides.items():
            geometry[shape].update({key: float(value) for key, value in values.items()})

    data = np.load(args.calibration, allow_pickle=False)
    measured_x = np.asarray(data["measured_extension"], dtype=np.float64)
    measured_tool = np.asarray(data["measured_tool"], dtype=np.float64)
    measured_clearance = np.asarray(data["measured_clearance"], dtype=np.float64)
    finite = (
        np.isfinite(measured_x).all(axis=1)
        & np.isfinite(measured_tool).all(axis=1)
        & np.isfinite(measured_clearance)
    )
    measured_x = measured_x[finite]
    measured_tool = measured_tool[finite]
    measured_clearance = measured_clearance[finite]

    workspace = json.loads(args.workspace.read_text())
    valid_rows = [row for row in workspace["samples"] if bool(row.get("valid", False))]
    workspace_cmd = np.asarray([row["cylinder_cmd"] for row in valid_rows], dtype=np.float64)
    workspace_x = np.asarray(
        [row.get("cylinder_measured", row["cylinder_cmd"]) for row in valid_rows],
        dtype=np.float64,
    )
    workspace_tool = np.asarray([row["cutter_truck"] for row in valid_rows], dtype=np.float64)
    workspace_clearance = np.asarray(
        [row["cutter_world_z"] for row in valid_rows], dtype=np.float64
    )
    workspace_head = np.asarray([row["head_alignment"] for row in valid_rows], dtype=np.float64)

    fit_x = np.concatenate((measured_x, workspace_x), axis=0)
    fit_tool = np.concatenate((measured_tool, workspace_tool), axis=0)
    fit_clearance = np.concatenate((measured_clearance, workspace_clearance), axis=0)
    for audit_path in args.direct_audit:
        audit = json.loads(audit_path.read_text())
        records = [
            record
            for shape in audit.get("shapes", {}).values()
            for record in shape.get("records", [])
        ]
        if not records:
            raise ValueError(f"direct audit contains no sample records: {audit_path}")
        # The deployed library commands cylinder targets.  Fit the direct
        # command->settled-tool map here, including linkage/servo bias that is
        # absent when measured extension alone is treated as the independent variable.
        fit_x = np.concatenate((
            fit_x,
            np.asarray([record["requested_cylinder"] for record in records], dtype=np.float64),
        ))
        fit_tool = np.concatenate((
            fit_tool,
            np.asarray([record["measured_tool"] for record in records], dtype=np.float64),
        ))
        fit_clearance = np.concatenate((
            fit_clearance,
            np.asarray([record["measured_clearance"] for record in records], dtype=np.float64),
        ))
    x_center = np.mean(fit_x, axis=0)
    x_scale = np.maximum(np.ptp(fit_x, axis=0), 1.0e-4)
    tool_center = np.mean(fit_tool, axis=0)
    tool_scale = np.maximum(np.ptp(fit_tool, axis=0), 0.05)
    neighbor_count = min(int(args.neighbors), len(fit_x))
    inverse = RBFInterpolator(
        (fit_tool - tool_center) / tool_scale,
        fit_x,
        kernel="thin_plate_spline",
        smoothing=1.0e-3,
        neighbors=neighbor_count,
    )
    forward = RBFInterpolator(
        (fit_x - x_center) / x_scale,
        np.column_stack((fit_tool, fit_clearance)),
        kernel="thin_plate_spline",
        smoothing=1.0e-5,
        neighbors=neighbor_count,
    )
    workspace_center = np.mean(workspace_x, axis=0)
    workspace_scale = np.maximum(np.ptp(workspace_x, axis=0), 1.0e-4)
    forward_head = RBFInterpolator(
        (workspace_x - workspace_center) / workspace_scale,
        workspace_head[:, None],
        kernel="thin_plate_spline",
        smoothing=1.0e-4,
    )

    margin = float(args.cylinder_margin_m)
    lower = np.min(workspace_cmd, axis=0) + margin
    upper = np.max(workspace_cmd, axis=0) - margin
    if np.any(lower >= upper):
        raise ValueError("cylinder margin collapses the verified workspace envelope")

    shapes: dict[str, object] = {}
    reports: dict[str, object] = {}
    for name in ALL_SHAPES:
        goals = desired_shape(name, int(args.points), geometry)
        initial = np.clip(inverse((goals - tool_center) / tool_scale), lower, upper)
        solved = np.empty_like(initial)
        predicted = np.empty_like(goals)
        predicted_clearance = np.empty(len(goals), dtype=np.float64)
        previous: np.ndarray | None = None
        for index, (goal, seed) in enumerate(zip(goals, initial, strict=True)):
            def residual(candidate: np.ndarray) -> np.ndarray:
                plant = forward(((candidate - x_center) / x_scale).reshape(1, -1))[0]
                terms = [plant[:3] - goal, 0.002 * (candidate - seed) / x_scale]
                if previous is not None:
                    terms.append(0.001 * (candidate - previous) / x_scale)
                return np.concatenate(terms)

            result = least_squares(
                residual,
                seed if previous is None else 0.8 * seed + 0.2 * previous,
                bounds=(lower, upper),
                max_nfev=60,
                xtol=1.0e-8,
                ftol=1.0e-8,
                gtol=1.0e-8,
            )
            solved[index] = result.x
            plant = forward(((result.x - x_center) / x_scale).reshape(1, -1))[0]
            predicted[index] = plant[:3]
            predicted_clearance[index] = plant[3]
            previous = result.x

        base_name = PROCEDURAL_VARIANTS.get(name, (name,))[0]
        if base_name in {"circle", "square"}:
            solved[-1] = solved[0]
            predicted[-1] = predicted[0]
            predicted_clearance[-1] = predicted_clearance[0]
        predicted_head = np.clip(
            forward_head((solved - workspace_center) / workspace_scale).reshape(-1),
            -1.0,
            1.0,
        )
        error = np.linalg.norm(predicted - goals, axis=1)
        path_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(goals, axis=0), axis=1))]
        normalized_jump = np.linalg.norm(np.diff(solved, axis=0) / x_scale, axis=1)
        cylinder_margin = np.minimum(solved - lower, upper - solved)
        report = {
            "plant_projection_mean_error_m": float(np.mean(error)),
            "plant_projection_p95_error_m": float(np.percentile(error, 95)),
            "plant_projection_max_error_m": float(np.max(error)),
            "minimum_interior_cylinder_margin_m": float(np.min(cylinder_margin)),
            "maximum_normalized_seed_jump": float(np.max(normalized_jump)),
            "minimum_predicted_head_alignment": float(np.min(predicted_head)),
            "tool_span_m": np.ptp(goals, axis=0).tolist(),
            "cylinder_span_m": np.ptp(solved, axis=0).tolist(),
            "length_m": float(path_s[-1]),
        }
        if (
            report["plant_projection_mean_error_m"] > float(args.max_mean_error_m)
            or report["plant_projection_p95_error_m"] > float(args.max_p95_error_m)
            or report["plant_projection_max_error_m"] > float(args.max_error_m)
            or report["minimum_predicted_head_alignment"] < float(args.min_head_alignment)
        ):
            raise RuntimeError(f"{name} is not plant-reachable: {json.dumps(report)}")
        reports[name] = report
        shapes[name] = {
            "base_shape": base_name,
            "procedural": name in PROCEDURAL_VARIANTS,
            "fixed_height_m": float(np.median(goals[:, 2])),
            "speed_factor": SPEED_FACTOR[base_name],
            "length_m": float(path_s[-1]),
            "tool_span_m": np.ptp(goals, axis=0).tolist(),
            "cylinder_span_m": np.ptp(solved, axis=0).tolist(),
            "plant_reachability": report,
            "samples": [
                {
                    "path_s_m": float(path_s[index]),
                    "cutter_truck": goals[index].tolist(),
                    "desired_cutter_truck": goals[index].tolist(),
                    "plant_projected_cutter_truck": predicted[index].tolist(),
                    "cylinder_cmd": solved[index].tolist(),
                    "cutter_world_z": float(predicted_clearance[index]),
                    "head_alignment": float(predicted_head[index]),
                }
                for index in range(len(goals))
            ],
        }
        print(f"[PLANT REACHABLE] {name} {json.dumps(report)}", flush=True)

    payload = {
        "schema_version": 3,
        "source_workspace": str(args.workspace.resolve()),
        "source_plant_calibration": str(args.calibration.resolve()),
        "source_direct_audits": [str(path.resolve()) for path in args.direct_audit],
        "generation_contract": {
            "reference": "exact requested geometry",
            "reachability": "measured PhysX cylinder-to-tool forward projection",
            "cylinder_interior_margin_m": margin,
            "max_mean_error_m": float(args.max_mean_error_m),
            "max_p95_error_m": float(args.max_p95_error_m),
            "max_error_m": float(args.max_error_m),
            "min_head_alignment": float(args.min_head_alignment),
            "shape_height_m": {
                shape: float(geometry[shape]["height_m"]) for shape in EXACT_SHAPES
            },
            "shape_speed_factor": SPEED_FACTOR,
            "shape_geometry": geometry,
            "square_corner_geometry": "sharp",
            "exact_shape_names": list(EXACT_SHAPES),
            "procedural_shape_names": list(PROCEDURAL_VARIANTS),
        },
        "shapes": shapes,
        "plant_reachability_report": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"[PLANT REACHABLE] wrote {args.out} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
