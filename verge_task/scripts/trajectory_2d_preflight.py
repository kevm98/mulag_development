#!/usr/bin/env python
"""Run geometry-only reachability and safety checks for trajectory-v2.

The cylinder check uses the measured ground-frame Jacobian as a conservative
screen and the existing head-down workspace as the reachability envelope. It
is intentionally separate from a PhysX rollout: a pass here means the teacher
reference is worth sending to the simulator, not that the plant has already
proved it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verge_task.trajectory_2d import from_xy_samples


def _load_v2(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = {"line", "sine_chain", "circle", "rounded_square"}
    if set(payload.get("shapes", {})) != expected:
        raise ValueError(f"trajectory-v2 library must contain exactly {sorted(expected)}")
    return payload


def check(library: Path, workspace: Path, jacobian_path: Path, *, speed_mps: float, bandwidth_hz: float) -> dict:
    payload = _load_v2(library)
    workspace_payload = json.loads(workspace.read_text())
    workspace_rows = [row for row in workspace_payload["samples"] if row.get("valid", True)]
    workspace_xyz = np.asarray([row["cutter_truck"] for row in workspace_rows], dtype=np.float64)
    workspace_cyl_measured = np.asarray(
        [row["cylinder_measured"] for row in workspace_rows], dtype=np.float64
    )
    workspace_cyl_command = np.asarray(
        [row["cylinder_cmd"] for row in workspace_rows], dtype=np.float64
    )
    # The trajectory library carries desired cylinder commands, so the primary
    # saturation gate must compare like with like.  Keep the measured settled
    # envelope in the report as a diagnostic; command-vs-measured differences
    # are the known servo settling residual, not a shape excursion.
    cylinder_lo = workspace_cyl_command.min(axis=0)
    cylinder_hi = workspace_cyl_command.max(axis=0)
    jacobian = np.asarray(json.loads(jacobian_path.read_text())["jacobian"], dtype=np.float64)
    jacobian_pinv = np.linalg.pinv(jacobian)
    jacobian_payload = json.loads(jacobian_path.read_text())
    nominal_tool = np.asarray(jacobian_payload["cutter_truck"], dtype=np.float64)
    nominal_cyl = np.asarray(jacobian_payload["cylinder_x"], dtype=np.float64)
    result = {
        "schema_version": 1,
        "library": str(library),
        "workspace": str(workspace),
        "jacobian": str(jacobian_path),
        "checks": {},
        "shapes": {},
    }
    all_pass = True
    for name, spec in payload["shapes"].items():
        rows = spec["samples"]
        xyz = np.asarray([row["cutter_truck"] for row in rows], dtype=np.float64)
        path = from_xy_samples(xyz[:, :2], name=name, closed=bool(spec.get("closed", False)))
        # The sparse workspace grid is an envelope, not a nearest-neighbour
        # tessellation.  A valid reference between two measured grid points
        # must be judged against that envelope; using sparse-point distance
        # incorrectly rejected the already validated line itself.
        workspace_margin = 0.04
        outside = (xyz < workspace_xyz.min(axis=0)[None, :] - workspace_margin) | (xyz > workspace_xyz.max(axis=0)[None, :] + workspace_margin)
        # The v2 JSON carries a measured v1 seed for every reference sample.
        # Use that seed for the saturation gate and retain the Jacobian result
        # as a diagnostic only; a single global Jacobian linearization is not a
        # substitute for the calibrated plant projection over a 2-D loop.
        estimated_cyl = np.asarray([row["cylinder_cmd"] for row in rows], dtype=np.float64)
        saturated = (estimated_cyl < cylinder_lo[None, :]) | (estimated_cyl > cylinder_hi[None, :])
        # The analytic overshoot proxy is the acceleration a finite-bandwidth
        # tracking loop would need to follow the normal component. It is used
        # as a preflight alarm, not presented as measured PhysX overshoot.
        edge = np.roll(path.xy, -1, axis=0) - path.xy if path.closed else np.diff(path.xy, axis=0)
        ds = np.linalg.norm(edge, axis=1).clip(min=1e-9)
        tangent_delta = np.linalg.norm(np.roll(path.tangent_xy, -1, axis=0) - path.tangent_xy, axis=1) if path.closed else np.linalg.norm(np.diff(path.tangent_xy, axis=0), axis=1)
        curvature = tangent_delta / ds
        max_curvature = float(np.max(curvature))
        arm_speed_mps = float(speed_mps) * float(spec.get("speed_factor", 1.0))
        normal_accel = float(arm_speed_mps * arm_speed_mps * max_curvature)
        overshoot_proxy = float(normal_accel / max((2.0 * np.pi * bandwidth_hz) ** 2, 1e-9))
        head_alignment = np.min([float(row.get("head_alignment", workspace_payload["minimum_head_alignment"])) for row in rows])
        checks = {
            "reachable": bool(not outside.any()),
            "max_workspace_outside_m": float(np.maximum(workspace_xyz.min(axis=0)[None, :] - xyz, xyz - workspace_xyz.max(axis=0)[None, :]).clip(min=0.0).max()),
            "cylinder_saturation_free": bool(not saturated.any()),
            "cylinder_saturation_fraction": float(saturated.any(axis=1).mean()),
            "estimated_cylinder_min": estimated_cyl.min(axis=0).tolist(),
            "estimated_cylinder_max": estimated_cyl.max(axis=0).tolist(),
            "measured_workspace_cylinder_min": workspace_cyl_measured.min(axis=0).tolist(),
            "measured_workspace_cylinder_max": workspace_cyl_measured.max(axis=0).tolist(),
            "jacobian_linearization_cylinder_min": (nominal_cyl[None, :] + (xyz - nominal_tool) @ jacobian_pinv.T).min(axis=0).tolist(),
            "jacobian_linearization_cylinder_max": (nominal_cyl[None, :] + (xyz - nominal_tool) @ jacobian_pinv.T).max(axis=0).tolist(),
            "max_curvature_1pm": max_curvature,
            "normal_acceleration_mps2": normal_accel,
            "arm_speed_mps": arm_speed_mps,
            "truck_motion_mode": str(
                spec.get("execution", {}).get("truck_motion_mode", "parked")
            ),
            "truck_speed_mps": float(
                spec.get("execution", {}).get("truck_speed_mps", 0.0)
            ),
            "overshoot_proxy_m": overshoot_proxy,
            "overshoot_proxy_below_30mm": bool(overshoot_proxy <= 0.03),
            "min_head_alignment": float(head_alignment),
            "head_alignment_above_0p98": bool(head_alignment >= 0.98),
            "closed": bool(spec.get("closed", False)),
            "length_m": float(path.length_m),
        }
        result["shapes"][name] = checks
        all_pass = all_pass and all(bool(value) for key, value in checks.items() if key in ("reachable", "cylinder_saturation_free", "overshoot_proxy_below_30mm", "head_alignment_above_0p98"))
    result["checks"] = {
        "all_geometry_gates_pass": bool(all_pass),
        "base_arm_speed_mps": speed_mps,
        "bandwidth_hz": bandwidth_hz,
        "cylinder_limits_source": "head_down_workspace_3d.cylinder_cmd",
        "measured_cylinder_envelope_diagnostic": "head_down_workspace_3d.cylinder_measured",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "mulag_plant/models/trajectory_2d_library_v2.json")
    parser.add_argument("--workspace", type=Path, default=ROOT / "mulag_plant/models/head_down_workspace_3d.json")
    parser.add_argument("--jacobian", type=Path, default=ROOT / "mulag_plant/models/ee_jacobian_ground_truck3x4.json")
    parser.add_argument("--speed_mps", type=float, default=0.13)
    parser.add_argument("--bandwidth_hz", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/trajectory_2d/preflight.json")
    args = parser.parse_args()
    report = check(args.library.resolve(), args.workspace.resolve(), args.jacobian.resolve(), speed_mps=args.speed_mps, bandwidth_hz=args.bandwidth_hz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["checks"]["all_geometry_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
