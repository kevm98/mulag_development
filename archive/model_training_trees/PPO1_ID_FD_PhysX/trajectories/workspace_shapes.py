"""Generate head-down tool paths inside the measured MULAG arm workspace.

The workspace sweep is the authority for both the inverse seed (tool position
to cylinder extension) and the forward reference (cylinder extension to tool
position).  Returning the forward-projected reference is intentional: it keeps
the deterministic ID/FD teacher and the Policy on the same physically
reachable path instead of rewarding an ideal but unreachable drawing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator


SUPPORTED_SHAPES = ("line", "arc", "circle", "sine")
HEIGHT_VARYING_SHAPES = ("arc", "circle", "sine")


@dataclass(frozen=True)
class WorkspaceTrajectory:
    name: str
    goals: np.ndarray
    cylinder_seeds: np.ndarray
    cutter_world_z: np.ndarray
    head_alignment: np.ndarray
    path_s: np.ndarray
    desired_goals: np.ndarray

    @property
    def length_m(self) -> float:
        return float(self.path_s[-1])

    def metadata(self) -> dict[str, object]:
        cylinder_span = np.ptp(self.cylinder_seeds, axis=0)
        tool_span = np.ptp(self.goals, axis=0)
        shape_error = np.linalg.norm(self.goals - self.desired_goals, axis=1)
        return {
            "length_m": self.length_m,
            "tool_span_m": tool_span.tolist(),
            "cylinder_span_m": cylinder_span.tolist(),
            "cylinder3_extension_span_m": float(cylinder_span[2]),
            "cutter_world_z_range_m": [
                float(np.min(self.cutter_world_z)),
                float(np.max(self.cutter_world_z)),
            ],
            "minimum_head_alignment": float(np.min(self.head_alignment)),
            "mean_shape_projection_error_m": float(np.mean(shape_error)),
            "max_shape_projection_error_m": float(np.max(shape_error)),
        }


def _valid_workspace_arrays(workspace_path: str | Path) -> tuple[np.ndarray, ...]:
    payload = json.loads(Path(workspace_path).expanduser().resolve().read_text())
    rows = [row for row in payload["samples"] if bool(row.get("valid", False))]
    if len(rows) < 16:
        raise ValueError(f"workspace needs at least 16 valid samples; found {len(rows)}")
    cylinder = np.asarray([row["cylinder_cmd"] for row in rows], dtype=np.float64)
    tool = np.asarray([row["cutter_truck"] for row in rows], dtype=np.float64)
    world_z = np.asarray([row["cutter_world_z"] for row in rows], dtype=np.float64)
    head = np.asarray([row["head_alignment"] for row in rows], dtype=np.float64)
    return cylinder, tool, world_z, head


def _desired_shape(name: str, u: np.ndarray) -> np.ndarray:
    """Return the requested drawing in the truck frame.

    The dimensions are deliberately inside the measured sweep.  The vertical
    modulation is large enough to exercise cylinder 2, while lateral/reach
    modulation forces cylinder 3 to expand and contract.
    """
    if name == "line":
        return np.column_stack((3.80 + 2.80 * u, np.full_like(u, -2.70), np.full_like(u, 1.50)))
    if name == "arc":
        angle = np.pi * (1.0 - u)
        return np.column_stack((
            5.15 + 1.25 * np.cos(angle),
            -2.70 + 0.50 * np.sin(angle),
            1.48 + 0.11 * np.sin(2.0 * angle),
        ))
    if name == "circle":
        angle = 2.0 * np.pi * u
        return np.column_stack((
            5.20 + 0.72 * np.cos(angle),
            -2.75 + 0.38 * np.sin(angle),
            1.50 + 0.11 * np.sin(angle),
        ))
    if name == "sine":
        return np.column_stack((
            3.80 + 2.80 * u,
            -2.70 + 0.32 * np.sin(4.0 * np.pi * u),
            1.50 + 0.11 * np.sin(2.0 * np.pi * u),
        ))
    raise ValueError(f"unsupported trajectory {name!r}; choose from {SUPPORTED_SHAPES}")


def generate_workspace_trajectory(
    workspace_path: str | Path,
    name: str,
    *,
    num_points: int = 241,
) -> WorkspaceTrajectory:
    """Build one reachable, head-down trajectory from the workspace sweep."""
    if name not in SUPPORTED_SHAPES:
        raise ValueError(f"unsupported trajectory {name!r}; choose from {SUPPORTED_SHAPES}")
    if num_points < 81:
        raise ValueError("num_points must be at least 81 for smooth interpolation")

    cylinder, tool, world_z, head = _valid_workspace_arrays(workspace_path)
    tool_scale = np.ptp(tool, axis=0)
    cylinder_scale = np.ptp(cylinder, axis=0)
    if np.any(tool_scale <= 0.0) or np.any(cylinder_scale <= 0.0):
        raise ValueError("workspace has a degenerate tool or cylinder dimension")

    tool_center = np.mean(tool, axis=0)
    cylinder_center = np.mean(cylinder, axis=0)
    inverse = RBFInterpolator(
        (tool - tool_center) / tool_scale,
        cylinder,
        kernel="thin_plate_spline",
        smoothing=1.0e-3,
    )
    forward_input = (cylinder - cylinder_center) / cylinder_scale
    forward_tool = RBFInterpolator(
        forward_input, tool, kernel="thin_plate_spline", smoothing=1.0e-4
    )
    forward_world_z = RBFInterpolator(
        forward_input, world_z[:, None], kernel="thin_plate_spline", smoothing=1.0e-4
    )
    forward_head = RBFInterpolator(
        forward_input, head[:, None], kernel="thin_plate_spline", smoothing=1.0e-4
    )

    desired = _desired_shape(name, np.linspace(0.0, 1.0, num_points))
    seeds = inverse((desired - tool_center) / tool_scale)
    seeds = np.clip(seeds, np.min(cylinder, axis=0), np.max(cylinder, axis=0))
    normalized_seed = (seeds - cylinder_center) / cylinder_scale
    goals = forward_tool(normalized_seed)
    predicted_world_z = forward_world_z(normalized_seed).reshape(-1)
    predicted_head = np.clip(forward_head(normalized_seed).reshape(-1), -1.0, 1.0)
    path_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(goals, axis=0), axis=1))]

    trajectory = WorkspaceTrajectory(
        name=name,
        goals=goals,
        cylinder_seeds=seeds,
        cutter_world_z=predicted_world_z,
        head_alignment=predicted_head,
        path_s=path_s,
        desired_goals=desired,
    )
    validate_workspace_trajectory(trajectory)
    return trajectory


def validate_workspace_trajectory(trajectory: WorkspaceTrajectory) -> None:
    """Enforce the excitation and head-down contracts before simulation."""
    arrays = (
        trajectory.goals,
        trajectory.cylinder_seeds,
        trajectory.cutter_world_z,
        trajectory.head_alignment,
        trajectory.path_s,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{trajectory.name}: non-finite trajectory value")
    if np.any(np.diff(trajectory.path_s) <= 0.0):
        raise ValueError(f"{trajectory.name}: path arc length is not strictly increasing")
    if trajectory.length_m < 2.5:
        raise ValueError(f"{trajectory.name}: path is too short ({trajectory.length_m:.3f} m)")
    cylinder_span = np.ptp(trajectory.cylinder_seeds, axis=0)
    if cylinder_span[2] < 0.05:
        raise ValueError(f"{trajectory.name}: cylinder 3 span is only {cylinder_span[2]:.3f} m")
    if float(np.min(trajectory.head_alignment)) < 0.995:
        raise ValueError(f"{trajectory.name}: predicted head-down alignment is below 0.995")
    if trajectory.name in HEIGHT_VARYING_SHAPES:
        if np.ptp(trajectory.goals[:, 2]) < 0.20:
            raise ValueError(f"{trajectory.name}: tool height span is below 0.20 m")
        if cylinder_span[1] < 0.02:
            raise ValueError(f"{trajectory.name}: cylinder 2 height span is below 0.02 m")


def trajectory_to_json(trajectory: WorkspaceTrajectory) -> dict[str, object]:
    return {
        **trajectory.metadata(),
        "samples": [
            {
                "path_s_m": float(trajectory.path_s[index]),
                "cutter_truck": trajectory.goals[index].tolist(),
                "desired_cutter_truck": trajectory.desired_goals[index].tolist(),
                "cylinder_cmd": trajectory.cylinder_seeds[index].tolist(),
                "cutter_world_z": float(trajectory.cutter_world_z[index]),
                "head_alignment": float(trajectory.head_alignment[index]),
            }
            for index in range(len(trajectory.path_s))
        ],
    }


def build_trajectory_library(
    workspace_path: str | Path, *, num_points: int = 241
) -> dict[str, object]:
    resolved = Path(workspace_path).expanduser().resolve()
    return {
        "schema_version": 1,
        "source_workspace": str(resolved),
        "generation_contract": {
            "tool_reference": "forward projection of clipped inverse-RBF cylinder seed",
            "minimum_length_m": 2.5,
            "minimum_cylinder3_span_m": 0.05,
            "minimum_height_span_m": 0.20,
            "minimum_head_alignment": 0.995,
        },
        "shapes": {
            name: trajectory_to_json(
                generate_workspace_trajectory(resolved, name, num_points=num_points)
            )
            for name in SUPPORTED_SHAPES
        },
    }
