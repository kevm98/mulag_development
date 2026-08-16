#!/usr/bin/env python
"""Build the plant-backed v2 arm trajectory library from shared 2-D geometry.

The existing v1 JSON remains untouched.  Cylinder seeds are selected from the
nearest measured v1 plant calibration sample; the runtime DLS then computes the
actual cylinder target for the new reference.  The report records that these
seeds are initialization hints, not a claim that the old sharp square is the
new rounded-square plant proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verge_task.trajectory_2d import circle, line, rounded_square, sine_chain


def _nearest_row(rows: np.ndarray, target: np.ndarray) -> int:
    return int(np.argmin(np.sum((rows - target[None, :]) ** 2, axis=1)))


def build(source: Path, output: Path, workspace: Path) -> dict:
    source_payload = json.loads(source.read_text())
    source_shapes = source_payload["shapes"]
    workspace_payload = json.loads(workspace.read_text())
    workspace_rows = [
        row for row in workspace_payload["samples"] if row.get("valid", True)
    ]
    if len(workspace_rows) < 8:
        raise RuntimeError("head-down workspace must contain at least eight valid samples")
    workspace_goals = np.asarray(
        [row["cutter_truck"] for row in workspace_rows], dtype=np.float64
    )
    # The old v1 paths were deliberately compact calibration paths.  v2 keeps
    # every point inside the measured envelope while using enough lateral span
    # to make the closed shapes visually meaningful on the road.  All four
    # references use the measured ground-contact layer (z ~= 1.528 m in the
    # truck frame), so the arm is visibly down on the green verge.  Closed
    # shapes are stop-and-mow stations; open paths use a separate, very slow
    # chassis creep so their world-frame motion remains realistic.
    specs = {
        "line": (
            line(length_m=2.5),
            np.array([5.05, -2.75, 1.528]),
            "line", False, 1.0, "creep", 0.035,
        ),
        "sine_chain": (
            sine_chain(length_m=2.8, amplitude_m=0.26, cycles=3.0),
            np.array([5.20, -2.75, 1.528]),
            "sine", False, 0.45, "creep", 0.030,
        ),
        "circle": (
            circle(radius_m=0.42),
            np.array([5.40, -2.80, 1.528]),
            "circle", True, 0.50, "parked", 0.0,
        ),
        "rounded_square": (
            rounded_square(side_m=0.82, corner_radius_m=0.14),
            np.array([5.50, -2.80, 1.528]),
            "square", True, 0.35, "parked", 0.0,
        ),
    }
    shapes: dict[str, dict] = {}
    for name, (geometry, centre, source_name, closed, speed_factor, truck_mode, truck_speed_mps) in specs.items():
        samples = []
        for point_xy, s_m in zip(geometry.xy, geometry.s_m):
            target = np.array([centre[0] + point_xy[0], centre[1] + point_xy[1], centre[2]])
            index = _nearest_row(workspace_goals, target)
            seed_row = workspace_rows[index]
            samples.append({
                "path_s_m": float(s_m),
                "timing_s_m": float(s_m),
                "cutter_truck": target.tolist(),
                "desired_cutter_truck": target.tolist(),
                "plant_projected_cutter_truck": target.tolist(),
                "cylinder_cmd": list(seed_row["cylinder_cmd"]),
                "cutter_world_z": float(seed_row["cutter_world_z"]),
                "head_alignment": float(seed_row["head_alignment"]),
                "seed_source_shape": source_name,
                "seed_source_index": index,
                "seed_source_workspace": str(workspace),
            })
        shapes[name] = {
            "base_shape": name,
            "closed": closed,
            "procedural": False,
            "fixed_height_m": float(centre[2]),
            "speed_factor": speed_factor,
            "execution": {
                "truck_motion_mode": truck_mode,
                "truck_speed_mps": truck_speed_mps,
                "closed_station_protocol": (
                    "park_track_stow_drive_park_repeat" if closed else None
                ),
            },
            "length_m": float(geometry.length_m),
            "geometry_metadata": geometry.metadata,
            "seed_contract": "nearest measured head-down workspace sample; runtime DLS owns target",
            "samples": samples,
        }
    payload = {
        "schema_version": 1,
        "source": str(source),
        "geometry_contract": {
            "coordinate_frame": "cutter truck frame XY plus fixed head-down height",
            "path_parameter": "arc_length_m",
            "nearest_point_tracking": True,
            "rounded_square_corner_geometry": "quarter-circle fillets",
            "shape_names": list(shapes),
            "working_layer_z_m": 1.528,
            "working_surface": "visual green verge at ground z=0; measured cutter contact layer",
        },
        "shapes": shapes,
        "notes": [
            "This v2 file is separate from plant_reachable_trajectory_library_v1.json.",
            "Cylinder seeds and head-down metadata come from head_down_workspace_3d.json.",
            "The v1 line-trained checkpoints are not changed or relabeled as v2 checkpoints.",
            "Run trajectory_2d_preflight.py and then a PhysX teacher rollout before promotion.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "mulag_plant/models/plant_reachable_trajectory_library_v1.json")
    parser.add_argument("--workspace", type=Path, default=ROOT / "mulag_plant/models/head_down_workspace_3d.json")
    parser.add_argument("--output", type=Path, default=ROOT / "mulag_plant/models/trajectory_2d_library_v2.json")
    args = parser.parse_args()
    payload = build(args.source.resolve(), args.output.resolve(), args.workspace.resolve())
    print(f"wrote {args.output} shapes={tuple(payload['shapes'])}")


if __name__ == "__main__":
    main()
