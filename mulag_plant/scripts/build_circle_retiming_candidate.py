#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Retiming only: keep the exact circle and cylinder path, vary speed locally."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "mulag_plant/models/plant_reachable_trajectory_library_v3_circle_c4_retime.json",
    )
    parser.add_argument(
        "--diagnostics", type=Path, nargs="+",
        default=[
            ROOT / "PPO2_ROM_B_PhysX/reports/circle_c4_profile_mc_r1/rom_b_diagnostics.json",
            ROOT / "PPO2_ROM_B_PhysX/reports/circle_c4_profile_mc_r4/rom_b_diagnostics.json",
            ROOT / "PPO2_ROM_B_PhysX/reports/circle_c4_profile_h0p5_r1/rom_b_diagnostics.json",
            ROOT / "PPO2_ROM_B_PhysX/reports/circle_c4_profile_h1p2_r3/rom_b_diagnostics.json",
        ],
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "mulag_plant/models/plant_reachable_trajectory_library_v3_circle_c7_retime.json",
    )
    parser.add_argument("--target_acceleration_mps2", type=float, default=38.0)
    parser.add_argument(
        "--activation_acceleration_mps2", type=float, default=40.0,
        help="Only retime bins above this measured value; default: target.",
    )
    parser.add_argument("--minimum_speed_scale", type=float, default=0.85)
    args = parser.parse_args()

    source = json.loads(args.source.expanduser().read_text())
    diagnostic_paths = (
        [args.diagnostics] if isinstance(args.diagnostics, Path) else args.diagnostics
    )
    profiles = []
    for path in diagnostic_paths:
        diagnostic = json.loads(path.expanduser().read_text())
        probe_rows = diagnostic["rows"]
        if len(probe_rows) != 1 or probe_rows[0]["shape"] != "circle":
            raise ValueError(f"expected one fresh-process circle diagnostic: {path}")
        profiles.append(probe_rows[0]["diagnostics"]["path_profile"])
    if any(len(profile) != len(profiles[0]) for profile in profiles):
        raise ValueError("diagnostic path profiles must use the same bins")
    rows = source["shapes"]["circle"]["samples"]
    geometry_s = np.asarray([row["path_s_m"] for row in rows], dtype=np.float64)
    existing_timing_s = np.asarray(
        [row.get("timing_s_m", row["path_s_m"]) for row in rows], dtype=np.float64
    )
    cylinder = np.asarray([row["cylinder_cmd"] for row in rows], dtype=np.float64)
    if geometry_s[0] != 0.0 or np.any(np.diff(geometry_s) <= 0.0):
        raise ValueError("circle path_s_m must increase strictly from zero")
    if not 0.0 < args.minimum_speed_scale <= 1.0:
        raise ValueError("minimum_speed_scale must be inside (0, 1]")

    # For a fixed path, the curvature part of acceleration is proportional to
    # speed squared. Each measured failing bin therefore requests
    # sqrt(target/measured). Gaussian shoulders keep speed and acceleration
    # changes smooth instead of introducing new bin-edge discontinuities.
    speed_scale = np.ones_like(geometry_s)
    active: list[dict[str, float]] = []
    cylinder_s = np.gradient(cylinder, geometry_s, axis=0)
    cylinder_ss = np.gradient(cylinder_s, geometry_s, axis=0)
    for items in zip(*profiles, strict=True):
        item = max(items, key=lambda row: float(row["p95_acceleration_mps2"]))
        measured = float(item["p95_acceleration_mps2"])
        activation = (
            float(args.activation_acceleration_mps2)
            if args.activation_acceleration_mps2 is not None
            else float(args.target_acceleration_mps2)
        )
        if measured <= activation:
            continue
        low = float(item["path_s_low_m"])
        high = float(item["path_s_high_m"])
        mask = (geometry_s >= low) & (geometry_s <= high)
        requested = max(
            float(args.minimum_speed_scale),
            float(np.sqrt(args.target_acceleration_mps2 / measured)),
        )
        center = 0.5 * (low + high)
        shoulder = max(high - low, 1.0e-6)
        deficit = (1.0 - requested) * np.exp(
            -0.5 * ((geometry_s - center) / shoulder) ** 2
        )
        speed_scale = np.minimum(speed_scale, 1.0 - deficit)
        active.append({
            "path_s_low_m": low,
            "path_s_high_m": high,
            "measured_p95_acceleration_mps2": measured,
            "binding_diagnostic_index": int(
                max(
                    range(len(items)),
                    key=lambda index: float(items[index]["p95_acceleration_mps2"]),
                )
            ),
            "center_speed_scale": requested,
            "cylinder_dq_ds_p95": float(np.percentile(
                np.linalg.norm(cylinder_s[mask], axis=1), 95
            )),
            "cylinder_d2q_ds2_p95": float(np.percentile(
                np.linalg.norm(cylinder_ss[mask], axis=1), 95
            )),
        })
    speed_scale = np.clip(speed_scale, args.minimum_speed_scale, 1.0)
    existing_du_ds = np.gradient(existing_timing_s, geometry_s)
    inverse_scale = existing_du_ds / speed_scale
    timing_s = np.r_[0.0, np.cumsum(
        np.diff(geometry_s) * 0.5 * (inverse_scale[:-1] + inverse_scale[1:])
    )]

    output = copy.deepcopy(source)
    for row, timing in zip(
        output["shapes"]["circle"]["samples"], timing_s, strict=True
    ):
        row["timing_s_m"] = float(timing)
    output["generation_contract"]["circle_retiming"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(args.source.expanduser().resolve()),
        "diagnostics": [str(path.expanduser().resolve()) for path in diagnostic_paths],
        "method": "worst-of-measured acceleration speed law on fixed cylinder path",
        "tool_geometry_changed": False,
        "cylinder_commands_changed": False,
        "geometric_start_phase_distribution_preserved": True,
        "target_acceleration_mps2": float(args.target_acceleration_mps2),
        "activation_acceleration_mps2": activation,
        "minimum_incremental_speed_scale": float(np.min(speed_scale)),
        "minimum_total_speed_scale": float(np.min(1.0 / inverse_scale)),
        "geometric_length_m": float(geometry_s[-1]),
        "timing_length_m": float(timing_s[-1]),
        "active_bins": active,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["generation_contract"]["circle_retiming"], indent=2))


if __name__ == "__main__":
    main()
