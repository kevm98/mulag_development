#!/usr/bin/env python3
"""Calibrate path clearance references from forward/reverse PhysX cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoothing_sigma", type=float, default=3.0)
    args = parser.parse_args()

    data = np.load(args.calibration, allow_pickle=False)
    payload = json.loads(args.library.read_text())
    report: dict[str, object] = {"source": str(args.calibration.resolve()), "shapes": {}}
    for shape_id, raw_name in enumerate(data["shape_names"]):
        name = str(raw_name)
        mask = data["shape_id"] == shape_id
        path_s = data["path_s_m"][mask].astype(np.float64)
        clearance = data["measured_clearance"][mask].astype(np.float64)
        direction = data["direction"][mask]
        rows = payload["shapes"][name]["samples"]
        grid = np.asarray([row["path_s_m"] for row in rows], dtype=np.float64)
        directional: list[np.ndarray] = []
        for sign in (1.0, -1.0):
            selected = direction * sign > 0.0
            order = np.argsort(path_s[selected])
            values = clearance[selected][order]
            if args.smoothing_sigma > 0.0:
                values = gaussian_filter1d(values, float(args.smoothing_sigma))
            directional.append(
                np.interp(grid, path_s[selected][order], values)
            )
        reference = 0.5 * (directional[0] + directional[1])
        for row, value in zip(rows, reference, strict=True):
            row["cutter_world_z"] = float(value)
        gap = np.abs(directional[0] - directional[1])
        report["shapes"][name] = {
            "reference_min_m": float(reference.min()),
            "reference_max_m": float(reference.max()),
            "direction_gap_max_m": float(gap.max()),
            "expected_half_gap_max_m": float(0.5 * gap.max()),
        }

    payload["clearance_calibration"] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.output.with_suffix(".clearance_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
