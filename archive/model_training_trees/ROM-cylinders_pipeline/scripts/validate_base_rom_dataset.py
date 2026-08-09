#!/usr/bin/env python3
"""Audit processed base-ROM splits, valve coverage and transition consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data import BASE_ROM_SCHEMA, load_dataset_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = load_dataset_bundle(args.dataset, schema=BASE_ROM_SCHEMA)
    splits = {"train": bundle.train, "validation": bundle.validation, "test": bundle.test}
    runs = {name: set(map(int, dataset.run_id)) for name, dataset in splits.items()}
    overlaps = {
        "train_validation": sorted(runs["train"] & runs["validation"]),
        "train_test": sorted(runs["train"] & runs["test"]),
        "validation_test": sorted(runs["validation"] & runs["test"]),
    }
    report = {"status": "PASS", "overlap": overlaps, "splits": {}}
    if any(overlaps.values()):
        report["status"] = "FAIL"
    schema = BASE_ROM_SCHEMA
    for name, dataset in splits.items():
        history = dataset.history
        target = dataset.target
        valve = history[:, -1, schema.valve_slice]
        velocity = history[:, -1, schema.velocity_slice]
        kinematic = 0.5 * (velocity + velocity + target[:, 4:8]) * schema.sample_time_s
        displacement_residual = target[:, 8:12] - kinematic
        pressure = history[:, -1, schema.pressure_slice]
        force = history[:, -1, schema.force_slice]
        reliable_force = []
        force_pressure_correlation = []
        force_pressure_ratio_median = []
        for cylinder in range(4):
            mask = np.abs(pressure[:, cylinder]) > 1.0e5
            corr = float(np.corrcoef(pressure[:, cylinder], force[:, cylinder])[0, 1])
            ratio = float(np.median(force[mask, cylinder] / pressure[mask, cylinder]))
            # The authored cylinders use AA=0.00785 and AB=0.00503 m^2.
            # A broad 0..0.02 m^2 interval tolerates asymmetry/load effects;
            # values far outside it are not a reliable hydraulic-force label.
            reliable_force.append(bool(0.0 < ratio < 0.02 and abs(corr) >= 0.5))
            force_pressure_correlation.append(corr)
            force_pressure_ratio_median.append(ratio)
        contiguous = all(
            np.all(np.diff(np.sort(dataset.time_index[dataset.run_id == run])) == 1)
            for run in np.unique(dataset.run_id)
        )
        split_report = {
            "samples": len(dataset),
            "runs": len(np.unique(dataset.run_id)),
            "finite": bool(np.isfinite(history).all() and np.isfinite(target).all()),
            "time_contiguous": bool(contiguous),
            "valve_min": valve.min(0).tolist(),
            "valve_max": valve.max(0).tolist(),
            "valve_std": valve.std(0).tolist(),
            "near_zero_valve_fraction": float(np.mean(np.max(np.abs(valve), axis=1) < 0.05)),
            "high_valve_fraction": float(np.mean(np.max(np.abs(valve), axis=1) >= 0.8)),
            "trapezoidal_displacement_residual_rmse_m": np.sqrt(np.mean(displacement_residual ** 2, axis=0)).tolist(),
            "force_dP_correlation": force_pressure_correlation,
            "median_force_over_dP_m2": force_pressure_ratio_median,
            "force_auxiliary_reliable_by_cylinder": reliable_force,
        }
        report["splits"][name] = split_report
        if not split_report["finite"] or not contiguous or np.any(valve.std(0) < 1.0e-4):
            report["status"] = "FAIL"
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
