#!/usr/bin/env python3
"""Estimate valve-to-pressure and valve-to-motion delay from training histories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data import BASE_ROM_SCHEMA, load_dataset_bundle  # noqa: E402


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1.0e-12 or y.std() < 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = load_dataset_bundle(args.dataset, schema=BASE_ROM_SCHEMA)
    history = bundle.train.history[:args.max_samples]
    target = bundle.train.target[:args.max_samples]
    schema = BASE_ROM_SCHEMA
    result = {"history_length": schema.history_length, "sample_time_s": schema.sample_time_s, "cylinders": []}
    for cylinder in range(4):
        pressure_corr = []
        velocity_corr = []
        displacement_corr = []
        for position in range(schema.history_length):
            valve = history[:, position, cylinder]
            pressure_corr.append(correlation(valve, target[:, cylinder]))
            velocity_corr.append(correlation(valve, target[:, 4 + cylinder]))
            displacement_corr.append(correlation(valve, target[:, 8 + cylinder]))
        def summary(values):
            index = int(np.argmax(np.abs(values)))
            lag = schema.history_length - 1 - index
            return {"correlations_oldest_to_current": values, "peak_lag_samples": lag,
                    "peak_lag_seconds": lag * schema.sample_time_s, "peak_correlation": values[index]}
        result["cylinders"].append({
            "cylinder": cylinder + 1,
            "valve_to_delta_pressure": summary(pressure_corr),
            "valve_to_delta_velocity": summary(velocity_corr),
            "valve_to_delta_extension": summary(displacement_corr),
        })
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
