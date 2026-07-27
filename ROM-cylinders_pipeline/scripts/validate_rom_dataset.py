#!/usr/bin/env python3
"""Validate dimensions, leakage boundaries, continuity and command coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ROM_pipeline.data import DEFAULT_SCHEMA, load_dataset_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = load_dataset_bundle(args.dataset)
    splits = {"train": bundle.train, "validation": bundle.validation, "test": bundle.test}
    run_sets = {name: set(map(int, dataset.run_id)) for name, dataset in splits.items()}
    overlap = {
        "train_validation": sorted(run_sets["train"] & run_sets["validation"]),
        "train_test": sorted(run_sets["train"] & run_sets["test"]),
        "validation_test": sorted(run_sets["validation"] & run_sets["test"]),
    }
    report = {"status": "PASS", "overlap": overlap, "splits": {}}
    if any(overlap.values()):
        report["status"] = "FAIL"
    for name, dataset in splits.items():
        command = dataset.history[:, -1, DEFAULT_SCHEMA.command_slice]
        finite = np.isfinite(dataset.history).all() and np.isfinite(dataset.target).all()
        time_contiguous = all(
            np.all(np.diff(np.sort(dataset.time_index[dataset.run_id == run])) == 1)
            for run in np.unique(dataset.run_id)
        )
        report["splits"][name] = {
            "samples": len(dataset),
            "runs": len(np.unique(dataset.run_id)),
            "finite": bool(finite),
            "time_monotonic": bool(time_contiguous),
            "command_std": command.std(axis=0).tolist(),
            "command_min": command.min(axis=0).tolist(),
            "command_max": command.max(axis=0).tolist(),
            "matched_intervention_samples": int(np.sum(dataset.intervention_id >= 0)),
        }
        if not finite or not time_contiguous or np.any(command.std(axis=0) <= 1.0e-10):
            report["status"] = "FAIL"
    if report["splits"]["train"]["matched_intervention_samples"] == 0:
        report["matched_intervention_warning"] = (
            "No matched-state intervention pairs are available; command authority must be established "
            "with valve-auxiliary supervision plus zeroed/shuffled-command and state-only ablations."
        )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
