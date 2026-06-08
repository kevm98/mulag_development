from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

try:
    from rom_dataset import DEFAULT_DATASET_PATH, LOCKED_SPLIT_MANIFEST_PATH, find_default_dataset
except ImportError:
    from kevin_integration.rom_cylinders.rom_dataset import (
        DEFAULT_DATASET_PATH,
        LOCKED_SPLIT_MANIFEST_PATH,
        find_default_dataset,
    )


ROM_DIR = Path(__file__).resolve().parent


def _run_payload(
    *,
    split_name: str,
    source_split: str,
    run_ids: np.ndarray,
    time_indices: np.ndarray,
    selected_runs: np.ndarray,
) -> dict[str, Any]:
    mask = np.isin(run_ids, selected_runs)
    sample_indices = np.flatnonzero(mask).astype(np.int64)
    order = np.lexsort((time_indices[sample_indices], run_ids[sample_indices]))
    sample_indices = sample_indices[order]
    selected_run_set = sorted(int(value) for value in np.unique(run_ids[sample_indices]))
    return {
        "split": split_name,
        "source_split": source_split,
        "sample_indices": torch.as_tensor(sample_indices, dtype=torch.long),
        "source_run_ids": torch.as_tensor(selected_run_set, dtype=torch.long),
        "time_indices": torch.as_tensor(time_indices[sample_indices], dtype=torch.long),
        "split_qualified_run_ids": [f"{split_name}:{source_split}:{run_id}" for run_id in selected_run_set],
        "source_qualified_run_ids": [f"{source_split}:{run_id}" for run_id in selected_run_set],
        "sample_count": int(sample_indices.size),
        "run_count": int(len(selected_run_set)),
    }


def _assert_no_overlap(split_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    names = sorted(split_payloads)
    run_sets = {
        name: set(split_payloads[name]["source_qualified_run_ids"])
        for name in names
    }
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            both = sorted(run_sets[left] & run_sets[right])
            if both:
                overlaps.append({"left": left, "right": right, "count": len(both), "preview": both[:20]})
    if overlaps:
        raise AssertionError(f"Locked split run overlap detected: {overlaps}")
    return overlaps


def _sequence_boundary_report(run_ids: np.ndarray, time_indices: np.ndarray, sample_indices: np.ndarray) -> dict[str, Any]:
    violations: list[dict[str, int]] = []
    selected_runs = np.unique(run_ids[sample_indices])
    for run_id in selected_runs:
        idx = sample_indices[run_ids[sample_indices] == run_id]
        times = time_indices[idx]
        if times.size > 1 and not np.all(np.diff(times) == 1):
            bad_positions = np.flatnonzero(np.diff(times) != 1)
            for pos in bad_positions[:10]:
                violations.append(
                    {
                        "run_id": int(run_id),
                        "left_time": int(times[pos]),
                        "right_time": int(times[pos + 1]),
                    }
                )
    return {"violation_count": len(violations), "violations_preview": violations[:20]}


def create_locked_splits(
    *,
    dataset_path: Path,
    output_dir: Path,
    seed: int,
    train_fraction: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    with h5py.File(dataset_path, "r") as handle:
        train_run_ids = np.asarray(handle["TrainRunID"][()]).reshape(-1).astype(np.int64)
        train_time_indices = np.asarray(handle["TrainTimeIndex"][()]).reshape(-1).astype(np.int64)
        val_run_ids = np.asarray(handle["ValRunID"][()]).reshape(-1).astype(np.int64)
        val_time_indices = np.asarray(handle["ValTimeIndex"][()]).reshape(-1).astype(np.int64)

    unique_train_runs = np.unique(train_run_ids)
    shuffled = unique_train_runs.copy()
    rng.shuffle(shuffled)
    train_count = int(round(float(train_fraction) * len(shuffled)))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    locked_train_runs = np.sort(shuffled[:train_count])
    locked_val_runs = np.sort(shuffled[train_count:])
    locked_test_runs = np.unique(val_run_ids)

    payloads = {
        "train": _run_payload(
            split_name="train",
            source_split="train",
            run_ids=train_run_ids,
            time_indices=train_time_indices,
            selected_runs=locked_train_runs,
        ),
        "val": _run_payload(
            split_name="val",
            source_split="train",
            run_ids=train_run_ids,
            time_indices=train_time_indices,
            selected_runs=locked_val_runs,
        ),
        "test": _run_payload(
            split_name="test",
            source_split="val",
            run_ids=val_run_ids,
            time_indices=val_time_indices,
            selected_runs=locked_test_runs,
        ),
    }
    overlap = _assert_no_overlap(payloads)
    boundary_reports = {
        "train": _sequence_boundary_report(train_run_ids, train_time_indices, payloads["train"]["sample_indices"].numpy()),
        "val": _sequence_boundary_report(train_run_ids, train_time_indices, payloads["val"]["sample_indices"].numpy()),
        "test": _sequence_boundary_report(val_run_ids, val_time_indices, payloads["test"]["sample_indices"].numpy()),
    }
    for name, report in boundary_reports.items():
        if report["violation_count"] != 0:
            raise AssertionError(f"Sequence boundary violations in locked {name}: {report}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_files = {
        "train": "rom_train_locked.pt",
        "val": "rom_val_locked.pt",
        "test": "rom_test_locked.pt",
    }
    for split_name, filename in split_files.items():
        torch.save(payloads[split_name], output_dir / filename)

    manifest = {
        "dataset_path": str(dataset_path),
        "created_at_unix": time.time(),
        "split_seed": int(seed),
        "train_fraction_from_original_train": float(train_fraction),
        "original_validation_used_as_locked_test": True,
        "complete_run_splitting_verified": True,
        "source_arrays": {
            "train_val_source": ["X_train_rom", "Y_train_rom", "TrainRunID", "TrainTimeIndex"],
            "test_source": ["X_val_rom", "Y_val_rom", "ValRunID", "ValTimeIndex"],
        },
        "overlap_checks": {
            "run_overlap_count": len(overlap),
            "overlaps": overlap,
        },
        "sequence_boundary_checks": boundary_reports,
        "splits": {
            name: {
                "file": split_files[name],
                "source_split": payloads[name]["source_split"],
                "sample_count": payloads[name]["sample_count"],
                "run_count": payloads[name]["run_count"],
                "source_qualified_run_ids": payloads[name]["source_qualified_run_ids"],
            }
            for name in ("train", "val", "test")
        },
    }
    manifest_path = output_dir / "rom_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def write_report(manifest: dict[str, Any], path: Path) -> None:
    lines = ["# Locked ROM Split Report", ""]
    lines.append(f"- Dataset: `{manifest['dataset_path']}`")
    lines.append(f"- Seed: `{manifest['split_seed']}`")
    lines.append(f"- Original validation used as locked test: `{manifest['original_validation_used_as_locked_test']}`")
    lines.append(f"- Complete-run splitting verified: `{manifest['complete_run_splitting_verified']}`")
    lines.append("")
    lines.append("| Split | Source Split | Runs | Samples |")
    lines.append("|---|---|---:|---:|")
    for name, info in manifest["splits"].items():
        lines.append(f"| {name} | {info['source_split']} | {info['run_count']} | {info['sample_count']} |")
    lines.append("")
    lines.append("## Overlap Checks")
    lines.append("")
    lines.append(f"- Run overlap count: `{manifest['overlap_checks']['run_overlap_count']}`")
    lines.append("")
    lines.append("## Sequence Boundary Checks")
    lines.append("")
    for name, report in manifest["sequence_boundary_checks"].items():
        lines.append(f"- `{name}` violations: `{report['violation_count']}`")
    lines.append("")
    lines.append("The locked `test` split is the original validation set and should not be used for iterative model development.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create locked complete-run ROM train/val/test split descriptors.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=ROM_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--report", type=Path, default=ROM_DIR / "results" / "locked_split_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    manifest = create_locked_splits(
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(manifest, args.report)
    print(f"Wrote {args.output_dir / 'rom_train_locked.pt'}")
    print(f"Wrote {args.output_dir / 'rom_val_locked.pt'}")
    print(f"Wrote {args.output_dir / 'rom_test_locked.pt'}")
    print(f"Wrote {args.output_dir / 'rom_split_manifest.json'}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
