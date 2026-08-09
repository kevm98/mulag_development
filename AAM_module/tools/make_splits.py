#!/usr/bin/env python
"""Create train/val/test run splits for the frozen master dataset (item 45).

Rules enforced here:

- Splits are by RUN and deployment domain; individual windows are never
  split. A run belongs to exactly one split.
- Validation is a disjoint set of *parameter configurations* (rounded factor
  keys) held out from the training pool per domain; runs with those keys form
  the validation split.
- The held-out test split contains only parameter configurations never seen
  in train/val (the manifest's D_TEST rows are already rejection-sampled;
  this tool re-checks the rounded keys and refuses to emit a leak).
- Nominal is the single baseline configuration: nominal runs stay in train
  (there is nothing to hold out for validation, and test is unseen
  *non-nominal* configurations).
- No window may cross a run boundary: splits are run-level, and
  ``runs_info.json`` records each run's step length so downstream windowing
  can be validated with ``aam_data.windows_within_run``.

Usage:
    python make_splits.py [--root datasets/master_v1_frozen]
        [--val-fraction 0.2] [--min-val-per-domain 1] [--seed 20260808]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import (  # noqa: E402
    raw_dir,
    split_key,
    splits_dir,
)


TRAIN_DOMAINS = ("nominal", "hydraulic", "mechanical", "combined")
VAR_DOMAINS = ("hydraulic", "mechanical", "combined")


def load_manifest(root: Path) -> list[dict]:
    with (raw_dir(root) / "manifest.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["run_id"] = int(r["run_id"])
    return rows


def run_length(root: Path, run_id: int) -> int:
    meta = json.loads(
        (raw_dir(root) / "runs" / f"{run_id:05d}" / "metadata.json").read_text()
    )
    return int(meta["num_control_steps"])


def seed_of(root: Path, run_id: int) -> int:
    meta = json.loads(
        (raw_dir(root) / "runs" / f"{run_id:05d}" / "metadata.json").read_text()
    )
    return int(meta["random_seed"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=str(AAM_ROOT / "datasets/master_v1_frozen"),
        help="frozen master dataset root (produced by freeze_dataset.py)",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--min-val-per-domain", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260808)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not (raw_dir(root) / "manifest.csv").is_file():
        raise SystemExit(f"not a frozen dataset: {root} (missing raw/manifest.csv)")
    rows = load_manifest(root)
    if not rows:
        raise SystemExit("empty manifest")

    # Group by manifest-level split (D_TEST rows are already disjoint).
    test_rows = [r for r in rows if r["split"] == "test"]
    train_pool = [r for r in rows if r["split"] == "train"]
    unknown = [r for r in rows if r["split"] not in ("train", "test")]
    if unknown:
        raise SystemExit(f"unexpected manifest split values: {sorted({r['split'] for r in unknown})}")

    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    per_domain = {d: [r for r in train_pool if r["domain"] == d] for d in TRAIN_DOMAINS}
    for d in TRAIN_DOMAINS:
        if d not in per_domain or not per_domain[d]:
            continue
        if d == "nominal":
            # Single baseline configuration; repeats stay in train.
            train_rows.extend(per_domain[d])
            continue
        rows_by_key: dict[tuple, list[dict]] = {}
        for r in per_domain[d]:
            rows_by_key.setdefault(split_key(r), []).append(r)
        keys = sorted(rows_by_key)
        rng.shuffle(keys)
        n_val = max(args.min_val_per_domain, int(round(args.val_fraction * len(keys))))
        n_val = min(n_val, len(keys) - 1)
        val_keys = set(keys[:n_val])
        for key, group in rows_by_key.items():
            (val_rows if key in val_keys else train_rows).extend(group)

    # Re-check config disjointness (train/val vs test) for varying domains.
    def key_set(rs: list[dict]) -> set:
        return {split_key(r) for r in rs if r["domain"] in VAR_DOMAINS}

    train_keys = key_set(train_rows)
    val_keys_all = key_set(val_rows)
    test_keys = key_set(test_rows)
    leaks = {
        "test_vs_train": sorted(test_keys & train_keys),
        "test_vs_val": sorted(test_keys & val_keys_all),
        "train_vs_val": sorted(train_keys & val_keys_all),
    }
    if leaks["test_vs_train"] or leaks["test_vs_val"]:
        raise SystemExit(
            f"configuration leak detected: test shares {len(leaks['test_vs_train'])} "
            f"keys with train and {len(leaks['test_vs_val'])} with val; refusing to write"
        )
    if leaks["train_vs_val"]:
        raise SystemExit(
            f"train/val share {len(leaks['train_vs_val'])} configuration keys; "
            "validation must be a disjoint set of parameter configurations"
        )

    train_ids = sorted(r["run_id"] for r in train_rows)
    val_ids = sorted(r["run_id"] for r in val_rows)
    test_ids = sorted(r["run_id"] for r in test_rows)
    if len(set(train_ids) | set(val_ids) | set(test_ids)) != len(rows):
        raise SystemExit("splits do not cover all runs exactly once")

    split_of = {}
    for run_id in train_ids:
        split_of[run_id] = "train"
    for run_id in val_ids:
        split_of[run_id] = "val"
    for run_id in test_ids:
        split_of[run_id] = "test"

    runs_info = {}
    for r in rows:
        run_id = r["run_id"]
        runs_info[f"{run_id:05d}"] = {
            "run_id": run_id,
            "domain": r["domain"],
            "manifest_split": r["split"],
            "frozen_split": split_of[run_id],
            "key": [float(r[k]) for k in ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")],
            "num_control_steps": run_length(root, run_id),
            "random_seed": seed_of(root, run_id),
        }

    summary = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "min_val_per_domain": args.min_val_per_domain,
        "counts": {
            split: {
                "runs": len(ids),
                "domains": dict(Counter(runs_info[f"{i:05d}"]["domain"] for i in ids)),
            }
            for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
        },
        "config_keys": {
            "train": len(train_keys),
            "val": len(val_keys_all),
            "test": len(test_keys),
            "nominal_repeats_in_train": sum(1 for r in train_rows if r["domain"] == "nominal"),
        },
        "disjointness": {
            "test_vs_train_leak_keys": len(leaks["test_vs_train"]),
            "test_vs_val_leak_keys": len(leaks["test_vs_val"]),
            "train_vs_val_leak_keys": len(leaks["train_vs_val"]),
        },
        "rules": {
            "split_by_run_and_domain": True,
            "no_windows_cross_run_boundaries": True,
            "windows_never_split_across_splits": True,
        },
    }

    out = splits_dir(root)
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        (out / f"{name}.json").write_text(json.dumps(ids) + "\n")
    (out / "runs_info.json").write_text(json.dumps(runs_info, indent=2) + "\n")
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text(
        "Run-level train/val/test splits (AAM Stage 16, item 45).\n"
        "- train.json / val.json / test.json: run-id lists; a run belongs to\n"
        "  exactly one split.\n"
        "- runs_info.json: per-run domain, manifest split, rounded factor key,\n"
        "  step count, and seed. Derived windows must stay inside one run\n"
        "  (validate with aam_data.windows_within_run).\n"
        "- test configurations are disjoint from train/val (rounded to 4\n"
        "  decimals); nominal repeats stay in train.\n"
    )

    print(f"splits -> {out}")
    for split in ("train", "val", "test"):
        c = summary["counts"][split]
        print(
            f"  {split:<5} runs={c['runs']:<4} "
            f"domains={json.dumps(c['domains'])}"
        )
    print(
        "  leak keys:",
        summary["disjointness"],
    )
    print(f"  wrote train/val/test + runs_info.json + split_summary.json")


if __name__ == "__main__":
    main()
