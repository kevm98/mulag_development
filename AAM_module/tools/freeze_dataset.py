#!/usr/bin/env python
"""Freeze a collected master dataset (AAM Stage 16, item 44).

Validates every run against the master schema, copies the raw trajectories
into an immutable ``raw/`` tree, records SHA-256 checksums, and writes full
provenance (source, collector task/ROM, schema version, domain/split counts,
environment, command line). The derived directories (``aam``, ``sd``,
``analysis``) are created empty with a README; every derived dataset must be
built from ``raw/`` by its own tool and must never write into ``raw/``.

The source dataset is left untouched (copy, never move). With ``--readonly``
the raw tree is chmod'ed read-only as a best-effort freeze.

Usage:
    python freeze_dataset.py [--src datasets/master_v1]
        [--out datasets/master_v1_frozen] [--readonly] [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import (  # noqa: E402
    DERIVED_DIRS,
    FROZEN_VERSION,
    metadata_dir,
    sha256_file,
    splits_dir,
)
from aam_data.frozen import CHECKSUM_FILE  # noqa: E402
from aam_data.master_schema import SCHEMA_VERSION, validate_run  # noqa: E402


DERIVED_README = """\
This directory is DERIVED from the frozen raw tree (AAM Stage 16, item 44).

Rules:
- Raw trajectories in ../raw/ are immutable. Never write into ../raw/.
- Build this dataset with its own tool, reading runs through
  ``aam_data.read_raw_run`` (checksum-verified) or ``aam_data.iter_raw_runs``.
- Record the producing tool, its arguments, the frozen root, and the
  frozen checksum file digest in this directory's provenance when generated.
- Do not overwrite raw trajectories during preprocessing.
"""


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["run_id"] = int(r["run_id"])
    return rows


def validate_source(src: Path, manifest_rows: list[dict]) -> tuple[int, list[str]]:
    """Validate every run dir against the schema; return (n_runs, issues)."""
    by_id = {int(r["run_id"]): r for r in manifest_rows}
    runs_dir = src / "runs"
    n = 0
    issues: list[str] = []
    run_dirs = [
        (int(child.name), child)
        for child in runs_dir.glob("*")
        if child.is_dir() and child.name.isdigit()
    ]
    for run_id, child in sorted(run_dirs):
        n += 1
        row = by_id.get(run_id)
        run_issues = validate_run(child, manifest_row=row)
        for msg in run_issues:
            issues.append(f"run {run_id:05d}: {msg}")
        if row is None:
            issues.append(f"run {run_id:05d}: not present in manifest")
    for row in manifest_rows:
        if not (src / "runs" / f"{int(row['run_id']):05d}").is_dir():
            issues.append(f"run {int(row['run_id']):05d}: manifest row has no run dir")
    return n, issues


def git_info() -> dict:
    out = {"commit": None, "branch": None, "describe": None, "dirty": None}
    try:
        res = subprocess.run(
            ["git", "-C", str(AAM_ROOT.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            out["commit"] = res.stdout.strip()
        res = subprocess.run(
            ["git", "-C", str(AAM_ROOT.parent), "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            out["branch"] = res.stdout.strip() or None
        res = subprocess.run(
            ["git", "-C", str(AAM_ROOT.parent), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            out["describe"] = res.stdout.strip()
    except Exception:
        pass
    return out


def environment() -> dict:
    import platform

    import numpy
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def run_metadata(src: Path, run_id: int) -> dict:
    return json.loads((src / "runs" / f"{run_id:05d}" / "metadata.json").read_text())


def summary_from_manifest(rows: list[dict]) -> dict:
    counts = Counter((r["domain"], r["split"]) for r in rows)
    return {
        "total_runs": len(rows),
        "by_domain_split": {
            f"{d}/{s}": counts[(d, s)] for d, s in sorted(counts)
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default=str(AAM_ROOT / "datasets/master_v1"),
        help="collected dataset dir (manifest.csv + runs/)",
    )
    p.add_argument("--out", default=None, help="frozen root (default: <src>_frozen)")
    p.add_argument(
        "--readonly",
        action="store_true",
        help="chmod raw files read-only after checksumming (best-effort freeze)",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    p.add_argument(
        "--ignore-validation",
        action="store_true",
        help="copy runs even if schema validation reports issues",
    )
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out or str(src) + "_frozen").expanduser().resolve()
    if out == src or src in out.parents:
        raise SystemExit(f"refusing to freeze into the source tree: {out}")
    if not (src / "manifest.csv").is_file():
        raise SystemExit(f"source manifest not found: {src / 'manifest.csv'}")
    if not (src / "runs").is_dir():
        raise SystemExit(f"source runs dir not found: {src / 'runs'}")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(
            f"target exists and is not empty: {out} (use --force to overwrite)"
        )

    rows = load_manifest(src / "manifest.csv")
    if not rows:
        raise SystemExit(f"manifest is empty: {src / 'manifest.csv'}")
    n_runs, issues = validate_source(src, rows)
    if issues and not args.ignore_validation:
        print(f"validation failed for {len(issues)} item(s):")
        for msg in issues[:20]:
            print(f"  {msg}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
        raise SystemExit(
            "freeze aborted; fix the source or pass --ignore-validation"
        )

    if out.exists():
        shutil.rmtree(out)
    (out / "raw" / "runs").mkdir(parents=True)
    metadata_dir(out).mkdir(parents=True, exist_ok=True)

    # Copy raw trajectories (never move; the source stays intact).
    shutil.copytree(src / "runs", out / "raw" / "runs", dirs_exist_ok=True)
    shutil.copyfile(src / "manifest.csv", out / "raw" / "manifest.csv")

    # Checksums over the raw tree.
    checksum_lines = []
    for path in sorted((out / "raw").rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(out / "raw"))
            checksum_lines.append(f"{sha256_file(path)}  {rel}")
    (out / "raw" / CHECKSUM_FILE).write_text("\n".join(sorted(checksum_lines)) + "\n")

    # Per-run metadata facts for provenance and summary (read once per run).
    metas = {r["run_id"]: run_metadata(src, r["run_id"]) for r in rows}
    roms = sorted({str(metas[r["run_id"]].get("rom_checkpoint")) for r in rows})
    tasks = sorted({str(metas[r["run_id"]].get("task_name")) for r in rows})
    schemas = sorted({str(metas[r["run_id"]].get("schema_version")) for r in rows})
    control_dts = sorted({float(metas[r["run_id"]].get("control_dt")) for r in rows})
    rom_dts = sorted({float(metas[r["run_id"]].get("rom_dt")) for r in rows})
    total_steps = sum(
        int(metas[r["run_id"]].get("num_control_steps", 0)) for r in rows
    )

    frozen_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "frozen_version": FROZEN_VERSION,
        "frozen_at": frozen_at,
        "source": str(src),
        "source_manifest": str(src / "manifest.csv"),
        "schema_version": SCHEMA_VERSION,
        "run_schema_versions": schemas,
        "collector_task_names": tasks,
        "rom_checkpoints": roms,
        "control_dt": control_dts,
        "rom_dt": rom_dts,
        "total_runs": len(rows),
        "total_control_steps": total_steps,
        "validation": {
            "runs_checked": n_runs,
            "issues": len(issues),
            "ignored": bool(args.ignore_validation),
        },
        "command": " ".join(sys.argv),
        "readonly": bool(args.readonly),
        "git": git_info(),
        "environment": environment(),
    }
    (metadata_dir(out) / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    # Domain/split summary with per-dimension factor ranges.
    dims = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
    summary = summary_from_manifest(rows)
    summary["factor_ranges"] = {}
    for domain in sorted({r["domain"] for r in rows}):
        domain_rows = [r for r in rows if r["domain"] == domain]
        summary["factor_ranges"][domain] = {
            dim: [round(min(float(r[dim]) for r in domain_rows), 6),
                  round(max(float(r[dim]) for r in domain_rows), 6)]
            for dim in dims
        }
    (metadata_dir(out) / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # Derived directories: separate, empty, documented.
    for name in DERIVED_DIRS:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(DERIVED_README)
    splits_dir(out).mkdir(parents=True, exist_ok=True)
    (splits_dir(out) / "README.md").write_text(
        "Run-level train/val/test splits written by tools/make_splits.py "
        "(AAM Stage 16, item 45).\n"
    )

    if args.readonly:
        for path in sorted((out / "raw").rglob("*")):
            if path.is_file():
                path.chmod(0o444)
            else:
                path.chmod(0o555)

    print(f"frozen {len(rows)} runs -> {out}")
    print(f"  schema: {SCHEMA_VERSION} | validation issues: {len(issues)}")
    print(f"  domains: {dict(summary['by_domain_split'])}")
    print(f"  total control steps: {total_steps}")
    print(f"  checksums: {out / 'raw' / CHECKSUM_FILE}")
    print(f"  provenance: {metadata_dir(out) / 'provenance.json'}")
    print(f"  readonly: {args.readonly}")
    if issues:
        print(f"  WARNING: validation issues were ignored ({len(issues)})")


if __name__ == "__main__":
    main()
