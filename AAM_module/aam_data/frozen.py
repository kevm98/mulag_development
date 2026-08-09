"""Read-only access contract for the frozen master dataset (AAM Stage 16).

A frozen dataset is a self-contained directory tree produced by
``tools/freeze_dataset.py``:

    <root>/
      raw/            immutable raw trajectories (never written by derived tools)
        manifest.csv
        runs/<run_id:05d>/{metadata.json,trajectory.npz}
        checksums.sha256
      metadata/       provenance.json (incl. environment), summary.json
      splits/         train/val/test run lists (+ runs_info.json)
      aam/            derived AAM windows (built separately, never into raw/)
      sd/             derived SD transitions (built separately)
      analysis/       reports/plots derived from raw or derived sets

Derived tools MUST read runs through ``read_raw_run`` / ``iter_raw_runs`` and
write only under their own derived directory. ``verify_frozen`` re-checks every
raw file against the recorded checksums so a derived run cannot silently use a
mutated raw file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .master_schema import read_run


RAW_DIR = "raw"
METADATA_DIR = "metadata"
SPLITS_DIR = "splits"
DERIVED_DIRS = ("aam", "sd", "analysis")
CHECKSUM_FILE = "checksums.sha256"
PROVENANCE_FILE = "provenance.json"
RUNS_INFO_FILE = "runs_info.json"
FROZEN_VERSION = "frozen_v1"


def frozen_root(path: str | Path) -> Path:
    return Path(path)


def raw_dir(root: str | Path) -> Path:
    return frozen_root(root) / RAW_DIR


def metadata_dir(root: str | Path) -> Path:
    return frozen_root(root) / METADATA_DIR


def splits_dir(root: str | Path) -> Path:
    return frozen_root(root) / SPLITS_DIR


def derived_dir(root: str | Path, name: str) -> Path:
    if name not in DERIVED_DIRS:
        raise ValueError(f"derived dir must be one of {DERIVED_DIRS}, got {name!r}")
    return frozen_root(root) / name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(root: str | Path, run_id: int) -> Path:
    return raw_dir(root) / "runs" / f"{int(run_id):05d}"


def iter_raw_runs(root: str | Path):
    """Yield ``(run_id, run_dir)`` for every raw run in id order."""
    runs = raw_dir(root) / "runs"
    entries = [
        (int(child.name), child)
        for child in runs.glob("*")
        if child.is_dir() and child.name.isdigit()
    ]
    for run_id, child in sorted(entries):
        yield run_id, child


def read_raw_run(root: str | Path, run_id: int, *, verify: bool = True) -> dict:
    """Read one raw run (metadata + arrays) through the master schema.

    ``verify=True`` checks the run files against the recorded checksums first;
    use it for any step that feeds training/evaluation data.
    """
    root = frozen_root(root)
    rdir = run_dir(root, run_id)
    if verify:
        ok, mismatches = verify_frozen(root, run_ids=[run_id])
        if not ok:
            raise RuntimeError(
                f"raw run {run_id:05d} failed checksum verification: {mismatches}"
            )
    return read_run(rdir)


def _load_checksums(root: Path) -> dict[str, str]:
    path = raw_dir(root) / CHECKSUM_FILE
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        out[rel] = digest
    return out


def verify_frozen(root: str | Path, run_ids: list[int] | None = None) -> tuple[bool, list[str]]:
    """Recompute SHA-256 of raw files and compare to the recorded checksums.

    Returns ``(ok, mismatches)`` where ``mismatches`` lists relative paths that
    are missing, extra, or different. ``run_ids`` restricts the check to a
    subset of runs (default: the whole raw tree).
    """
    root = frozen_root(root)
    recorded_all = _load_checksums(root)
    mismatches: list[str] = []
    if not recorded_all:
        return False, ["missing checksums file"]
    wanted = None if run_ids is None else {int(r) for r in run_ids}
    recorded = {
        rel: digest
        for rel, digest in recorded_all.items()
        if wanted is None
        or rel == "manifest.csv"
        or (rel.startswith("runs/") and int(rel.split("/")[1]) in wanted)
    }

    targets: list[Path] = []
    manifest = raw_dir(root) / "manifest.csv"
    if manifest.is_file():
        targets.append(manifest)
    for run_id, rdir in iter_raw_runs(root):
        if run_ids is not None and int(run_id) not in run_ids:
            continue
        targets.extend(sorted(p for p in rdir.iterdir() if p.is_file()))

    seen: set[str] = set()
    for path in targets:
        rel = str(path.relative_to(raw_dir(root)))
        seen.add(rel)
        digest = sha256_file(path)
        if recorded.get(rel) != digest:
            mismatches.append(f"{rel}: {recorded.get(rel)} != {digest}")

    for rel in sorted(recorded):
        if rel not in seen:
            mismatches.append(f"{rel}: recorded but missing")
    for rel in sorted(seen - set(recorded)):
        mismatches.append(f"{rel}: present but not recorded")
    return not mismatches, mismatches


def load_provenance(root: str | Path) -> dict:
    return json.loads((metadata_dir(root) / PROVENANCE_FILE).read_text())


def load_split(root: str | Path, name: str) -> list[int]:
    """Load a split (train/val/test) as an ordered run-id list."""
    path = splits_dir(root) / f"{name}.json"
    return [int(run_id) for run_id in json.loads(path.read_text())]


def load_runs_info(root: str | Path) -> dict[str, dict]:
    return json.loads((splits_dir(root) / RUNS_INFO_FILE).read_text())


def split_key(row: dict) -> tuple[float, ...]:
    """Rounded factor key used to keep parameter configurations disjoint."""
    return tuple(
        round(float(row[k]), 4)
        for k in ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
    )


def windows_within_run(runs_info: dict[str, dict], windows: list[tuple[int, int, int]]) -> list[str]:
    """Validate derived windows: each window ``(run_id, start, end)`` must lie
    inside one raw run and never span a run boundary."""
    issues: list[str] = []
    for run_id, start, end in windows:
        info = runs_info.get(f"{int(run_id):05d}")
        if info is None:
            issues.append(f"run {run_id:05d} unknown")
            continue
        n = int(info["num_control_steps"])
        if not (0 <= start < end <= n + 1):
            issues.append(
                f"window ({run_id:05d}, {start}, {end}) outside run length {n + 1}"
            )
    return issues
