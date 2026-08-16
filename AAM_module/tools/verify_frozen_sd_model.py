#!/usr/bin/env python
"""Verify the frozen privileged representation (AAM Stage 23 gate).

Stage 23 gate: the frozen copy must reproduce the Stage 22 rollout metrics.
Given a frozen model directory produced by ``freeze_sd_model.py``, this tool

1. recomputes SHA-256 over every frozen file and compares against the
   recorded ``checksums.sha256`` (a single mutated file fails the gate);
2. re-runs ``evaluate_sd_z_rollout.py`` on the FROZEN copy with exactly the
   Stage 22 gate protocol read from ``evidence/rollout_gate.json`` (latent
   modes, horizons, hold-start q) against the same frozen dataset;
3. compares the reproduced rollout report to the stored gate evidence:
   aggregate ``rollout_rmse``, per-split means, paired diffs and bootstrap
   CIs, and the zero/persistence baseline;
4. re-runs the Stage 22 formal checks (compare vs the stored no-latent
   control, then ``check_stage22_gate.py``) and requires ``strict_pass``;
5. writes a machine-readable verdict JSON.

The gate passes only if the checksums match AND every reproduced metric is
within a small relative tolerance of the recorded Stage 22 evidence.

Usage:
    python verify_frozen_sd_model.py \\
        --model-dir sd_model/stage23_frozen/dz2_nodp_holdq \\
        --root AAM_module/datasets/master_v3_stage22_wide_frozen \\
        --out AAM_module/results/stage23_frozen_verify/dz2_nodp_holdq/verdict.json

All verification artifacts are written OUTSIDE the frozen tree so the frozen
copy stays byte-identical to the moment it was frozen.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import sha256_file  # noqa: E402


CHECKSUM_FILE = "checksums.sha256"
REF_ROLLOUT = "evidence/rollout_gate.json"
NONE_CONTROL = "evidence/none_rollout_hold-start.json"
EVALUATOR = AAM_ROOT / "tools/evaluate_sd_z_rollout.py"
COMPARER = AAM_ROOT / "tools/compare_sd_z_rollouts.py"
GATE_CHECKER = AAM_ROOT / "tools/check_stage22_gate.py"
RTOL = 1e-4
ATOL = 1e-9


def load_checksums(model_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (model_dir / CHECKSUM_FILE).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition("  ")
        out[rel] = digest
    return out


def verify_checksums(model_dir: Path) -> tuple[bool, list[str]]:
    recorded = load_checksums(model_dir)
    mismatches = []
    for rel, digest in sorted(recorded.items()):
        actual = sha256_file(model_dir / rel)
        if actual != digest:
            mismatches.append(f"{rel}: recorded {digest}, actual {actual}")
    # Any file in the frozen tree that is not recorded is also a violation.
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and path.name != CHECKSUM_FILE:
            rel = str(path.relative_to(model_dir))
            if rel not in recorded:
                mismatches.append(f"{rel}: present but not recorded in checksums")
    return not mismatches, mismatches


def run(tool: Path, args: list[str], model_dir: Path) -> Path:
    cmd = [sys.executable, str(tool), *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(
            f"subprocess failed: {' '.join(cmd)}\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    return model_dir


def max_rel_diff(a: dict, b: dict, *, path: str = "") -> tuple[float, str]:
    """Largest relative (and absolute) difference between two JSON subtrees.

    Only numeric leaves are compared. Returns ``(max_rel, max_abs)`` where
    both are encoded as ``(value, path)``.
    """
    best_rel: tuple[float, str] = (0.0, "")
    best_abs: tuple[float, str] = (0.0, "")
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a:
            if key not in b:
                continue
            r, ab = max_rel_diff(a[key], b[key], path=f"{path}.{key}")
            if r[0] > best_rel[0]:
                best_rel = r
            if ab[0] > best_abs[0]:
                best_abs = ab
    elif isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            r, ab = max_rel_diff(x, y, path=f"{path}[{i}]")
            if r[0] > best_rel[0]:
                best_rel = r
            if ab[0] > best_abs[0]:
                best_abs = ab
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        denom = abs(b)
        rel = abs(a - b) / denom if denom > 0 else abs(a - b)
        best_rel = (rel, path)
        best_abs = (abs(a - b), path)
    return best_rel, best_abs


def compare_reports(repro: dict, ref: dict) -> dict:
    """Compare the reproduced rollout report to the Stage 22 gate evidence."""
    compared = {}
    sections = {
        "rollout_rmse": "aggregate rollout RMSE per mode/horizon/channel",
        "splits": "per-split (val/test) rollout means",
        "paired_vs_true": "paired true-control diffs + bootstrap CIs",
        "paired_vs_true_by_split": "paired diffs by split",
        "zero_rmse": "zero/persistence baseline",
    }
    worst_rel: tuple[float, str] = (0.0, "")
    worst_abs: tuple[float, str] = (0.0, "")
    for key, desc in sections.items():
        rel, ab = max_rel_diff(repro[key], ref[key])
        compared[key] = {
            "description": desc,
            "max_abs": ab[0],
            "max_abs_path": ab[1],
            "max_rel": rel[0],
            "max_rel_path": rel[1],
        }
        if rel[0] > worst_rel[0]:
            worst_rel = rel
        if ab[0] > worst_abs[0]:
            worst_abs = ab
    compared["_worst"] = {
        "max_abs": worst_abs[0],
        "max_abs_path": worst_abs[1],
        "max_rel": worst_rel[0],
        "max_rel_path": worst_rel[1],
    }
    return compared


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument(
        "--root",
        default=str(AAM_ROOT / "datasets/master_v3_stage22_wide_frozen"),
        help="frozen dataset root (must match the one used for Stage 22)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="verdict JSON path (default: AAM_module/results/"
        "stage23_frozen_verify/<model-name>/verdict.json)",
    )
    p.add_argument("--rtol", type=float, default=RTOL)
    p.add_argument("--atol", type=float, default=ATOL)
    args = p.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if args.out:
        out = Path(args.out)
    else:
        out = (
            AAM_ROOT / "results" / "stage23_frozen_verify"
            / model_dir.name / "verdict.json"
        )
    verify_dir = out.parent
    verify_dir.mkdir(parents=True, exist_ok=True)

    checksum_ok, checksum_issues = verify_checksums(model_dir)
    print(f"checksums: {'OK' if checksum_ok else 'FAILED'}")
    for msg in checksum_issues:
        print(f"  {msg}")
    if not checksum_ok:
        raise SystemExit("checksum verification failed; aborting gate")

    ref = json.loads((model_dir / REF_ROLLOUT).read_text())
    modes = ref["modes"]
    horizons = [int(h) for h in ref["horizons_ticks"]]
    q_rollout = ref["q_rollout"]
    print(f"protocol from evidence: modes={modes} horizons={horizons} "
          f"q_rollout={q_rollout}")

    repro_out = verify_dir / "rollout_reproduced.json"
    run(
        EVALUATOR,
        [
            "--root", str(Path(args.root).expanduser().resolve()),
            "--model-dir", str(model_dir),
            "--latent-mode", *modes,
            "--horizons", *(str(h) for h in horizons),
            "--q-rollout", q_rollout,
            "--out", str(repro_out),
        ],
        model_dir,
    )
    repro = json.loads(repro_out.read_text())
    assert repro["modes"] == modes, "reproduced modes differ from evidence"

    # Stage 22 formal checks on the reproduced rollout.
    vs_none_out = verify_dir / "true_vs_none_reproduced.json"
    run(
        COMPARER,
        [
            "--candidate", str(repro_out),
            "--control", str(model_dir / NONE_CONTROL),
            "--horizon", "20",
            "--out", str(vs_none_out),
        ],
        model_dir,
    )
    gate_out = verify_dir / "gate_verdict_reproduced.json"
    run(
        GATE_CHECKER,
        [
            "--rollout", str(repro_out),
            "--vs-none", str(vs_none_out),
            "--out", str(gate_out),
        ],
        model_dir,
    )
    gate = json.loads(gate_out.read_text())
    strict_pass = bool(gate["strict_pass"])

    compared = compare_reports(repro, ref)
    max_abs = compared["_worst"]["max_abs"]
    max_abs_path = compared["_worst"]["max_abs_path"]
    max_rel = compared["_worst"]["max_rel"]
    max_rel_path = compared["_worst"]["max_rel_path"]

    # Leaf-wise tolerance: every numeric leaf must satisfy
    # |repro - ref| <= atol + rtol * |ref|.
    def leaves_ok(a: dict, b: dict, *, rtol: float, atol: float) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            return all(leaves_ok(a[k], b[k], rtol=rtol, atol=atol)
                       for k in a if k in b)
        if isinstance(a, list) and isinstance(b, list):
            return all(leaves_ok(x, y, rtol=rtol, atol=atol)
                       for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) <= atol + rtol * abs(b)
        return True

    metrics_ok = all(
        leaves_ok(repro[key], ref[key], rtol=args.rtol, atol=args.atol)
        for key in ("rollout_rmse", "splits", "paired_vs_true",
                    "paired_vs_true_by_split", "zero_rmse")
    )

    pass_ = checksum_ok and strict_pass and metrics_ok
    verdict = {
        "stage": "23",
        "gate": "frozen copy reproduces Stage-22 rollout metrics",
        "pass": bool(pass_),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "dataset_root": str(Path(args.root).expanduser().resolve()),
        "checksums_ok": checksum_ok,
        "checksum_issues": checksum_issues,
        "protocol_from_evidence": {
            "modes": modes,
            "horizons_ticks": horizons,
            "q_rollout": q_rollout,
        },
        "strict_pass_reproduced": strict_pass,
        "max_abs_diff": max_abs,
        "max_abs_diff_path": max_abs_path,
        "max_rel_diff": max_rel,
        "max_rel_diff_path": max_rel_path,
        "tolerance": {"rtol": args.rtol, "atol": args.atol},
        "metric_comparison": compared,
        "artifacts": {
            "rollout_reproduced": str(repro_out),
            "true_vs_none_reproduced": str(vs_none_out),
            "gate_verdict_reproduced": str(gate_out),
        },
    }
    out.write_text(json.dumps(verdict, indent=2) + "\n")

    print(f"reproduced rollout: {repro_out}")
    print(f"strict_pass (reproduced gate): {strict_pass}")
    print(f"max |repro - evidence|: {max_abs:.3e} at {max_abs_path}")
    print(f"max rel diff (rollout_rmse): {max_rel:.3e} at {max_rel_path}")
    print(f"GATE {'PASS' if pass_ else 'FAIL'}; wrote {out}")
    raise SystemExit(0 if pass_ else 1)


if __name__ == "__main__":
    main()
