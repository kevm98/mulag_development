#!/usr/bin/env python3
"""Orchestrator: runs all extensive debug tests sequentially.

Usage:
    python run_all_debug_tests.py --device cuda:0

Tests are run in order. Failures are logged but do not stop subsequent tests.
A summary is printed at the end.
"""
from __future__ import annotations
import subprocess
import sys
import os
import argparse
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

TESTS = [
    ("Test 1: Articulation Inspection",         "test1_articulation_inspection.py"),
    ("Test 3: Single-Joint Perturbation",       "test3_single_joint_perturbation.py"),
    ("Test 4: Single-Cylinder Perturbation",    "test4_single_cylinder_perturbation.py"),
    ("Test 5: Cylinder Sign/Order Matrix",      "test5_cylinder_sign_order_matrix.py"),
    ("Test 6: Numerical Cylinder Jacobian",     "test6_numerical_cylinder_jacobian.py"),
    ("Test 7: q4/q5 Coupling Check",            "test7_q4_q5_coupling.py"),
    ("Test 8: Safe Velocity Tracking",          "test8_safe_velocity_tracking.py"),
    ("Test 9: A/B Comparison",                  "test9_ab_comparison.py"),
]

PYTHON = "/home/rrlab/miniconda3/envs/env_isaaclab/bin/python"


def run_test(script_name: str, device: str) -> tuple[bool, str, float]:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        return False, f"Script not found: {script_path}", 0.0
    cmd = [PYTHON, str(script_path), "--device", device]
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            return True, result.stdout[-500:] if len(result.stdout) > 500 else result.stdout, elapsed
        else:
            msg = (result.stderr[-500:] if result.stderr else "") + "\n" + (result.stdout[-500:] if result.stdout else "")
            return False, msg.strip(), elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return False, "TIMEOUT (600s)", elapsed
    except Exception as exc:
        elapsed = time.time() - start
        return False, str(exc), elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all extensive debug tests")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip", nargs="*", default=[], help="Test numbers to skip (e.g. 8 9)")
    args = parser.parse_args()

    device = args.device
    skip_nums = set(args.skip)

    print("=" * 72)
    print("Extensive Debug — Run All Tests")
    print(f"Device: {device}")
    print(f"Tests: {len(TESTS)}")
    print("=" * 72)

    results = []
    for i, (label, script) in enumerate(TESTS):
        test_num = i + 1
        if str(test_num) in skip_nums:
            print(f"\n[{test_num}/{len(TESTS)}] SKIP {label}")
            results.append((label, "SKIPPED", "", 0.0))
            continue

        print(f"\n[{test_num}/{len(TESTS)}] RUNNING {label}...")
        success, output, elapsed = run_test(script, device)
        status = "PASS" if success else "FAIL"
        print(f"  Status: {status} ({elapsed:.1f}s)")
        if not success:
            print(f"  Output: {output[:300]}")
        results.append((label, status, output, elapsed))

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    failed = sum(1 for _, s, _, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _, _ in results if s == "SKIPPED")
    for label, status, _, elapsed in results:
        print(f"  [{status:8s}] {label} ({elapsed:.1f}s)")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped out of {len(results)} tests")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
