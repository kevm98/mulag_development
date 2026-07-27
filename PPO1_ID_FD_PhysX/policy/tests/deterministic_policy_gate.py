"""Offline acceptance gate for the live JSON produced by evaluate.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_policy_improvement(policy: dict, baseline: dict) -> tuple[bool, dict[str, bool]]:
    checks = {
        "completed_cycles": bool(policy.get("completed_all_cycles", False)),
        "mean_improved": policy.get("mean_tool_error", 999.0) < baseline.get("mean_tool_error", 0.0),
        "p95_improved": policy.get("p95_tool_error", 999.0) < baseline.get("p95_tool_error", 0.0),
        "max_not_regressed": policy.get("max_tool_error", 999.0) <= 1.05 * baseline.get("max_tool_error", 0.0),
        "ground": policy.get("ground_fraction", 0.0) >= 0.99,
        "head": policy.get("min_head_alignment", 0.0) >= 0.995,
        "truck": policy.get("max_truck_drift", 999.0) < 0.002,
        "nonvacuous_path": 0.95 <= policy.get("filtered_path_ratio", 0.0) <= 1.25,
    }
    return all(checks.values()), checks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evaluation_json", help="Output from policy/scripts/evaluate_v3.py")
    args = p.parse_args()
    payload = json.loads(Path(args.evaluation_json).read_text())
    if "baseline" not in payload or "policy" not in payload:
        raise ValueError("evaluation JSON must contain live 'baseline' and 'policy' objects")
    passed, checks = check_policy_improvement(payload["policy"], payload["baseline"])
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
