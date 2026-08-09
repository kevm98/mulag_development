"""Offline acceptance gate for the live JSON produced by evaluate.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_policy_improvement(
    policy: dict,
    baseline: dict,
    shape_regressions: dict[str, float] | None = None,
    maximum_shape_regression_m: float = 0.002,
) -> tuple[bool, dict[str, bool]]:
    shape_regressions = shape_regressions or {}
    checks = {
        "baseline_completed": bool(baseline.get("completed_all_cycles", False)),
        "completed_cycles": bool(policy.get("completed_all_cycles", False)),
        "mean_improved": policy.get("mean_tool_error_m", 999.0)
        < baseline.get("mean_tool_error_m", 0.0),
        "p95_improved": policy.get("p95_tool_error_m", 999.0)
        < baseline.get("p95_tool_error_m", 0.0),
        "acceleration_not_increased": policy.get("p95_tool_acceleration_mps2", 999.0)
        <= baseline.get("p95_tool_acceleration_mps2", 0.0),
        "shape_regression": bool(shape_regressions) and max(shape_regressions.values())
        <= float(maximum_shape_regression_m),
        "clearance": policy.get("max_clearance_error_m", 999.0) <= 0.20,
        "head": policy.get("min_head_alignment", 0.0) >= 0.98,
        "truck": policy.get("max_truck_post_constraint_drift_m", 999.0) <= 0.001,
        "residual_authority": policy.get("mean_abs_residual_m", 999.0) <= 0.002,
    }
    return all(checks.values()), checks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evaluation_json", help="Output from policy/scripts/evaluate.py")
    args = p.parse_args()
    payload = json.loads(Path(args.evaluation_json).read_text())
    if "baseline" not in payload or "policy" not in payload:
        raise ValueError("evaluation JSON must contain live 'baseline' and 'policy' objects")
    passed, checks = check_policy_improvement(
        payload["policy"],
        payload["baseline"],
        payload.get("shape_mean_error_regression_m", {}),
        float(payload.get("maximum_allowed_shape_mean_regression_m", 0.002)),
    )
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
