#!/usr/bin/env python
"""Apply the strict Stage 22 H20 ext/vel gate to completed JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHANNELS = ("extension", "velocity")
REQUIRED_CONTROLS = ("zero", "shuffled")
DIAGNOSTIC_CONTROLS = ("random", "mean", "nominal")


def ci_pass(row: dict) -> bool:
    ci = row.get("bootstrap_95ci") or row.get("ci95")
    return bool(ci and ci[1] < 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout", required=True)
    p.add_argument("--vs-none", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=20)
    args = p.parse_args()

    rollout = json.loads(Path(args.rollout).read_text())
    versus_none = json.loads(Path(args.vs_none).read_text())
    horizon = str(args.horizon)
    checks = {}
    for scope, table in (
        ("pooled", rollout["paired_vs_true"]),
        ("test", rollout["paired_vs_true_by_split"]["test"]),
    ):
        checks[scope] = {
            control: {
                channel: ci_pass(table[control][horizon][channel])
                for channel in CHANNELS
            }
            for control in REQUIRED_CONTROLS + DIAGNOSTIC_CONTROLS
        }
    checks["pooled"]["no_latent"] = {
        channel: ci_pass(versus_none["pooled"][channel]) for channel in CHANNELS
    }
    checks["test"]["no_latent"] = {
        channel: ci_pass(versus_none["by_split"]["test"][channel])
        for channel in CHANNELS
    }
    required_checks = {
        scope: {
            control: checks[scope][control]
            for control in REQUIRED_CONTROLS + ("no_latent",)
        }
        for scope in checks
    }
    passed = all(
        value
        for scope in required_checks.values()
        for control in scope.values()
        for value in control.values()
    )
    report = {
        "strict_pass": passed,
        "rule": (
            "H20 true-control paired bootstrap upper CI < 0 for extension "
            "and velocity, pooled and reserved test, against zero/shuffled/"
            "protocol-matched no-latent (deployment_adaptation README item 55)"
        ),
        "rollout": args.rollout,
        "vs_none": args.vs_none,
        "checks": checks,
        "required_checks": required_checks,
        "diagnostic_only_controls": list(DIAGNOSTIC_CONTROLS),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"strict_pass={passed}; wrote {out}")


if __name__ == "__main__":
    main()
