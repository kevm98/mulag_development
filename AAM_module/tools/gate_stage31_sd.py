#!/usr/bin/env python3
"""Gate the Stage 31 privileged SD on paired held-out latent evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and float(value) == float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()

    rollout = json.loads(Path(args.rollout).read_text())
    metrics = json.loads(Path(args.metrics).read_text())
    failures: list[str] = []
    test_rmse = metrics.get("test", {}).get("rmse", {})
    if not test_rmse or not all(finite(value) for value in test_rmse.values()):
        failures.append("SD test one-step RMSE is missing or non-finite")

    paired = rollout.get("paired_vs_true_by_split", {})
    controls = [
        name for name in ("zero", "mean", "shuffled")
        if any(name in split_values for split_values in paired.values())
    ]
    if "true" not in rollout.get("modes", []) or "zero" not in controls:
        failures.append("rollout must contain paired true and zero latent modes")
    evidence: dict[str, object] = {}
    for split in ("val", "test"):
        split_evidence: dict[str, object] = {}
        zero = paired.get(split, {}).get("zero", {}).get(str(args.horizon), {}).get("all")
        if not isinstance(zero, dict) or not finite(zero.get("mean_diff_true_minus_control")):
            failures.append(f"missing paired zero-latent evidence for {split} H{args.horizon}")
            continue
        diff = float(zero["mean_diff_true_minus_control"])
        ci = zero.get("bootstrap_95ci")
        # true - zero <= 0 means the correctly conditioned latent is no worse;
        # require a strict improvement on at least one held-out split.
        if diff > 1.0e-9:
            failures.append(f"true latent is worse than zero latent on {split}: diff={diff}")
        split_evidence["true_minus_zero_all"] = diff
        split_evidence["bootstrap_95ci"] = ci
        split_evidence["n_runs"] = zero.get("n_runs")
        evidence[split] = split_evidence
    diffs = [float(evidence[s]["true_minus_zero_all"]) for s in evidence]
    if diffs and not any(value < -1.0e-9 for value in diffs):
        failures.append("no strict held-out latent advantage over zero-latent control")

    report = {
        "format": "stage31_sd_latent_gate_v1",
        "strict_pass": not failures,
        "horizon_ticks": args.horizon,
        "evidence": evidence,
        "failures": failures,
        "rollout": str(Path(args.rollout).resolve()),
        "metrics": str(Path(args.metrics).resolve()),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("Stage 31 SD latent gate failed")


if __name__ == "__main__":
    main()
