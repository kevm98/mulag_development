#!/usr/bin/env python
"""Paired comparison of two completed SD rollout reports.

The independent unit is a run. Reports must use the same rollout horizon and
future-q protocol. This is used for the protocol-matched Stage 22 comparison
between a privileged/direct candidate and the separately trained no-latent
control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CHANNELS = ("extension", "velocity", "dP", "all")


def bootstrap_ci(diffs: np.ndarray, seed: int, n: int = 10000) -> list[float] | None:
    if diffs.size < 2:
        return None
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, diffs.size, size=(n, diffs.size))
    means = diffs[picks].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--candidate-mode", default="true")
    p.add_argument("--control-mode", default="true")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    candidate = json.loads(Path(args.candidate).read_text())
    control = json.loads(Path(args.control).read_text())
    q_candidate = candidate.get("q_rollout", "recorded")
    q_control = control.get("q_rollout", "recorded")
    if q_candidate != q_control:
        raise SystemExit(
            f"future-q protocol mismatch: candidate={q_candidate}, control={q_control}"
        )

    c_runs = candidate["per_run"][args.candidate_mode]
    b_runs = control["per_run"][args.control_mode]
    common = sorted(set(c_runs) & set(b_runs), key=int)
    horizon = str(args.horizon)

    def summarize(run_ids: list[str]) -> dict:
        report = {}
        for channel in CHANNELS:
            diffs = np.asarray(
                [c_runs[r]["rollout_rmse"][horizon][channel]
                 - b_runs[r]["rollout_rmse"][horizon][channel]
                 for r in run_ids],
                dtype=np.float64,
            )
            report[channel] = {
                "candidate_minus_control": float(diffs.mean()) if diffs.size else None,
                "ci95": bootstrap_ci(diffs, args.seed),
                "candidate_wins": int((diffs < 0).sum()),
                "n_runs": int(diffs.size),
            }
        return report

    by_split = {
        split: summarize([r for r in common if c_runs[r].get("split") == split])
        for split in ("val", "test")
    }
    report = {
        "candidate": args.candidate,
        "candidate_mode": args.candidate_mode,
        "control": args.control,
        "control_mode": args.control_mode,
        "horizon": args.horizon,
        "q_rollout": q_candidate,
        "pooled": summarize(common),
        "by_split": by_split,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["pooled"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
