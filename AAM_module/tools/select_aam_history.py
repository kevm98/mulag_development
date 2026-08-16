#!/usr/bin/env python
"""Select the AAM history length (AAM Stage 25).

Stage 25 rule: compare a small predeclared set (0.25/0.5/1.0/2.0 s) and
select the SHORTEST history whose downstream result is within uncertainty of
the best. Downstream result = ``SD_z(s, u, z_hat)`` rollout RMSE at the
selection horizon, scored per validation run. The reserved test split is
never used for selection.

Input: one ``evaluate_sd_z_rollout.py`` report per candidate lag, produced
with ``--latent-mode zhat`` (per-run ``z_hat`` from ``train_aam.py``). The
per-run downstream score is the mean of the extension and velocity H20
rollout RMSE (dP is the hidden channel by construction and is reported
separately as diagnostics).

Selection: let ``best`` be the lag with the lowest mean validation score.
A candidate lag is acceptable when the paired per-run difference
``score(candidate) - score(best)`` has a percentile bootstrap 95% CI that
contains zero (i.e. the candidate is not significantly worse than the best).
The selected lag is the SHORTEST acceptable candidate.

Usage:
    python select_aam_history.py --results outputs/aam_stage25 \\
        --lags 5 10 20 40 --out outputs/aam_stage25/selection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CI_SEED = 725
CI_BOOTSTRAPS = 2000
SELECT_CHANNELS = ("extension", "velocity")


def bootstrap_ci(diffs: np.ndarray, seed: int, n: int = CI_BOOTSTRAPS,
                 alpha: float = 0.05) -> list[float] | None:
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, d.size, size=d.size)
        means[i] = d[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results",
        required=True,
        help="directory with one rollout_L<lag>.json per candidate",
    )
    p.add_argument("--lags", type=int, nargs="+", required=True)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--seed", type=int, default=CI_SEED)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    results_dir = Path(args.results).expanduser().resolve()
    horizon = str(args.horizon)

    per_run_score: dict[int, dict[int, float]] = {}  # lag -> run -> score
    per_run_channels: dict[int, dict[int, dict[str, float]]] = {}
    for lag in args.lags:
        report_path = results_dir / f"rollout_L{lag}.json"
        report = json.loads(report_path.read_text())
        runs = report["per_run"]["zhat"]
        score = {}
        channels = {}
        for rid, r in runs.items():
            if r["split"] != "val":
                continue  # never select on test
            rm = r["rollout_rmse"][horizon]
            score[int(rid)] = float(np.mean([rm[c] for c in SELECT_CHANNELS]))
            channels[int(rid)] = {c: float(rm[c]) for c in rm}
        per_run_score[lag] = score
        per_run_channels[lag] = channels
        if not score:
            raise SystemExit(f"lag {lag}: no validation runs in {report_path}")

    means = {lag: float(np.mean(list(per_run_score[lag].values())))
             for lag in args.lags}
    best_lag = min(args.lags, key=lambda lag: means[lag])
    common_runs = set(per_run_score[best_lag])
    for lag in args.lags:
        common_runs &= set(per_run_score[lag])
    common_runs = sorted(common_runs)

    table = {}
    for lag in args.lags:
        diffs = np.asarray(
            [per_run_score[lag][r] - per_run_score[best_lag][r]
             for r in common_runs],
            dtype=np.float64,
        )
        ci = bootstrap_ci(diffs, args.seed)
        acceptable = bool(ci and ci[0] <= 0 <= ci[1])
        table[str(lag)] = {
            "mean_score": means[lag],
            "n_val_runs": len(common_runs),
            "paired_vs_best": {
                "mean_diff_candidate_minus_best": (
                    float(diffs.mean()) if diffs.size else None
                ),
                "bootstrap_95ci": ci,
                "acceptable_within_uncertainty": acceptable,
            },
            "per_channel_mean": {
                c: float(np.mean([per_run_channels[lag][r][c] for r in common_runs]))
                for c in ("extension", "velocity", "dP", "all")
            },
        }

    acceptable_lags = [lag for lag in args.lags if table[str(lag)]["paired_vs_best"]["acceptable_within_uncertainty"]]
    selected = min(acceptable_lags) if acceptable_lags else best_lag

    verdict = {
        "stage": "25",
        "rule": (
            "select the SHORTEST candidate lag whose per-run validation "
            "downstream score (mean ext/vel H20 rollout RMSE) is within a "
            "paired bootstrap 95% CI of the best lag; test split unused"
        ),
        "horizon_ticks": args.horizon,
        "candidate_lags": args.lags,
        "best_lag_by_mean": best_lag,
        "selected_lag": selected,
        "selected_seconds": selected * 0.05,
        "per_lag": table,
        "selected_because": (
            f"shortest acceptable lag (acceptable = CI of candidate-best "
            f"contains 0); acceptable set {acceptable_lags}"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2) + "\n")

    print("per-lag validation downstream score (mean ext/vel H20 rollout RMSE):")
    for lag in args.lags:
        row = table[str(lag)]
        print(
            f"  L{lag:>2} ({lag * 0.05:>4.2f}s): {row['mean_score']:.6g} "
            f"CI {row['paired_vs_best']['bootstrap_95ci']} "
            f"acceptable={row['paired_vs_best']['acceptable_within_uncertainty']}"
        )
    print(f"best by mean: L{best_lag}; selected: L{selected} "
          f"({selected * 0.05}s)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
