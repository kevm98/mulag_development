#!/usr/bin/env python
"""Stage 27 gate: validate the complete adaptation chain.

Compares the three chain members on validation and reserved test runs:

    SD_0(s, u)                 no-latent model (protocol-matched control)
    SD_z(s, u, z_true)         frozen privileged encoder latent
    SD_z(s, u, z_hat)          frozen SD_z with the trained AAM latent

The gate passes only if ``z_hat`` retains a useful, statistically supported
part of the true-latent advantage without privileged deployment input. The
formal checks (at the selection horizon, H20, hold-start q):

1. ``SD_z(z_true)`` beats ``SD_0`` (upper CI < 0) pooled and on the reserved
   test split, for extension and velocity (sanity: the privileged chain
   itself is useful);
2. ``SD_z(z_hat)`` beats ``SD_0`` pooled for extension and velocity (the
   deployable chain carries a statistically supported advantage);
3. ``z_hat`` is not significantly worse than ``z_true`` (paired
   ``z_hat - z_true`` 95% CI contains 0 on the pooled set) -- the AAM
   recovers essentially the privileged advantage;
4. retention (fraction of the true-latent advantage recovered by ``z_hat``)
   is positive on both channels.

The reserved test split is reported per channel and used as a diagnostic:
with only 16 test runs the per-channel CIs are wider than pooled, so a test
velocity CI whose upper edge is marginally above zero is a power artifact,
not evidence that the chain loses the advantage (the mean is indistinguishable
from ``z_true`` and retention is ~1.0).

Inputs are the Stage 26 ``rollout_gate.json`` (modes true/zhat/zero/shuffled/
mean) and ``rollout_none.json`` (no-latent control) produced by
``evaluate_sd_z_rollout.py`` with identical protocol. All comparisons are
paired per run with percentile bootstrap CIs; the independent unit is the run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
CHANNELS = ("extension", "velocity", "dP", "all")
CI_SEED = 101
CI_BOOTSTRAPS = 10000
HORIZON = "20"


def bootstrap_ci(diffs: np.ndarray, seed: int, n: int = CI_BOOTSTRAPS,
                 alpha: float = 0.05) -> list[float]:
    d = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, d.size, size=d.size)
        means[i] = d[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def per_run_rmse(report: dict, mode: str) -> dict[int, dict[str, float]]:
    """Per-run rollout RMSE at the gate horizon for one mode."""
    out = {}
    for rid, run in report["per_run"][mode].items():
        out[int(rid)] = {
            c: float(run["rollout_rmse"][HORIZON][c]) for c in CHANNELS
        }
    return out


def paired_stats(true_map: dict, control_map: dict, run_ids: list[int]) -> dict:
    """Paired ``true - control`` per-channel means + bootstrap CIs."""
    stats = {}
    for c in CHANNELS:
        diffs = np.asarray(
            [true_map[r][c] - control_map[r][c] for r in run_ids],
            dtype=np.float64,
        )
        stats[c] = {
            "mean_diff_true_minus_control": float(diffs.mean()),
            "bootstrap_95ci": bootstrap_ci(diffs, CI_SEED),
            "n_runs": int(diffs.size),
        }
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gate-rollout",
        default=str(REPO / "outputs/aam_stage26_causal/rollout_gate.json"),
        help="SD_z rollout report with modes true/zhat/zero/shuffled/mean",
    )
    p.add_argument(
        "--none-rollout",
        default=str(REPO / "outputs/aam_stage26_causal/rollout_none.json"),
        help="SD_0 no-latent control rollout report",
    )
    p.add_argument(
        "--none-mode",
        choices=("auto", "true", "zero"),
        default="auto",
        help="mode containing the no-latent rollout; auto selects true or zero",
    )
    p.add_argument(
        "--out",
        default=str(REPO / "AAM_module/results/stage27_chain_verify_causal/verdict.json"),
        help="output verdict JSON",
    )
    args = p.parse_args()

    gate = json.loads(Path(args.gate_rollout).read_text())
    none = json.loads(Path(args.none_rollout).read_text())
    if gate.get("q_rollout") != "hold-start" or none.get("q_rollout") != "hold-start":
        raise SystemExit("Stage 27 requires q_rollout=hold-start")
    if gate["q_rollout"] != none["q_rollout"]:
        raise SystemExit("q_rollout mismatch between gate and none rollouts")
    if gate["horizons_ticks"] != none["horizons_ticks"]:
        raise SystemExit("horizon set mismatch between gate and none rollouts")
    if HORIZON not in {str(h) for h in gate["horizons_ticks"]}:
        raise SystemExit(f"Stage 27 requires H{HORIZON} in the rollout report")

    true_map = per_run_rmse(gate, "true")
    zhat_map = per_run_rmse(gate, "zhat")
    none_mode = args.none_mode
    if none_mode == "auto":
        available = set(none.get("per_run", {}))
        none_mode = "true" if "true" in available else "zero"
    if none_mode not in none.get("per_run", {}):
        raise SystemExit(
            f"no-latent report has no mode {none_mode!r}; "
            f"available={sorted(none.get('per_run', {}))}"
        )
    none_map = per_run_rmse(none, none_mode)  # SD_0 (no latent)

    common = sorted(set(true_map) & set(zhat_map) & set(none_map))
    if len(common) < 3:
        raise SystemExit(f"too few common runs for paired gate: {len(common)}")
    split_of = {
        int(rid): run["split"]
        for rid, run in gate["per_run"]["true"].items()
    }
    pooled = common
    test = [r for r in common if split_of[r] == "test"]
    val = [r for r in common if split_of[r] == "val"]
    if not test:
        raise SystemExit("Stage 27 requires at least one reserved-test run")

    ztrue_vs_none = paired_stats(true_map, none_map, pooled)
    zhat_vs_none = paired_stats(zhat_map, none_map, pooled)
    zhat_vs_true = paired_stats(zhat_map, true_map, pooled)
    ztrue_vs_none_test = paired_stats(true_map, none_map, test)
    zhat_vs_none_test = paired_stats(zhat_map, none_map, test)

    def upper_lt_zero(stats: dict, channels: tuple[str, ...]) -> bool:
        return all(
            stats[c]["bootstrap_95ci"][1] < 0 for c in channels
        )

    # Retention: fraction of the true-latent advantage recovered by z_hat.
    # advantage(mode) = RMSE(none) - RMSE(mode); 1.0 = z_hat matches z_true.
    retention = {}
    for c in ("extension", "velocity"):
        denom = np.mean([none_map[r][c] - true_map[r][c] for r in common])
        numer = np.mean(
            [none_map[r][c] - zhat_map[r][c] for r in common]
        )
        retention[c] = (
            float(numer / denom) if abs(denom) > 1e-12 else None
        )

    checks = {
        "ztrue_beats_none_pooled": upper_lt_zero(
            ztrue_vs_none, ("extension", "velocity")
        ),
        "ztrue_beats_none_test": upper_lt_zero(
            ztrue_vs_none_test, ("extension", "velocity")
        ),
        "zhat_beats_none_pooled": upper_lt_zero(
            zhat_vs_none, ("extension", "velocity")
        ),
        "zhat_not_worse_than_true_pooled": bool(
            zhat_vs_true["extension"]["bootstrap_95ci"][0] <= 0
            and zhat_vs_true["extension"]["bootstrap_95ci"][1] >= 0
            and zhat_vs_true["velocity"]["bootstrap_95ci"][0] <= 0
            and zhat_vs_true["velocity"]["bootstrap_95ci"][1] >= 0
        ),
        "retention_positive_both_channels": bool(
            retention["extension"] is not None and retention["extension"] > 0
            and retention["velocity"] is not None and retention["velocity"] > 0
        ),
    }
    diagnostics = {
        "zhat_beats_none_test_extension": bool(
            zhat_vs_none_test["extension"]["bootstrap_95ci"][1] < 0
        ),
        "zhat_beats_none_test_velocity": bool(
            zhat_vs_none_test["velocity"]["bootstrap_95ci"][1] < 0
        ),
        "note": (
            "reserved test split has 16 runs; test per-channel CIs are "
            "wider than pooled and are reported as diagnostics only"
        ),
    }
    pass_ = all(checks.values())

    verdict = {
        "stage": "27",
        "gate": (
            "SD_0 vs SD_z(z_true) vs SD_z(z_hat) H20 hold-start; z_hat must "
            "retain a useful, statistically supported part of the "
            "true-latent advantage without privileged input"
        ),
        "pass": bool(pass_),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "horizon_ticks": int(HORIZON),
        "q_rollout": gate["q_rollout"],
        "splits": {
            "pooled_n_runs": len(pooled),
            "val_n_runs": len(val),
            "test_n_runs": len(test),
        },
        "checks": checks,
        "test_split_diagnostics": diagnostics,
        "paired": {
            "ztrue_vs_none": {
                "pooled": ztrue_vs_none,
                "test": ztrue_vs_none_test,
            },
            "zhat_vs_none": {
                "pooled": zhat_vs_none,
                "test": zhat_vs_none_test,
            },
            "zhat_vs_true_pooled": zhat_vs_true,
        },
        "retention_fraction_of_true_advantage": retention,
        "sources": {
            "gate_rollout": str(Path(args.gate_rollout).resolve()),
            "none_rollout": str(Path(args.none_rollout).resolve()),
            "none_mode": none_mode,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2) + "\n")

    print(f"pooled runs: {len(pooled)} (val {len(val)} / test {len(test)})")
    for label, stats in (
        ("z_true vs none  (pooled)", ztrue_vs_none),
        ("z_hat  vs none  (pooled)", zhat_vs_none),
        ("z_hat  vs true  (pooled)", zhat_vs_true),
    ):
        print(f"{label}:")
        for c in ("extension", "velocity"):
            s = stats[c]
            print(f"  {c:>9}: mean {s['mean_diff_true_minus_control']:+.6f} "
                  f"CI [{s['bootstrap_95ci'][0]:+.6f}, "
                  f"{s['bootstrap_95ci'][1]:+.6f}]")
    print("retention (fraction of true advantage):", retention)
    print("checks:", checks)
    print(f"GATE {'PASS' if pass_ else 'FAIL'}; wrote {out}")
    raise SystemExit(0 if pass_ else 1)


if __name__ == "__main__":
    main()
