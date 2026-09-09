#!/usr/bin/env python3
"""Paired per-seed comparison of SD_0 against SD+AAM e2e on the open-loop data.

Both arms are trained on the identical frozen dataset, split and objective, and
report the same estimand: free-running accumulated normalised state error at
H=5 with tf_ratio=0. Only the latent differs.

The pilot showed -21.5% on a single unmatched pair. That is the kind of margin
the closed-loop collection never produced, but a single seed cannot separate a
real effect from one outlier carrying the mean -- which is exactly how the
earlier `zhat<frozen` result failed. This reports the paired differences, the
sign count, and a distribution-free interval, and refuses to declare a win on
the mean alone.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASE_KEY = "best_val_loss"       # train_sd_z.py
E2E_KEY = "val_rollout_state_mse"  # train_sd_aam_e2e.py


def read(p: Path, key: str) -> float | None:
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    return float(d[key]) if key in d else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    seeds = sorted({int(p.name.split("_s")[-1])
                    for p in a.root.glob("*_s*") if p.is_dir()})
    rows, missing = [], []
    for s in seeds:
        b = read(a.root / f"sd_none_s{s}/metrics.json", BASE_KEY)
        e = read(a.root / f"e2e_s{s}/metrics.json", E2E_KEY)
        if b is None or e is None:
            missing.append((s, b is None, e is None))
            continue
        rows.append((s, b, e))

    if missing:
        for s, nb, ne in missing:
            print(f"  seed {s}: incomplete pair "
                  f"({'sd_none ' if nb else ''}{'e2e' if ne else ''} missing)")
    if not rows:
        raise SystemExit("no complete pairs yet")

    print(f"\n  {'seed':>9}  {'SD_0':>9}  {'e2e':>9}  {'delta':>9}  {'rel':>8}")
    for s, b, e in rows:
        print(f"  {s:>9}  {b:9.5f}  {e:9.5f}  {e - b:+9.5f}  "
              f"{100 * (e / b - 1):+7.1f}%")

    base = np.array([r[1] for r in rows])
    e2e = np.array([r[2] for r in rows])
    d = e2e - base
    rel = 100.0 * (e2e / base - 1.0)
    n = len(d)
    wins = int((d < 0).sum())

    # Paired bootstrap over seeds. n=5 is small, so this is reported as a range
    # rather than a p-value; the sign count is the primary evidence.
    rng = np.random.default_rng(0)
    boot = np.array([rel[rng.integers(0, n, n)].mean() for _ in range(20000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n  n = {n} matched pairs")
    print(f"  e2e better in {wins}/{n} seeds")
    print(f"  mean relative change  {rel.mean():+.1f}%   "
          f"(per-seed {rel.min():+.1f}% .. {rel.max():+.1f}%)")
    print(f"  paired bootstrap 95% CI on the mean  [{lo:+.1f}%, {hi:+.1f}%]")

    unanimous = wins == n
    ci_excludes_zero = hi < 0.0
    verdict = "PASS" if (unanimous and ci_excludes_zero) else "NOT ESTABLISHED"
    print(f"\n  claim 2 ({verdict}): the latent lowers free-running error on "
          f"open-loop data")
    if not unanimous:
        print("    sign count is not unanimous -- do not report the mean alone")
    if not ci_excludes_zero:
        print("    bootstrap interval includes zero")

    if a.out:
        a.out.write_text(json.dumps({
            "estimand": "free-running accumulated normalised state error, "
                        "tf_ratio=0, H=5",
            "n_pairs": n, "seeds": [r[0] for r in rows],
            "sd_none": base.tolist(), "e2e": e2e.tolist(),
            "relative_pct": rel.tolist(),
            "mean_relative_pct": float(rel.mean()),
            "wins": wins, "bootstrap_ci_pct": [float(lo), float(hi)],
            "unanimous": bool(unanimous),
            "verdict": verdict,
        }, indent=2))
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
