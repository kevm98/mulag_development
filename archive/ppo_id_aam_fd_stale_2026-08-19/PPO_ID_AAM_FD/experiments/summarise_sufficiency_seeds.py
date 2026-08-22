#!/usr/bin/env python3
"""Does the sufficiency flip survive reseeding?

One seed gave `none < zhat < true` with the full feature set (the latent HURTS)
and `true < zhat < none` with dP/force masked (the latent HELPS), on
differences of 0.6-0.9%. The flip is the claim; the magnitudes are small enough
that a single seed cannot carry it.

Reports, per configuration, the mean and spread across seeds, and -- the part
that matters -- how many seeds reproduce the ordering. An ordering that holds
in 5/5 seeds is a result; 3/5 is not, however good the means look.

Paired by seed: every variant within a seed sees the same initialisation and
the same data, so the within-seed differences are the comparison. Reporting
only across-seed means would hide that.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from statistics import mean, stdev

VARIANTS = ("none", "zhat", "true")
CONFIGS = {"none": "full features", "no_dp_force": "dP/force masked"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "outputs/stage31_rom/seeds")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    runs: dict[tuple[str, int], dict[str, float]] = {}
    for d in sorted(a.dir.glob("*_s*")):
        abl, _, seed = d.name.rpartition("_s")
        if not seed.isdigit():
            continue
        vals = {}
        for v in VARIANTS:
            f = d / f"{v}_metrics.json"
            if f.is_file():
                vals[v] = float(json.loads(f.read_text())["best_val_loss"])
        if len(vals) == len(VARIANTS):
            runs[(abl, int(seed))] = vals
    if not runs:
        raise SystemExit(f"no complete seed runs under {a.dir}")

    report = {}
    for abl, label in CONFIGS.items():
        seeds = sorted(s for (c, s) in runs if c == abl)
        if not seeds:
            continue
        print(f"\n=== {label}  ({abl})   {len(seeds)} seeds ===")
        print(f"{'seed':>5}  " + "  ".join(f"{v:>10}" for v in VARIANTS)
              + "   ordering")
        helps = 0
        for s in seeds:
            v = runs[(abl, s)]
            order = " < ".join(sorted(VARIANTS, key=lambda k: v[k]))
            # "helps" = the privileged latent is best, i.e. true < none
            if v["true"] < v["none"]:
                helps += 1
            print(f"{s:>5}  " + "  ".join(f"{v[k]:>10.5f}" for k in VARIANTS)
                  + f"   {order}")
        agg = {}
        for k in VARIANTS:
            xs = [runs[(abl, s)][k] for s in seeds]
            agg[k] = {"mean": mean(xs),
                      "sd": stdev(xs) if len(xs) > 1 else 0.0}
            print(f"  {k:>5} mean {agg[k]['mean']:.5f}  sd {agg[k]['sd']:.5f}")
        # paired within-seed deltas: the actual comparison
        dt = [runs[(abl, s)]["true"] - runs[(abl, s)]["none"] for s in seeds]
        dz = [runs[(abl, s)]["zhat"] - runs[(abl, s)]["none"] for s in seeds]
        print(f"  paired true-none  {mean(dt):+.5f}"
              + (f" +- {stdev(dt):.5f}" if len(dt) > 1 else "")
              + f"   negative in {sum(x<0 for x in dt)}/{len(dt)} seeds")
        print(f"  paired zhat-none  {mean(dz):+.5f}"
              + (f" +- {stdev(dz):.5f}" if len(dz) > 1 else "")
              + f"   negative in {sum(x<0 for x in dz)}/{len(dz)} seeds")
        print(f"  latent HELPS (true<none) in {helps}/{len(seeds)} seeds")
        report[abl] = {"label": label, "n_seeds": len(seeds), "per_variant": agg,
                       "paired_true_minus_none": dt, "paired_zhat_minus_none": dz,
                       "helps_in_seeds": helps}

    if len(report) == 2:
        f, m = report.get("none"), report.get("no_dp_force")
        print("\n=== THE FLIP ===")
        print(f"  full features   : latent helps in {f['helps_in_seeds']}/{f['n_seeds']} seeds")
        print(f"  dP/force masked : latent helps in {m['helps_in_seeds']}/{m['n_seeds']} seeds")
        if m["helps_in_seeds"] > f["helps_in_seeds"]:
            print("  -> the flip reproduces: masking the observation is what")
            print("     makes the latent useful.")
        else:
            print("  -> the flip does NOT reproduce across seeds. The single-seed")
            print("     result was noise; do not build the claim on it.")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
