#!/usr/bin/env python3
"""Operating-point drift across plants, per excitation-amplitude tier.

Every run in the open-loop collection executes the same excitation schedule, so
at a matched tick the only thing that differs between runs is the plant. That
makes the across-run spread of cylinder extension a direct measure of how far
the plants have DESYNCHRONISED in position by that point in the episode.

This is the measurement that explains why plant identity stops being readable
above amplitude 0.35: not actuator saturation (only ~10% of steps saturate even
at 0.70) but operating-point divergence. Once runs are half a probe-stroke
apart, a one-second window cannot separate "different because leaky" from
"different because sitting five centimetres lower".

Writes a JSON the Chapter 7 figure script reads, so nothing downstream is
recomputed by hand.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def tiers_of(meta: dict) -> dict[float, tuple[int, int]]:
    out: dict[float, list[int]] = {}
    for s in meta["segment_ids"]:
        if s.get("kind") != "probe":
            continue
        m = re.search(r"_a([0-9.]+)_", s["name"])
        if not m:
            continue
        amp = float(m.group(1))
        lo, hi = out.setdefault(amp, [s["start"], s["end"]])
        out[amp] = [min(lo, s["start"]), max(hi, s["end"])]
    return {a: (lo, hi) for a, (lo, hi) in sorted(out.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--max-runs", type=int, default=40,
                    help="runs to pool; the statistic is a spread ACROSS runs, "
                         "so this bounds cost, not validity")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    runs = sorted((a.raw_root / "runs").glob("[0-9]*"))[:a.max_runs]
    if len(runs) < 8:
        raise SystemExit(f"only {len(runs)} runs; spread would be noise")
    meta = json.loads((runs[0] / "metadata.json").read_text())
    tiers = tiers_of(meta)
    n_ctrl = int(meta["num_control_steps"])

    ext, cmd, sat, lim = [], [], [], []
    for r in runs:
        d = np.load(r / "trajectory.npz")
        ext.append(d["cylinder_extension"][:n_ctrl])
        cmd.append(d["command_after_limits"])
        sat.append(d["command_saturation_flag"])
        lim.append(d["cylinder_limit_flag"])
    ext, cmd = np.stack(ext), np.stack(cmd)
    sat, lim = np.stack(sat), np.stack(lim)

    rows = []
    for amp, (lo, hi) in tiers.items():
        sl = slice(lo, hi)
        rows.append({
            "amplitude": amp,
            "spread_at_tier_start_mm": float(ext[:, lo].std(0).mean() * 1e3),
            "spread_at_tier_end_mm": float(ext[:, hi - 1].std(0).mean() * 1e3),
            "mean_spread_mm": float(ext[:, sl].std(0).mean() * 1e3),
            "within_tier_travel_mm": float(
                (ext[:, sl].max(1) - ext[:, sl].min(1)).mean() * 1e3),
            "command_saturation_pct": float(100 * sat[:, sl].mean()),
            "cylinder_limit_pct": float(100 * lim[:, sl].mean()),
            "command_spread_across_plants": float(cmd[:, sl].std(0).mean()),
        })
    for r in rows:
        r["drift_over_travel"] = r["mean_spread_mm"] / r["within_tier_travel_mm"]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "source": str(a.raw_root), "runs_pooled": len(runs),
        "note": "all runs execute an identical excitation schedule; spread "
                "across runs at a matched tick is therefore plant-induced "
                "desynchronisation",
        "tiers": rows,
    }, indent=2))

    print(f"  {len(runs)} runs pooled")
    print(f"  {'amp':>6} {'spread':>9} {'travel':>9} {'ratio':>7} "
          f"{'cmd sat %':>10} {'cyl lim %':>10}")
    for r in rows:
        print(f"  {r['amplitude']:>6.2f} {r['mean_spread_mm']:>8.1f}mm "
              f"{r['within_tier_travel_mm']:>8.1f}mm {r['drift_over_travel']:>7.2f} "
              f"{r['command_saturation_pct']:>10.2f} {r['cylinder_limit_pct']:>10.2f}")
    print(f"\n  cumulative: {rows[0]['spread_at_tier_start_mm']:.1f}mm -> "
          f"{rows[-1]['spread_at_tier_end_mm']:.1f}mm across the episode")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
