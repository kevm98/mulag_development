#!/usr/bin/env python3
"""Read a training curve CYCLE-ALIGNED, because raw first-vs-last lies.

An episode here is ~47.3 s at a 0.0167 s step (~2840 steps) while one PPO
iteration collects `num_steps_per_env` steps per environment. At 32 steps that
is **~89 iterations per episode**, so the logged per-iteration metrics trace the
WITHIN-EPISODE error profile: excellent just after a reset, decaying through the
episode, snapping back at the next one.

Consequences, all of which were hit for real on 2026-08-29:

* Comparing the first logged value to the last compares two different phases of
  that cycle and **manufactures apparent collapses**. Several "the policy is
  degrading" readings that day were exactly this.
* It explains the otherwise impossible observation that metrics "degraded"
  during a critic warmup while the actor was FROZEN. A constant policy cannot
  degrade itself; the cycle was moving, not the policy.
* Genuine failures remain visible: a truly broken run holds a bad value across
  ALL phases (one held lateral ~1.25 m throughout, far outside the healthy
  0.03-0.25 band).

So this tool reports each cycle's BEST phase, which is the like-for-like
comparison, alongside the trough. Read peak-to-peak.

Usage:
    read_training_curve.py <run-dir-or-name> [--cycle 89] [--logroot ...]
"""

from __future__ import annotations

import argparse
import glob
import statistics as st
from pathlib import Path

DEFAULT_LOGROOT = "logs/rsl_rl/ppo_id_aam_fd"
# 47.3 s episode / 0.0167 s step = ~2840 steps; 2840 / 32 steps-per-iteration.
DEFAULT_CYCLE = 89

METRICS = (
    ("Episode_Reward/total", "total"),
    ("Metrics/lateral_error_m", "lateral_m"),
    ("Metrics/cut_height_error_m", "cut_m"),
    ("Metrics/ground_contact_fraction", "ground_contact"),
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--cycle", type=int, default=DEFAULT_CYCLE)
    p.add_argument("--logroot", default=DEFAULT_LOGROOT)
    a = p.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = Path(a.run)
    if not run_dir.is_dir():
        run_dir = Path(a.logroot) / a.run
    files = sorted(glob.glob(str(run_dir / "events*")))
    if not files:
        raise SystemExit(f"no tensorboard events under {run_dir}")

    ea = EventAccumulator(files[-1])
    ea.Reload()
    have = ea.Tags()["scalars"]
    series = {label: [x.value for x in ea.Scalars(tag)]
              for tag, label in METRICS if tag in have}
    if "total" not in series:
        raise SystemExit(f"Episode_Reward/total not logged; have {sorted(have)[:10]}")

    n = len(series["total"])
    cycle = int(a.cycle)
    print(f"{run_dir}")
    print(f"iterations={n}  cycle={cycle} (~one episode)  cycles={max(1, n // cycle)}")
    if n < cycle:
        print(f"\n*** WARNING: fewer iterations ({n}) than one episode cycle ({cycle}).")
        print("    No cycle-aligned comparison is possible; do NOT read a trend "
              "from this run.")

    labels = [l for _, l in METRICS if l in series]
    head = "  ".join(f"{l:>15s}" for l in labels)
    print(f"\n{'cycle':>5} {'range':>13} {head}   {'trough_total':>12}")
    peaks = []
    for c in range(0, n, cycle):
        idx = [i for i in range(c, min(c + cycle, n))]
        # A partial cycle need not contain the cycle's true peak, so it is shown
        # for completeness but EXCLUDED from the peak-to-peak comparison. Including
        # one turned a genuine +1.5% into a reported -2.4% on the first run of
        # this tool.
        partial = len(idx) < int(0.9 * cycle)
        best = max(idx, key=lambda i: series["total"][i])
        worst = min(idx, key=lambda i: series["total"][i])
        row = "  ".join(f"{series[l][best]:15.4f}" for l in labels)
        mark = "  (partial -- excluded)" if partial else ""
        print(f"{c//cycle:>5} {str((idx[0], idx[-1])):>13} {row}   "
              f"{series['total'][worst]:12.4f}{mark}")
        if not partial:
            peaks.append(series["total"][best])

    if len(peaks) >= 2:
        drift = peaks[-1] - peaks[0]
        pct = 100.0 * drift / abs(peaks[0]) if peaks[0] else float("nan")
        verdict = ("STABLE / IMPROVING" if drift >= -0.05 * abs(peaks[0])
                   else "DEGRADING across cycles")
        print(f"\npeak-to-peak (complete cycles only): {peaks[0]:.3f} -> {peaks[-1]:.3f} "
              f"({drift:+.3f}, {pct:+.1f}%)  ->  {verdict}")
        print(f"peak median across cycles: {st.median(peaks):.3f}")
    else:
        print("\nfewer than two complete cycles: no peak-to-peak comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
