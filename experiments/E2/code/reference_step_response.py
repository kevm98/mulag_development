#!/usr/bin/env python3
"""Step response of each control stack to a step in the COMMANDED line.

The plant's dead time is a property of the hydraulics and is the same for every
stack. What differs is how each control stack handles it, and that cannot be
recovered from operational data: five different phase and correlation
estimators applied to a normal drive returned five mutually inconsistent
answers, because at the frequencies where the reference and the achieved path
are coherent the phase is dominated by controller dynamics rather than by
transport.

So the delay is measured by injecting one. A 100 mm lateral step is added to the
commanded line partway through the scored window, and each stack is scored on
the same mesh, seed and speed. The step reaches the policy's own preview of the
line, so the controller sees the new target rather than only being charged for
missing it.

Reported per stack:
    dead time      first sample after the step at which the cutter leaves its
                   pre-step band, in ms
    rise time      10% to 90% of the commanded 100 mm
    settled        first time |error| stays under 20 mm for 0.5 s
    overshoot      peak excursion beyond the new line, as a percentage

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FS = 60.0
STACKS = [("ppo_rom", "PPO-ROM"), ("ppo_id_fd", "PPO-ID-FD"),
          ("ppo_aam_id_fd", "PPO-(AAM)-ID-FD")]


def response(y, k0, step, pre=30, k_sigma=4.0, settle_tol=0.020,
             settle_hold=30):
    """(dead_ms, rise_ms, settle_ms, overshoot_pct) for one environment.

    `y` must already have the underlying commanded path removed, so that the
    only thing left to follow is the step itself. Measuring the raw cutter
    position against a frozen pre-step baseline conflates the response with the
    trajectory continuing to curve, which inflates settling time and overshoot
    over the seconds that follow.
    """
    base = y[k0 - pre:k0]
    mu, sd = base.mean(), base.std()
    target = mu + step
    seg = y[k0:]
    if sd < 1e-9:
        sd = 1e-9
    hit = np.flatnonzero(np.abs(seg - mu) > k_sigma * sd)
    dead = hit[0] / FS * 1000 if hit.size else np.nan
    prog = (seg - mu) / step
    i10 = np.flatnonzero(prog >= 0.1)
    i90 = np.flatnonzero(prog >= 0.9)
    rise = ((i90[0] - i10[0]) / FS * 1000
            if i10.size and i90.size and i90[0] >= i10[0] else np.nan)
    ok = np.abs(seg - target) < settle_tol
    settle = np.nan
    for i in range(len(ok) - settle_hold):
        if ok[i:i + settle_hold].all():
            settle = i / FS * 1000
            break
    over = (prog.max() - 1.0) * 100 if prog.size else np.nan
    return dead, rise, settle, max(over, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mesh", default="1337")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    print(f"\n  {'stack':<20}{'dead time':>11}{'rise':>10}{'settled':>10}"
          f"{'overshoot':>11}{'n':>5}")
    report = {"step_m": 0.10, "stacks": {}}
    for key, label in STACKS:
        p = a.root / f"{key}_m{a.mesh}.npz"
        if not p.is_file():
            print(f"  {label}: missing")
            continue
        d = np.load(p)
        ref = d["reference_y"][:, :, 0]
        cut = d["cutter_y"][:, :, 0]
        rows = []
        for e in range(ref.shape[1]):
            dr = np.diff(ref[:, e])
            k = int(np.argmax(np.abs(dr))) + 1
            step = float(dr[k - 1])
            if abs(step) < 0.05:          # this environment saw no step
                continue
            if k < 40 or k > len(ref) - 120:
                continue
            # Reconstruct the line WITHOUT the step and subtract it, leaving a
            # signal whose only content is the response to the step.
            base_ref = ref[:, e].copy()
            base_ref[k:] -= step
            rows.append(response(cut[:, e] - base_ref, k, step))
        if not rows:
            print(f"  {label}: no environment shows the step")
            continue
        arr = np.array(rows, dtype=float)
        med = np.nanmedian(arr, axis=0)
        print(f"  {label:<20}{med[0]:8.0f} ms{med[1]:7.0f} ms{med[2]:7.0f} ms"
              f"{med[3]:10.0f}%{len(rows):5d}")
        report["stacks"][label] = {
            "dead_time_ms": float(med[0]), "rise_ms": float(med[1]),
            "settle_ms": float(med[2]), "overshoot_pct": float(med[3]),
            "n_envs": len(rows)}
    print("\n  plant dead time from open-loop excitation: 117 ms")
    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
