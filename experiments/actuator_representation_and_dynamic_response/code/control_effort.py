#!/usr/bin/env python3
"""Control effort: what each stack spends to achieve its tracking.

Tracking error alone ranks outcomes, not cost. Two stacks with the same error
are not equivalent if one commands twice the actuation, and a stack that spends
its actuation reversing direction is failing differently from one that simply
tracks poorly.

Three measures, chosen so that none is a restatement of another:

  RMS command      how much actuation is used
  rate RMS         how fast it changes -- smoothness
  reversals/s      how often the command changes sign of motion. This is
                   scale-free: a stack can have a low rate RMS and still
                   chatter, and chatter is what damages the machine.

Effort must be read WITH the error it buys, never alone: a stack that gives up
on tracking trivially spends less. Both are therefore reported together.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FS = 60.0


def measure(path: Path) -> dict:
    d = np.load(path)
    u = d["valve_cmd"]
    du = np.diff(u, axis=0)
    sgn = np.sign(du)
    return {
        "rms_command": float(np.sqrt((u ** 2).mean())),
        "rate_rms_per_s": float(np.sqrt((du ** 2).mean()) * FS),
        "reversals_per_s": float((np.diff(sgn, axis=0) != 0).mean() * FS),
        "saturation_fraction": float((np.abs(u) > 0.99).mean()),
        "mean_abs_lateral_m": float(np.abs(d["lateral_error"][:, :, 0]).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True,
                    help="NAME=PATH pairs")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = {}
    for spec in a.traces:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.is_file():
            print(f"  missing {p}")
            continue
        rows[name] = measure(p)
    print(f"\n  {'arm':<20}{'RMS cmd':>10}{'rate RMS':>11}{'revers/s':>10}"
          f"{'sat':>8}{'|lat| mm':>10}")
    for name, r in rows.items():
        print(f"  {name:<20}{r['rms_command']:10.4f}{r['rate_rms_per_s']:11.2f}"
              f"{r['reversals_per_s']:10.1f}{r['saturation_fraction']:8.3f}"
              f"{1000*r['mean_abs_lateral_m']:10.1f}")
    print("\n  Effort is only interpretable beside the error it buys.")
    if a.out:
        a.out.write_text(json.dumps(rows, indent=2))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
