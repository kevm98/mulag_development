#!/usr/bin/env python3
"""Is boom_sweep infeasible by construction? Pure geometry, no GPU, no sim.

Question 10 of the review brief asks what evidence would justify an
"infeasible by construction" verdict. This script supplies the geometric half:
it regenerates the ACTUAL tracked path for every family on every held-out mesh
and measures what the path DEMANDS, independent of any policy.

``verge_task/terrain.py`` is pure numpy + trimesh, so the exact same
generator the simulator uses can be called directly on the CPU.

Three demands are measured per family:

  peak |offset|      how far from strip centre the path goes            [m]
  peak |dy/dx|       path slope in the strip frame                      [m/m]
  peak |dy/dt|       required LATERAL SPEED at the truck's ground speed [m/s]

The last is the one that matters. The truck drives at a fixed 1.111 m/s and
cannot slow down for a corner, so a path with slope s demands a lateral cutter
speed of s * 1.111 m/s. If that exceeds what the cylinders can deliver, the
error is accumulated by construction and no policy can remove it.

Capability ceiling, from the shipped configuration:

  cylinder action scale   0.004 m per control step   (c1-c3)
  control step            0.02 s  (rom_dt 0.01 x decimation 2)
  => cylinder rate        0.200 m/s

which is an UPPER bound on cutter lateral speed: it assumes a lateral Jacobian
of 1.0, i.e. that a metre of cylinder extension buys a full metre of lateral
cutter travel, and that the whole cylinder budget is spent on lateral motion
rather than on holding height and head angle. Real capability is lower.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verge_task.terrain import (  # noqa: E402
    MOW_STRIP_EVAL_CFG,
    TRACKING_PATH_FAMILIES,
    apply_relief_profile,
    tracking_line_center_y_at,
)

TRUCK_SPEED_MPS = 4.00 / 3.6          # the scripted drive speed, 4 km/h
CONTROL_DT_S = 0.02                   # rom_dt 0.01 * decimation 2
CYLINDER_RATE_MPS = 0.004 / CONTROL_DT_S   # 0.200 m/s, optimistic ceiling


def measure(cfg, family: str, samples: int = 4000) -> dict:
    """Demand statistics for one family on one mesh."""
    cfg = replace(cfg, tracking_path_family=family)
    x0 = float(cfg.tracking_reference_x_m)
    x1 = x0 + float(cfg.tracking_active_length_m)
    x = np.linspace(x0, x1, samples)
    y = np.asarray(tracking_line_center_y_at(cfg, x), dtype=float)

    offset = y - float(cfg.center_y_m)
    dydx = np.gradient(y, x)
    lateral_speed = np.abs(dydx) * TRUCK_SPEED_MPS

    return {
        "family": family,
        "peak_abs_offset_m": float(np.abs(offset).max()),
        "rms_offset_m": float(np.sqrt(np.mean(offset**2))),
        "peak_abs_slope": float(np.abs(dydx).max()),
        "peak_lateral_speed_mps": float(lateral_speed.max()),
        "p95_lateral_speed_mps": float(np.quantile(lateral_speed, 0.95)),
        "fraction_over_capability": float(
            (lateral_speed > CYLINDER_RATE_MPS).mean()
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meshes", default="1337,17,909")
    p.add_argument("--amplitude_scale", type=float, default=2.2)
    p.add_argument("--width_m", type=float, default=1.8)
    p.add_argument("--length_m", type=float, default=26.0)
    p.add_argument("--active_length_m", type=float, default=16.0)
    p.add_argument("--relief", default="flat_patches")
    p.add_argument("--report", default="")
    a = p.parse_args()

    meshes = [int(m) for m in a.meshes.split(",") if m.strip()]
    results: dict[str, list[dict]] = {}

    for mesh in meshes:
        cfg = apply_relief_profile(
            replace(
                MOW_STRIP_EVAL_CFG,
                seed=mesh,
                width_m=float(a.width_m),
                length_m=float(a.length_m),
                tracking_active_length_m=float(a.active_length_m),
                tracking_path_amplitude_scale=float(a.amplitude_scale),
            ),
            str(a.relief),
        )
        results[str(mesh)] = [
            measure(cfg, family) for family in sorted(TRACKING_PATH_FAMILIES)
        ]

    print(f"truck speed          {TRUCK_SPEED_MPS:.3f} m/s")
    print(f"control step         {CONTROL_DT_S:.3f} s")
    print(f"cylinder rate ceil   {CYLINDER_RATE_MPS:.3f} m/s  (optimistic: Jacobian=1.0)")
    print(f"amplitude scale      {a.amplitude_scale}   relief={a.relief}   width={a.width_m} m")
    print()

    for mesh in meshes:
        rows = sorted(
            results[str(mesh)], key=lambda r: -r["peak_lateral_speed_mps"]
        )
        print(f"=== mesh {mesh} " + "=" * 58)
        print(
            "%-18s %10s %10s %12s %12s"
            % ("family", "pk|off|", "rms off", "pk lat m/s", "%over cap")
        )
        for r in rows:
            flag = "  <== OVER CEILING" if r["peak_lateral_speed_mps"] > CYLINDER_RATE_MPS else ""
            print(
                "%-18s %10.4f %10.4f %12.4f %11.1f%%%s"
                % (
                    r["family"],
                    r["peak_abs_offset_m"],
                    r["rms_offset_m"],
                    r["peak_lateral_speed_mps"],
                    100.0 * r["fraction_over_capability"],
                    flag,
                )
            )
        print()

    if a.report:
        out = Path(a.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "truck_speed_mps": TRUCK_SPEED_MPS,
                    "control_dt_s": CONTROL_DT_S,
                    "cylinder_rate_ceiling_mps": CYLINDER_RATE_MPS,
                    "amplitude_scale": float(a.amplitude_scale),
                    "relief_profile": str(a.relief),
                    "width_m": float(a.width_m),
                    "per_mesh": results,
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"[PATH-FEASIBILITY] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
