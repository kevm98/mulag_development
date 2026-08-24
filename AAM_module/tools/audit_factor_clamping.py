#!/usr/bin/env python3
"""Audit what a deployment-factor sweep REQUESTS against what the plant APPLIES.

The randomisation sweep stretches each manifest draw away from the nominal
plant, ``value -> nominal + scale * (value - nominal)``
(:func:`verge_task.deployment_factors.select_factors`).  That stretch is never
validated against the audited factor ranges, because scaled rows are built
through the dataclass constructor and do not pass ``_parse_row``.

The plant clamps them instead.  ``HydraulicROM.forward`` pins every hydraulic
factor back into the range the model was identified on::

    compliance -> [-0.25, 0.10]      flow     -> [-0.25, 0.25]
    pressure   -> [-0.15, 0.15]      seal     -> [ 0.00, 1.00]

and then bounds the realised actuator response gain to ``[0.35, 1.50]``.  So a
sweep point above ``scale = 1`` does not deliver the hydraulic deviation its
label claims: only ``mass_scale`` and ``com_z``, applied through
``set_masses``/``set_coms`` without a clamp, keep scaling.

This script quantifies that per scale, so a degradation curve is read against
the stress that reached the plant rather than the stress that was requested.
It reads the manifest only and needs neither torch nor IsaacLab.

Usage::

    python AAM_module/tools/audit_factor_clamping.py
    python AAM_module/tools/audit_factor_clamping.py --scales 1 1.5 2 3 --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO / "AAM_module/datasets/rom_deployment_stage31_v1_frozen/raw/manifest.csv"
)

DIM_ORDER = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
# The nominal plant, i.e. the point every draw is stretched away from.
NOMINAL = {"flow": 0.0, "compliance": 0.0, "seal": 0.0,
           "pressure": 0.0, "mass_scale": 1.0, "com_z": 0.0}
# What HydraulicROM.forward actually enforces on the hydraulic channels.
ROM_CLAMP = {"compliance": (-0.25, 0.10), "flow": (-0.25, 0.25),
             "pressure": (-0.15, 0.15), "seal": (0.0, 1.0)}
RESPONSE_GAIN_CLAMP = (0.35, 1.50)


def response_gain(compliance: float, flow: float, pressure: float, seal: float) -> float:
    """Reproduce HydraulicROM's ``factor_response_gain`` (pre-clamp)."""
    return (
        math.exp(0.5 * compliance)
        * (1.0 + flow)
        * math.sqrt(max(1.0 + pressure, 0.1))
        / (1.0 + 0.625 * seal)
    )


def clamp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return min(max(value, lo), hi)


def load_rows(manifest: Path, domain: str) -> list[dict[str, float]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if domain == "held_out":
        keep = [r for r in rows if r["domain"] == "combined" and r["split"] == "test"]
    elif domain == "all":
        keep = rows
    else:
        keep = [r for r in rows if r["domain"] == domain]
    if not keep:
        raise SystemExit(f"no manifest rows for domain {domain!r}")
    return [dict({name: float(r[name]) for name in DIM_ORDER},
                 run_id=int(r["run_id"])) for r in keep]


def audit(rows: list[dict[str, float]], scale: float) -> dict:
    per_dim = {name: 0 for name in ROM_CLAMP}
    rows_clamped = 0
    delivered: list[float] = []
    gains: list[float] = []
    for row in rows:
        hit = False
        applied: dict[str, float] = {}
        requested: dict[str, float] = {}
        for name in DIM_ORDER:
            base = NOMINAL[name]
            requested[name] = base + scale * (row[name] - base)
            if name in ROM_CLAMP:
                applied[name] = clamp(requested[name], ROM_CLAMP[name])
                if abs(applied[name] - requested[name]) > 1e-9:
                    per_dim[name] += 1
                    hit = True
            else:
                applied[name] = requested[name]
        rows_clamped += int(hit)
        # Delivered hydraulic deviation as a fraction of the requested one.
        req_norm = math.hypot(*(requested[n] - NOMINAL[n] for n in ROM_CLAMP))
        app_norm = math.hypot(*(applied[n] - NOMINAL[n] for n in ROM_CLAMP))
        delivered.append(app_norm / req_norm if req_norm > 1e-12 else 1.0)
        gains.append(clamp(
            response_gain(applied["compliance"], applied["flow"],
                          applied["pressure"], applied["seal"]),
            RESPONSE_GAIN_CLAMP,
        ))
    n = len(rows)
    return {
        "scale": scale,
        "rows": n,
        "rows_with_any_clamped_factor": rows_clamped,
        "per_factor_clamped": per_dim,
        "delivered_hydraulic_fraction_mean": sum(delivered) / n,
        "delivered_hydraulic_fraction_min": min(delivered),
        "response_gain_min": min(gains),
        "response_gain_max": max(gains),
    }


def authority_floor() -> dict:
    """Least actuator authority reachable through the factor path at all."""
    worst = response_gain(ROM_CLAMP["compliance"][0], ROM_CLAMP["flow"][0],
                          ROM_CLAMP["pressure"][0], ROM_CLAMP["seal"][1])
    best = response_gain(ROM_CLAMP["compliance"][1], ROM_CLAMP["flow"][1],
                         ROM_CLAMP["pressure"][1], ROM_CLAMP["seal"][0])
    floor = max(worst, RESPONSE_GAIN_CLAMP[0])
    return {
        "response_gain_worst_case": worst,
        "response_gain_best_case": best,
        "hard_clamp": list(RESPONSE_GAIN_CLAMP),
        "reachable_floor": floor,
        "max_authority_removable_pct": 100.0 * (1.0 - floor),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--domain", default="held_out",
                   choices=("held_out", "all", "combined", "hydraulic",
                            "mechanical", "nominal"))
    p.add_argument("--scales", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--json", default=None, help="write the report to this path")
    a = p.parse_args()

    rows = load_rows(Path(a.manifest).expanduser().resolve(), a.domain)
    report = {
        "manifest": a.manifest,
        "domain": a.domain,
        "rows": len(rows),
        "authority": authority_floor(),
        "scales": [audit(rows, s) for s in a.scales],
    }

    auth = report["authority"]
    print(f"manifest {a.manifest}")
    print(f"domain {a.domain}: {len(rows)} factor rows\n")
    print("Response gain -- the factor that multiplies BOTH the velocity and the")
    print("extension increment the ROM emits, so it scales realised cylinder motion:")
    print(f"  reachable {auth['response_gain_worst_case']:.4f} .. "
          f"{auth['response_gain_best_case']:.4f}; explicit limiter {auth['hard_clamp']} "
          f"alone permits a {100*(1-auth['hard_clamp'][0]):.0f}% cut")
    print(f"  => the factor path cannot scale the ROM's realised actuator motion")
    print(f"     below {auth['reachable_floor']:.3f} of nominal, a reduction of at most "
          f"{auth['max_authority_removable_pct']:.1f}%.")
    print("     A knockout to zero is NOT reachable by this route, and on the ID/FD")
    print("     plant these factors scale nothing at all (they are not consumed).\n")
    head = (f"{'scale':>6} {'clamped rows':>14} {'delivered frac':>15} "
            f"{'gain min':>9}  per-factor")
    print(head)
    print("-" * len(head))
    for entry in report["scales"]:
        print(f"{entry['scale']:>6.2f} "
              f"{entry['rows_with_any_clamped_factor']:>7}/{entry['rows']:<6} "
              f"{entry['delivered_hydraulic_fraction_mean']:>15.3f} "
              f"{entry['response_gain_min']:>9.3f}  {entry['per_factor_clamped']}")
    print("\n'delivered frac' is the applied hydraulic deviation as a fraction of the")
    print("requested one -- a ratio in FACTOR SPACE, not a measure of authority.")
    print("1.000 means the sweep point reached the plant intact. It is a different")
    print("quantity from the response gain above; do not translate one into the other.")
    print("mass_scale and com_z are never clamped and keep scaling throughout.")

    if a.json:
        out = Path(a.json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
