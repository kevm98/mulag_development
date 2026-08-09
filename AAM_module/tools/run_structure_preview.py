#!/usr/bin/env python
"""Preview and validate the Stage 13 run structure (headless, no Isaac).

Builds the 11-step run plan, prints the segment table, and validates:
- segment boundaries are contiguous and cover the whole run;
- every segment's reference seeds stay inside the learned stroke limits;
- the composite probe schedule (ID segments only) is bounded, +- signed,
  independent across channels, and full-rank (input diagnostics);
- the probe amplitude is the Stage 12 measured safe value by default.

Usage:
    python run_structure_preview.py [--id-s 12.0] [--probe prbs|multisine]
        [--amplitude 0.015]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import build_run_plan  # noqa: E402
from tools.action_identifiability import input_diagnostics  # noqa: E402


LEARNED_LIMITS = [
    (0.0643, 0.4193),
    (0.4147, 0.7387),
    (0.3776, 0.6780),
    (0.5335, 0.7033),
]


def stroke_check(plan) -> list[dict]:
    """Worst-case reference extension per segment vs learned limits."""
    out = []
    for seg in plan.segments:
        ref = seg.reference_seeds
        if ref is None:
            continue
        row = {"name": seg.name, "kind": seg.kind}
        ok = True
        for c in range(4):
            lo, hi = LEARNED_LIMITS[c]
            mn, mx = float(ref[:, c].min()), float(ref[:, c].max())
            row[f"cyl{c}"] = {
                "min": round(mn, 4),
                "max": round(mx, 4),
                "ok": bool(mn >= lo and mx <= hi),
            }
            ok = ok and mn >= lo and mx <= hi
        row["ok"] = ok
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--id-s",
        type=float,
        default=30.0,
        help="ID segment length; keep >= the Stage 12 designed probe segment "
        "(30 s / 1800 steps) so PRBS channel independence (0.102) holds.",
    )
    p.add_argument("--probe", choices=("prbs", "multisine"), default="prbs")
    p.add_argument("--amplitude", type=float, default=0.015)
    p.add_argument("--probe-seed", type=int, default=20260807)
    p.add_argument("--probe-hold-steps", type=int, default=10)
    p.add_argument("--trajectory-stride", type=int, default=1)
    p.add_argument("--out", default=str(AAM_ROOT / "results/run_structure_preview.json"))
    args = p.parse_args()

    plan = build_run_plan(
        id_s=args.id_s,
        probe_name=args.probe,
        probe_amplitude=args.amplitude,
        probe_seed=args.probe_seed,
        probe_hold_steps=args.probe_hold_steps,
        trajectory_stride=args.trajectory_stride,
    )

    # Contiguity.
    starts = [s.start for s in plan.segments]
    ends = [s.end for s in plan.segments]
    contiguous = starts == [0] + ends[:-1] and ends[-1] == plan.total_steps

    # Probe schedule diagnostics.
    U = plan.probe_schedule()
    diag = input_diagnostics(U, "composite_probe", (1, 3, 6))
    corr = np.corrcoef(U.T)
    indep = all(abs(corr[i, j]) < 0.15 for i in range(4) for j in range(i + 1, 4))
    both_signs = all(bool(U[:, c].max() > 0.0 and U[:, c].min() < 0.0) for c in range(4))

    # Independence is a per-segment property: the zeros between ID segments
    # inflate the composite correlation statistic without adding excitation.
    per_segment = []
    for seg in plan.segments:
        if seg.probe_action is None:
            continue
        P = seg.probe_action
        c = np.corrcoef(P.T)
        per_segment.append(
            {
                "name": seg.name,
                "steps": seg.steps,
                "max_cross_corr": round(float(np.abs(c - np.eye(4)).max()), 3),
                "independent": bool(
                    all(abs(c[i, j]) < 0.15 for i in range(4) for j in range(i + 1, 4))
                ),
            }
        )

    stroke = stroke_check(plan)
    stroke_ok = all(r["ok"] for r in stroke)

    table = plan.segment_table()
    probe_steps = sum(s.steps for s in plan.segments if s.kind == "id_excitation")
    dls_steps = sum(s.steps for s in plan.segments if s.kind == "dls_trajectory")
    report = {
        "total_steps": plan.total_steps,
        "total_duration_s": round(plan.total_steps * plan.control_dt, 2),
        "probe": args.probe,
        "probe_amplitude": args.amplitude,
        "probe_seed": args.probe_seed,
        "probe_hold_steps": args.probe_hold_steps,
        "trajectory_stride": args.trajectory_stride,
        "contiguous": contiguous,
        "stroke_ok": stroke_ok,
        "segments": table,
        "stroke": stroke,
        "probe_schedule": {
            "probe_steps": probe_steps,
            "dls_trajectory_steps": dls_steps,
            "id_fraction": round(probe_steps / plan.total_steps, 3),
            "bounded": bool(np.abs(U).max() <= 1.0),
            "both_signs": both_signs,
            "independent_composite": indep,
            "max_cross_corr_composite": round(float(np.abs(corr - np.eye(4)).max()), 3),
            "per_segment": per_segment,
            "rank": diag["rank"],
            "num_rank": diag["effective_numerical_rank"],
            "kappa": round(diag["condition_number"], 3),
            "lagged": {
                L: {
                    "rank": diag[f"lagged_L{L}"]["rank"],
                    "num_rank": diag[f"lagged_L{L}"]["effective_numerical_rank"],
                    "kappa": round(diag[f"lagged_L{L}"]["condition_number"], 3),
                }
                for L in (1, 3, 6)
                if f"lagged_L{L}" in diag
            },
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"Stage 13 run structure ({args.probe}, amp={args.amplitude})")
    print(f"total: {plan.total_steps} steps = {report['total_duration_s']} s | "
          f"contiguous={contiguous} stroke_ok={stroke_ok}")
    print(f"{'name':<18}{'kind':<16}{'start':>6}{'end':>6}{'steps':>7}{'dur_s':>8}")
    for seg in table:
        print(
            f"{seg['name']:<18}{seg['kind']:<16}{seg['start']:>6}{seg['end']:>6}"
            f"{seg['steps']:>7}{seg['duration_s']:>8}"
        )
    ps = report["probe_schedule"]
    print(
        f"\nprobe: bounded={ps['bounded']} both_signs={ps['both_signs']} "
        f"composite_indep={ps['independent_composite']} "
        f"(max_corr={ps['max_cross_corr_composite']}) "
        f"rank={ps['rank']} kappa={ps['kappa']} id_fraction={ps['id_fraction']}"
    )
    for ps_seg in ps["per_segment"]:
        print(
            f"  {ps_seg['name']:<6} independent={ps_seg['independent']} "
            f"(max_corr={ps_seg['max_cross_corr']})"
        )
    for L, lag in ps["lagged"].items():
        print(f"  U_L L={L}: rank {lag['rank']} num {lag['num_rank']} kappa {lag['kappa']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
