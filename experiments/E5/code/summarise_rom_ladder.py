#!/usr/bin/env python3
"""Reduce the ROM authority ladder to a robustness curve and a breaking point.

Reads the per-level, per-mesh reports written by
``run_rom_authority_ladder.sh`` and applies the thresholds predeclared in
``experiments/E5/README.md``:

    degradation onset    median lateral error >= 1.5x its own alpha=1 value,
                         or ground contact >= 5 percentage points below it
    functional breakdown ground contact < 0.700, or safety termination > 0.080

Three rules keep this honest, and each exists because the experiment already
tripped over the failure it prevents.

1.  Thresholds are the constants below. They were fixed before any of these
    runs existed and are never refitted to the curve.

2.  Spread is measured ACROSS TERRAIN MESHES, not across seeds. Varying
    ``--seed`` alone returns bit-identical numbers (``gates.py`` says so, and
    it was confirmed here), so seed "repeats" would report a spread of zero and
    understate the uncertainty. The mesh is the real source of variation.

3.  The lateral-error onset criterion is reported but flagged INAPPLICABLE on
    this axis. Removing actuator authority makes the arm progressively
    stiller, and lateral error, cut-height error, safety terminations and
    impacts all measure DEVIATION -- an arm that barely moves scores better on
    every one of them. Measured at alpha=0.01: lateral error fell to 0.3x
    nominal while ground contact fell 15 pp. Quietly dropping the criterion
    that cannot fire and reporting only the metric that moves is the
    cherry-pick the predeclaration exists to prevent, so it is named instead.

Ground contact is therefore the primary failure metric for this axis.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ONSET_LATERAL_MULTIPLE = 1.5
ONSET_CONTACT_DROP_PP = 5.0
BREAKDOWN_CONTACT = 0.700
BREAKDOWN_SAFETY_TERMINATION = 0.080

REPORT_RE = re.compile(r"rom_a([0-9.]+)_r(\d+)_m(\d+)\.json$")


def band_mean(mesh_report: dict, key: str) -> float | None:
    bands = mesh_report.get("distance_results") or []
    vals = [b[key] for b in bands if isinstance(b, dict) and key in b]
    return sum(vals) / len(vals) if vals else None


def load(report_dir: Path) -> dict[float, dict[str, dict]]:
    levels: dict[float, dict[str, dict]] = defaultdict(dict)
    for path in sorted(report_dir.glob("rom_a*_r*_m*.json")):
        m = REPORT_RE.search(path.name)
        if not m:
            continue
        alpha, mesh = float(m.group(1)), m.group(3)
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rep = (doc.get("per_mesh") or {}).get(mesh)
        if not rep:
            continue
        levels[alpha][mesh] = {
            # PPO_ROM/gates.py defines the 0.700 gate on
            # worst_mesh_worst_family_ground_contact_fraction, so THAT is the
            # metric the predeclared threshold belongs to. The band mean is
            # kept for context but is not what the threshold was written for;
            # scoring the threshold against the mean understates failure.
            "contact": rep.get("worst_family_ground_contact_fraction"),
            "band_mean_contact": band_mean(rep, "ground_contact_fraction"),
            "lateral": band_mean(rep, "mean_abs_lateral_error_m"),
            "cut_height": band_mean(rep, "mean_abs_cut_height_error_m"),
            "worst_contact": rep.get("worst_family_ground_contact_fraction"),
            "safety": rep.get("safety_termination_rate"),
        }
    return levels


def stats(per_mesh: dict[str, dict], key: str) -> dict:
    vals = [v[key] for v in per_mesh.values() if v.get(key) is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "min": min(vals),
        "max": max(vals),
        "spread": max(vals) - min(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--min-meshes", type=int, default=1,
                   help="levels measured on fewer meshes than this are listed but "
                        "excluded from the reported crossings")
    p.add_argument("--out", default="")
    a = p.parse_args()

    levels = load(Path(a.report_dir))
    if not levels:
        print("no ladder reports found")
        return 1

    rows = []
    for alpha in sorted(levels, reverse=True):
        pm = levels[alpha]
        rows.append({
            "alpha": alpha,
            "meshes": sorted(pm),
            "n_meshes": len(pm),
            "contact": stats(pm, "contact"),
            "worst_mesh_contact": min(
                (v["contact"] for v in pm.values() if v.get("contact") is not None),
                default=None),
            "band_mean_contact": stats(pm, "band_mean_contact"),
            "worst_contact": stats(pm, "worst_contact"),
            "lateral": stats(pm, "lateral"),
            "safety": stats(pm, "safety"),
            "per_mesh_contact": {k: v["contact"] for k, v in sorted(pm.items())},
        })

    nominal = next((r for r in rows if abs(r["alpha"] - 1.0) < 1e-9), None)
    if nominal is None:
        print("no alpha=1.00 level; onset is defined relative to nominal")
        return 1

    nom_c = nominal["contact"]["mean"]
    nom_l = nominal["lateral"]["mean"]
    noise_c = nominal["contact"]["spread"]
    onset_c = nom_c - ONSET_CONTACT_DROP_PP / 100.0

    print("ROM ACTUATOR-AUTHORITY ROBUSTNESS CURVE")
    print("=" * 78)
    print(f"nominal (alpha=1.00, {nominal['n_meshes']} meshes): "
          f"contact={nom_c:.4f}  lateral={nom_l:.5f} m")
    print(f"across-mesh spread at nominal: {noise_c * 100:.2f} pp")
    if noise_c * 100 >= ONSET_CONTACT_DROP_PP:
        print(f"  WARNING: nominal spread ({noise_c * 100:.2f} pp) is at or above the "
              f"{ONSET_CONTACT_DROP_PP} pp onset criterion.")
        print(f"  Only crossings substantially larger than the spread are defensible.")
    print()
    print(f"PRIMARY metric: ground contact")
    print(f"  onset      contact <= {onset_c:.4f}   ({ONSET_CONTACT_DROP_PP} pp below nominal)")
    print(f"  breakdown  contact <  {BREAKDOWN_CONTACT}  or safety termination > "
          f"{BREAKDOWN_SAFETY_TERMINATION}")
    print(f"lateral-error onset criterion: INAPPLICABLE on this axis "
          f"(a starved arm is a still arm; see module docstring)")
    print()

    hdr = (f"{'alpha':>6s} {'msh':>4s} {'wf-mean':>9s} {'+-pp':>6s} "
           f"{'WORST-MESH':>9s} {'band-mean':>9s} {'safety':>8s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        c, w, l, s = r["contact"], r["worst_contact"], r["lateral"], r["safety"]
        marks = []
        usable = r["n_meshes"] >= a.min_meshes
        drop_pp = (nom_c - c["mean"]) * 100
        # An effect must clear BOTH the predeclared threshold and the measured
        # scatter before it is called resolved.
        resolved = drop_pp > max(ONSET_CONTACT_DROP_PP, noise_c * 100, c["spread"] * 100)
        if c["mean"] <= onset_c:
            marks.append("ONSET" if resolved else "onset?")
        # The gate is worst_mesh_worst_family_*: a MIN across meshes, so it is
        # scored on the worst mesh, never the mean of them.
        if r["worst_mesh_contact"] is not None and r["worst_mesh_contact"] < BREAKDOWN_CONTACT:
            marks.append("BREAKDOWN-contact")
        if s.get("mean", 0) > BREAKDOWN_SAFETY_TERMINATION:
            marks.append("BREAKDOWN-safety")
        verdict = ",".join(marks) if marks else "ok"
        if not usable:
            verdict += f" (only {r['n_meshes']} mesh)"
        r["verdict"] = verdict
        r["contact_drop_pp"] = drop_pp
        r["resolved"] = bool(resolved)
        print(f"{r['alpha']:6.3f} {r['n_meshes']:4d} {c['mean']:9.4f} "
              f"{c['spread'] * 100:6.2f} {r['worst_mesh_contact']:9.4f} "
              f"{r['band_mean_contact'].get('mean', float('nan')):9.4f} "
              f"{s.get('mean', float('nan')):8.4f}  {verdict}")

    ok = [r for r in rows if r["n_meshes"] >= a.min_meshes]
    onset_alpha = max((r["alpha"] for r in ok
                       if "ONSET" in r["verdict"] and r["resolved"]), default=None)
    brk_alpha = max((r["alpha"] for r in ok if "BREAKDOWN" in r["verdict"]), default=None)
    plateau = [r["alpha"] for r in ok if r["contact_drop_pp"] <= max(ONSET_CONTACT_DROP_PP,
                                                                    noise_c * 100)]

    print()
    print(f"robustness plateau : alpha >= {min(plateau):.3f}" if plateau else "no plateau")
    print(f"degradation onset  : alpha = {onset_alpha}")
    print(f"functional breakdown: alpha = {brk_alpha}")

    summary = {
        "primary_metric": "ground_contact_fraction",
        "lateral_onset_criterion": "INAPPLICABLE - starved arm is stiller, deviation metrics improve",
        "nominal": {"contact": nom_c, "lateral": nom_l,
                    "across_mesh_spread_pp": noise_c * 100},
        "thresholds": {
            "onset_contact_drop_pp": ONSET_CONTACT_DROP_PP,
            "breakdown_contact": BREAKDOWN_CONTACT,
            "breakdown_safety_termination": BREAKDOWN_SAFETY_TERMINATION,
            "onset_lateral_multiple": ONSET_LATERAL_MULTIPLE,
        },
        "plateau_min_alpha": min(plateau) if plateau else None,
        "degradation_onset_alpha": onset_alpha,
        "functional_breakdown_alpha": brk_alpha,
        "levels": rows,
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
