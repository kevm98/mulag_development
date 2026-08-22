#!/usr/bin/env python3
"""E5: how each stack degrades as the plant moves away from nominal.

Reports a curve, not a verdict. A gate that goes green says nothing about where
a stack stops working; the breakdown point does.

The sweep varies only the MAGNITUDE of the deployment-factor randomisation.
Each environment keeps the draw it would have had, stretched away from nominal
along the same line, so the curve is not confounded with visiting different
plants. Scale 0 is the nominal plant and 1 reproduces the manifest.

One mesh, so the unit is the trajectory family: 11 cells per point. That
resolves a trend across five points but makes the absolute values
non-comparable with the three-mesh tables elsewhere.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STACKS = [("ppo_rom", "PPO-ROM"), ("ppo_id_fd", "PPO-ID-FD"),
          ("ppo_aam_id_fd", "PPO-(AAM)-ID-FD")]
METRICS = {"mean_abs_lateral_error_m": ("lateral error (m)", True),
           "mean_abs_cut_height_error_m": ("cut-height error (m)", True),
           "ground_contact_fraction": ("ground contact", False)}


def families(root: Path, stack: str, scale: str, mesh: str):
    p = root / f"{stack}_s{scale}_m{mesh}.json"
    if not p.is_file():
        return None
    blk = json.loads(p.read_text())["per_mesh"].get(mesh)
    if not blk:
        return None
    return {f["family"]: f for f in blk.get("path_family_results", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mesh", default="1337")
    ap.add_argument("--scales", default="0.0 0.5 1.0 1.5 2.0")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    scales = a.scales.split()

    data = {(k, sc): families(a.root, k, sc, a.mesh)
            for k, _ in STACKS for sc in scales}
    missing = [f"{k}@{sc}" for (k, sc), v in data.items() if v is None]
    if missing:
        raise SystemExit(f"incomplete sweep: {missing}")
    keys = sorted(set.intersection(*(set(v) for v in data.values())))
    print(f"\n  {len(keys)} families per point, scales {scales}\n")

    report = {"scales": [float(s) for s in scales], "n_families": len(keys),
              "metrics": {}}
    for key, (name, lower_better) in METRICS.items():
        print(f"  {name}")
        print(f"    {'stack':<20}" + "".join(f"{s:>10}" for s in scales)
              + f"{'x0 -> x2':>12}")
        rec = {}
        for k, label in STACKS:
            vals = [float(np.mean([data[(k, sc)][f][key] for f in keys]))
                    for sc in scales]
            ratio = vals[-1] / vals[0] if vals[0] else float("nan")
            print(f"    {label:<20}" + "".join(f"{v:10.4f}" for v in vals)
                  + f"{ratio:11.2f}x")
            rec[label] = {"values": vals, "ratio_x0_to_x2": float(ratio)}
        report["metrics"][key] = rec
        print()

    # breakdown point: first scale at which a stack leaves its own nominal band
    print("  breakdown (first scale exceeding 1.5x the nominal value)")
    lat = "mean_abs_lateral_error_m"
    for k, label in STACKS:
        vals = [float(np.mean([data[(k, sc)][f][lat] for f in keys]))
                for sc in scales]
        hit = next((s for s, v in zip(scales, vals) if v > 1.5 * vals[0]), None)
        print(f"    {label:<20}{hit if hit else 'not reached by 2.0'}")
        report.setdefault("breakdown_scale", {})[label] = hit
    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
