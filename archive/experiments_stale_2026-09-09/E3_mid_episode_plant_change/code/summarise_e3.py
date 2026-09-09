#!/usr/bin/env python3
"""E3: does the adaptive stack recover from a mid-episode plant change?

The plant's valve effectiveness is scaled at a known time inside the scored
window. Arms differ only in what occupies the latent slot, so the comparison
that carries the claim is `estimated` against `frozen`: both receive the same
plant change with identical input dimensionality, and only one keeps updating.

Controls, and what each rules out:

  shuffled   right latent distribution, wrong plant -- rules out "extra input
             dimensions help"
  mean       in-distribution, carries no plant information. `zero` is NOT used:
             measured on a real trace both channels are <= 0 with means
             -1.19/-2.34, so zero sits 2-3 s.d. outside the distribution and is
             an out-of-distribution arm rather than an ablation
  sham       identical to `estimated` but the plant does NOT change. Without it
             a difference across the step is not evidence of tracking, because
             the latent drifts as the trajectory evolves regardless
  idfd_ref   the non-adaptive stack under the same schedule, so each arm can be
             read against what the stack degrades TO

Unit of analysis is the (mesh, family) cell, as in E1: the evaluation is
deterministic and the seed is not a unit of replication.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = [("aam_estimated", "online estimate"),
        ("aam_frozen", "frozen latent (pre-change)"),
        ("aam_shuffled", "shuffled latent"),
        ("aam_mean", "mean latent"),
        ("aam_sham", "online, plant unchanged"),
        ("idfd_ref", "PPO-ID-FD reference")]
METRICS = {"mean_abs_lateral_error_m": ("lateral error (m)", True),
           "mean_abs_cut_height_error_m": ("cut-height error (m)", True),
           "ground_contact_fraction": ("ground contact", False)}


def cells(root: Path, arm: str, meshes: list[str]):
    out = {}
    for m in meshes:
        p = root / f"{arm}_m{m}.json"
        if not p.is_file():
            return None
        blk = json.loads(p.read_text())["per_mesh"].get(m)
        if not blk:
            return None
        for fam in blk.get("path_family_results", []):
            out[(m, fam["family"])] = fam
    return out


def boot(d, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--meshes", default="1337 17 909")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    meshes = a.meshes.split()

    data = {k: cells(a.root, k, meshes) for k, _ in ARMS}
    missing = [k for k, v in data.items() if v is None]
    if missing:
        raise SystemExit(f"missing arms: {missing}")
    keys = sorted(set.intersection(*(set(v) for v in data.values())))
    print(f"\n  {len(keys)} paired (mesh, family) cells per arm\n")

    report = {"n_cells": len(keys), "metrics": {}}
    for key, (name, lower_better) in METRICS.items():
        vals = {k: np.array([data[k][c][key] for c in keys]) for k, _ in ARMS}
        print(f"  {name}")
        for k, label in ARMS:
            print(f"    {label:<28}{vals[k].mean():10.5f}")
        rec = {"per_arm_mean": {k: float(vals[k].mean()) for k, _ in ARMS}}

        # the contrast that carries the claim
        for other, tag in (("aam_frozen", "vs frozen"),
                           ("aam_shuffled", "vs shuffled"),
                           ("aam_mean", "vs mean")):
            d = vals["aam_estimated"] - vals[other]
            better = int((d < 0).sum()) if lower_better else int((d > 0).sum())
            lo, hi = boot(d)
            excl = (hi < 0) if lower_better else (lo > 0)
            ok = better == len(keys) and excl
            rel = 100 * (vals["aam_estimated"].mean() / vals[other].mean() - 1)
            print(f"      online {tag:<12}{rel:+7.1f}%  better in "
                  f"{better}/{len(keys)}  CI [{lo:+.5f}, {hi:+.5f}]"
                  f"  {'supported' if ok else 'not established'}")
            rec[tag] = {"rel_pct": float(rel), "better": better,
                        "n": len(keys), "ci": [lo, hi], "supported": bool(ok)}
        report["metrics"][key] = rec
        print()

    # sham check: if the perturbed and unperturbed runs are identical, the
    # scheduled plant change did not reach the plant and nothing above is a
    # statement about adaptation.
    lat = "mean_abs_lateral_error_m"
    e = np.array([data["aam_estimated"][c][lat] for c in keys])
    sh = np.array([data["aam_sham"][c][lat] for c in keys])
    delta = float(np.abs(e - sh).mean())
    rel = 100 * (e.mean() / sh.mean() - 1)
    print(f"  SHAM CHECK: perturbed vs unperturbed differ by {delta:.5f} m "
          f"({rel:+.1f}%)")
    if delta < 1e-6:
        print("    IDENTICAL -- the plant change never reached the plant; "
              "no arm above can be read as adaptation")
    report["sham_delta_m"] = delta
    report["sham_rel_pct"] = float(rel)
    report["plant_change_detected"] = bool(delta >= 1e-6)

    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
