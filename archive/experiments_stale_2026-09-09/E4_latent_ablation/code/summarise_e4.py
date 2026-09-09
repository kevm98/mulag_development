#!/usr/bin/env python3
"""E4: does knowing which plant you are on help in closed loop, and how much could it?

The plant is drawn per episode from the held-out factor set and is NOT perturbed
mid-episode, so a latent has genuine identity to convey and the arms differ only
in the quality of that identity.

The `true` arm is the point of the experiment. It reads the true factors through
the privileged encoder and bounds what any estimator could achieve. If it does
not beat `mean`, then no latent of any quality helps here and the estimator's
accuracy is not the limiting factor.

CAVEAT on that arm: an adapter must be trained for the latent it consumes, so
the oracle necessarily uses a different adapter checkpoint as well as a
different latent source. It is not a pure single-variable swap.

Unit is the (mesh, family) cell; the evaluation is deterministic and the seed is
not replication.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = [("aam_true", "privileged latent (oracle)"),
        ("aam_estimated", "estimated latent (deployed)"),
        ("aam_mean", "mean latent (no plant info)"),
        ("aam_shuffled", "shuffled latent (wrong plant)"),
        ("rom_ref", "PPO-ROM reference"),
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


def contrast(name, a, b, lower_better, keys):
    d = a - b
    better = int((d < 0).sum()) if lower_better else int((d > 0).sum())
    lo, hi = boot(d)
    excl = (hi < 0) if lower_better else (lo > 0)
    ok = better == len(keys) and excl
    rel = 100 * (a.mean() / b.mean() - 1) if b.mean() else float("nan")
    print(f"      {name:<34}{rel:+7.1f}%  {better}/{len(keys)}  "
          f"CI [{lo:+.5f}, {hi:+.5f}]  {'supported' if ok else 'not established'}")
    return {"rel_pct": float(rel), "better": better, "n": len(keys),
            "ci": [lo, hi], "supported": bool(ok)}


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
        raise SystemExit(f"incomplete grid, refusing to summarise: {missing}")
    keys = sorted(set.intersection(*(set(v) for v in data.values())))
    print(f"\n  {len(keys)} paired (mesh, family) cells per arm\n")

    report = {"n_cells": len(keys), "metrics": {}}
    for key, (name, lower_better) in METRICS.items():
        vals = {k: np.array([data[k][c][key] for c in keys]) for k, _ in ARMS}
        print(f"  {name}")
        for k, label in ARMS:
            print(f"    {label:<32}{vals[k].mean():10.5f}")
        rec = {"per_arm_mean": {k: float(vals[k].mean()) for k, _ in ARMS}}
        rec["oracle_vs_mean"] = contrast(
            "oracle vs mean  (is there ANY gain?)",
            vals["aam_true"], vals["aam_mean"], lower_better, keys)
        rec["estimated_vs_mean"] = contrast(
            "estimated vs mean (deployed gain)",
            vals["aam_estimated"], vals["aam_mean"], lower_better, keys)
        rec["estimated_vs_oracle"] = contrast(
            "estimated vs oracle (cost of error)",
            vals["aam_estimated"], vals["aam_true"], lower_better, keys)
        rec["estimated_vs_shuffled"] = contrast(
            "estimated vs shuffled (content)",
            vals["aam_estimated"], vals["aam_shuffled"], lower_better, keys)
        report["metrics"][key] = rec
        print()

    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
