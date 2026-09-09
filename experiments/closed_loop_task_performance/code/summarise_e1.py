#!/usr/bin/env python3
"""E1: paired closed-loop tracking comparison across the three control stacks.

UNIT OF ANALYSIS. The evaluation is deterministic: terrain is fixed by the mesh
id, the policy is deterministic at evaluation time and initial conditions do not
vary, so ``--seed`` is recorded in the report and changes nothing. Five seeds
were run and returned bit-identical numbers to nine decimal places. The seed is
therefore NOT a unit of replication here and is ignored; quoting a sign count
over seeds would report one observation five times.

What does vary is the (mesh, path family) cell: 3 held-out terrain meshes by 11
trajectory families gives 33 paired conditions per stack, and the same cell
presents every stack with an identical task. Metrics available only per mesh
(impacts, head alignment) are paired over the 3 meshes and are labelled n=3 so
the weaker evidence is visible rather than pooled away.

Lateral and cut-height error are reported as the mean over families, not the
worst family. The worst-family statistic is a maximum, and on this task it is
dominated by ``double_s_curve`` -- roughly twenty times the lateral error of a
typical family -- so it measures one hard trajectory rather than tracking
quality.

The primary contrast is PPO-(AAM)-ID-FD against PPO-ID-FD: they share the frozen
inverse and forward models and differ only in the adaptive latent, which
isolates adaptation. PPO-ROM is the reference condition, not a competitor -- its
actuator IS the deployment plant.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STACKS = ["ppo_rom", "ppo_id_fd", "ppo_aam_id_fd"]
LABEL = {"ppo_rom": "PPO-ROM", "ppo_id_fd": "PPO-ID-FD",
         "ppo_aam_id_fd": "PPO-(AAM)-ID-FD"}

FAMILY_METRICS = {
    "mean_abs_lateral_error_m": ("lateral error (m)", True),
    "mean_abs_cut_height_error_m": ("cut-height error (m)", True),
    "ground_contact_fraction": ("ground contact", False),
}
MESH_METRICS = {
    "hard_impact_step_fraction": ("hard impacts", True),
    "p01_head_alignment": ("head alignment", False),
    "p95_tool_acceleration_mps2": ("tool accel p95 (m/s^2)", True),
}


def load(root: Path, stack: str, seed: int, meshes: list[str]):
    """(cells, per_mesh) keyed so every stack lines up on identical conditions."""
    cells, per_mesh = {}, {}
    for m in meshes:
        p = root / f"{stack}_s{seed}_m{m}.json"
        if not p.is_file():
            return None, None
        block = json.loads(p.read_text()).get("per_mesh", {}).get(m)
        if not block:
            return None, None
        for fam in block.get("path_family_results", []):
            cells[(m, fam["family"])] = fam
        per_mesh[m] = block
    return cells, per_mesh


def contrast(name, lower_better, a_vals, b_vals, n_label):
    """b (adaptive) against a (frozen baseline), paired elementwise."""
    d = b_vals - a_vals
    better = int((d < 0).sum()) if lower_better else int((d > 0).sum())
    n = len(d)
    rel = 100.0 * (b_vals.mean() / a_vals.mean() - 1.0) if a_vals.mean() else float("nan")
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(20000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    excludes = (hi < 0) if lower_better else (lo > 0)
    ok = better == n and excludes
    print(f"    AAM vs ID-FD: {rel:+.1f}%, better in {better}/{n} {n_label}, "
          f"paired 95% CI [{lo:+.5f}, {hi:+.5f}]  "
          f"-> {'supported' if ok else 'NOT established'}")
    return {"rel_pct": float(rel), "better": better, "n": n,
            "ci": [float(lo), float(hi)], "supported": bool(ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--meshes", default="1337 17 909")
    ap.add_argument("--seed", type=int, default=17,
                    help="which recorded run to read; the others are identical")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    meshes = a.meshes.split()

    loaded = {s: load(a.root, s, a.seed, meshes) for s in STACKS}
    missing = [LABEL[s] for s in STACKS if loaded[s][0] is None]
    if missing:
        raise SystemExit(f"missing reports for {', '.join(missing)}")

    keys = sorted(set.intersection(*(set(loaded[s][0]) for s in STACKS)))
    print(f"\n  {len(meshes)} meshes x {len(keys)//len(meshes)} families "
          f"= {len(keys)} paired conditions per stack")
    print("  (evaluation is deterministic; the seed dimension is not "
          "replication and is ignored)")
    report = {"n_cells": len(keys), "meshes": meshes, "metrics": {}}

    for key, (name, lower_better) in FAMILY_METRICS.items():
        vals = {s: np.array([loaded[s][0][k][key] for k in keys]) for s in STACKS}
        print(f"\n  {name}   [n={len(keys)} mesh x family cells]")
        for s in STACKS:
            v = vals[s]
            print(f"    {LABEL[s]:<20}mean {v.mean():9.5f}   median {np.median(v):9.5f}"
                  f"   p95 {np.percentile(v, 95):9.5f}")
        report["metrics"][name] = {
            "unit": "mesh x family", "n": len(keys),
            "per_stack_mean": {s: float(vals[s].mean()) for s in STACKS},
            "per_stack_median": {s: float(np.median(vals[s])) for s in STACKS},
            "contrast": contrast(name, lower_better,
                                 vals["ppo_id_fd"], vals["ppo_aam_id_fd"],
                                 "cells"),
        }

    for key, (name, lower_better) in MESH_METRICS.items():
        vals = {s: np.array([loaded[s][1][m][key] for m in meshes])
                for s in STACKS if all(key in loaded[s][1][m] for m in meshes)}
        if len(vals) < len(STACKS):
            continue
        print(f"\n  {name}   [n={len(meshes)} meshes -- weak evidence]")
        for s in STACKS:
            print(f"    {LABEL[s]:<20}mean {vals[s].mean():9.5f}   "
                  f"per mesh {np.round(vals[s], 5).tolist()}")
        report["metrics"][key] = {
            "unit": "mesh", "n": len(meshes),
            "per_stack_mean": {s: float(vals[s].mean()) for s in STACKS},
            "per_mesh": {s: vals[s].tolist() for s in STACKS},
            "contrast": contrast(name, lower_better,
                                 vals["ppo_id_fd"], vals["ppo_aam_id_fd"],
                                 "meshes"),
        }

    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
