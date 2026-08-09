#!/usr/bin/env python
"""Preflight/gate the Stage-22-wide matched-excitation collection design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data.master_schema import DIM_ORDER, read_run, validate_run  # noqa: E402
from aam_data.timebase import check_rom_grid, rom_tick_indices  # noqa: E402
from tools.domain_sampler import STAGE22_WIDE_RANGES, round_key  # noqa: E402


HYDRAULIC = ("flow", "compliance", "seal", "pressure")
ACTIVE_DIMS = {
    ("hydraulic", "train"): HYDRAULIC,
    ("mechanical", "train"): ("mass_scale", "com_z"),
    ("combined", "train"): DIM_ORDER,
    ("combined", "test"): DIM_ORDER,
}
SAFETY_LIMITS = {
    # Empirical envelopes from all 89 already accepted v2 runs. Command clamps
    # are a protective mechanism and were not zero in v2 (max entry rate
    # 0.107); valve saturation and invalid physics should remain effectively
    # absent. The cylinder-limit ceiling equals the v2 max (0.059) plus margin.
    "invalid": 0.0,
    "cylinder_limit": 0.06,
    "command_saturation": 0.11,
    "valve_saturation": 0.001,
}


def load_manifest(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(newline="")))
    for row in rows:
        row["run_id"] = int(row["run_id"])
        for name in DIM_ORDER:
            row[name] = float(row[name])
    return rows


def validate_design(rows: list[dict], *, require_full: bool) -> tuple[dict, list[str]]:
    issues: list[str] = []
    for row in rows:
        for name, (lo, hi) in STAGE22_WIDE_RANGES.items():
            value = row[name]
            if not lo - 1e-9 <= value <= hi + 1e-9:
                issues.append(f"run {row['run_id']}: {name}={value} outside [{lo}, {hi}]")
    train = {round_key(r) for r in rows if r["split"] == "train"}
    test = {round_key(r) for r in rows if r["split"] == "test"}
    if train & test:
        issues.append(f"train/test factor-key overlap: {len(train & test)}")

    key_counts = Counter(round_key(r) for r in rows)
    unique_fraction = len(key_counts) / max(len(rows), 1)
    # Two nominal replicates plus the zero-seal hydraulic boundary anchor share
    # the nominal factor key intentionally. They measure simulator repeatability;
    # all other configurations remain space-filling and unique.
    nominal_key = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    unexpected_duplicates = {
        str(key): count
        for key, count in key_counts.items()
        if count > 1 and key != nominal_key
    }
    if unique_fraction < 0.95:
        issues.append(
            f"effective unique factor fraction {unique_fraction:.3f} < 0.95"
        )
    if unexpected_duplicates:
        issues.append(f"unexpected duplicate factor configurations: {unexpected_duplicates}")

    space_filling = {}
    for (domain, split), active in ACTIVE_DIMS.items():
        group = [r for r in rows if r["domain"] == domain and r["split"] == split]
        if not group:
            continue
        lo = np.asarray([STAGE22_WIDE_RANGES[name][0] for name in active])
        hi = np.asarray([STAGE22_WIDE_RANGES[name][1] for name in active])
        x = np.asarray([[r[name] for name in active] for r in group], dtype=np.float64)
        x = (x - lo) / (hi - lo)
        rank = int(np.linalg.matrix_rank(x - x.mean(axis=0)))
        distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        min_distance = float(distances.min()) if len(group) > 1 else float("inf")
        median_nearest = (
            float(np.median(distances.min(axis=1))) if len(group) > 1 else float("inf")
        )
        quartiles = [
            np.histogram(x[:, j], bins=[0.0, 0.25, 0.5, 0.75, 1.000001])[0].tolist()
            for j in range(len(active))
        ]
        label = f"{domain}/{split}"
        space_filling[label] = {
            "active_dims": list(active),
            "rank": rank,
            "min_normalized_distance": min_distance,
            "median_nearest_distance": median_nearest,
            "quartile_counts": quartiles,
        }
        if rank != len(active):
            issues.append(f"{label} factor design rank {rank} != {len(active)}")
        if min_distance < 0.05:
            issues.append(f"{label} min normalized factor distance {min_distance:.3f} < 0.05")
        if any(0 in counts for counts in quartiles):
            issues.append(f"{label} leaves an active-factor quartile empty")
    anchors = {}
    hyd_rows = [r for r in rows if r["domain"] == "hydraulic"]
    if hyd_rows or require_full:
        for name in HYDRAULIC:
            lo, hi = STAGE22_WIDE_RANGES[name]
            anchors[name] = {
                "low": any(abs(r[name] - lo) < 1e-9 for r in hyd_rows),
                "high": any(abs(r[name] - hi) < 1e-9 for r in hyd_rows),
            }
            if not all(anchors[name].values()):
                issues.append(f"missing hydraulic boundary anchor(s) for {name}")
    if require_full:
        counts = {
            f"{domain}/{split}": sum(
                r["domain"] == domain and r["split"] == split for r in rows
            )
            for domain, split in (
                ("nominal", "train"), ("hydraulic", "train"),
                ("mechanical", "train"), ("combined", "train"),
                ("combined", "test"),
            )
        }
        expected = {
            "nominal/train": 2, "hydraulic/train": 40,
            "mechanical/train": 12, "combined/train": 48,
            "combined/test": 16,
        }
        if counts != expected:
            issues.append(f"full-manifest allocation {counts} != {expected}")
    else:
        counts = {}
    return {
        "rows": len(rows),
        "counts": counts,
        "anchors": anchors,
        "adequacy": {
            "unique_factor_keys": len(key_counts),
            "unique_fraction": unique_fraction,
            "intentional_nominal_key_repeats": key_counts.get(nominal_key, 0),
            "unexpected_duplicates": unexpected_duplicates,
            "space_filling": space_filling,
        },
    }, issues


def id_segments(metadata: dict) -> list[dict]:
    return [s for s in metadata["segment_ids"] if s["kind"] == "id_excitation"]


def hydraulic_state(arrays: dict) -> np.ndarray:
    return np.concatenate(
        (arrays["cylinder_extension"], arrays["cylinder_velocity"], arrays["dP"]),
        axis=1,
    ).astype(np.float64)


def matched_effect(nominal: dict, candidate: dict) -> dict:
    na, ca = nominal["arrays"], candidate["arrays"]
    nm, cm = nominal["metadata"], candidate["metadata"]
    if na["rom_tick"].shape != ca["rom_tick"].shape:
        return {"error": "trajectory lengths differ"}
    mask = np.zeros(na["rom_tick"].shape[0], dtype=bool)
    for ns, cs in zip(id_segments(nm), id_segments(cm), strict=True):
        if (ns["start_step"], ns["end_step"]) != (cs["start_step"], cs["end_step"]):
            return {"error": "ID segment boundaries differ"}
        mask[ns["start_step"] : ns["end_step"]] = True
    ticks = rom_tick_indices(na)
    ticks = ticks[mask[ticks] & (ticks + 3 < len(na["cylinder_extension"]))]
    sn, sc = hydraulic_state(na), hydraulic_state(ca)
    dn, dc = sn[ticks + 3] - sn[ticks], sc[ticks + 3] - sc[ticks]
    out = {}
    for name, sl in (("extension", slice(0, 4)), ("velocity", slice(4, 8)), ("dP", slice(8, 12))):
        level_scale = np.sqrt(np.mean((sn[ticks, sl] - sn[ticks, sl].mean(0)) ** 2))
        delta_scale = np.sqrt(np.mean(dn[:, sl] ** 2))
        out[name] = {
            "level_effect_ratio": float(
                np.sqrt(np.mean((sc[ticks, sl] - sn[ticks, sl]) ** 2))
                / max(level_scale, 1e-12)
            ),
            "delta_effect_ratio": float(
                np.sqrt(np.mean((dc[:, sl] - dn[:, sl]) ** 2))
                / max(delta_scale, 1e-12)
            ),
        }
    return out


def validate_flights(dataset: Path, rows: list[dict]) -> tuple[dict, list[str]]:
    issues: list[str] = []
    by_id = {r["run_id"]: r for r in rows}
    data = {}
    probe_hashes: dict[str, set[str]] = {}
    safety = {}
    reset_seeds = set()
    for run_id, row in sorted(by_id.items()):
        run_dir = dataset / "runs" / f"{run_id:05d}"
        if not run_dir.is_dir():
            issues.append(f"missing test flight run {run_id:05d}")
            continue
        run_issues = validate_run(run_dir, manifest_row=row)
        issues.extend(f"run {run_id:05d}: {msg}" for msg in run_issues)
        d = read_run(run_dir)
        data[run_id] = d
        m, a = d["metadata"], d["arrays"]
        reset_seeds.add(int(m["random_seed"]))
        if m.get("factor_profile") != "stage22_wide":
            issues.append(f"run {run_id:05d}: wrong factor_profile")
        if not m.get("excitation_config", {}).get("matched_reset_seed"):
            issues.append(f"run {run_id:05d}: matched reset seed not recorded")
        grid = check_rom_grid(a, control_dt=m["control_dt"], rom_dt=m["rom_dt"])
        issues.extend(f"run {run_id:05d}: {msg}" for msg in grid["issues"])
        for seg in id_segments(m):
            block = a["probe_excitation"][seg["start_step"] : seg["end_step"]]
            probe_hashes.setdefault(seg["name"], set()).add(
                hashlib.sha256(block.tobytes()).hexdigest()
            )
        rates = {
            "invalid": float(np.mean(a["invalid_physics_flag"] > 0.5)),
            "cylinder_limit": float(np.mean(a["cylinder_limit_flag"] > 0.5)),
            "command_saturation": float(np.mean(a["command_saturation_flag"] > 0.5)),
            "valve_saturation": float(np.mean(a["valve_saturation_flag"] > 0.5)),
        }
        safety[str(run_id)] = rates
        for key, limit in SAFETY_LIMITS.items():
            if rates[key] > limit:
                issues.append(
                    f"run {run_id:05d}: {key} rate {rates[key]:.3g} > {limit:.3g}"
                )
    if len(reset_seeds) > 1:
        issues.append(f"reset seeds are not matched: {sorted(reset_seeds)}")
    for name, hashes in probe_hashes.items():
        if len(hashes) != 1:
            issues.append(f"{name} probe differs across factor configurations")

    nominal_ids = [r["run_id"] for r in rows if r["domain"] == "nominal" and r["run_id"] in data]
    effects = {}
    if not nominal_ids:
        issues.append("no nominal flight for matched effect test")
    else:
        nominal = data[nominal_ids[0]]
        for run_id, d in data.items():
            if run_id in nominal_ids:
                continue
            effect = matched_effect(nominal, d)
            effects[str(run_id)] = effect
            if "error" in effect:
                issues.append(f"run {run_id:05d}: {effect['error']}")
                continue
            row = by_id[run_id]
            if row["domain"] in ("hydraulic", "combined"):
                useful = max(
                    effect[sig][kind]
                    for sig in ("extension", "velocity")
                    for kind in ("level_effect_ratio", "delta_effect_ratio")
                )
                if useful < 0.05:
                    issues.append(
                        f"run {run_id:05d}: ext/vel matched effect {useful:.3g} < 0.05"
                    )
    return {
        "runs": len(data),
        "matched_reset_seeds": sorted(reset_seeds),
        "probe_hash_counts": {k: len(v) for k, v in probe_hashes.items()},
        "safety_rates": safety,
        "safety_limits": SAFETY_LIMITS,
        "matched_effects": effects,
    }, issues


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--require-full", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    manifest = Path(args.manifest)
    rows = load_manifest(manifest)
    design, issues = validate_design(rows, require_full=args.require_full)
    flights = None
    if args.dataset:
        flights, flight_issues = validate_flights(Path(args.dataset), rows)
        issues.extend(flight_issues)
    report = {
        "pass": not issues,
        "manifest": str(manifest),
        "dataset": args.dataset,
        "design": design,
        "flights": flights,
        "issues": issues,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"pass={report['pass']} issues={len(issues)} -> {out}")
    for issue in issues:
        print(f"  {issue}")
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
