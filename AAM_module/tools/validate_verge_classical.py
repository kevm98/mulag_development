#!/usr/bin/env python3
"""Gate fresh final-task classical-controller identification data.

This is deliberately separate from the historical Stage-22 validator: the
new collector uses the final ``RRLAB-Mulag-Verge-*`` tasks and stores one
privileged sidecar per run. A dataset is usable for SD/AAM training only when
the raw schema, exact 50 ms grid, safety envelope, controller provenance and
excitation rank all pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AAM_ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, AAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aam_data import read_run, validate_run  # noqa: E402
from aam_data.timebase import check_rom_grid  # noqa: E402


DIM_ORDER = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
SAFETY_LIMITS = {
    "invalid": 0.0,
    "cylinder_limit": 0.06,
    "command_saturation": 0.11,
    # The command-ROM's internal valve can touch a bound for a few samples
    # during a DLS posture transition.  A 0.5% cap still rejects sustained
    # saturation while not rejecting a valid informative transition window.
    "valve_saturation": 0.005,
}


def load_manifest(path: Path) -> dict[int, dict]:
    with path.open(newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            row["run_id"] = int(row["run_id"])
            for name in DIM_ORDER:
                row[name] = float(row[name])
            rows[row["run_id"]] = row
    return rows


def block_hankel_rank(x: np.ndarray, block_rows: int = 10) -> dict:
    """Return numerical rank/conditioning of a stacked input Hankel matrix."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 * block_rows:
        return {"rows": 0, "rank": 0, "condition": None}
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1.0e-12)
    H = np.concatenate(
        [x[i : x.shape[0] - block_rows + i] for i in range(block_rows)], axis=1
    )
    singular = np.linalg.svd(H, compute_uv=False)
    threshold = max(float(singular[0]) * 1.0e-8, 1.0e-10)
    rank = int(np.sum(singular > threshold))
    positive = singular[singular > threshold]
    return {
        "rows": int(H.shape[0]),
        "columns": int(H.shape[1]),
        "rank": rank,
        "required_rank": int(min(H.shape[1], x.shape[1] * block_rows)),
        "condition": float(positive[0] / positive[-1]) if positive.size else None,
        "singular_values": [float(v) for v in singular[: min(16, singular.size)]],
    }


def probe_segments(metadata: dict) -> list[dict]:
    return [s for s in metadata.get("segment_ids", []) if s.get("kind") == "probe"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    manifest = Path(args.manifest or dataset / "manifest.csv").expanduser().resolve()
    rows = load_manifest(manifest)
    run_dirs = sorted(
        (p for p in (dataset / "runs").glob("*") if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    issues: list[str] = []
    reports: list[dict] = []
    probe_hashes: dict[str, set[str]] = {}

    for run_dir in run_dirs:
        run_id = int(run_dir.name)
        row = rows.get(run_id)
        if row is None:
            issues.append(f"run {run_id:05d}: absent from manifest")
            continue
        run_issues = validate_run(run_dir, manifest_row=row)
        issues.extend(f"run {run_id:05d}: {msg}" for msg in run_issues)
        try:
            loaded = read_run(run_dir)
            metadata, arrays = loaded["metadata"], loaded["arrays"]
        except Exception as exc:
            issues.append(f"run {run_id:05d}: cannot read trajectory: {exc}")
            continue

        expected_task = "RRLAB-Mulag-Verge-ROM-v0"
        if metadata.get("task_name") != expected_task:
            issues.append(
                f"run {run_id:05d}: task {metadata.get('task_name')!r} != {expected_task!r}"
            )
        collection = metadata.get("collection", {})
        if collection.get("controller") != "DLS":
            issues.append(f"run {run_id:05d}: collector is not DLS")
        if collection.get("ppo_used") is not False:
            issues.append(f"run {run_id:05d}: PPO provenance is not explicitly false")
        if collection.get("data_source") != "rom_deployment_plant":
            issues.append(f"run {run_id:05d}: data source is not the ROM deployment plant")
        if collection.get("id_fd_dataset_collected") is not False:
            issues.append(f"run {run_id:05d}: ID-FD dataset provenance is not explicitly false")
        if collection.get("sd_action_contract") != "previous_valve_4 + current_valve_4":
            issues.append(f"run {run_id:05d}: SD action contract mismatch")
        if collection.get("aam_feature_contract") != "joint_q_5 + joint_qdot_5 + dP_4 + previous_valve_4":
            issues.append(f"run {run_id:05d}: AAM feature contract mismatch")
        if metadata.get("actuator") != "rom":
            issues.append(f"run {run_id:05d}: actuator metadata mismatch")
        contract = collection.get("plant_contract", {})
        if contract.get("shared_signature", {}).get("mow_strip_length_m") != 26.0:
            issues.append(f"run {run_id:05d}: missing final 26 m plant contract")
        grid = check_rom_grid(
            arrays,
            control_dt=float(metadata.get("control_dt", 0.0)),
            rom_dt=float(metadata.get("rom_dt", 0.0)),
        )
        issues.extend(f"run {run_id:05d}: {msg}" for msg in grid["issues"])

        with np.load(run_dir / "privileged.npz", allow_pickle=False) as sidecar:
            side = {key: sidecar[key] for key in sidecar.files}
        n = int(metadata["num_control_steps"])
        for key, shape in {
            "factor_e": (6,),
            "classical_controller_action": (n, 4),
            "probe_excitation": (n, 4),
            "rom_tick_control_step": (grid["n_ticks"],),
        }.items():
            if key not in side or side[key].shape != shape:
                issues.append(
                    f"run {run_id:05d}: sidecar {key} shape "
                    f"{None if key not in side else side[key].shape} != {shape}"
                )
        if "rom_gru_hidden" not in side:
            issues.append(f"run {run_id:05d}: ROM hidden-state sidecar missing")
        if "dls_singular_values" in side:
            singular = side["dls_singular_values"]
            if not np.isfinite(singular).all() or (singular <= 0.0).any():
                issues.append(f"run {run_id:05d}: invalid DLS singular-value diagnostic")

        rates = {
            "invalid": float(np.mean(arrays["invalid_physics_flag"] > 0.5)),
            "cylinder_limit": float(np.mean(arrays["cylinder_limit_flag"] > 0.5)),
            "command_saturation": float(np.mean(arrays["command_saturation_flag"] > 0.5)),
            "valve_saturation": float(np.mean(arrays["valve_saturation_flag"] > 0.5)),
        }
        for name, limit in SAFETY_LIMITS.items():
            if rates[name] > limit:
                issues.append(
                    f"run {run_id:05d}: {name} rate {rates[name]:.4g} > {limit:.4g}"
                )

        probe = arrays["probe_excitation"]
        for segment in probe_segments(metadata):
            start, end = int(segment["start"]), int(segment["end"])
            block = probe[start:end]
            digest = hashlib.sha256(block.tobytes()).hexdigest()
            probe_hashes.setdefault(str(segment["name"]), set()).add(digest)
        probe_rank = block_hankel_rank(probe)
        if probe_rank["rank"] < 4:
            issues.append(f"run {run_id:05d}: probe input rank < 4")
        reports.append({
            "run_id": run_id,
            "domain": metadata.get("domain"),
            "split": metadata.get("split"),
            "num_control_steps": n,
            "rom_ticks": grid["n_ticks"],
            "safety_rates": rates,
            "probe_rank": probe_rank,
            "probe_std": np.std(probe, axis=0).tolist(),
            "state_std": np.std(arrays["cylinder_extension"], axis=0).tolist(),
            "hidden_shape": list(side["rom_gru_hidden"].shape)
            if "rom_gru_hidden" in side else None,
        })

    # Probe schedules must be matched across factor runs for paired dynamics
    # comparisons; the DLS response is allowed to differ.
    for name, hashes in probe_hashes.items():
        if len(hashes) != 1:
            issues.append(f"probe {name} differs across factor configurations")

    report = {
        "format": "verge_classical_collection_gate_v1",
        "passed": not issues,
        "dataset": str(dataset),
        "manifest": str(manifest),
        "actuator": "rom",
        "task": "RRLAB-Mulag-Verge-ROM-v0",
        "safety_limits": SAFETY_LIMITS,
        "probe_hash_counts": {key: len(value) for key, value in probe_hashes.items()},
        "runs": reports,
        "issues": issues,
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "runs": len(reports), "issues": issues[:12]}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
