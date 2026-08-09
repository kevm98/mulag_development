#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Run and aggregate the repeated-seed circle switching-oracle screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT / "PPO2_ROM_B_PhysX/scripts/screen_switching_oracle.py"
DEFAULT_BASELINE = (
    ROOT / "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14"
    / "model_53_ppo2_rom_b_final.pt"
)
DEFAULT_ORACLE = (
    ROOT / "logs/rsl_rl/ppo2_rom_b_privileged_oracle/2026-07-25_22-25-49"
    / "model_49_ppo2_rom_b_final.pt"
)


def _numbers(text: str, cast):
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def _tag(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--target_hs", default="0.4,0.7,1.0,1.2")
p.add_argument("--seeds", default="17,29,41")
p.add_argument("--shape", choices=("line", "arc", "sine", "circle", "square"), default="circle")
p.add_argument("--device", default="cuda:0")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--baseline_checkpoint", default=str(DEFAULT_BASELINE))
p.add_argument("--oracle_checkpoint", default=str(DEFAULT_ORACLE))
p.add_argument("--output_dir", default="")
p.add_argument("--dry_run", action="store_true")
a = p.parse_args()

target_hs = _numbers(a.target_hs, float)
seeds = _numbers(a.seeds, int)
if not target_hs or any(not 0.4 <= value <= 1.2 for value in target_hs):
    p.error("--target_hs must contain values in [0.4, 1.2]")
if 1.0 not in target_hs:
    p.error("--target_hs must include 1.0 as the no-change control")
if not seeds:
    p.error("--seeds must not be empty")

output_dir = (
    Path(a.output_dir).expanduser().resolve()
    if a.output_dir
    else ROOT / "PPO2_ROM_B_PhysX/reports" / (
        "switching_oracle_matrix_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
)


def _condition_path(seed: int, target_h: float) -> Path:
    return output_dir / f"seed_{seed}_h_{_tag(target_h)}.json"


def _valid_condition(path: Path, seed: int, target_h: float) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        protocol = payload["protocol"]
        return (
            int(protocol["seed"]) == seed
            and float(protocol["target_h"]) == target_h
            and str(protocol["shape"]) == str(a.shape)
            and set(payload["results"]) == {"romb", "oracle"}
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_conditions() -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [(seed, target_h) for seed in seeds for target_h in target_hs]
    for index, (seed, target_h) in enumerate(conditions, start=1):
        path = _condition_path(seed, target_h)
        if _valid_condition(path, seed, target_h):
            print(f"[MATRIX {index:02d}/{len(conditions):02d}] reuse seed={seed} h={target_h:.1f}", flush=True)
            continue
        command = [
            sys.executable, str(SCREEN),
            "--device", str(a.device),
            "--shape", str(a.shape),
            "--target_h", str(target_h),
            "--seed", str(seed),
            "--path_speed_mps", str(a.path_speed_mps),
            "--baseline_checkpoint", str(a.baseline_checkpoint),
            "--oracle_checkpoint", str(a.oracle_checkpoint),
            "--output", str(path),
        ]
        print(f"[MATRIX {index:02d}/{len(conditions):02d}] run seed={seed} h={target_h:.1f}", flush=True)
        if a.dry_run:
            print("  " + " ".join(command), flush=True)
            continue
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-100:])
            raise RuntimeError(
                f"condition seed={seed} h={target_h:.1f} failed; rerun the same "
                f"matrix command to resume completed conditions:\n{tail}"
            )
        if not _valid_condition(path, seed, target_h):
            raise RuntimeError(f"condition did not write a valid artifact: {path}")
    if a.dry_run:
        return []
    return [json.loads(_condition_path(seed, h).read_text()) for seed, h in conditions]


def _aggregate(payloads: list[dict]) -> None:
    by_condition = {
        (int(row["protocol"]["seed"]), float(row["protocol"]["target_h"])): row
        for row in payloads
    }
    rows = []
    for seed in seeds:
        control = by_condition[(seed, 1.0)]["results"]
        control_gap = (
            float(control["romb"]["post_switch_p95_acceleration_mps2"])
            - float(control["oracle"]["post_switch_p95_acceleration_mps2"])
        )
        for target_h in target_hs:
            result = by_condition[(seed, target_h)]["results"]
            romb = result["romb"]
            oracle = result["oracle"]
            romb_acc = float(romb["post_switch_p95_acceleration_mps2"])
            oracle_acc = float(oracle["post_switch_p95_acceleration_mps2"])
            gap = romb_acc - oracle_acc
            romb_rmse = float(romb["post_switch_rmse_m"])
            oracle_rmse = float(oracle["post_switch_rmse_m"])
            rows.append({
                "seed": seed,
                "target_h": target_h,
                "romb_rmse_mm": 1000.0 * romb_rmse,
                "oracle_rmse_mm": 1000.0 * oracle_rmse,
                "oracle_rmse_improvement_percent": 100.0 * (romb_rmse - oracle_rmse) / max(romb_rmse, 1.0e-9),
                "romb_acc95_mps2": romb_acc,
                "oracle_acc95_mps2": oracle_acc,
                "oracle_acceleration_advantage_mps2": gap,
                "no_change_control_advantage_mps2": control_gap,
                "acceleration_interaction_mps2": gap - control_gap,
            })

    fieldnames = list(rows[0])
    csv_path = output_dir / "matrix_rows.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for target_h in target_hs:
        selected = [row for row in rows if row["target_h"] == target_h]
        interactions = [row["acceleration_interaction_mps2"] for row in selected]
        metric_vectors = {
            (
                row["romb_rmse_mm"], row["oracle_rmse_mm"],
                row["romb_acc95_mps2"], row["oracle_acc95_mps2"],
            )
            for row in selected
        }
        summary_rows.append({
            "target_h": target_h,
            "seed_count": len(selected),
            "mean_romb_rmse_mm": statistics.mean(row["romb_rmse_mm"] for row in selected),
            "mean_oracle_rmse_mm": statistics.mean(row["oracle_rmse_mm"] for row in selected),
            "mean_romb_acc95_mps2": statistics.mean(row["romb_acc95_mps2"] for row in selected),
            "mean_oracle_acc95_mps2": statistics.mean(row["oracle_acc95_mps2"] for row in selected),
            "mean_acceleration_interaction_mps2": statistics.mean(interactions),
            "positive_interaction_seeds": sum(value > 0.0 for value in interactions),
            "unique_metric_outcomes": len(metric_vectors),
            "seed_variation_detected": len(metric_vectors) > 1,
        })

    independent_seed_replication = all(
        row["seed_variation_detected"] for row in summary_rows
    )

    summary = {
        "diagnostic_only": True,
        "claim": "repeated-seed switching-oracle mechanism screen; not certification",
        "protocol": {
            "shape": str(a.shape),
            "initial_h": 1.0,
            "target_hs": list(target_hs),
            "seeds": list(seeds),
            "no_change_control_h": 1.0,
            "process_isolation": "each controller and condition uses a fresh process",
            "interaction_definition": "(ROM-B acc95 - oracle acc95) at target h minus the same gap at h=1.0 for the matched seed",
            "independent_seed_replication": independent_seed_replication,
            "seed_warning": (
                None if independent_seed_replication else
                "At least one h condition produced bit-identical metrics across seeds; "
                "those rows are deterministic repeats, not independent replications."
            ),
        },
        "summary_by_target_h": summary_rows,
    }
    summary_path = output_dir / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    markdown = [
        "# Switching-oracle matrix summary",
        "",
        "Diagnostic only; this is not ROM-B certification evidence.",
        "",
        "| target h | ROM-B RMSE mm | oracle RMSE mm | ROM-B acc95 | oracle acc95 | controlled acceleration interaction | positive rows | unique outcomes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            f"| {row['target_h']:.1f} | {row['mean_romb_rmse_mm']:.2f} | "
            f"{row['mean_oracle_rmse_mm']:.2f} | {row['mean_romb_acc95_mps2']:.2f} | "
            f"{row['mean_oracle_acc95_mps2']:.2f} | "
            f"{row['mean_acceleration_interaction_mps2']:+.2f} m/s² | "
            f"{row['positive_interaction_seeds']}/{row['seed_count']} | "
            f"{row['unique_metric_outcomes']} |"
        )
    if not independent_seed_replication:
        markdown.extend([
            "",
            "**Seed warning:** metrics were bit-identical across the requested seeds. "
            "The rows are deterministic repeats and do not establish repeated-seed evidence.",
        ])
    markdown_path = output_dir / "matrix_summary.md"
    markdown_path.write_text("\n".join(markdown) + "\n")

    print("\n" + "\n".join(markdown[4:]), flush=True)
    print(f"\n[MATRIX] rows: {csv_path}")
    print(f"[MATRIX] summary: {summary_path}")
    print(f"[MATRIX] report: {markdown_path}")


payloads = _run_conditions()
if payloads:
    _aggregate(payloads)
