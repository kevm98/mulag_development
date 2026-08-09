#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Run independent five-factor switches plus one no-change oracle control."""

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

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--seeds", default="17,29,41")
p.add_argument("--shape", choices=("line", "arc", "sine", "circle", "square"), default="circle")
p.add_argument("--device", default="cuda:0")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--output_dir", default="")
p.add_argument("--dry_run", action="store_true")
a = p.parse_args()

seeds = tuple(int(item.strip()) for item in a.seeds.split(",") if item.strip())
if not seeds:
    p.error("--seeds must not be empty")
output_dir = (
    Path(a.output_dir).expanduser().resolve()
    if a.output_dir
    else ROOT / "PPO2_ROM_B_PhysX/reports" / (
        "switching_oracle_independent_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
)


def valid(path: Path, *, mode: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        protocol = payload["protocol"]
        return (
            protocol["target_mode"] == mode
            and int(protocol["seed"]) == seed
            and str(protocol["shape"]) == str(a.shape)
            and set(payload["results"]) == {"romb", "oracle"}
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run(path: Path, *, mode: str, seed: int) -> None:
    if valid(path, mode=mode, seed=seed):
        print(f"[INDEPENDENT] reuse mode={mode} seed={seed}", flush=True)
        return
    command = [
        sys.executable, str(SCREEN),
        "--device", str(a.device),
        "--shape", str(a.shape),
        "--target_mode", mode,
        "--seed", str(seed),
        "--path_speed_mps", str(a.path_speed_mps),
        "--output", str(path),
    ]
    if mode == "health":
        command.extend(("--target_h", "1.0"))
    print(f"[INDEPENDENT] run mode={mode} seed={seed}", flush=True)
    if a.dry_run:
        print("  " + " ".join(command), flush=True)
        return
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-100:])
        raise RuntimeError(
            f"mode={mode} seed={seed} failed; rerun this command to resume:\n{tail}"
        )
    if not valid(path, mode=mode, seed=seed):
        raise RuntimeError(f"invalid condition artifact: {path}")


output_dir.mkdir(parents=True, exist_ok=True)
control_path = output_dir / "control_h_1p0.json"
run(control_path, mode="health", seed=seeds[0])
condition_paths = []
for seed in seeds:
    path = output_dir / f"independent_seed_{seed}.json"
    run(path, mode="independent", seed=seed)
    condition_paths.append(path)

if not a.dry_run:
    control = json.loads(control_path.read_text())["results"]
    control_gap = (
        float(control["romb"]["post_switch_p95_acceleration_mps2"])
        - float(control["oracle"]["post_switch_p95_acceleration_mps2"])
    )
    rows = []
    factor_vectors = set()
    for seed, path in zip(seeds, condition_paths, strict=True):
        payload = json.loads(path.read_text())
        factors = tuple(float(value) for value in payload["protocol"]["target_factors"])
        factor_vectors.add(factors)
        result = payload["results"]
        romb = result["romb"]
        oracle = result["oracle"]
        romb_acc = float(romb["post_switch_p95_acceleration_mps2"])
        oracle_acc = float(oracle["post_switch_p95_acceleration_mps2"])
        romb_rmse = 1000.0 * float(romb["post_switch_rmse_m"])
        oracle_rmse = 1000.0 * float(oracle["post_switch_rmse_m"])
        rows.append({
            "seed": seed,
            "e_comp": factors[0],
            "d_seal_1": factors[1],
            "d_seal_2": factors[2],
            "d_seal_3": factors[3],
            "d_seal_4": factors[4],
            "romb_rmse_mm": romb_rmse,
            "oracle_rmse_mm": oracle_rmse,
            "oracle_rmse_improvement_percent": 100.0 * (romb_rmse - oracle_rmse) / max(romb_rmse, 1.0e-9),
            "romb_acc95_mps2": romb_acc,
            "oracle_acc95_mps2": oracle_acc,
            "raw_oracle_acceleration_advantage_mps2": romb_acc - oracle_acc,
            "no_change_control_advantage_mps2": control_gap,
            "controlled_acceleration_interaction_mps2": romb_acc - oracle_acc - control_gap,
        })
    if len(factor_vectors) != len(seeds):
        raise RuntimeError("independent seeds did not produce distinct factor vectors")

    csv_path = output_dir / "independent_rows.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    interactions = [row["controlled_acceleration_interaction_mps2"] for row in rows]
    summary = {
        "diagnostic_only": True,
        "claim": "independent-factor switching-oracle screen; not certification",
        "protocol": {
            "shape": str(a.shape),
            "initial_factors": [0.0] * 5,
            "target_distribution": {
                "e_comp": "Uniform(-0.125, 0.020)",
                "d_seal_1_to_4": "independent Uniform(0, 1)",
            },
            "seeds": list(seeds),
            "distinct_target_factor_vectors": len(factor_vectors),
            "no_change_control_seed": seeds[0],
            "process_isolation": True,
        },
        "no_change_control_acceleration_advantage_mps2": control_gap,
        "mean_controlled_acceleration_interaction_mps2": statistics.mean(interactions),
        "positive_interaction_count": sum(value > 0.0 for value in interactions),
        "condition_count": len(rows),
        "rows": rows,
    }
    summary_path = output_dir / "independent_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    markdown = [
        "# Independent-factor switching-oracle summary",
        "",
        "Diagnostic only; not certification evidence.",
        "",
        "| seed | target factors [e_comp, seals 1..4] | ROM-B RMSE mm | oracle RMSE mm | ROM-B acc95 | oracle acc95 | controlled interaction |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        factors = [row["e_comp"], row["d_seal_1"], row["d_seal_2"], row["d_seal_3"], row["d_seal_4"]]
        factor_text = "[" + ", ".join(f"{value:.3f}" for value in factors) + "]"
        markdown.append(
            f"| {row['seed']} | {factor_text} | {row['romb_rmse_mm']:.2f} | "
            f"{row['oracle_rmse_mm']:.2f} | {row['romb_acc95_mps2']:.2f} | "
            f"{row['oracle_acc95_mps2']:.2f} | "
            f"{row['controlled_acceleration_interaction_mps2']:+.2f} m/s² |"
        )
    markdown.extend([
        "",
        f"Positive controlled interactions: {summary['positive_interaction_count']}/{summary['condition_count']}.",
    ])
    markdown_path = output_dir / "independent_summary.md"
    markdown_path.write_text("\n".join(markdown) + "\n")
    print("\n" + "\n".join(markdown[4:]), flush=True)
    print(f"\n[INDEPENDENT] rows: {csv_path}")
    print(f"[INDEPENDENT] summary: {summary_path}")
    print(f"[INDEPENDENT] report: {markdown_path}")
