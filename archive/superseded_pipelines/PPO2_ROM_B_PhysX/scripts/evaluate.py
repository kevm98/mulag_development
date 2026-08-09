#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Run held-out Monte Carlo and correlated h-sweep audits for ROM-B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


ORIGINAL_CLI_ARGS = sys.argv[1:]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=17)
p.add_argument(
    "--monte_carlo_rounds",
    type=int,
    default=5,
    help="Independent held-out seed groups evaluated over every shape.",
)
p.add_argument(
    "--h_sweep_rounds",
    type=int,
    default=3,
    help="Matched held-out seed groups evaluated at every h value.",
)
p.add_argument("--eval_steps", type=int, default=120)
p.add_argument(
    "--force_zero_policy",
    action="store_true",
    help=(
        "Diagnostic baseline: force PPO residual action to zero. Reports from "
        "this ablation are never ROM-B certification results."
    ),
)
p.add_argument("--trajectory_shapes", default="line,arc,sine,circle,square")
p.add_argument("--h_values", default="0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument(
    "--dls_velocity_filter_beta",
    type=float,
    default=0.0,
    help=(
        "EMA applied to inverse-cylinder reference velocity. Use only with a "
        "predeclared candidate; 0 preserves the accepted production contract."
    ),
)
p.add_argument("--residual_scale_m", type=float, default=0.002)
p.add_argument("--cylinder4_residual_scale_m", type=float, default=0.00025)
p.add_argument(
    "--checkpoint",
    default=str(
        ROOT / "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14"
        / "model_53_ppo2_rom_b_final.pt"
    ),
)
p.add_argument(
    "--command_rom_checkpoint",
    default=str(
        ROOT / "ROM-cylinders_pipeline/checkpoints"
        / "plant_reachable_v1_finetune_blend25/best_rom.pt"
    ),
)
p.add_argument(
    "--trajectory_library_path",
    default=str(
        ROOT / "PPO2_ROM_PhysX/models"
        / "plant_reachable_trajectory_library_v3_circle_c7_retime.json"
    ),
)
p.add_argument(
    "--clearance_frame_consistent",
    action="store_true",
    help=(
        "Score against the frame-consistent clearance signal. Required to read "
        "acceptance on the corrected measurement; not comparable with earlier runs."
    ),
)
p.add_argument(
    "--output_dir",
    default="",
    help="Default: PPO2_ROM_B_PhysX/reports/evaluation_<timestamp>.",
)
p.add_argument(
    "--privileged_oracle",
    action="store_true",
    help=(
        "Score a privileged-oracle checkpoint (observation 42 -> 47, true factors "
        "appended). DIAGNOSTIC ONLY: the resulting report is an upper bound on "
        "adaptation, never a ROM-B robustness result. Use the identical seeds, "
        "suites and isolation as the factor-blind audit so the two are comparable."
    ),
)
p.add_argument(
    "--isolate_conditions",
    action="store_true",
    help=(
        "Run each rollout seed in its own subprocess, then aggregate. REQUIRED "
        "FOR CERTIFICATION. Measured 2026-07-25: running multiple rollouts in "
        "250 rollouts in one process leaks simulator state between them and "
        "fabricates truck_raw_reaction terminations. Across the six failing h "
        "conditions, isolation cut truck failures 100 -> 14 while leaving head "
        "alignment at 50 -> 47, and all six then passed (worst bound 0.0352 vs "
        "the 0.05 gate). A later matched check on 2026-07-26 proved that keeping "
        "seed groups in one condition process still leaks state: h=1.0 circle "
        "round 3 was 44.08 m/s^2 after rounds 1-2 but 39.99 in a fresh process."
    ),
)
p.add_argument(
    "--shard",
    default="",
    help=(
        "Internal. Run a single condition and dump its rows as JSON. Format "
        "'mc:<shape_index>:<round_index>' or "
        "'h:<h_index>:<shape_index>:<round_index>'. Set by "
        "--isolate_conditions; not intended for direct use."
    ),
)
p.add_argument(
    "--shard_output",
    default="",
    help="Internal. Destination JSON for --shard rows.",
)
p.add_argument("--assemble_rows", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

shapes = tuple(value.strip() for value in a.trajectory_shapes.split(",") if value.strip())
CANONICAL_SHAPES = ("line", "arc", "sine", "circle", "square")
try:
    h_values = tuple(float(value) for value in a.h_values.split(",") if value.strip())
except ValueError as exc:
    p.error(f"invalid --h_values: {exc}")
if not shapes:
    p.error("at least one trajectory shape is required")
if any(shape not in CANONICAL_SHAPES for shape in shapes):
    p.error(f"trajectory shapes must be drawn from {CANONICAL_SHAPES}")
if not 0.0 <= a.dls_velocity_filter_beta < 1.0:
    p.error("--dls_velocity_filter_beta must be in [0, 1)")
if not h_values or any(value < 0.4 or value > 1.2 for value in h_values):
    p.error("h values must be in [0.4, 1.2]")
if (
    a.num_envs <= 0
    or a.monte_carlo_rounds <= 0
    or a.h_sweep_rounds <= 0
    or a.eval_steps < 60
):
    p.error("num_envs and rounds must be positive; eval_steps must be at least 60")
if a.shard and a.isolate_conditions:
    p.error("--shard is set by --isolate_conditions; do not pass both")
if a.shard and not a.shard_output:
    p.error("--shard requires --shard_output")

# The isolation orchestrator must run before AppLauncher. Otherwise this parent
# owns a Vulkan/Kit context while its child shards start theirs, which caused a
# measured GPU crash on 2026-07-26.
if a.isolate_conditions and not a.shard:
    import subprocess

    raise SystemExit(subprocess.run(
        [
            sys.executable,
            str(ROOT / "PPO2_ROM_B_PhysX/scripts/evaluate_isolated.py"),
            *ORIGINAL_CLI_ARGS,
        ],
        check=False,
    ).returncode)


def _parse_shard(spec: str):
    """Parse one fresh-process rollout shard (round index is zero-based)."""

    parts = spec.split(":")
    if parts[0] == "mc" and len(parts) == 3:
        return ("monte_carlo", None, int(parts[1]), int(parts[2]))
    if parts[0] == "h" and len(parts) == 4:
        return ("h_sweep", int(parts[1]), int(parts[2]), int(parts[3]))
    raise SystemExit(f"malformed --shard {spec!r}")


def _shard_specs() -> list[str]:
    """Every rollout in a fresh process, in canonical audit order."""

    specs = [
        f"mc:{shape_index}:{round_index}"
        for shape_index in range(len(shapes))
        for round_index in range(int(a.monte_carlo_rounds))
    ]
    specs += [
        f"h:{h_index}:{shape_index}:{round_index}"
        for h_index in range(len(h_values))
        for shape_index in range(len(shapes))
        for round_index in range(int(a.h_sweep_rounds))
    ]
    return specs


def _child_passthrough_args(checkpoint: Path) -> list[str]:
    """Child command line: everything except the isolation flags themselves."""

    passthrough: list[str] = []
    for flag, value in (
        ("--num_envs", a.num_envs), ("--seed", a.seed),
        ("--monte_carlo_rounds", a.monte_carlo_rounds),
        ("--h_sweep_rounds", a.h_sweep_rounds),
        ("--eval_steps", a.eval_steps),
        ("--trajectory_shapes", a.trajectory_shapes),
        ("--h_values", a.h_values),
        ("--path_speed_mps", a.path_speed_mps),
        ("--dls_velocity_filter_beta", a.dls_velocity_filter_beta),
        ("--residual_scale_m", a.residual_scale_m),
        ("--cylinder4_residual_scale_m", a.cylinder4_residual_scale_m),
        ("--checkpoint", str(checkpoint)),
        ("--command_rom_checkpoint", a.command_rom_checkpoint),
        ("--trajectory_library_path", a.trajectory_library_path),
        ("--device", a.device),
    ):
        passthrough += [flag, str(value)]
    if a.clearance_frame_consistent:
        passthrough.append("--clearance_frame_consistent")
    if a.privileged_oracle:
        passthrough.append("--privileged_oracle")
    if a.force_zero_policy:
        passthrough.append("--force_zero_policy")
    if a.headless:
        passthrough.append("--headless")
    return passthrough


launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import PPO2_ROM_B_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_B_PhysX.env_cfg import PPO2ROMBPPORunnerCfg
from PPO2_ROM_B_PhysX.runtime import (
    FACTOR_NAMES,
    ROMBCurriculumThresholds,
    condition_bound,
    health_scale_to_factors,
    run_deterministic_probe,
    summarize_acceptance,
)
from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage


METRIC_COLUMNS = (
    "mean_error_m",
    "p95_error_m",
    "p95_acceleration_mps2",
    "min_head_alignment",
    "max_clearance_error_m",
    "max_post_constraint_truck_drift_m",
    "max_termination_rate",
    "completion_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "max_shape_mean_error_m": max(float(row["mean_error_m"]) for row in rows),
        "max_shape_p95_error_m": max(float(row["p95_error_m"]) for row in rows),
        "max_shape_p95_acceleration_mps2": max(
            float(row["p95_acceleration_mps2"]) for row in rows
        ),
        "min_head_alignment": min(float(row["min_head_alignment"]) for row in rows),
        "max_clearance_error_m": max(
            float(row["max_clearance_error_m"]) for row in rows
        ),
        "max_post_constraint_truck_drift_m": max(
            float(row["max_post_constraint_truck_drift_m"]) for row in rows
        ),
        "max_termination_rate": max(
            float(row["max_termination_rate"]) for row in rows
        ),
        "min_completion_fraction": min(
            float(row["completion_fraction"]) for row in rows
        ),
    }


def _failures(row: dict[str, object]) -> tuple[int, int]:
    """Exact (failures, replicas) for one seed group."""

    replicas = int(row.get("replicas") or row["evaluated_environments"])
    count = row.get("unsuccessful_environments")
    if count is None:  # tolerate rows produced before the field existed
        count = round(float(row["max_termination_rate"]) * replicas)
    return int(count), replicas


def _acceptance_bounds(
    rows: list[dict[str, object]], *, per_h: bool
) -> list:
    """One bound per shape, or per (h, shape). Never pooled across shapes or h."""

    grouped: dict[tuple[float | None, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (float(row["h"]) if per_h else None, str(row["shape"]))
        grouped.setdefault(key, []).append(row)
    bounds = []
    for (h, shape), shape_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0] or 0.0, item[0][1])
    ):
        counts = [_failures(row) for row in shape_rows]
        replicas = {replica for _count, replica in counts}
        if len(replicas) != 1:
            raise ValueError(
                f"seed groups for {shape} have differing replica counts: {replicas}"
            )
        bounds.append(
            condition_bound(
                shape=shape,
                h=h,
                failures_per_group=[count for count, _replica in counts],
                replicas_per_group=replicas.pop(),
                threshold=ROMBCurriculumThresholds().max_termination_rate,
            )
        )
    return bounds


def _print_acceptance(label: str, bounds: list) -> None:
    for bound in bounds:
        tag = "" if bound.h is None else f"h={bound.h:.1f} "
        group_upper = (
            "n/a" if bound.seed_group_t_upper is None
            else f"{bound.seed_group_t_upper:.5f}"
        )
        phi = "n/a" if bound.dispersion_phi is None else f"{bound.dispersion_phi:.2f}"
        print(
            f"[ROM-B ACCEPT] {label} {tag}{bound.shape:7s} "
            f"{bound.total_failures:>3d}/{bound.total_replicas:<5d} "
            f"rate={bound.rate:.5f} cp={bound.clopper_pearson_upper:.5f} "
            f"group_t={group_upper} upper={bound.upper_bound:.5f} "
            f"({bound.binding_term}) phi={phi} "
            f"{'PASS' if bound.passed else 'FAIL'}",
            flush=True,
        )


def _gate_failures(metrics: dict[str, float], thresholds) -> list[str]:
    checks = {
        "mean_error": metrics["max_shape_mean_error_m"] <= thresholds.max_shape_mean_error_m,
        "p95_error": metrics["max_shape_p95_error_m"] <= thresholds.max_shape_p95_error_m,
        "p95_acceleration": metrics["max_shape_p95_acceleration_mps2"] <= thresholds.max_shape_p95_acceleration_mps2,
        "head_alignment": metrics["min_head_alignment"] >= thresholds.min_head_alignment,
        "clearance": metrics["max_clearance_error_m"] <= thresholds.max_clearance_error_m,
        "truck_drift": metrics["max_post_constraint_truck_drift_m"] <= thresholds.max_post_constraint_truck_drift_m,
        "termination_rate": metrics["max_termination_rate"] <= thresholds.max_termination_rate,
    }
    return [name for name, passed in checks.items() if not passed]


def _print_row(label: str, row: dict[str, object]) -> None:
    print(
        f"[ROM-B AUDIT] {label} shape={row['shape']} "
        f"mean={float(row['mean_error_m']):.5f}m "
        f"p95={float(row['p95_error_m']):.5f}m "
        f"acc95={float(row['p95_acceleration_mps2']):.2f}m/s2 "
        f"head={float(row['min_head_alignment']):.5f} "
        f"clearance={float(row['max_clearance_error_m']):.5f}m "
        f"term={float(row['max_termination_rate']):.5f}",
        flush=True,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "suite", "round", "h", "shape", "seed", *FACTOR_NAMES,
        "stiffness_multiplier", "leakage_multiplier",
        *METRIC_COLUMNS,
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            factors = row.get("fixed_factor_vector") or row.get("factor_mean")
            record = {name: row.get(name, "") for name in fields}
            if factors:
                for index, name in enumerate(FACTOR_NAMES):
                    record[name] = factors[index]
                record["stiffness_multiplier"] = float(torch.exp(torch.tensor(factors[0])))
                record["leakage_multiplier"] = float(
                    (1.0 + 0.25 * torch.tensor(factors[1:5])).pow(3).mean()
                )
            writer.writerow(record)


def _write_plot(
    path: Path,
    h_groups: list[dict[str, object]],
    monte_carlo: dict[str, object],
) -> None:
    h = [float(group["h"]) for group in h_groups]
    figure, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
    series = (
        ("mean_error_m", "max_shape_mean_error_m", "Mean tracking error", "m", 0.040),
        ("p95_error_m", "max_shape_p95_error_m", "P95 tracking error", "m", 0.080),
        ("p95_acceleration_mps2", "max_shape_p95_acceleration_mps2", "P95 tool acceleration", "m/s^2", 40.0),
        ("min_head_alignment", "min_head_alignment", "Minimum head alignment", "alignment", 0.980),
        ("max_clearance_error_m", "max_clearance_error_m", "Maximum clearance error", "m", 0.200),
        ("max_post_constraint_truck_drift_m", "max_post_constraint_truck_drift_m", "Maximum truck drift", "m", 0.001),
        ("max_termination_rate", "max_termination_rate", "Termination rate", "fraction", 0.05),
        ("completion_fraction", "min_completion_fraction", "Completion fraction", "fraction", 0.95),
    )
    shape_colors = {
        shape: plt.get_cmap("tab10")(index)
        for index, shape in enumerate(shapes)
    }
    for axis, (row_key, aggregate_key, title, unit, gate) in zip(
        axes.flat, series
    ):
        for shape in shapes:
            values = []
            for group in h_groups:
                shape_rows = [row for row in group["rows"] if row["shape"] == shape]
                values.append(
                    sum(float(row[row_key]) for row in shape_rows) / len(shape_rows)
                )
            axis.plot(
                h, values, marker=".", linewidth=1, alpha=0.55,
                color=shape_colors[shape], label=shape,
            )
        round_values = [
            [float(row[aggregate_key]) for row in group["round_aggregates"]]
            for group in h_groups
        ]
        centers = [sum(values) / len(values) for values in round_values]
        lows = [min(values) for values in round_values]
        highs = [max(values) for values in round_values]
        axis.fill_between(h, lows, highs, color="#111111", alpha=0.12)
        axis.plot(
            h, centers, marker="o", linewidth=2.2, color="#111111",
            label="worst-shape mean",
        )
        for x, group, center in zip(h, h_groups, centers):
            axis.scatter(
                [x], [center], s=48, zorder=4,
                color="#16803C" if group["passed"] else "#B42318",
            )
        axis.axhline(gate, color="#B42318", linestyle="--", linewidth=1, label="gate")
        axis.axvline(1.0, color="#555555", linestyle=":", linewidth=1, label="nominal")
        axis.set(title=title, xlabel="health index h", ylabel=unit)
        axis.grid(alpha=0.25)
    mc_axis = axes.flat[-1]
    mc_rows = monte_carlo["rows"]
    distributions = [
        [
            float(row["max_termination_rate"])
            for row in mc_rows if row["shape"] == shape
        ]
        for shape in shapes
    ]
    mc_axis.boxplot(distributions, tick_labels=shapes, showmeans=True)
    mc_axis.axhline(0.05, color="#B42318", linestyle="--", linewidth=1)
    mc_axis.set(
        title="Monte Carlo termination by trajectory",
        xlabel="trajectory",
        ylabel="termination fraction",
    )
    mc_axis.tick_params(axis="x", rotation=25)
    mc_axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=4, frameon=False,
    )
    figure.suptitle(
        "PPO2 ROM-B final robustness audit"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_markdown(
    path: Path,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    monte_carlo: dict[str, object],
    h_groups: list[dict[str, object]],
) -> None:
    mc_metrics = monte_carlo["aggregate"]
    lines = [
        "# PPO2 ROM-B held-out robustness audit",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- SHA-256: `{checkpoint_sha256}`",
        f"- Held-out Monte Carlo result: **{'PASS' if monte_carlo['passed'] else 'FAIL'}**",
        f"- Held-out failures: `{monte_carlo['failed_checks'] or []}`",
        f"- Correlated h seed groups per condition: `{len(h_groups[0]['round_aggregates'])}`",
        "",
        "## Held-out Monte Carlo worst cases",
        "",
        "| mean error | p95 error | p95 acceleration | head alignment | clearance | truck drift | termination |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {mc_metrics['max_shape_mean_error_m']:.5f} m | {mc_metrics['max_shape_p95_error_m']:.5f} m | {mc_metrics['max_shape_p95_acceleration_mps2']:.2f} m/s^2 | {mc_metrics['min_head_alignment']:.5f} | {mc_metrics['max_clearance_error_m']:.5f} m | {mc_metrics['max_post_constraint_truck_drift_m']:.6f} m | {mc_metrics['max_termination_rate']:.5f} |",
        "",
        "## Correlated h sweep",
        "",
        "The h sweep is evaluation-only. Its factors are applied inside the hydraulic ROM before pressure integration. Each result is the conservative worst case across all repeated seed groups and trajectory shapes.",
        "",
        "| h | e_comp | seal | stiffness | leakage | mean error | p95 error | acc95 | term | gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for group in h_groups:
        factors = group["factors"]
        physical = group["physical_multipliers"]
        metrics = group["aggregate"]
        lines.append(
            f"| {group['h']:.1f} | {factors[0]:.5f} | {factors[1]:.5f} | "
            f"{physical['stiffness']:.4f} | {physical['leakage']:.4f} | "
            f"{metrics['max_shape_mean_error_m']:.5f} | "
            f"{metrics['max_shape_p95_error_m']:.5f} | "
            f"{metrics['max_shape_p95_acceleration_mps2']:.2f} | "
            f"{metrics['max_termination_rate']:.5f} | "
            f"{'PASS' if group['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `rom_b_held_out_report.json`: complete contract and rollout summaries",
        "- `rom_b_held_out_rows.csv`: one row per rollout condition",
        "- `rom_b_final_robustness.png`: all gates, per-shape curves, repeated-seed envelopes, and Monte Carlo terminations",
        "",
    ])
    path.write_text("\n".join(lines))


def _regroup(all_rows: list[dict[str, object]]) -> tuple[list, list]:
    """Split a flat row list back into the Monte Carlo rows and h groups.

    Sharded runs produce rows condition-by-condition rather than in the nested
    order the single-process loop used, so grouping is reconstructed from the
    rows themselves. Every seed is a pure function of round and canonical shape
    index (`seed = base + offset + round * 5 + canonical_shape_index`) and does
    not depend on h, execution order, or evaluating a shape subset.
    """

    monte_carlo_rows = [r for r in all_rows if r["suite"] == "monte_carlo"]
    thresholds = ROMBCurriculumThresholds()
    h_groups: list[dict[str, object]] = []
    for h in h_values:
        rows = [
            r for r in all_rows
            if r["suite"] == "h_sweep" and float(r["h"]) == float(h)
        ]
        if not rows:
            continue
        factor_values = list(rows[0]["fixed_factor_vector"])
        factors = torch.tensor(factor_values)
        round_aggregates = [
            _aggregate([r for r in rows if r["round"] == index + 1])
            for index in range(int(a.h_sweep_rounds))
        ]
        aggregate = _aggregate(rows)
        failed = _gate_failures(aggregate, thresholds)
        h_groups.append({
            "h": float(h),
            "factors": factor_values,
            "physical_multipliers": {
                "stiffness": float(torch.exp(factors[0])),
                "leakage": float((1.0 + 0.25 * factors[1:5]).pow(3).mean()),
            },
            "aggregate": aggregate,
            "round_aggregates": round_aggregates,
            "failed_checks": failed,
            "passed": not failed,
            "rows": rows,
        })
    return monte_carlo_rows, h_groups


def _run_isolated() -> None:
    """Parent mode: one subprocess per condition, then aggregate normally."""

    import subprocess
    import tempfile

    checkpoint = Path(a.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(a.output_dir).expanduser().resolve()
        if a.output_dir
        else ROOT / "PPO2_ROM_B_PhysX/reports" / f"evaluation_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    specs = _shard_specs()
    passthrough = _child_passthrough_args(checkpoint)
    all_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, spec in enumerate(specs, start=1):
            shard_out = Path(scratch) / f"shard_{index}.json"
            print(f"[ROM-B ISOLATE] {index}/{len(specs)} {spec}", flush=True)
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--shard", spec, "--shard_output", str(shard_out),
                 *passthrough],
                check=False,
            )
            if result.returncode != 0 or not shard_out.is_file():
                # A certification audit must never silently drop a condition:
                # a missing shard would shrink the denominator and flatter the
                # bound. Fail loudly instead.
                raise RuntimeError(
                    f"shard {spec} failed with code {result.returncode}"
                )
            all_rows.extend(json.loads(shard_out.read_text()))
    _assemble_report(all_rows, checkpoint, output_dir, isolated=True)


def main() -> None:
    checkpoint = Path(a.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(a.output_dir).expanduser().resolve()
        if a.output_dir
        else ROOT / "PPO2_ROM_B_PhysX/reports" / f"evaluation_{timestamp}"
    )
    if not a.shard and not hasattr(a, "assemble_rows"):
        output_dir.mkdir(parents=True, exist_ok=False)
    if hasattr(a, "assemble_rows"):
        rows_path = Path(a.assemble_rows).expanduser().resolve()
        if not rows_path.is_file():
            raise FileNotFoundError(f"isolated rows do not exist: {rows_path}")
        _assemble_report(
            json.loads(rows_path.read_text()), checkpoint, output_dir, isolated=True
        )
        return

    task = "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.rom_b_privileged_factors = bool(a.privileged_oracle)
    cfg.observation_space = 47 if a.privileged_oracle else 42
    cfg.command_rom_checkpoint = str(Path(a.command_rom_checkpoint).resolve())
    cfg.trajectory_library_path = str(Path(a.trajectory_library_path).resolve())
    # Construction state must not depend on whether a caller evaluates a shape
    # subset.  The full audit constructs on canonical line before selecting the
    # requested shape; constructing directly on circle changes the rollout.
    cfg.trajectory_shapes = CANONICAL_SHAPES
    cfg.trajectory_shape = CANONICAL_SHAPES[0]
    cfg.randomize_trajectory_shape = False
    cfg.procedural_trajectory_probability = 0.0
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.dls_velocity_filter_beta = float(a.dls_velocity_filter_beta)
    cfg.residual_scale_m = float(a.residual_scale_m)
    cfg.cylinder4_residual_scale_m = float(a.cylinder4_residual_scale_m)
    cfg.command_rom_direction_guard = True
    cfg.defer_safety_terminations = True
    cfg.rom_b_mode = "random"
    cfg.rom_b_randomization_scale = 1.0
    cfg.clearance_frame_consistent = bool(a.clearance_frame_consistent)
    cfg.log_dir = str(output_dir)

    agent_cfg = PPO2ROMBPPORunnerCfg()
    agent_cfg.seed = int(a.seed)
    agent_cfg.device = a.device
    all_rows: list[dict[str, object]] = []

    gym_env = gym.make(task, cfg=cfg)
    try:
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(output_dir), device=agent_cfg.device
        )
        runner.load(
            str(checkpoint),
            load_cfg={
                "actor": True, "critic": True, "optimizer": False,
                "iteration": False, "rnd": False,
            },
        )
        policy = runner.get_inference_policy(device=gym_env.unwrapped.device)

        # A shard runs exactly one condition; the default runs all of them in
        # this single process. Seeds depend only on round and shape index, so
        # both paths execute identical rollouts.
        if a.shard:
            suite, h_index, shape_index, selected_round = _parse_shard(a.shard)
            work = [(suite, h_index, shape_index, selected_round)]
        else:
            work = [("monte_carlo", None, i, None) for i in range(len(shapes))]
            work += [
                ("h_sweep", h_i, s_i, None)
                for h_i in range(len(h_values))
                for s_i in range(len(shapes))
            ]

        for suite, h_index, shape_index, selected_round in work:
            shape = shapes[shape_index]
            if suite == "monte_carlo":
                offset, rounds, factors = 300000, int(a.monte_carlo_rounds), None
            else:
                h = h_values[h_index]
                factors = health_scale_to_factors(
                    h, device=gym_env.unwrapped.device
                )[0]
                offset, rounds = 400000, int(a.h_sweep_rounds)
            round_indices = (
                (selected_round,) if selected_round is not None else range(rounds)
            )
            if selected_round is not None and not 0 <= selected_round < rounds:
                raise ValueError(
                    f"round index {selected_round} is outside [0, {rounds})"
                )
            for round_index in round_indices:
                # A subset audit must reproduce the same seed as the complete
                # five-shape audit; otherwise a circle-only experiment silently
                # changes both the plant draws and randomized start phases.
                canonical_shape_index = CANONICAL_SHAPES.index(shape)
                seed = (
                    int(a.seed) + offset
                    + round_index * len(CANONICAL_SHAPES)
                    + canonical_shape_index
                )
                row = run_deterministic_probe(
                    gym_env.unwrapped, env, policy, shape=shape, seed=seed,
                    scored_steps=int(a.eval_steps), fixed_factors=factors,
                    force_zero_policy=bool(a.force_zero_policy),
                )
                if suite == "monte_carlo":
                    row.update({
                        "suite": "monte_carlo",
                        "round": round_index + 1,
                        "h": None,
                    })
                    _print_row(f"mc={round_index + 1}/{rounds}", row)
                else:
                    row.update({
                        "suite": "h_sweep", "round": round_index + 1,
                        "h": float(h),
                        "fixed_factor_vector": factors.detach().cpu().tolist(),
                    })
                    _print_row(
                        f"h={h:.1f} seed_group={round_index + 1}/{rounds}", row
                    )
                all_rows.append(row)
    finally:
        gym_env.close()

    if a.shard:
        Path(a.shard_output).write_text(json.dumps(all_rows) + "\n")
        print(
            f"[ROM-B SHARD] {a.shard} wrote {len(all_rows)} rows", flush=True
        )
        return

    _assemble_report(all_rows, checkpoint, output_dir, isolated=False)


def _assemble_report(
    all_rows: list[dict[str, object]],
    checkpoint: Path,
    output_dir: Path,
    *,
    isolated: bool,
) -> None:
    thresholds = ROMBCurriculumThresholds()
    monte_carlo_rows, h_groups = _regroup(all_rows)
    mc_aggregate = _aggregate(monte_carlo_rows)
    mc_failed = _gate_failures(mc_aggregate, thresholds)
    mc_bounds = _acceptance_bounds(monte_carlo_rows, per_h=False)
    _print_acceptance("mc", mc_bounds)
    mc_acceptance = summarize_acceptance(mc_bounds)
    monte_carlo = {
        "rounds": int(a.monte_carlo_rounds),
        "aggregate": mc_aggregate,
        "failed_checks": mc_failed,
        "passed": not mc_failed,
        "termination_acceptance": mc_acceptance,
        "rows": monte_carlo_rows,
    }
    h_bounds = _acceptance_bounds(
        [row for group in h_groups for row in group["rows"]], per_h=True
    )
    _print_acceptance("h_sweep", h_bounds)
    h_acceptance = summarize_acceptance(h_bounds)
    for group in h_groups:
        group["termination_acceptance"] = summarize_acceptance(
            [bound for bound in h_bounds if bound.h == float(group["h"])]
        )
    checkpoint_sha256 = _sha256(checkpoint)
    trajectory_library = Path(a.trajectory_library_path).expanduser().resolve()
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "trajectory_library": str(trajectory_library),
        "trajectory_library_sha256": _sha256(trajectory_library),
        "policy": (
            "forced_zero_residual" if a.force_zero_policy
            else "deterministic_actor_mean"
        ),
        "force_zero_policy": bool(a.force_zero_policy),
        "num_envs": int(a.num_envs),
        "eval_steps": int(a.eval_steps),
        "dls_velocity_filter_beta": float(a.dls_velocity_filter_beta),
        "shapes": list(shapes),
        "factor_order": list(FACTOR_NAMES),
        "factor_sampling": "independent_uniform_per_environment_at_reset",
        "clearance_frame_consistent": bool(a.clearance_frame_consistent),
        # An oracle report is an adaptation upper bound, never a robustness result.
        "privileged_oracle": bool(a.privileged_oracle),
        "factor_blind": not bool(a.privileged_oracle),
        "certifiable": (
            bool(isolated)
            and not bool(a.privileged_oracle)
            and not bool(a.force_zero_policy)
        ),
        "monte_carlo_seed_offset": 300000,
        "h_sweep_seed_offset": 400000,
        "h_sweep_rounds": int(a.h_sweep_rounds),
        "h_sweep_uses_matched_shape_seeds": True,
        # Certification requires a fresh process for every rollout.  Grouping
        # all seed rounds for one condition still leaks simulator state.
        "condition_process_isolation": bool(isolated),
        "rollout_process_isolation": bool(isolated),
        "thresholds": asdict(thresholds),
        "termination_acceptance_statistic": {
            "definition": (
                "max(pooled one-sided Clopper-Pearson upper, one-sided "
                "seed-group t upper); pooled only across seed groups"
            ),
            "confidence": 0.95,
            "threshold": thresholds.max_termination_rate,
            "granularity": "per shape (monte carlo), per (h, shape) (h sweep)",
            "aggregation": "worst shape controls; never pooled across shapes or h",
            "note": (
                "Reported alongside the legacy row-wise point-estimate gate. "
                "Neither term alone is safe: pooled Clopper-Pearson assumes "
                "exchangeable replicas, and the seed-group t bound carries no "
                "within-group binomial uncertainty."
            ),
        },
        "monte_carlo_termination_acceptance": mc_acceptance,
        "h_sweep_termination_acceptance": h_acceptance,
        "monte_carlo": monte_carlo,
        "h_sweep": h_groups,
    }
    (output_dir / "rom_b_held_out_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    _write_csv(output_dir / "rom_b_held_out_rows.csv", all_rows)
    _write_plot(
        output_dir / "rom_b_final_robustness.png", h_groups, monte_carlo
    )
    _write_markdown(
        output_dir / "rom_b_held_out_report.md",
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        monte_carlo=monte_carlo,
        h_groups=h_groups,
    )
    print(
        f"[ROM-B AUDIT] complete mc={'PASS' if not mc_failed else 'FAIL'} "
        f"h_passes={sum(group['passed'] for group in h_groups)}/{len(h_groups)} "
        f"report={output_dir / 'rom_b_held_out_report.md'}",
        flush=True,
    )
    print(
        "[ROM-B ACCEPT] predeclared termination statistic: "
        f"mc={'PASS' if mc_acceptance['passed'] else 'FAIL'} "
        f"(worst {mc_acceptance['worst_condition']['shape']} "
        f"upper={mc_acceptance['worst_condition']['upper_bound']:.5f}) "
        f"h_sweep={'PASS' if h_acceptance['passed'] else 'FAIL'} "
        f"(worst h={h_acceptance['worst_condition']['h']} "
        f"{h_acceptance['worst_condition']['shape']} "
        f"upper={h_acceptance['worst_condition']['upper_bound']:.5f})",
        flush=True,
    )
    if not isolated:
        print(
            "[ROM-B ACCEPT] WARNING: single-process audit. Simulator state leaks "
            "between rollouts and inflates termination rates -- measured "
            "2026-07-25, truck_raw_reaction failures fell 100 -> 14 under "
            "isolation while head_alignment held at 50 -> 47. NOT valid for "
            "certification; re-run with --isolate_conditions.",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()
