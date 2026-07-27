#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Per-failed-environment diagnostics for the accepted ROM-B checkpoint.

This runs the same deterministic probe as `evaluate.py` with observational
diagnostics enabled. It changes no gate calculation and selects no checkpoint;
it exists to explain *why* individual replicas terminate.

Two suites are available and both reuse the audit's seed offsets so a row here
is directly comparable to the same row in `rom_b_held_out_report.json`:

  monte_carlo  seeds = seed + 300000 + round * len(shapes) + shape_index
  h_sweep      seeds = seed + 400000 + round * len(shapes) + shape_index
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=17)
p.add_argument(
    "--suite",
    default="monte_carlo",
    choices=("monte_carlo", "h_sweep"),
    help="Which seed family to reproduce.",
)
p.add_argument(
    "--rounds",
    default="1,2,3,4,5",
    help="1-based seed groups to reproduce, comma separated.",
)
p.add_argument(
    "--trajectory_shapes",
    default="line,circle",
    help="Shapes to diagnose. The audit order is line,arc,sine,circle,square "
    "and shape_index is taken from that canonical order.",
)
p.add_argument(
    "--h_values",
    default="",
    help="Required for --suite h_sweep, e.g. 0.6,0.9.",
)
p.add_argument("--eval_steps", type=int, default=120)
p.add_argument("--diagnostic_window", type=int, default=12)
p.add_argument(
    "--force_zero_policy",
    action="store_true",
    help=(
        "Diagnostic ablation: force the PPO residual action to zero while "
        "leaving the reference, inner controller, ROM plant, and PhysX intact."
    ),
)
p.add_argument(
    "--hard_reset",
    action="store_true",
    help=(
        "EXPERIMENTAL, default off. Attempts to restore the articulation spawn "
        "state before each probe. Measured insufficient: probes remain "
        "position-dependent. See restore_canonical_physical_state.__doc__."
    ),
)
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument(
    "--dls_velocity_filter_beta",
    type=float,
    default=0.0,
    help=(
        "Diagnostic EMA on inverse-cylinder reference velocity; 0 preserves "
        "the production contract and larger values add more smoothing."
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
p.add_argument("--clearance_frame_consistent", action="store_true")
p.add_argument("--output_dir", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

# The audit indexes shapes against this fixed order; diagnosing a subset must
# not shift the seeds relative to the report being explained.
CANONICAL_SHAPES = ("line", "arc", "sine", "circle", "square")
SEED_OFFSETS = {"monte_carlo": 300000, "h_sweep": 400000}

shapes = tuple(v.strip() for v in a.trajectory_shapes.split(",") if v.strip())
unknown_shapes = [s for s in shapes if s not in CANONICAL_SHAPES]
if not shapes or unknown_shapes:
    p.error(f"shapes must be drawn from {CANONICAL_SHAPES}; got {unknown_shapes}")
try:
    rounds = tuple(int(v) for v in a.rounds.split(",") if v.strip())
except ValueError as exc:
    p.error(f"invalid --rounds: {exc}")
if not rounds or any(value < 1 for value in rounds):
    p.error("rounds must be positive 1-based indices")
h_values: tuple[float, ...] = ()
if a.suite == "h_sweep":
    try:
        h_values = tuple(float(v) for v in a.h_values.split(",") if v.strip())
    except ValueError as exc:
        p.error(f"invalid --h_values: {exc}")
    if not h_values or any(v < 0.4 or v > 1.2 for v in h_values):
        p.error("--suite h_sweep requires --h_values inside [0.4, 1.2]")
if a.eval_steps < 60:
    p.error("eval_steps must be at least 60")
if not 0.0 <= a.dls_velocity_filter_beta < 1.0:
    p.error("--dls_velocity_filter_beta must be in [0, 1)")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import PPO2_ROM_B_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_B_PhysX.env_cfg import PPO2ROMBPPORunnerCfg
from PPO2_ROM_B_PhysX.runtime import (
    FACTOR_NAMES,
    health_scale_to_factors,
    run_deterministic_probe,
)
from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage


def _failure_rows(row: dict[str, object]) -> list[dict[str, object]]:
    """Flatten one probe's failed-environment records for CSV export."""

    diagnostics = row.get("diagnostics") or {}
    out: list[dict[str, object]] = []
    for record in diagnostics.get("failed_environments", []):
        terminal = record["at_termination"]
        factors = record["factors"] or [None] * len(FACTOR_NAMES)
        flat: dict[str, object] = {
            "suite": row["suite"],
            "round": row["round"],
            "h": row.get("h"),
            "shape": row["shape"],
            "seed": row["seed"],
            "env_id": record["env_id"],
            "simulator_step": record["simulator_step"],
            "scored_step": record["scored_step"],
            "causes": "|".join(record["termination_causes"]),
        }
        flat.update(dict(zip(FACTOR_NAMES, factors)))
        for name, value in terminal.items():
            if isinstance(value, list):
                for index, element in enumerate(value):
                    flat[f"{name}_{index + 1}"] = element
            else:
                flat[name] = value
        out.append(flat)
    return out


def _print_summary(row: dict[str, object]) -> None:
    diagnostics = row.get("diagnostics") or {}
    records = diagnostics.get("failed_environments", [])
    label = f"h={row['h']}" if row.get("h") is not None else "mc"
    print(
        f"[ROM-B DIAG] {label} round={row['round']} shape={row['shape']} "
        f"seed={row['seed']} term={float(row['max_termination_rate']):.5f} "
        f"acc95={float(row['p95_acceleration_mps2']):.2f} failures={len(records)}",
        flush=True,
    )
    if not records:
        return
    causes = Counter(
        cause for record in records for cause in record["termination_causes"]
    )
    print(f"    causes: {dict(causes)}", flush=True)
    for record in records:
        terminal = record["at_termination"]
        print(
            f"    env={record['env_id']:>3} step={record['scored_step']:>3} "
            f"cause={','.join(record['termination_causes'])} "
            f"path_s={terminal['path_s_m']:.3f} "
            f"phase={terminal['reference_phase']:.0f} "
            f"clearance={terminal['clearance_signed_m']:+.4f} "
            f"head={terminal['head_alignment']:.5f} "
            f"acc={terminal['acceleration_mps2']:.2f} "
            f"raw_drift={terminal['truck_raw_drift_m']:.5f} "
            f"e_comp={record['factors'][0]:+.4f} "
            f"seal_mean={sum(record['factors'][1:5]) / 4:.4f}",
            flush=True,
        )


def main() -> None:
    checkpoint = Path(a.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(a.output_dir).expanduser().resolve()
        if a.output_dir
        else ROOT / "PPO2_ROM_B_PhysX/reports" / f"diagnostics_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    task = "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 42
    cfg.command_rom_checkpoint = str(Path(a.command_rom_checkpoint).resolve())
    cfg.trajectory_library_path = str(Path(a.trajectory_library_path).resolve())
    cfg.trajectory_shapes = CANONICAL_SHAPES
    # Match the certification evaluator's construction state exactly.  The
    # environment is constructed on the first canonical/audit shape (line)
    # before run_deterministic_probe selects the requested diagnostic shape.
    # Constructing directly on circle changes the subsequent PhysX rollout
    # despite an identical reset seed (measured 2026-07-26: 40.50 vs 44.08
    # m/s^2 for h=1.0, circle, round 3).
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
    offset = SEED_OFFSETS[a.suite]
    rows: list[dict[str, object]] = []

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
        conditions = h_values if a.suite == "h_sweep" else (None,)
        for h in conditions:
            factors = (
                health_scale_to_factors(h, device=gym_env.unwrapped.device)[0]
                if h is not None else None
            )
            for round_index in rounds:
                for shape in shapes:
                    shape_index = CANONICAL_SHAPES.index(shape)
                    seed = (
                        int(a.seed) + offset
                        + (round_index - 1) * len(CANONICAL_SHAPES) + shape_index
                    )
                    row = run_deterministic_probe(
                        gym_env.unwrapped, env, policy, shape=shape, seed=seed,
                        scored_steps=int(a.eval_steps),
                        force_zero_policy=bool(a.force_zero_policy),
                        fixed_factors=factors,
                        collect_diagnostics=True,
                        diagnostic_window=int(a.diagnostic_window),
                        hard_reset=bool(a.hard_reset),
                    )
                    row.update({
                        "suite": a.suite,
                        "round": round_index,
                        "h": float(h) if h is not None else None,
                    })
                    rows.append(row)
                    _print_summary(row)
    finally:
        gym_env.close()

    (output_dir / "rom_b_diagnostics.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "checkpoint": str(checkpoint),
                "suite": a.suite,
                "canonical_shapes": list(CANONICAL_SHAPES),
                "seed_offset": offset,
                "num_envs": int(a.num_envs),
                "eval_steps": int(a.eval_steps),
                "force_zero_policy": bool(a.force_zero_policy),
                "dls_velocity_filter_beta": float(a.dls_velocity_filter_beta),
                "note": (
                    "Diagnostics are observational. Gate metrics in these rows "
                    "are computed exactly as in evaluate.py."
                ),
                "rows": rows,
            },
            indent=2,
        ) + "\n"
    )

    failures = [record for row in rows for record in _failure_rows(row)]
    if failures:
        fields: list[str] = []
        for record in failures:
            for key in record:
                if key not in fields:
                    fields.append(key)
        with (output_dir / "rom_b_failed_environments.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(failures)

    by_shape_cause: dict[tuple[str, str], int] = defaultdict(int)
    for record in failures:
        for cause in str(record["causes"]).split("|"):
            by_shape_cause[(str(record["shape"]), cause)] += 1
    print("[ROM-B DIAG] failure counts by shape and cause:", flush=True)
    for (shape, cause), count in sorted(by_shape_cause.items()):
        print(f"    {shape:7s} {cause:22s} {count}", flush=True)
    print(f"[ROM-B DIAG] complete output={output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
