#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Paired deterministic gate for a four-cylinder residual checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", required=True)
p.add_argument(
    "--baseline_checkpoint",
    default=str(
        ROOT
        / "logs/rsl_rl/ppo2_rom_residual/2026-07-19_18-55-23/model_249.pt"
    ),
)
p.add_argument("--shapes", default="line,arc,sine,circle,square")
p.add_argument("--seed", type=int, default=17)
p.add_argument("--num_seeds", type=int, default=1)
p.add_argument("--scored_steps", type=int, default=480)
p.add_argument("--full_cycles", action="store_true")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--cylinder4_residual_scale_m", type=float, default=0.00025)
p.add_argument("--trajectory_library_path", default="")
p.add_argument("--command_rom_checkpoint", default="")
p.add_argument(
    "--command_rom_direction_guard",
    action=argparse.BooleanOptionalAction,
    default=None,
)
p.add_argument(
    "--out",
    default="outputs/ppo2_rom_residual_4cyl_pitch_branch/five_shape_gate.json",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

shapes = tuple(part.strip() for part in a.shapes.split(",") if part.strip())
valid_shapes = {"line", "arc", "sine", "circle", "square"}
if not shapes or not set(shapes).issubset(valid_shapes):
    p.error("--shapes must contain line, arc, sine, circle, and/or square")
if a.num_seeds < 1 or a.scored_steps < 60:
    p.error("--num_seeds must be positive and --scored_steps must be at least 60")
if not 0.0 <= a.cylinder4_residual_scale_m <= 0.0005:
    p.error("cylinder-4 authority must be in [0, 0.0005] m")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import PPO2_ROM_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage
from PPO2_ROM_PhysX.policy_4cyl.checkpoint_transfer import (
    transfer_checkpoint_into_runner,
)
from PPO2_ROM_PhysX.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyPPORunnerCfg,
)
from isaaclab_tasks.utils import parse_env_cfg


def _run_policy(raw, env, policy, *, shape: str, seed: int) -> dict[str, object]:
    raw.set_trajectory_shape(shape)
    raw.reset(seed=int(seed))
    obs = env.get_observations()
    errors: list[float] = []
    tools: list[np.ndarray] = []
    heads: list[float] = []
    clearance_errors: list[float] = []
    drifts: list[float] = []
    cylinder4_residuals: list[float] = []
    all_residuals: list[float] = []
    terminated = False
    total_steps = 0
    max_total_steps = 30000 if a.full_cycles else int(a.scored_steps) + 4000
    scales = raw._residual_scale_vector()

    while total_steps < max_total_steps:
        with torch.no_grad():
            obs, _, dones, _ = env.step(policy(obs))
        total_steps += 1
        active = bool(
            raw._preposition_counter[0] <= 0 and raw._settle_counter[0] <= 0
        )
        if active:
            tool = raw._compute_ee_pos_local()[0]
            error = torch.linalg.norm(raw._original_reference_tool[0] - tool)
            clearance_error = torch.abs(
                raw._compute_cutter_world_z()[0] - raw._original_reference_world_z[0]
            )
            physical_residual = torch.abs(raw._residual_filtered[0]) * scales
            errors.append(float(error))
            tools.append(tool.detach().cpu().numpy().copy())
            heads.append(float(raw._measure_head_down_alignment()[0]))
            clearance_errors.append(float(clearance_error))
            drifts.append(float(raw._truck_post_constraint_drift_xy[0]))
            cylinder4_residuals.append(float(physical_residual[3]))
            all_residuals.append(float(torch.mean(physical_residual)))
        if bool(dones.any()):
            terminated = True
            break
        if a.full_cycles:
            if active and float(raw._scored_elapsed[0]) >= float(raw._complete_cycle_time):
                break
        elif len(errors) >= int(a.scored_steps):
            break

    if len(errors) < 20:
        return {
            "completed": False,
            "terminated": terminated,
            "active_steps": len(errors),
            "total_steps": total_steps,
        }
    tool_array = np.asarray(tools)
    velocity = np.diff(tool_array, axis=0) / float(raw.step_dt)
    acceleration = np.diff(velocity, axis=0) / float(raw.step_dt)
    return {
        "completed": bool(not terminated and total_steps < max_total_steps),
        "terminated": terminated,
        "active_steps": len(errors),
        "total_steps": total_steps,
        "mean_tool_error_m": float(np.mean(errors)),
        "p95_tool_error_m": float(np.percentile(errors, 95)),
        "p95_tool_acceleration_mps2": float(
            np.percentile(np.linalg.norm(acceleration, axis=-1), 95)
        ),
        "min_head_alignment": float(np.min(heads)),
        "max_clearance_error_m": float(np.max(clearance_errors)),
        "max_truck_post_constraint_drift_m": float(np.max(drifts)),
        "mean_cylinder4_abs_residual_m": float(np.mean(cylinder4_residuals)),
        "max_cylinder4_abs_residual_m": float(np.max(cylinder4_residuals)),
        "mean_abs_residual_m": float(np.mean(all_residuals)),
    }


def _aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    mean_keys = (
        "mean_tool_error_m",
        "p95_tool_error_m",
        "p95_tool_acceleration_mps2",
        "mean_cylinder4_abs_residual_m",
        "mean_abs_residual_m",
    )
    return {
        "completed_all": all(bool(run.get("completed")) for run in runs),
        **{
            key: float(np.mean([float(run.get(key, np.inf)) for run in runs]))
            for key in mean_keys
        },
        "min_head_alignment": float(
            min(float(run.get("min_head_alignment", -np.inf)) for run in runs)
        ),
        "max_clearance_error_m": float(
            max(float(run.get("max_clearance_error_m", np.inf)) for run in runs)
        ),
        "max_truck_post_constraint_drift_m": float(
            max(
                float(run.get("max_truck_post_constraint_drift_m", np.inf))
                for run in runs
            )
        ),
        "max_cylinder4_abs_residual_m": float(
            max(
                float(run.get("max_cylinder4_abs_residual_m", np.inf))
                for run in runs
            )
        ),
    }


def main() -> None:
    task = "RRLAB-Mulag-PPO2-A-Policy-V3-PLAY-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=1)
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 42
    if a.trajectory_library_path:
        cfg.trajectory_library_path = str(
            Path(a.trajectory_library_path).expanduser().resolve()
        )
    if a.command_rom_checkpoint:
        cfg.command_rom_checkpoint = str(
            Path(a.command_rom_checkpoint).expanduser().resolve()
        )
    if a.command_rom_direction_guard is not None:
        cfg.command_rom_direction_guard = bool(a.command_rom_direction_guard)
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.randomize_trajectory_shape = False
    cfg.randomize_start_phase = False
    cfg.cylinder4_residual_scale_m = float(a.cylinder4_residual_scale_m)
    cfg.clearance_relative_to_reference = True
    cfg.cutter_clearance_hard_band_m = 0.20
    cfg.head_down_termination_alignment = 0.98
    cfg.episode_length_s = 300.0

    gym_env = gym.make(task, cfg=cfg)
    raw = gym_env.unwrapped
    agent_cfg = FourCylinderResidualPolicyPPORunnerCfg()
    agent_cfg.device = a.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    baseline_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=a.device
    )
    transfer_checkpoint_into_runner(
        baseline_runner,
        a.baseline_checkpoint,
        cylinder123_std=0.001,
        cylinder4_std=0.08,
    )
    baseline_policy = baseline_runner.get_inference_policy(device=raw.device)
    candidate_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=a.device
    )
    candidate_runner.load(str(Path(a.checkpoint).expanduser().resolve()))
    candidate_policy = candidate_runner.get_inference_policy(device=raw.device)

    runs: dict[str, list[dict[str, object]]] = {"baseline": [], "candidate": []}
    try:
        for shape in shapes:
            for seed in range(int(a.seed), int(a.seed) + int(a.num_seeds)):
                baseline = _run_policy(
                    raw, env, baseline_policy, shape=shape, seed=seed
                )
                candidate = _run_policy(
                    raw, env, candidate_policy, shape=shape, seed=seed
                )
                baseline.update({"shape": shape, "seed": seed})
                candidate.update({"shape": shape, "seed": seed})
                runs["baseline"].append(baseline)
                runs["candidate"].append(candidate)
                print(
                    f"[4CYL EVAL] {shape} seed={seed} "
                    f"mean={baseline.get('mean_tool_error_m', np.inf):.5f}->"
                    f"{candidate.get('mean_tool_error_m', np.inf):.5f}m "
                    f"p95={baseline.get('p95_tool_error_m', np.inf):.5f}->"
                    f"{candidate.get('p95_tool_error_m', np.inf):.5f}m "
                    f"acc95={baseline.get('p95_tool_acceleration_mps2', np.inf):.2f}->"
                    f"{candidate.get('p95_tool_acceleration_mps2', np.inf):.2f}m/s^2",
                    flush=True,
                )
    finally:
        env.close()

    aggregate = {name: _aggregate(values) for name, values in runs.items()}
    per_shape: dict[str, dict[str, float | bool]] = {}
    for shape in shapes:
        base = [run for run in runs["baseline"] if run["shape"] == shape]
        cand = [run for run in runs["candidate"] if run["shape"] == shape]
        base_mean = float(np.mean([run["mean_tool_error_m"] for run in base]))
        cand_mean = float(np.mean([run["mean_tool_error_m"] for run in cand]))
        base_acc = float(np.mean([run["p95_tool_acceleration_mps2"] for run in base]))
        cand_acc = float(np.mean([run["p95_tool_acceleration_mps2"] for run in cand]))
        per_shape[shape] = {
            "mean_error_delta_m": cand_mean - base_mean,
            "p95_acceleration_delta_mps2": cand_acc - base_acc,
            "mean_error_not_regressed": cand_mean <= base_mean + 0.002,
            "acceleration_not_increased": cand_acc <= base_acc,
        }

    baseline = aggregate["baseline"]
    candidate = aggregate["candidate"]
    checks = {
        "completed_all": bool(baseline["completed_all"] and candidate["completed_all"]),
        "aggregate_mean_error_better": bool(
            candidate["mean_tool_error_m"] < baseline["mean_tool_error_m"]
        ),
        "aggregate_p95_error_better": bool(
            candidate["p95_tool_error_m"] < baseline["p95_tool_error_m"]
        ),
        "all_shape_mean_errors_within_2mm": all(
            bool(values["mean_error_not_regressed"])
            for values in per_shape.values()
        ),
        "all_shape_acceleration_not_increased": all(
            bool(values["acceleration_not_increased"])
            for values in per_shape.values()
        ),
        "head_alignment_safe_and_not_regressed": bool(
            candidate["min_head_alignment"] >= 0.98
            and candidate["min_head_alignment"] >= baseline["min_head_alignment"] - 0.001
        ),
        "clearance_within_20cm": bool(candidate["max_clearance_error_m"] <= 0.20),
        "truck_drift_within_1mm": bool(
            candidate["max_truck_post_constraint_drift_m"] <= 0.001
        ),
        "cylinder4_within_authority": bool(
            candidate["max_cylinder4_abs_residual_m"]
            <= float(a.cylinder4_residual_scale_m) + 1.0e-9
        ),
    }
    passed = all(checks.values())
    result = {
        "passed": passed,
        "mode": "full_cycles" if a.full_cycles else "bounded_screen",
        "checkpoint": str(Path(a.checkpoint).expanduser().resolve()),
        "baseline_checkpoint": str(Path(a.baseline_checkpoint).expanduser().resolve()),
        "trajectory_library": str(
            Path(a.trajectory_library_path).expanduser().resolve()
        ) if a.trajectory_library_path else "environment_default",
        "command_rom_checkpoint": str(
            Path(a.command_rom_checkpoint).expanduser().resolve()
        ) if a.command_rom_checkpoint else "environment_default",
        "shapes": list(shapes),
        "seeds": list(range(int(a.seed), int(a.seed) + int(a.num_seeds))),
        "checks": checks,
        "per_shape": per_shape,
        "aggregate": aggregate,
        "runs": runs,
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[4CYL EVAL] {'PASS' if passed else 'FAIL'} wrote {out}", flush=True)
    if not passed:
        raise SystemExit(2)


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
