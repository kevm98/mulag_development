#!/usr/bin/env python3
"""Root-cause validation for the active PPO1 horizon-IDFD pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        choices=("ground_debug", "workspace", "trajectory", "compare", "ppo_smoke"),
        default="ground_debug",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--horizon_steps", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--workspace_samples", type=int, default=200)
    parser.add_argument(
        "--trajectory_mode",
        choices=("ground_straight_line", "ground_semicircle", "ground_mower_sweep"),
        default="ground_straight_line",
    )
    parser.add_argument("--trajectory_plane", choices=("ground_xy", "base_yz"), default="ground_xy")
    parser.add_argument("--trajectory_scale", type=float, default=0.8)
    parser.add_argument("--trajectory_points", type=int, default=200)
    parser.add_argument(
        "--tool_contact_mode",
        choices=("geometry_aabb", "manual_offset", "ee_origin"),
        default="geometry_aabb",
    )
    parser.add_argument("--tool_ground_clearance", type=float, default=0.03)
    parser.add_argument("--ground_z", type=float, default=0.0)
    parser.add_argument("--greedy_gain", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--ppo_best", type=str, default="")
    parser.add_argument("--ppo_final", type=str, default="")
    parser.add_argument("--ppo_iterations", type=int, default=1)
    parser.add_argument("--ppo_steps_per_env", type=int, default=2)
    parser.add_argument("--log_dir", type=str, default="outputs/ppo1_horizon_idfd/root_cause_validation")
    parser.add_argument("--fast_exit", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import (
    HorizonIDFDEnv,
    HorizonIDFDRslRlVecEnvWrapper,
)
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import (
    MulagUpperLevelEnvCfg,
    MulagUpperLevelPPORunnerCfg,
)


def _as_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _print_json(label: str, payload: dict) -> None:
    print(f"{label} {json.dumps(_as_jsonable(payload), sort_keys=True)}", flush=True)


def _make_cfg(trajectory_mode: str | None = None) -> MulagUpperLevelEnvCfg:
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.sim.device = args_cli.device
    cfg.scene.replicate_physics = False
    cfg.ppo_stage = "full_mowing"
    cfg.ee_action_mode = "task_space_delta"
    cfg.ee_action_dim = 3
    cfg.low_level_servo_mode = "none"
    cfg.lowlevel_actuator_backend = "explicit_effort_pid_bias"
    cfg.enable_scripted_vehicle_motion = False
    cfg.horizon_pin_root_pose = True
    cfg.horizon_reset_settle_steps = 1
    cfg.trajectory_mode = trajectory_mode or args_cli.trajectory_mode
    cfg.trajectory_plane = args_cli.trajectory_plane
    cfg.trajectory_scale = float(args_cli.trajectory_scale)
    cfg.ground_trajectory_points = int(args_cli.trajectory_points)
    cfg.tool_contact_mode = args_cli.tool_contact_mode
    cfg.tool_ground_clearance = float(args_cli.tool_ground_clearance)
    cfg.ground_z = float(args_cli.ground_z)
    cfg.debug_draw_tool_contact_point = True
    cfg.debug_draw_tool_aabb = True
    cfg.visualize_target_trajectory = True
    cfg.visualize_actual_ee_trail = True
    return cfg


def _make_env(trajectory_mode: str | None = None) -> HorizonIDFDEnv:
    torch.manual_seed(int(args_cli.random_seed))
    cfg = _make_cfg(trajectory_mode)
    env = HorizonIDFDEnv(
        cfg,
        horizon_steps=int(args_cli.horizon_steps),
        device=args_cli.device,
    )
    env.reset()
    return env


def _summarize_rollout(env: HorizonIDFDEnv, records: list[dict], done_count: int, cause_counts: dict[str, int]) -> dict:
    if not records:
        return {"records": 0, "pass": False}
    keys = records[0].keys()
    summary = {"records": len(records), "done_count": done_count, "cause_counts": cause_counts}
    for key in keys:
        values = [float(row[key]) for row in records if row[key] is not None]
        if values:
            summary[f"min_{key}"] = min(values)
            summary[f"mean_{key}"] = sum(values) / len(values)
            summary[f"max_{key}"] = max(values)
    summary["control_pipeline"] = env.control_pipeline_report()
    summary["physics_valid"] = (
        summary.get("min_tool_bottom_clearance", 0.0) >= -0.01
        and summary.get("min_arm_geometry_clearance", 0.0) >= -0.02
        and summary.get("max_max_abs_body_world_position", 0.0) <= 100.0
        and summary.get("max_nan_inf_count", 1.0) == 0.0
        and summary.get("max_huge_transform_warning_count", 1.0) == 0.0
    )
    summary["idfd_active"] = (
        summary["control_pipeline"]["ppo_action_is_upper_level_delta_ee"]
        and summary["control_pipeline"]["empirical_j_eff_used"]
        and all(bool(v) for v in summary["control_pipeline"]["jacobian_valid_mask"])
        and float(summary["control_pipeline"].get("jacobian_max_singular_value") or 0.0) > 0.0
        and summary["control_pipeline"]["id_forward_calls"] > 0
        and summary["control_pipeline"]["fd_forward_calls"] > 0
        and summary["control_pipeline"]["horizon_cylinder_target_writes"] > 0
        and not summary["control_pipeline"]["direct_cylinder_bypass_used"]
    )
    summary["pass"] = bool(summary["physics_valid"] and summary["idfd_active"] and done_count == 0)
    return summary


def _metrics_row(env: HorizonIDFDEnv) -> dict:
    metrics = env.ground_safety_metrics()
    workspace_violation, cab_violation = env._workspace_guard_components()
    base_masks = env._base_done_masks()
    return {
        "tracking_error": float(metrics["tracking_error"].mean().detach().cpu().item()),
        "tool_bottom_clearance": float(metrics["tool_bottom_clearance"].min().detach().cpu().item()),
        "arm_geometry_clearance": float(metrics["arm_geometry_clearance"].min().detach().cpu().item()),
        "target_tool_bottom_clearance": float(metrics["target_tool_bottom_clearance"].min().detach().cpu().item()),
        "max_abs_body_world_position": float(metrics["max_abs_body_world_position"].max().detach().cpu().item()),
        "nan_inf_count": float(metrics["nan_inf_count"].sum().detach().cpu().item()),
        "huge_transform_warning_count": float((metrics["max_abs_body_world_position"] > 100.0).sum().detach().cpu().item()),
        "root_motion_norm": float(metrics["root_motion_norm"].max().detach().cpu().item()),
        "path_completion_fraction": float(metrics["path_completion_fraction"].mean().detach().cpu().item()),
        "ee_origin_world_z": float(metrics["ee_origin_world_z"].mean().detach().cpu().item()),
        "tool_bottom_world_z": float(metrics["tool_bottom_world_z"].mean().detach().cpu().item()),
        "ground_z": float(metrics["ground_z"].mean().detach().cpu().item()),
        "horizon_workspace_violation": float(workspace_violation.sum().detach().cpu().item()),
        "horizon_cab_violation": float(cab_violation.sum().detach().cpu().item()),
        "base_workspace_violation": float(base_masks["workspace_violation"].sum().detach().cpu().item()),
        "base_joint_limit": float(base_masks["joint_limit"].sum().detach().cpu().item()),
        "base_flying": float(base_masks["base_flying"].sum().detach().cpu().item()),
        "base_nonfinite": float(base_masks["NaN/nonfinite"].sum().detach().cpu().item()),
    }


def _policy_action(env: HorizonIDFDEnv, policy_name: str, policy_fn=None) -> torch.Tensor:
    if policy_name == "zero":
        return torch.zeros((env.num_envs, env.num_actions), device=env.device)
    if policy_name == "random":
        return torch.empty((env.num_envs, env.num_actions), device=env.device).uniform_(-1.0, 1.0)
    if policy_name == "greedy":
        return env.compute_ground_tracking_action(gain=float(args_cli.greedy_gain))
    if policy_fn is None:
        raise RuntimeError(f"No policy function for {policy_name}")
    obs = env._build_observation()
    return torch.clamp(policy_fn(obs), -1.0, 1.0)


def run_policy(policy_name: str, policy_fn=None, *, steps: int | None = None, trajectory_mode: str | None = None) -> dict:
    env = _make_env(trajectory_mode)
    records = []
    done_count = 0
    cause_counts = {name: 0 for name in env.DONE_CAUSES}
    component_counts = {}
    last_cause_values = {}
    try:
        _print_json(f"[initial_geometry/{policy_name}]", env.tool_geometry_report())
        _print_json(f"[initial_frame/{policy_name}]", env.discover_controlled_frame())
        for _ in range(int(steps or args_cli.num_steps)):
            action = _policy_action(env, policy_name, policy_fn)
            _, _, terminated, truncated, extras = env.step(action)
            records.append(_metrics_row(env))
            done_count += int((terminated | truncated).sum().detach().cpu().item())
            for name, mask in extras.get("cause_masks", {}).items():
                cause_counts[name] = cause_counts.get(name, 0) + int(mask.sum().detach().cpu().item())
            for name, mask in extras.get("cause_components", {}).items():
                component_counts[name] = component_counts.get(name, 0) + int(mask.sum().detach().cpu().item())
            if extras.get("cause_values"):
                last_cause_values = _as_jsonable(extras["cause_values"])
        summary = _summarize_rollout(env, records, done_count, cause_counts)
        summary["component_counts"] = component_counts
        summary["last_cause_values"] = last_cause_values
        _print_json(f"[summary/{policy_name}]", summary)
        return summary
    finally:
        env.close()


def run_ground_debug() -> int:
    summary = run_policy("greedy", steps=args_cli.num_steps, trajectory_mode=args_cli.trajectory_mode)
    return 0 if summary["pass"] else 2


def run_workspace() -> int:
    steps = max(int(args_cli.workspace_samples), 1)
    summary = run_policy("greedy", steps=steps, trajectory_mode=args_cli.trajectory_mode)
    summary["workspace_samples"] = steps
    _print_json("[workspace_result]", summary)
    return 0 if summary["pass"] else 2


def run_trajectory() -> int:
    summary = run_policy("greedy", steps=args_cli.num_steps, trajectory_mode=args_cli.trajectory_mode)
    summary["trajectory_mode"] = args_cli.trajectory_mode
    _print_json("[trajectory_result]", summary)
    return 0 if summary["pass"] else 2


def _load_rsl_policy(checkpoint: str):
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.is_file():
        print(f"[ppo_policy] missing_checkpoint={checkpoint}", flush=True)
        return None
    env = _make_env(args_cli.trajectory_mode)
    vec_env = HorizonIDFDRslRlVecEnvWrapper(env)
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = MulagUpperLevelPPORunnerCfg()
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(str(path))
    policy = runner.get_inference_policy(device=args_cli.device)

    def _policy(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return policy(obs)

    return _policy


def run_compare() -> int:
    results = {
        "zero": run_policy("zero", trajectory_mode=args_cli.trajectory_mode),
        "random": run_policy("random", trajectory_mode=args_cli.trajectory_mode),
        "greedy": run_policy("greedy", trajectory_mode=args_cli.trajectory_mode),
    }
    for label, checkpoint in (("ppo_best", args_cli.ppo_best), ("ppo_final", args_cli.ppo_final)):
        policy = _load_rsl_policy(checkpoint)
        if policy is None:
            results[label] = {"evaluated": False, "checkpoint": checkpoint}
        else:
            results[label] = run_policy(label, policy, trajectory_mode=args_cli.trajectory_mode)
    greedy_err = results["greedy"].get("mean_tracking_error")
    zero_err = results["zero"].get("mean_tracking_error")
    learning_verdict = "not_claimed"
    if results.get("ppo_best", {}).get("evaluated"):
        ppo_err = results["ppo_best"].get("mean_tracking_error", float("inf"))
        ppo_valid = bool(results["ppo_best"].get("physics_valid", False))
        if ppo_valid and ppo_err < min(float(zero_err), results["random"].get("mean_tracking_error", float("inf"))):
            learning_verdict = "ppo_best_improves_tracking_over_zero_and_random"
        else:
            learning_verdict = "ppo_best_does_not_clear_validation_baselines"
    _print_json("[compare_result]", {"results": results, "learning_verdict": learning_verdict, "greedy_mean_error": greedy_err})
    return 0 if all(results[name].get("pass", False) for name in ("zero", "random", "greedy")) else 2


def run_ppo_smoke() -> int:
    env = _make_env(args_cli.trajectory_mode)
    vec_env = HorizonIDFDRslRlVecEnvWrapper(env)
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = MulagUpperLevelPPORunnerCfg()
    agent_cfg.num_steps_per_env = int(args_cli.ppo_steps_per_env)
    agent_cfg.max_iterations = int(args_cli.ppo_iterations)
    agent_cfg.save_interval = max(1, int(args_cli.ppo_iterations))
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = 1
    log_dir = Path(args_cli.log_dir) / "ppo_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=args_cli.device)
    runner.learn(num_learning_iterations=int(args_cli.ppo_iterations), init_at_random_ep_len=False)
    summary = env.control_pipeline_report()
    metrics = _metrics_row(env)
    _print_json("[ppo_smoke_result]", {"control_pipeline": summary, "metrics": metrics})
    env.close()
    valid = (
        metrics["tool_bottom_clearance"] >= -0.01
        and metrics["arm_geometry_clearance"] >= -0.02
        and metrics["max_abs_body_world_position"] <= 100.0
        and metrics["nan_inf_count"] == 0.0
        and metrics["huge_transform_warning_count"] == 0.0
        and summary["id_forward_calls"] > 0
        and summary["fd_forward_calls"] > 0
        and not summary["direct_cylinder_bypass_used"]
    )
    return 0 if valid else 2


def main() -> int:
    try:
        if args_cli.gate == "ground_debug":
            return run_ground_debug()
        if args_cli.gate == "workspace":
            return run_workspace()
        if args_cli.gate == "trajectory":
            return run_trajectory()
        if args_cli.gate == "compare":
            return run_compare()
        if args_cli.gate == "ppo_smoke":
            return run_ppo_smoke()
        raise RuntimeError(f"Unknown gate {args_cli.gate}")
    finally:
        if not args_cli.fast_exit:
            simulation_app.close()


if __name__ == "__main__":
    code = main()
    print(f"[gate_exit_code] {int(code)}", flush=True)
    if args_cli.fast_exit:
        os._exit(int(code))
    raise SystemExit(code)
