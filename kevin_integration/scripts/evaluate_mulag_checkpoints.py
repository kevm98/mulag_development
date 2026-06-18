"""Deterministic checkpoint discovery, evaluation, and best-checkpoint ranking."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "/home/rrlab/IsaacLab"))
ISAACLAB_RSL_RL_SCRIPTS = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if ISAACLAB_RSL_RL_SCRIPTS.is_dir() and str(ISAACLAB_RSL_RL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_RSL_RL_SCRIPTS))


CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")
RANKING_FIELDS = (
    "strict_success_rate",
    "hold_completion_fraction",
    "mean_max_hold_steps",
    "final_position_error_mean",
    "final_orientation_error_mean",
    "final_linear_speed_mean",
    "final_angular_speed_mean",
    "clamp_activation_fraction_mean",
    "valve_saturation_fraction_mean",
)


def _is_missing_cfg_value(value: object) -> bool:
    return isinstance(value, type(MISSING))


def _has_non_missing_attr(obj: object, attr_name: str) -> bool:
    return hasattr(obj, attr_name) and not _is_missing_cfg_value(getattr(obj, attr_name))


def _uses_current_distribution_schema(agent_cfg, installed_rsl_rl_version: str, version_module) -> bool:
    """Return True when the agent config is already in the RSL-RL 5 distribution schema."""
    if version_module.parse(installed_rsl_rl_version) < version_module.parse("5.0.0"):
        return False
    if _has_non_missing_attr(agent_cfg, "policy"):
        return False

    saw_model_cfg = False
    for model_name in ("actor", "critic", "student", "teacher"):
        if not _has_non_missing_attr(agent_cfg, model_name):
            continue
        model_cfg = getattr(agent_cfg, model_name)
        if not hasattr(model_cfg, "distribution_cfg"):
            return False
        saw_model_cfg = True
        for attr_name in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            if _has_non_missing_attr(model_cfg, attr_name):
                return False
    return saw_model_cfg


def prepare_rsl_rl_agent_cfg(agent_cfg, installed_rsl_rl_version: str, version_module, migration_fn):
    """Apply IsaacLab migration only when the Kevin config is not already current-schema."""
    if _uses_current_distribution_schema(agent_cfg, installed_rsl_rl_version, version_module):
        print(
            "[INFO] RSL-RL agent config already uses distribution_cfg; "
            "skipping deprecated stochastic-field migration.",
            flush=True,
        )
        return agent_cfg
    return migration_fn(agent_cfg, installed_rsl_rl_version)


@dataclass(frozen=True)
class CheckpointCandidate:
    path: str
    iteration: int


@dataclass
class CheckpointEvaluationResult:
    checkpoint_path: str
    iteration: int
    episodes: int
    evaluation_seeds: list[int]
    strict_success_rate: float
    hold_completion_fraction: float
    mean_max_hold_steps: float
    final_position_error_mean: float
    final_orientation_error_mean: float
    final_linear_speed_mean: float
    final_angular_speed_mean: float
    clamp_activation_fraction_mean: float
    valve_saturation_fraction_mean: float
    mean_reward: float = 0.0


def checkpoint_iteration(path: str | Path) -> int:
    match = CHECKPOINT_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"Invalid checkpoint filename: {path}")
    return int(match.group(1))


def discover_checkpoints(run_dir: str | Path) -> list[CheckpointCandidate]:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    checkpoints = [
        CheckpointCandidate(path=str(path), iteration=checkpoint_iteration(path))
        for path in run_path.iterdir()
        if path.is_file() and CHECKPOINT_RE.match(path.name)
    ]
    checkpoints.sort(key=lambda item: item.iteration)
    if not checkpoints:
        raise FileNotFoundError(f"No valid model_*.pt checkpoints found in {run_path}")
    return checkpoints


def select_checkpoints(
    candidates: list[CheckpointCandidate],
    *,
    checkpoint_stride: int | None = None,
    explicit_checkpoints: Iterable[str] | None = None,
) -> list[CheckpointCandidate]:
    if explicit_checkpoints:
        by_path = {str(Path(item.path).resolve()): item for item in candidates}
        by_name = {Path(item.path).name: item for item in candidates}
        selected: list[CheckpointCandidate] = []
        for requested in explicit_checkpoints:
            requested_path = Path(requested).expanduser()
            key_path = str(requested_path.resolve()) if requested_path.exists() else str(requested_path)
            candidate = by_path.get(key_path) or by_name.get(requested_path.name)
            if candidate is None:
                raise FileNotFoundError(f"Requested checkpoint is not in run directory: {requested}")
            selected.append(candidate)
        return sorted(selected, key=lambda item: item.iteration)
    if checkpoint_stride is None or checkpoint_stride <= 1:
        return candidates
    return [item for index, item in enumerate(candidates) if index % checkpoint_stride == 0]


def ranking_key(result: CheckpointEvaluationResult) -> tuple[float, ...]:
    """Lexicographic key where Python's ``max`` returns the desired checkpoint."""
    return (
        result.strict_success_rate,
        result.hold_completion_fraction,
        result.mean_max_hold_steps,
        -result.final_position_error_mean,
        -result.final_orientation_error_mean,
        -result.final_linear_speed_mean,
        -result.final_angular_speed_mean,
        -result.clamp_activation_fraction_mean,
        -result.valve_saturation_fraction_mean,
    )


def choose_best_checkpoint(results: list[CheckpointEvaluationResult]) -> CheckpointEvaluationResult:
    if not results:
        raise ValueError("No checkpoint evaluation results were provided.")
    return max(results, key=ranking_key)


def ranking_explanation(best: CheckpointEvaluationResult) -> str:
    return (
        "Lexicographic ranking: highest strict success rate, highest 50-step hold completion fraction, "
        "highest mean maximum hold duration, then lowest final position error, orientation error, linear speed, "
        "angular speed, avoidable clamp activity, and avoidable valve saturation. Mean reward is recorded but "
        "does not participate in best-checkpoint selection."
    )


def write_results(run_dir: str | Path, results: list[CheckpointEvaluationResult]) -> dict[str, Path]:
    run_path = Path(run_dir).expanduser().resolve()
    best = choose_best_checkpoint(results)
    json_path = run_path / "checkpoint_evaluation.json"
    csv_path = run_path / "checkpoint_evaluation.csv"
    best_path = run_path / "best_checkpoint.json"

    serializable = [asdict(result) for result in results]
    json_path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(serializable[0].keys()))
        writer.writeheader()
        writer.writerows(serializable)
    best_payload = asdict(best)
    best_payload["ranking_metrics"] = {field: getattr(best, field) for field in RANKING_FIELDS}
    best_payload["ranking_explanation"] = ranking_explanation(best)
    best_payload["checkpoint_path"] = best.checkpoint_path
    best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "best": best_path}


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _extract_scalar(log: dict[str, object], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = log.get(key)
        if value is None:
            continue
        try:
            if hasattr(value, "detach"):
                value = value.detach().float().mean().cpu().item()
            return float(value)
        except Exception:
            continue
    return float(default)


def evaluate_checkpoint_runtime(
    *,
    checkpoint: CheckpointCandidate,
    task: str,
    env,
    runner,
    episodes: int,
    seeds: list[int],
    rsl_rl_version: str,
    version_module,
    torch_module,
) -> CheckpointEvaluationResult:
    """Evaluate one checkpoint in an already-created IsaacLab/RSL-RL env."""
    runner.load(checkpoint.path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    completed = 0
    rewards: list[float] = []
    successes: list[float] = []
    hold_steps: list[float] = []
    pos_errors: list[float] = []
    orient_errors: list[float] = []
    linear_speeds: list[float] = []
    angular_speeds: list[float] = []
    clamp_fractions: list[float] = []
    valve_fractions: list[float] = []
    while completed < episodes:
        with torch_module.inference_mode():
            actions = policy(obs)
            obs, reward, dones, extras = env.step(actions)
            rewards.append(float(reward.detach().mean().cpu()))
            if version_module.parse(rsl_rl_version) >= version_module.parse("4.0.0"):
                policy.reset(dones)
        done_count = int(dones.detach().sum().cpu())
        if done_count <= 0:
            continue
        completed += done_count
        log = extras.get("log", {}) if isinstance(extras, dict) else {}
        successes.append(_extract_scalar(log, "Episode/final_pose_success"))
        hold_steps.append(_extract_scalar(log, "Episode/final_success_hold_steps"))
        pos_errors.append(_extract_scalar(log, "Episode/final_position_error"))
        orient_errors.append(_extract_scalar(log, "Episode/final_orientation_error_rad"))
        linear_speeds.append(_extract_scalar(log, "Episode/final_linear_speed"))
        angular_speeds.append(_extract_scalar(log, "Episode/final_angular_speed"))
        clamp_fractions.append(_extract_scalar(log, "Episode/clamp_activation_fraction"))
        valve_fractions.append(_extract_scalar(log, "Episode/valve_saturation_fraction"))
    required_hold = float(getattr(env.unwrapped.cfg, "final_pose_success_hold_steps", 50))
    mean_hold = _mean(hold_steps)
    return CheckpointEvaluationResult(
        checkpoint_path=checkpoint.path,
        iteration=checkpoint.iteration,
        episodes=episodes,
        evaluation_seeds=seeds,
        strict_success_rate=_mean(successes),
        hold_completion_fraction=min(mean_hold / max(required_hold, 1.0), 1.0),
        mean_max_hold_steps=mean_hold,
        final_position_error_mean=_mean(pos_errors),
        final_orientation_error_mean=_mean(orient_errors),
        final_linear_speed_mean=_mean(linear_speeds),
        final_angular_speed_mean=_mean(angular_speeds),
        clamp_activation_fraction_mean=_mean(clamp_fractions),
        valve_saturation_fraction_mean=_mean(valve_fractions),
        mean_reward=_mean(rewards),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Kevin/Mulag RSL-RL checkpoints deterministically.")
    parser.add_argument("--task", type=str, default="Kevin-Mulag-HydraulicROM-TaskSpace-v0")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--checkpoint_stride", type=int, default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    return parser


def main() -> None:
    parser = build_arg_parser()
    from isaaclab.app import AppLauncher

    import cli_args  # isort: skip

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        import importlib.metadata as metadata

        import gymnasium as gym
        import torch
        from packaging import version
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils.hydra import hydra_task_config

        import isaaclab_tasks  # noqa: F401
        import kevin_integration.tasks  # noqa: F401

        installed_version = metadata.version("rsl-rl-lib")
        candidates = select_checkpoints(
            discover_checkpoints(args_cli.run_dir),
            checkpoint_stride=args_cli.checkpoint_stride,
            explicit_checkpoints=args_cli.checkpoints,
        )

        @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
        def _run(env_cfg, agent_cfg):
            agent_cfg = prepare_rsl_rl_agent_cfg(
                agent_cfg,
                installed_version,
                version,
                handle_deprecated_rsl_rl_cfg,
            )
            env_cfg.scene.num_envs = int(args_cli.num_envs)
            env_cfg.seed = int(args_cli.seeds[0])
            agent_cfg.seed = int(args_cli.seeds[0])
            if getattr(args_cli, "device", None) is not None:
                env_cfg.sim.device = args_cli.device
                agent_cfg.device = args_cli.device
            env_cfg.log_dir = str(Path(args_cli.run_dir).expanduser().resolve())
            env = gym.make(args_cli.task, cfg=env_cfg)
            try:
                if isinstance(env.unwrapped, DirectMARLEnv):
                    env = multi_agent_to_single_agent(env)
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                results = [
                    evaluate_checkpoint_runtime(
                        checkpoint=candidate,
                        task=args_cli.task,
                        env=env,
                        runner=runner,
                        episodes=int(args_cli.episodes),
                        seeds=list(args_cli.seeds),
                        rsl_rl_version=installed_version,
                        version_module=version,
                        torch_module=torch,
                    )
                    for candidate in candidates
                ]
                paths = write_results(args_cli.run_dir, results)
                best = choose_best_checkpoint(results)
                print(f"[INFO] Evaluated {len(results)} checkpoints.")
                print(f"[INFO] Best checkpoint: {best.checkpoint_path}")
                print(f"[INFO] Wrote {paths['json']}, {paths['csv']}, {paths['best']}")
            finally:
                env.close()

        _run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
