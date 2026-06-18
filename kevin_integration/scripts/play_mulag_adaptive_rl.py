# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local RSL-RL player for the Kevin/Mulag task-space checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import MISSING
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "/home/rrlab/IsaacLab"))
ISAACLAB_RSL_RL_SCRIPTS = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if ISAACLAB_RSL_RL_SCRIPTS.is_dir() and str(ISAACLAB_RSL_RL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_RSL_RL_SCRIPTS))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

DEFAULT_HYDRAULIC_ROM_TASK = "Kevin-Mulag-HydraulicROM-TaskSpace-v0"
EXPECTED_REGISTRATION_MODULE = "kevin_integration.tasks"

parser = argparse.ArgumentParser(description="Play a Kevin/Mulag RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=DEFAULT_HYDRAULIC_ROM_TASK, help="Gym task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time, if possible.")
parser.add_argument(
    "--best_from_run",
    type=str,
    default=None,
    help="Run directory containing best_checkpoint.json from evaluate_mulag_checkpoints.py.",
)
parser.add_argument(
    "--evaluate_best_from_run",
    action="store_true",
    default=False,
    help="Require the evaluator to be run explicitly when best_checkpoint.json is missing.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.checkpoint and args_cli.best_from_run:
    raise ValueError("Use either --checkpoint or --best_from_run, not both.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
import kevin_integration.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def _is_missing_cfg_value(value: object) -> bool:
    return isinstance(value, type(MISSING))


def _has_non_missing_attr(obj: object, attr_name: str) -> bool:
    return hasattr(obj, attr_name) and not _is_missing_cfg_value(getattr(obj, attr_name))


def _uses_current_distribution_schema(agent_cfg: RslRlBaseRunnerCfg, installed_rsl_rl_version: str) -> bool:
    """Return True when the agent config is already in the RSL-RL 5 distribution schema."""
    if version.parse(installed_rsl_rl_version) < version.parse("5.0.0"):
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


def _prepare_rsl_rl_agent_cfg(
    agent_cfg: RslRlBaseRunnerCfg,
    installed_rsl_rl_version: str,
) -> RslRlBaseRunnerCfg:
    """Apply IsaacLab migration only when the Kevin config is not already current-schema."""
    if _uses_current_distribution_schema(agent_cfg, installed_rsl_rl_version):
        print(
            "[INFO] RSL-RL agent config already uses distribution_cfg; "
            "skipping deprecated stochastic-field migration.",
            flush=True,
        )
        return agent_cfg
    return handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)


def _verify_task_registration(task_name: str | None) -> None:
    if not task_name:
        raise RuntimeError("No --task was provided for the Kevin local player.")
    try:
        spec = gym.spec(task_name)
    except Exception as exc:
        raise RuntimeError(
            "Kevin Gym task registration failed before Hydra resolution.\n"
            f"requested task: {task_name}\n"
            f"expected registration module: {EXPECTED_REGISTRATION_MODULE}\n"
            f"current working directory: {os.getcwd()}\n"
            f"Python executable: {sys.executable}\n"
            "suggested editable-install check: run `python -m pip install -e .` "
            "from the Kevin project or ensure the repository root is on PYTHONPATH."
        ) from exc
    print(
        "[INFO] Gym task registration resolved: "
        f"id={spec.id} entry_point={spec.entry_point} kwargs={sorted(spec.kwargs)}",
        flush=True,
    )


def _resolve_best_checkpoint(run_dir: str, *, evaluate_if_missing: bool = False) -> str:
    run_path = Path(run_dir).expanduser().resolve()
    best_path = run_path / "best_checkpoint.json"
    if not best_path.is_file():
        if evaluate_if_missing:
            raise RuntimeError(
                "best_checkpoint.json is missing. Run the deterministic evaluator explicitly first:\n"
                "./rrlab.sh -p kevin_integration/scripts/evaluate_mulag_checkpoints.py "
                f"--task {args_cli.task} --run_dir {run_path} --headless"
            )
        raise FileNotFoundError(
            f"Missing best checkpoint metadata: {best_path}. "
            "Run evaluate_mulag_checkpoints.py before using --best_from_run."
        )
    data = json.loads(best_path.read_text(encoding="utf-8"))
    checkpoint = data.get("checkpoint_path") or data.get("checkpoint path")
    if not checkpoint:
        raise RuntimeError(f"{best_path} does not contain checkpoint_path.")
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_path / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint path from {best_path} does not exist: {checkpoint_path}")
    print(f"[INFO] Resolved best checkpoint from {best_path}: {checkpoint_path}", flush=True)
    return str(checkpoint_path)


def _agent_architecture(agent_cfg: RslRlBaseRunnerCfg) -> dict[str, object]:
    actor = getattr(agent_cfg, "actor", None)
    critic = getattr(agent_cfg, "critic", None)
    algorithm = getattr(agent_cfg, "algorithm", None)
    return {
        "runner_class": getattr(agent_cfg, "class_name", None),
        "actor_hidden_dims": list(getattr(actor, "hidden_dims", []) or []),
        "critic_hidden_dims": list(getattr(critic, "hidden_dims", []) or []),
        "actor_activation": getattr(actor, "activation", None),
        "critic_activation": getattr(critic, "activation", None),
        "gamma": getattr(algorithm, "gamma", None),
    }


def _checkpoint_contract_payload(checkpoint_path: str) -> dict[str, object]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] Could not inspect checkpoint metadata before loading: {exc}", flush=True)
        return {}
    if not isinstance(checkpoint, dict):
        return {}
    infos = checkpoint.get("infos", {})
    if isinstance(infos, dict):
        for key in ("kevin_checkpoint_contract", "environment_contract", "env_contract"):
            value = infos.get(key)
            if isinstance(value, dict):
                return value
    for key in ("kevin_checkpoint_contract", "environment_contract", "env_contract"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _validate_checkpoint_compatibility(
    checkpoint_path: str,
    *,
    task_name: str,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    env,
) -> None:
    current = {
        "task_name": task_name,
        "observation_dimension": int(getattr(env_cfg, "observation_space", -1)),
        "action_dimension": int(getattr(env_cfg, "action_space", -1)),
        "runner_class": getattr(agent_cfg, "class_name", None),
        "policy_architecture": _agent_architecture(agent_cfg),
        "normalizer_present": bool(getattr(agent_cfg, "empirical_normalization", False))
        or bool(getattr(getattr(agent_cfg, "actor", None), "obs_normalization", False)),
        "environment_contract_version": str(getattr(env_cfg, "kevin_fixed_pose_contract_version", "")),
        "reward_contract_version": str(getattr(env_cfg, "reward_contract_version", "")),
    }
    actual_obs = env.unwrapped.get_observations()
    if isinstance(actual_obs, dict) and "policy" in actual_obs:
        current["observation_dimension"] = int(actual_obs["policy"].shape[-1])
    current["action_dimension"] = int(getattr(env.unwrapped, "num_actions", current["action_dimension"]))

    saved = _checkpoint_contract_payload(checkpoint_path)
    if not saved:
        print(
            "[WARN] Checkpoint has no Kevin environment/reward contract metadata; "
            "visualization is allowed only because tensor dimensions will be checked by RSL-RL load.",
            flush=True,
        )
        return
    strict_keys = ("task_name", "observation_dimension", "action_dimension", "runner_class")
    mismatches = []
    warnings = []
    for key in strict_keys:
        if key in saved and saved[key] != current[key]:
            mismatches.append(f"{key}: checkpoint={saved[key]!r} current={current[key]!r}")
    for key in ("policy_architecture", "normalizer_present", "environment_contract_version", "reward_contract_version"):
        if key in saved and saved[key] != current[key]:
            warnings.append(f"{key}: checkpoint={saved[key]!r} current={current[key]!r}")
    if mismatches:
        raise RuntimeError("Checkpoint is incompatible with the current task contract:\n" + "\n".join(mismatches))
    if warnings:
        print(
            "[WARN] Checkpoint contract differs from current visualization contract:\n"
            + "\n".join(warnings)
            + "\nDo not resume training silently under a changed reward or environment contract.",
            flush=True,
        )


_verify_task_registration(args_cli.task)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = _prepare_rsl_rl_agent_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.best_from_run:
        resume_path = _resolve_best_checkpoint(
            args_cli.best_from_run,
            evaluate_if_missing=bool(args_cli.evaluate_best_from_run),
        )
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        _validate_checkpoint_compatibility(
            resume_path,
            task_name=task_name,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            env=env,
        )

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during play.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        policy_nn = None
        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        else:
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        timestep = 0
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                elif policy_nn is not None:
                    policy_nn.reset(dones)
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
