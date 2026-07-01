# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Train an imitation (DAgger) agent in IsaacLab using imitation + SB3 policies.

This script uses:
- IsaacLab AppLauncher (Omniverse/Kit runtime)
- Hydra task + agent config entry points (isaaclab_tasks.utils.hydra.hydra_task_config)
- SB3 teacher checkpoint (PPO.load(...).policy)
- SB3 student policy class (MultiInputActorCriticPolicy / ActorCriticPolicy / custom) built from config
- imitation.algorithms.dagger.SimpleDAggerTrainer for DAgger
- imitation.algorithms.bc.BC for supervised learning

All params are expected to come from the agent YAML via Hydra under keys:
  teacher, student, bc, dagger, logging (plus your env config handled by IsaacLab)
"""

import argparse
import signal
import sys
import os
import random
import tempfile
import importlib
import inspect
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import numpy as np
import gymnasium as gym

import torch as th
import torch.nn as nn

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI / AppLauncher / Hydra setup (IsaacLab style)
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train an imitation agent with imitation + SB3 policies in IsaacLab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="imitation_cfg_entry_point", help="Agent config entry point key.")
parser.add_argument("--seed", type=int, default=None, help="Seed override.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--keep_all_info", action="store_true", default=False, help="Use slower SB3 wrapper but keep info.")

# optional overrides (fully optional; you can keep everything in YAML)
parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Override teacher.checkpoint.")
parser.add_argument("--teacher_vecnormalize", type=str, default=None, help="Override teacher.vecnormalize_path.")
parser.add_argument("--dagger_timesteps", type=int, default=None, help="Override dagger.total_timesteps.")
parser.add_argument("--student_checkpoint", type=str, default=None, help="Override student.checkpoint.")

# Append Isaac Sim / AppLauncher args
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# IsaacLab env types/utilities
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import rrlab_tasks
# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# imitation
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

# Optional imitation logger (nice but not mandatory)
try:
    from imitation.util import logger as imit_logger
except Exception:
    imit_logger = None

def cleanup_pbar(*_args):
    """Stop training and cleanup progress bar properly on ctrl+c."""
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, cleanup_pbar)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def import_from_string(path: str):
    """Import 'module.sub:Attr' -> object."""
    module_name, attr_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_activation(act: str):
    """Convert 'nn.ELU' -> torch.nn.ELU class."""
    if isinstance(act, str) and act.startswith("nn."):
        return getattr(nn, act[3:])
    raise ValueError(f"Unsupported activation_fn: {act}")


def make_lr_schedule(cfg: dict):
    """Build SB3 Schedule callable from config."""
    t = str(cfg.get("type", "constant")).lower()
    v = float(cfg.get("value", 3e-4))
    if t == "constant":
        return lambda _: v
    if t == "linear":
        return lambda progress_remaining: progress_remaining * v
    raise ValueError(f"Unknown lr_schedule type: {t}")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def setup_run_dirs(task_name: str, agent_cfg: dict) -> dict:
    """Create run/log folders based on config + timestamp."""
    log_cfg = agent_cfg.get("logging", {}) or {}

    root_dir = log_cfg.get("root_dir", os.path.join("logs", "dagger"))
    run_name = log_cfg.get("run_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    log_root_path = os.path.abspath(os.path.join(root_dir, task_name))
    log_dir = os.path.join(log_root_path, run_name)

    params_dir = ensure_dir(os.path.join(log_dir, "params"))
    scratch_dir = ensure_dir(os.path.join(log_dir, log_cfg.get("scratch_subdir", "dagger_scratch")))
    models_dir = ensure_dir(os.path.join(log_dir, log_cfg.get("models_subdir", "models")))
    tb_dir = ensure_dir(os.path.join(log_dir, log_cfg.get("tb_subdir", "tb")))

    return {
        "log_dir": log_dir,
        "params_dir": params_dir,
        "scratch_dir": scratch_dir,
        "models_dir": models_dir,
        "tb_dir": tb_dir,
    }


def maybe_configure_imitation_logger(agent_cfg: dict, tb_dir: str):
    """Configure imitation logger to write to tensorboard/stdout if requested."""
    log_cfg = agent_cfg.get("logging", {}) or {}
    use_im_logger = bool(log_cfg.get("use_imitation_logger", False))
    if not use_im_logger:
        return
    if imit_logger is None:
        print("[WARN] imitation logger not available; skipping imitation logger configuration.")
        return

    formats = log_cfg.get("imitation_logger_formats", ["stdout"])
    # mimic imitation's logger.configure API
    imit_logger.configure(folder=tb_dir, format_strs=formats)
    print(f"[INFO] imitation logger configured at: {tb_dir} with formats={formats}")


def maybe_wrap_vecnormalize(venv, vecnorm_path: str | None, training: bool = True, norm_reward: bool = False):
    """Load VecNormalize stats if provided."""
    if not vecnorm_path:
        return venv
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = bool(training)
    venv.norm_reward = bool(norm_reward)
    return venv


def build_student_policy(venv, student_cfg: dict, device: str):
    """
    Build SB3 policy instance from config.

    student_cfg:
      policy_cls: "stable_baselines3.ppo.policies:MultiInputActorCriticPolicy"
      policy_kwargs: {...}  # passed to policy ctor, with conversions:
          lr_schedule: {type, value} -> callable
          activation_fn: "nn.ELU" -> nn.ELU class
          features_extractor_class: "module:Class" -> class
    """
    PolicyCls = import_from_string(student_cfg["policy_cls"])
    pk = dict(student_cfg.get("policy_kwargs", {}) or {})

    # lr_schedule (required by SB3 policy ctor)
    if "lr_schedule" in pk and isinstance(pk["lr_schedule"], dict):
        pk["lr_schedule"] = make_lr_schedule(pk["lr_schedule"])
    elif "lr_schedule" not in pk:
        # SB3 policy requires lr_schedule, so provide a safe default
        pk["lr_schedule"] = make_lr_schedule({"type": "constant", "value": 3e-4})

    # activation_fn
    if "activation_fn" in pk and isinstance(pk["activation_fn"], str):
        pk["activation_fn"] = resolve_activation(pk["activation_fn"])

    # features extractor class
    if "features_extractor_class" in pk and isinstance(pk["features_extractor_class"], str):
        pk["features_extractor_class"] = import_from_string(pk["features_extractor_class"])

    # Ensure extractor kwargs is plain dict (Hydra may give OmegaConf containers)
    if "features_extractor_kwargs" in pk and pk["features_extractor_kwargs"] is not None:
        pk["features_extractor_kwargs"] = dict(pk["features_extractor_kwargs"])

    policy = PolicyCls(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        **pk,
    )
    policy = policy.to(th.device(device))
    return policy


def load_teacher_policy(venv, teacher_cfg: dict, device: str):
    """
    Load SB3 teacher algorithm from checkpoint and return policy (BasePolicy).

    teacher_cfg:
      checkpoint: "/path/model.zip"
      algo_cls: "stable_baselines3:PPO"  # optional
      use_policy_only: true
    """
    if "checkpoint" not in teacher_cfg or teacher_cfg["checkpoint"] is None:
        raise ValueError("teacher.checkpoint must be set in agent config (or via --teacher_checkpoint).")

    AlgoCls = import_from_string(teacher_cfg.get("algo_cls", "stable_baselines3:PPO"))
    algo = AlgoCls.load(teacher_cfg["checkpoint"], env=venv, device=device)

    # Return BasePolicy for SimpleDAggerTrainer signature
    if bool(teacher_cfg.get("use_policy_only", True)):
        return algo.policy
    return algo  # works in rollout helper, but may violate type hints


def filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    """Drop kwargs not accepted by fn (prevents 'unexpected keyword' errors)."""
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    # allow **kwargs
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}


def resolve_bc_kwargs(bc_kwargs: dict) -> dict:
    """Resolve import strings inside bc.kwargs."""
    out = dict(bc_kwargs or {})

    if "device" in out and out["device"] is None:
        out.pop("device")

    if "optimizer_cls" in out and isinstance(out["optimizer_cls"], str):
        out["optimizer_cls"] = import_from_string(out["optimizer_cls"])

    if "optimizer_kwargs" in out and out["optimizer_kwargs"] is not None:
        out["optimizer_kwargs"] = dict(out["optimizer_kwargs"])

    float_keys = ("l2_weight", "ent_weight", "grad_clip_norm")
    int_keys = ("batch_size", "minibatch_size", "log_interval")

    for k in float_keys:
        if k in out and out[k] is not None:
            out[k] = float(out[k])

    for k in int_keys:
        if k in out and out[k] is not None:
            out[k] = int(out[k])

    return out


# -----------------------------------------------------------------------------
# Main (Hydra-provided env_cfg + agent_cfg)
# -----------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    # -----------------------------
    # seed + cli overrides
    # -----------------------------
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    seed = int(agent_cfg.get("seed", 0))

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.seed = seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    device = args_cli.device if args_cli.device is not None else agent_cfg.get("device", "cuda")

    # CLI overrides for teacher/dagger
    if args_cli.teacher_checkpoint is not None:
        agent_cfg.setdefault("teacher", {})
        agent_cfg["teacher"]["checkpoint"] = args_cli.teacher_checkpoint
    if args_cli.teacher_vecnormalize is not None:
        agent_cfg.setdefault("teacher", {})
        agent_cfg["teacher"]["vecnormalize_path"] = args_cli.teacher_vecnormalize
    if args_cli.dagger_timesteps is not None:
        agent_cfg.setdefault("dagger", {})
        agent_cfg["dagger"]["total_timesteps"] = args_cli.dagger_timesteps

    # -----------------------------
    # logging directories
    # -----------------------------
    dirs = setup_run_dirs(args_cli.task, agent_cfg)
    log_dir = dirs["log_dir"]
    params_dir = dirs["params_dir"]
    scratch_dir = dirs["scratch_dir"]
    models_dir = dirs["models_dir"]
    tb_dir = dirs["tb_dir"]

    print(f"[INFO] Logging directory: {log_dir}")

    # dump configs
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(params_dir, "env.pkl"), env_cfg)
    dump_pickle(os.path.join(params_dir, "agent.pkl"), agent_cfg)

    # save command
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    maybe_configure_imitation_logger(agent_cfg, tb_dir)

    # set env log_dir (IsaacLab uses this internally)
    env_cfg.log_dir = log_dir

    # IO descriptors export flag
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    # -----------------------------
    # Create IsaacLab env
    # -----------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    policy_space = env.unwrapped.single_observation_space["policy"]
    policy_space.spaces["height_map"] = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=policy_space.spaces["height_map"].shape,
        dtype=policy_space.spaces["height_map"].dtype,
    )
    policy_space.spaces["camera"] = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=policy_space.spaces["camera"].shape,
        dtype=policy_space.spaces["camera"].dtype,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "dagger"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap as SB3 VecEnv
    venv = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    # -----------------------------
    # VecNormalize (load from teacher cfg if present)
    # -----------------------------
    teacher_cfg = agent_cfg.get("teacher", {}) or {}
    # print("teacher_cfg:", teacher_cfg)
    # vecnorm_path = teacher_cfg.get("vecnormalize_path", None)

    # DAgger collects data; use training=True typically, but disable reward norm
    vecnorm_training = True
    vecnorm_norm_reward = False

    # venv = maybe_wrap_vecnormalize(
    #     venv,
    #     vecnorm_path=vecnorm_path,
    #     training=vecnorm_training,
    #     norm_reward=vecnorm_norm_reward,
    # )

    # -----------------------------
    # Build teacher policy (BasePolicy)
    # -----------------------------
    teacher_policy = load_teacher_policy(venv, teacher_cfg, device=device)
    # print(f"[INFO] Teacher policy type: {type(teacher_policy)}")

    # Optionally export teacher policy as .pt for easier reuse
    if bool(teacher_cfg.get("export_policy_pt", False)):
        teacher_policy_path = os.path.join(models_dir, "teacher_policy.pt")
        teacher_policy.save(teacher_policy_path)
        print(f"[INFO] Saved teacher policy to: {teacher_policy_path}")

    # -----------------------------
    # Build student policy (SB3 policy instance)
    # -----------------------------
    student_cfg = agent_cfg.get("student", {}) or {}
    student_policy = build_student_policy(venv, student_cfg, device=device)
    print(f"[INFO] Student policy type: {type(student_policy)}")
    # print("student_cfg:", student_cfg) 
    # print("student_policy obs:", student_policy.observation_space)
    if args_cli.student_checkpoint is not None:
        print(f"[INFO] Loading student model checkpoint from: {args_cli.student_checkpoint}")
        student_policy = student_policy.load(args_cli.student_checkpoint, device=device)

    # -----------------------------
    # Build BC trainer (from bc: section)
    # -----------------------------
    bc_cfg = agent_cfg.get("bc", {}) or {}
    # print("bc_cfg:", bc_cfg)

    # Start from bc_cfg directly (YAML style), not bc_cfg["kwargs"]
    bc_kwargs = dict(bc_cfg)

    # Convert import strings to real classes (optimizer_cls etc.)
    bc_kwargs = resolve_bc_kwargs(bc_kwargs)

    # Filter to match your installed imitation version
    bc_kwargs = filter_kwargs_for_callable(bc.BC.__init__, bc_kwargs)

    # print("bc_kwargs:", bc_kwargs)

    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=student_policy,
        rng=np.random.default_rng(seed),
        **bc_kwargs,
    )

    # -----------------------------
    # DAgger training (from dagger: section)
    # -----------------------------
    dagger_cfg = agent_cfg.get("dagger", {}) or {}
    # print("dagger_cfg:", dagger_cfg)

    total_timesteps = int(dagger_cfg.get("total_timesteps", 2000))
    rollout_round_min_episodes = int(dagger_cfg.get("rollout_round_min_episodes", 3))
    rollout_round_min_timesteps = int(dagger_cfg.get("rollout_round_min_timesteps", 500))

    # Optional BC.train kwargs (logging/eval/progress bar/etc.)
    bc_train_kwargs = dict(dagger_cfg.get("bc_train_kwargs", {}) or {})

    # Filter to match your imitation version
    bc_train_kwargs = filter_kwargs_for_callable(bc.BC.train, bc_train_kwargs)

    # print("bc_train_kwargs:", bc_train_kwargs)

    dagger_trainer = SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=scratch_dir,
        expert_policy=teacher_policy,
        bc_trainer=bc_trainer,
        rng=np.random.default_rng(seed),
    )

    print(
        f"[INFO] Starting DAgger training: total_timesteps={total_timesteps}, "
        f"rollout_round_min_episodes={rollout_round_min_episodes}, "
        f"rollout_round_min_timesteps={rollout_round_min_timesteps}, "
        f"scratch_dir={scratch_dir}"
    )

    save_every_rounds = int(dagger_cfg.get("save_every_rounds", 0) or 0)
    save_on_last_round = bool(dagger_cfg.get("save_on_last_round", True))
    save_prefix = str(dagger_cfg.get("save_prefix", "dagger"))

    dagger_trainer.train(
        total_timesteps=total_timesteps,
        rollout_round_min_episodes=rollout_round_min_episodes,
        rollout_round_min_timesteps=rollout_round_min_timesteps,
        bc_train_kwargs=bc_train_kwargs if len(bc_train_kwargs) > 0 else None,
        save_every_rounds=save_every_rounds,
        save_on_last_round=save_on_last_round,
        save_prefix=save_prefix,
    )

    # (Optional) Save DAgger trainer snapshot + policy into scratch_dir (creates checkpoint-*.pt, policy-*.pt)
    if hasattr(dagger_trainer, "save_trainer"):
        ckpt_path, pol_path = dagger_trainer.save_trainer()
        print(f"[INFO] Saved DAgger snapshot: {ckpt_path}")
        print(f"[INFO] Saved DAgger policy: {pol_path}")

    # -----------------------------
    # Save student artifacts (your canonical export)
    # -----------------------------
    student_policy_path = os.path.join(models_dir, "student_policy.pt")
    student_policy.save(student_policy_path)
    print(f"[INFO] Saved student policy to: {student_policy_path}")


    # Save VecNormalize stats if active
    if isinstance(venv, VecNormalize):
        vecnorm_out = os.path.join(models_dir, "vecnormalize.pkl")
        venv.save(vecnorm_out)
        print(f"[INFO] Saved VecNormalize stats to: {vecnorm_out}")

    # close env
    venv.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
