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
import time

import numpy as np
import gymnasium as gym

import torch
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

parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

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


# -----------------------------------------------------------------------------
# Main (Hydra-provided env_cfg + agent_cfg)
# -----------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
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

    checkpoint_path = args_cli.checkpoint
    if checkpoint_path is None:
        raise ValueError("--checkpoint must be provided and point to student_policy.pt")

    log_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    env_cfg.log_dir = log_dir


    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    policy_space = env.unwrapped.single_observation_space["policy"]
    print(f"Original policy space: {policy_space}")
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
    
    # SB3 VecEnv wrapper
    venv = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    # (Optional) If you use VecNormalize in training and saved stats, load them here.
    # If you did NOT use VecNormalize, skip this block.
    vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        print(f"[INFO] Loading VecNormalize stats from: {vecnorm_path}")
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    # Load student policy
    student_cfg = agent_cfg.get("student", {}) or {}
    PolicyCls = import_from_string(student_cfg["policy_cls"])

    student_policy = PolicyCls.load(checkpoint_path, device=device)
    student_policy.set_training_mode(False)  # SB3 helper (puts modules in eval-like mode)
    print(f"[INFO] Loaded student policy: {type(student_policy)} on {device}")
    print(f"[INFO] checkpoint: {checkpoint_path}")

    # dt for real-time loop (IsaacLab env has step_dt on unwrapped)
    dt = getattr(venv.unwrapped, "step_dt", None)

    obs = venv.reset()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # SB3 policy.predict expects numpy/dict-of-numpy; venv gives that
            actions, _ = student_policy.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        # if args_cli.real_time and dt is not None:
        #     sleep_time = dt - (time.time() - start_time)
        #     if sleep_time > 0:
        #         time.sleep(sleep_time)

    venv.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()