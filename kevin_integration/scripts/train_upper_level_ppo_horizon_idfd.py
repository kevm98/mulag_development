#!/usr/bin/env python3
"""Train horizon-displacement PPO with ID/FD closed-loop pipeline."""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime
from pathlib import Path
import pickle
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Horizon-IDFD PPO training.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_iterations", type=int, default=200)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--horizon_action_scale_m", type=float, default=0.01)
parser.add_argument("--fd_horizon_gain_per_cylinder", type=str, default="2.0,3.0,3.0,3.0")
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""): args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np, torch
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg, MulagUpperLevelPPORunnerCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import (
    HorizonIDFDEnv,
    HorizonIDFDRslRlVecEnvWrapper,
)
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml


def dump_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(data, stream)

# ---- SAFETY GUARDS ----
env_cfg = MulagUpperLevelEnvCfg()
if env_cfg.low_level_servo_mode == "position_ramp_pi":
    raise RuntimeError("REFUSING: position_ramp_pi is diagnostic only, not for PPO training.")
if not env_cfg.use_sequence_cylinder_models:
    raise RuntimeError("REFUSING: ID/FD must be enabled for horizon PPO.")

id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
if not id_path.is_file():
    raise RuntimeError(f"Fine-tuned ID checkpoint not found at {id_path}")

fd_gains = [float(x.strip()) for x in args_cli.fd_horizon_gain_per_cylinder.split(",")]
if len(fd_gains) != 4:
    raise RuntimeError(f"fd_horizon_gain_per_cylinder must have 4 values, got {len(fd_gains)}")

print(f"SAFETY CHECKS PASSED", flush=True)
print(f"  ID checkpoint: {id_path}", flush=True)
print(f"  fd_gains: {fd_gains}", flush=True)
print(f"  horizon_steps: {args_cli.horizon_steps}", flush=True)
print(f"  action_scale: {args_cli.horizon_action_scale_m}m", flush=True)

# ---- Build env ----
env_cfg.scene.num_envs = args_cli.num_envs
if getattr(args_cli, "device", None): env_cfg.device = args_cli.device; env_cfg.sim.device = args_cli.device
apply_kevin_sim_memory_optimizations(env_cfg.sim)

base_env = HorizonIDFDEnv(
    base_cfg=env_cfg,
    horizon_steps=args_cli.horizon_steps,
    action_scale_m=args_cli.horizon_action_scale_m,
    fd_gains=fd_gains,
    device=args_cli.device,
)

env = HorizonIDFDRslRlVecEnvWrapper(base_env)
print(f"Env wrapped: {args_cli.num_envs} envs, obs={base_env.observation_space}, act={base_env.num_actions}", flush=True)

# ---- Build PPO runner using existing config ----
agent_cfg = MulagUpperLevelPPORunnerCfg()
agent_cfg.max_iterations = args_cli.max_iterations
agent_cfg.device = args_cli.device
agent_cfg.seed = 42

# Log directory
log_root = os.path.join("logs", "rsl_rl", "horizon_idfd_ppo")
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_root, run_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
print(f"Log dir: {log_dir}", flush=True)

# Save horizon config
try:
    import yaml as _yaml
    _yaml.dump({"horizon_steps": args_cli.horizon_steps,
                "action_scale": args_cli.horizon_action_scale_m,
                "fd_gains": fd_gains,
                "num_envs": args_cli.num_envs,
                "max_iterations": args_cli.max_iterations},
               open(os.path.join(log_dir, "params", "horizon_config.yaml"), "w"))
except Exception:
    pass

# Create runner using agent_cfg.to_dict() — same pattern as reference training script
runner = OnPolicyRunner(
    env,
    agent_cfg.to_dict(),
    log_dir=log_dir,
    device=agent_cfg.device,
)

if args_cli.checkpoint:
    print(f"Resuming from checkpoint: {args_cli.checkpoint}", flush=True)
    runner.load(args_cli.checkpoint)

dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

print(f"Starting PPO training: {args_cli.max_iterations} iterations...", flush=True)
try:
    with torch.inference_mode(False):
        runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    print("PPO training finished; closing simulation.", flush=True)
finally:
    env.close()
    simulation_app.close()
    print("Training complete.", flush=True)
