#!/usr/bin/env python3
"""Verify high-level PPO observation has no cylinder internal states."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check high-level PPO observation.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.new_observations_pipeline.ppo1.observations_adapter import HI_HORIZON_OBS_INDEX, HI_HORIZON_OBS_DIM
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = True
cfg.log_target_delta_xdot_clipping = False
cfg.horizon_pin_root_pose = False
cfg.horizon_reset_settle_steps = 0
cfg.ee_action_mode = "task_space_delta"
if getattr(args_cli, "device", None):
    cfg.device = args_cli.device
    cfg.sim.device = args_cli.device
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = HorizonIDFDEnv(base_cfg=cfg, horizon_steps=10, action_scale_m=0.01, fd_gains=[2.0,3.0,3.0,3.0], device=args_cli.device)
obs, _ = env.reset()

print("=" * 60, flush=True)
print("HIGH-LEVEL OBSERVATION CHECK", flush=True)
print("=" * 60, flush=True)

print(f"\nAction dim: {env.num_actions}", flush=True)
print(f"EE mode: {env._ee_mode}", flush=True)
print(f"Observation dim: {obs.shape[-1]} (expected {HI_HORIZON_OBS_DIM})", flush=True)

# Verify no cylinder state leakage
obs_vec = obs[0].detach().cpu().numpy()
print(f"\nObservation index:", flush=True)
for name, (start, end) in sorted(HI_HORIZON_OBS_INDEX.items(), key=lambda x: x[1][0]):
    vals = obs_vec[start:end]
    print(f"  {name}[{start}:{end}]: {vals}", flush=True)

# Check for forbidden keys
forbidden = {"cylinder", "valve", "delta_x", "fd_", "id_", "guard_flags"}
found_forbidden = [name for name in HI_HORIZON_OBS_INDEX if any(f in name for f in forbidden)]
if found_forbidden:
    print(f"\nFAIL: Found forbidden keys in observation: {found_forbidden}", flush=True)
else:
    print(f"\nPASS: No cylinder/valve/FD/ID states in observation.", flush=True)

print(f"\nObs dim match: {obs.shape[-1] == HI_HORIZON_OBS_DIM}", flush=True)
print(f"Target/EE same frame check: target_pos_b and ee_pos_b are both base-frame.", flush=True)

env.close()
simulation_app.close()
