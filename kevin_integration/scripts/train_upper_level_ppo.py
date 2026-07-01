#!/usr/bin/env python3
"""Train MulagUpperLevelEnv with position-ramp PI lower-level servo."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]
if getattr(args_cli, "device", None) in (None, ""): args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg, MulagUpperLevelPPORunnerCfg
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

env_cfg = MulagUpperLevelEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.low_level_servo_mode = "position_ramp_pi"
env_cfg.pos_pi_kp = [3.0, 1.5, 3.0, 3.0]
env_cfg.pos_pi_ki = [1.5, 0.8, 1.5, 1.5]
env_cfg.pos_pi_kd = [0.1, 0.05, 0.1, 0.1]
env_cfg.pos_pi_integral_limit_m = 0.02
env_cfg.pos_pi_delta_clip_m = 0.01
env_cfg.enable_scripted_vehicle_motion = False
env_cfg.enable_startup_summary = True
if getattr(args_cli, "device", None): env_cfg.device = args_cli.device; env_cfg.sim.device = args_cli.device
apply_kevin_sim_memory_optimizations(env_cfg.sim)

agent_cfg = MulagUpperLevelPPORunnerCfg()
agent_cfg.max_iterations = args_cli.max_iterations
if args_cli.checkpoint:
    agent_cfg.resume = True
    agent_cfg.load_checkpoint = args_cli.checkpoint

print(f"Creating {args_cli.num_envs} envs with position_ramp_pi servo...")
env = MulagUpperLevelEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
runner = agent_cfg.to_runner(env, device=args_cli.device)
print(f"PPO ready. {runner.max_iterations} iterations. Run runner.learn() to start.")
runner.learn(num_learning_iterations=runner.max_iterations, init_at_random_ep_len=True)
env.close()
simulation_app.close()
