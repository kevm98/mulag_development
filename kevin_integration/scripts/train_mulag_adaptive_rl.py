from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Train the Mulag adaptive RL joint-reaching policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of vectorized Isaac Lab environments.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--seed", type=int, default=None, help="Training seed. Use -1 for a random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override PPO training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL checkpoint path to resume from.")
parser.add_argument("--control_mode", choices=["position_delta", "effort"], default=None, help="Override env control mode.")
parser.add_argument(
    "--position_delta_source",
    choices=["direct_action", "id_valve"],
    default=None,
    help="Override position-delta command source.",
)
parser.add_argument("--use_fd_effort", action="store_true", default=None, help="Run FD and apply effort targets.")
parser.add_argument("--use_sd_feedback", action="store_true", default=None, help="Enable learned SD feedback.")
parser.add_argument("--max_abs_torque", type=float, default=None, help="Override FD torque clamp.")
parser.add_argument("--torque_ramp_steps", type=int, default=None, help="Override effort torque ramp length.")
parser.add_argument("--torque_safety_scale", type=float, default=None, help="Override effort torque safety multiplier.")
parser.add_argument("--action_to_qddot_scale", type=float, default=None, help="Override action-to-qddot scale.")
parser.add_argument("--qddot_cmd_clip", type=float, default=None, help="Override qddot command clamp.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

# Isaac Lab version compatibility fix:
# Some Isaac Lab versions do not expose dump_pickle / dump_yaml from isaaclab.utils.io.
try:
    from isaaclab.utils.io import dump_pickle, dump_yaml
except ImportError:
    import pickle
    import yaml

    def dump_pickle(filename, obj):
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    def dump_yaml(filename, obj):
        with open(filename, "w") as f:
            try:
                if hasattr(obj, "to_dict"):
                    obj = obj.to_dict()
                yaml.safe_dump(obj, f, sort_keys=False)
            except Exception:
                # Fallback for Isaac Lab config objects that are not directly YAML serializable.
                yaml.safe_dump(str(obj), f, sort_keys=False)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import kevin_integration.tasks  # noqa: F401


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if args_cli.control_mode is not None:
        env_cfg.control_mode = args_cli.control_mode
    if args_cli.position_delta_source is not None:
        env_cfg.position_delta_source = args_cli.position_delta_source
    if args_cli.use_fd_effort is not None:
        env_cfg.use_fd_effort = args_cli.use_fd_effort
    if args_cli.use_sd_feedback is not None:
        env_cfg.use_sd_feedback = args_cli.use_sd_feedback
    if args_cli.max_abs_torque is not None:
        env_cfg.max_abs_torque = args_cli.max_abs_torque
    if args_cli.torque_ramp_steps is not None:
        env_cfg.torque_ramp_steps = args_cli.torque_ramp_steps
    if args_cli.torque_safety_scale is not None:
        env_cfg.torque_safety_scale = args_cli.torque_safety_scale
    if args_cli.action_to_qddot_scale is not None:
        env_cfg.action_to_qddot_scale = args_cli.action_to_qddot_scale
    if args_cli.qddot_cmd_clip is not None:
        env_cfg.qddot_cmd_clip = args_cli.qddot_cmd_clip


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    _apply_env_overrides(env_cfg)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"

    log_dir = os.path.join(log_root_path, run_name)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env_cfg.log_dir = log_dir

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    try:
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env)

        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )

        if args_cli.checkpoint is not None:
            print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
            runner.load(args_cli.checkpoint)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

        with torch.inference_mode(False):
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
            )

    finally:
        env.close()  # always close env before simulation app


if __name__ == "__main__":
    try:
        main() # type: ignore
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise
    finally:
        simulation_app.close()  # always runs after env.close() inside main()
