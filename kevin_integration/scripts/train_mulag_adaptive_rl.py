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

from kevin_integration.rl.torque_adapter import (  # noqa: E402
    TORQUE_ADAPTER_MODES,
    TORQUE_ADAPTER_PRESETS,
    resolve_torque_adapter_scale,
)

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
parser.add_argument("--use_sd_observer", action="store_true", default=None, help="Run SD as a passive observer.")
parser.add_argument("--use_sd_feedback", action="store_true", default=None, help="Enable learned SD feedback.")
parser.add_argument("--sd_feedback_beta", type=float, default=None, help="Blend factor for SD feedback.")
parser.add_argument("--sd_dP_rate_limit", type=float, default=None, help="Per-step SD dP feedback rate limit.")
parser.add_argument("--sd_fnet_rate_limit", type=float, default=None, help="Per-step SD Fnet feedback rate limit.")
parser.add_argument(
    "--use_reduced_hydraulic_rom",
    action="store_true",
    default=None,
    help="Enable safe reduced hydraulic ROM position-target mode.",
)
parser.add_argument("--rom_use_sd_feedback", action="store_true", default=None, help="Use SD dP/Fnet as ROM input.")
parser.add_argument(
    "--rom_use_id_valve_cmd",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Use learned ID valve command as ROM input; disable to feed policy actions directly as valve commands.",
)
parser.add_argument("--rom_dt", type=float, default=None, help="Override reduced hydraulic ROM integration dt.")
parser.add_argument("--max_abs_torque", type=float, default=None, help="Override FD torque clamp.")
parser.add_argument("--fd_residual_alpha", type=float, default=None, help="Override FD residual gain.")
parser.add_argument("--torque_ramp_steps", type=int, default=None, help="Override effort torque ramp length.")
parser.add_argument("--torque_safety_scale", type=float, default=None, help="Override effort torque safety multiplier.")
parser.add_argument(
    "--torque_adapter_mode",
    choices=TORQUE_ADAPTER_MODES,
    default=None,
    help="Override Simscape-to-Isaac torque adapter mode.",
)
parser.add_argument(
    "--torque_adapter_preset",
    choices=TORQUE_ADAPTER_PRESETS,
    default=None,
    help="Override named torque adapter scale preset.",
)
parser.add_argument("--action_to_qddot_scale", type=float, default=None, help="Override action-to-qddot scale.")
parser.add_argument("--qddot_cmd_clip", type=float, default=None, help="Override qddot command clamp.")
parser.add_argument("--use_q_ref_ramp", action="store_true", default=False, help="Enable q_ref command ramping.")
parser.add_argument("--disable_q_ref_ramp", action="store_true", default=False, help="Disable q_ref command ramping.")
parser.add_argument("--q_ref_ramp_time_s", type=float, default=None, help="Override q_ref commanded ramp time.")
parser.add_argument("--use_q_ref_curriculum", action="store_true", default=False, help="Enable q_ref target curriculum.")
parser.add_argument(
    "--disable_q_ref_curriculum",
    action="store_true",
    default=False,
    help="Disable q_ref target curriculum.",
)
parser.add_argument("--q_ref_target_scale", type=float, default=None, help="Use a fixed q_ref target scale.")
parser.add_argument("--q_ref_target_scale_initial", type=float, default=None, help="Initial q_ref curriculum scale.")
parser.add_argument("--q_ref_target_scale_final", type=float, default=None, help="Final q_ref curriculum scale.")
parser.add_argument("--q_ref_curriculum_steps", type=int, default=None, help="Steps over which q_ref scale increases.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Override episode duration.")

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
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if args_cli.control_mode is not None:
        env_cfg.control_mode = args_cli.control_mode
    if args_cli.position_delta_source is not None:
        env_cfg.position_delta_source = args_cli.position_delta_source
    if args_cli.use_fd_effort:
        env_cfg.use_fd_effort = True
    if args_cli.max_abs_torque is not None:
        env_cfg.max_abs_torque = args_cli.max_abs_torque
    if args_cli.fd_residual_alpha is not None:
        env_cfg.fd_residual_alpha = args_cli.fd_residual_alpha
        env_cfg.use_fd_residual_alpha = True
    if args_cli.use_sd_observer:
        env_cfg.use_sd_observer = True
    if args_cli.use_sd_feedback:
        env_cfg.use_sd_observer = True
        env_cfg.use_sd_feedback = True
    if args_cli.sd_feedback_beta is not None:
        env_cfg.sd_feedback_beta = args_cli.sd_feedback_beta
    if args_cli.sd_dP_rate_limit is not None:
        env_cfg.sd_dP_rate_limit = args_cli.sd_dP_rate_limit
    if args_cli.sd_fnet_rate_limit is not None:
        env_cfg.sd_fnet_rate_limit = args_cli.sd_fnet_rate_limit
    if args_cli.use_reduced_hydraulic_rom is not None:
        env_cfg.use_reduced_hydraulic_rom = args_cli.use_reduced_hydraulic_rom
        if args_cli.use_reduced_hydraulic_rom:
            env_cfg.use_fd_effort = False
            env_cfg.control_mode = "position_delta"
    if args_cli.rom_use_sd_feedback is not None:
        env_cfg.rom_use_sd_feedback = args_cli.rom_use_sd_feedback
    if args_cli.rom_use_id_valve_cmd is not None:
        env_cfg.rom_use_id_valve_cmd = args_cli.rom_use_id_valve_cmd
    if args_cli.rom_dt is not None:
        env_cfg.rom_dt = args_cli.rom_dt
    if args_cli.torque_adapter_preset is not None:
        env_cfg.torque_adapter_preset = args_cli.torque_adapter_preset
        env_cfg.torque_adapter_scale = resolve_torque_adapter_scale(args_cli.torque_adapter_preset)
    if args_cli.torque_adapter_mode is not None:
        env_cfg.torque_adapter_mode = args_cli.torque_adapter_mode
    if args_cli.torque_ramp_steps is not None:
        env_cfg.torque_ramp_steps = args_cli.torque_ramp_steps
    if args_cli.torque_safety_scale is not None:
        env_cfg.torque_safety_scale = args_cli.torque_safety_scale
    if args_cli.action_to_qddot_scale is not None:
        env_cfg.action_to_qddot_scale = args_cli.action_to_qddot_scale
    if args_cli.qddot_cmd_clip is not None:
        env_cfg.qddot_cmd_clip = args_cli.qddot_cmd_clip
    if args_cli.disable_q_ref_ramp:
        env_cfg.use_q_ref_ramp = False
    elif args_cli.use_q_ref_ramp:
        env_cfg.use_q_ref_ramp = True
    if args_cli.q_ref_ramp_time_s is not None:
        env_cfg.q_ref_ramp_time_s = args_cli.q_ref_ramp_time_s
    if args_cli.disable_q_ref_curriculum:
        env_cfg.use_q_ref_curriculum = False
    elif args_cli.use_q_ref_curriculum:
        env_cfg.use_q_ref_curriculum = True
    if args_cli.q_ref_target_scale is not None:
        env_cfg.q_ref_target_scale_initial = args_cli.q_ref_target_scale
        env_cfg.q_ref_target_scale_final = args_cli.q_ref_target_scale
        env_cfg.use_q_ref_curriculum = False
    if args_cli.q_ref_target_scale_initial is not None:
        env_cfg.q_ref_target_scale_initial = args_cli.q_ref_target_scale_initial
    if args_cli.q_ref_target_scale_final is not None:
        env_cfg.q_ref_target_scale_final = args_cli.q_ref_target_scale_final
    if args_cli.q_ref_curriculum_steps is not None:
        env_cfg.q_ref_curriculum_steps = args_cli.q_ref_curriculum_steps
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s


def _print_adaptive_control_config(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    print(
        "[INFO] Adaptive control config: "
        f"control_mode={env_cfg.control_mode} "
        f"use_fd_effort={env_cfg.use_fd_effort} "
        f"max_abs_torque={env_cfg.max_abs_torque} "
        f"fd_residual_alpha={env_cfg.fd_residual_alpha} "
        f"use_sd_observer={env_cfg.use_sd_observer} "
        f"use_sd_feedback={env_cfg.use_sd_feedback} "
        f"sd_feedback_beta={env_cfg.sd_feedback_beta} "
        f"sd_dP_rate_limit={env_cfg.sd_dP_rate_limit} "
        f"sd_fnet_rate_limit={env_cfg.sd_fnet_rate_limit} "
        f"use_reduced_hydraulic_rom={env_cfg.use_reduced_hydraulic_rom} "
        f"rom_output_mode={env_cfg.rom_output_mode} "
        f"rom_use_sd_feedback={env_cfg.rom_use_sd_feedback} "
        f"rom_use_id_valve_cmd={env_cfg.rom_use_id_valve_cmd} "
        f"rom_dt={env_cfg.rom_dt} "
        f"rom_xdot_max={env_cfg.rom_xdot_max} "
        f"rom_dx_max={env_cfg.rom_dx_max} "
        f"qddot_cmd_scale={env_cfg.action_to_qddot_scale} "
        f"use_joint5_coupling={env_cfg.use_joint5_coupling} "
        f"joint5_coupling_source_joint={env_cfg.joint5_coupling_source_joint} "
        f"joint5_coupling_target_joint={env_cfg.joint5_coupling_target_joint} "
        f"joint5_coupling_gain={env_cfg.joint5_coupling_gain} "
        f"joint5_coupling_bias={env_cfg.joint5_coupling_bias} "
        f"q_tracking_reward_joint_weights={getattr(env_cfg, 'q_tracking_reward_joint_weights', env_cfg.q_tracking_joint_weights)} "
        f"q_tracking_debug_joint_weights={getattr(env_cfg, 'q_tracking_debug_joint_weights', env_cfg.q_tracking_joint_weights)} "
        f"rew_q_tracking={env_cfg.rew_q_tracking} "
        f"rew_action_saturation={env_cfg.rew_action_saturation} "
        f"action_saturation_threshold={env_cfg.action_saturation_threshold} "
        f"rew_action_magnitude={env_cfg.rew_action_magnitude} "
        f"rew_action_rate={env_cfg.rew_action_rate} "
        f"rew_progress={env_cfg.rew_progress} "
        f"progress_ema_alpha={env_cfg.progress_ema_alpha} "
        f"use_settle_tracking_reward={env_cfg.use_settle_tracking_reward} "
        f"settle_ramp_progress_threshold={env_cfg.settle_ramp_progress_threshold} "
        f"rew_settle_tracking={env_cfg.rew_settle_tracking} "
        f"use_joint1_settle_tracking_reward={env_cfg.use_joint1_settle_tracking_reward} "
        f"rew_joint1_settle_tracking={env_cfg.rew_joint1_settle_tracking} "
        f"joint1_settle_error_deadband={env_cfg.joint1_settle_error_deadband} "
        f"use_q_ref_ramp={env_cfg.use_q_ref_ramp} "
        f"q_ref_ramp_time_s={env_cfg.q_ref_ramp_time_s} "
        f"q_ref_ramp_profile={env_cfg.q_ref_ramp_profile} "
        f"use_q_ref_curriculum={env_cfg.use_q_ref_curriculum} "
        f"q_ref_target_scale_initial={env_cfg.q_ref_target_scale_initial} "
        f"q_ref_target_scale_final={env_cfg.q_ref_target_scale_final} "
        f"q_ref_curriculum_steps={env_cfg.q_ref_curriculum_steps} "
        f"q_ref_target_amplitude={env_cfg.q_ref_target_amplitude} "
        f"q_ref_sample_relative_to_current={env_cfg.q_ref_sample_relative_to_current} "
        f"episode_length_s={env_cfg.episode_length_s} "
        f"torque_adapter_preset={env_cfg.torque_adapter_preset} "
        f"torque_adapter_mode={env_cfg.torque_adapter_mode}",
        flush=True,
    )


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
    _print_adaptive_control_config(env_cfg)
    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

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
