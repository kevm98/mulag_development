from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kevin_integration.rl.torque_adapter import (  # noqa: E402
    DEFAULT_TORQUE_ADAPTER_PRESET,
    TORQUE_ADAPTER_MODES,
    TORQUE_ADAPTER_PRESETS,
    resolve_torque_adapter_scale,
)

parser = argparse.ArgumentParser(description="Evaluate Mulag adaptive RL reward/debug diagnostics.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional RSL-RL checkpoint path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--steps", type=int, default=1000, help="Number of policy steps to run.")
parser.add_argument("--print_interval", type=int, default=50, help="Step interval for terminal diagnostics.")
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
parser.add_argument(
    "--torque_adapter_scale",
    type=str,
    default=None,
    help='Override per-joint scale as comma-separated values, e.g. "0.0001,0.001,0.0005,0.0001".',
)
parser.add_argument("--fd_torque_scale", type=float, default=None, help="Override FD scale for tanh_squash mode.")
parser.add_argument(
    "--use_fd_residual_alpha",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable/disable deterministic residual authority scaling after the torque adapter.",
)
parser.add_argument("--fd_residual_alpha", type=float, default=None, help="Override FD residual gain.")
parser.add_argument("--torque_rate_limit", type=float, default=None, help="Override per-step torque rate limit.")
parser.add_argument("--torque_lowpass_alpha", type=float, default=None, help="Override torque low-pass alpha.")
parser.add_argument("--action_to_qddot_scale", type=float, default=None, help="Override action-to-qddot scale.")
parser.add_argument("--qddot_cmd_clip", type=float, default=None, help="Override qddot command clamp.")
parser.add_argument("--q_ref_ramp_time_s", type=float, default=None, help="Override q_ref commanded ramp time.")
parser.add_argument("--q_ref_target_scale", type=float, default=None, help="Use a fixed q_ref target scale.")
parser.add_argument("--disable_q_ref_ramp", action="store_true", default=False, help="Disable q_ref command ramping.")
parser.add_argument(
    "--disable_q_ref_curriculum",
    action="store_true",
    default=False,
    help="Disable q_ref target-scale curriculum.",
)
parser.add_argument("--episode_length_s", type=float, default=None, help="Override episode duration.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.torque_adapter_preset is not None or args_cli.torque_adapter_scale is not None:
    args_cli.torque_adapter_scale = resolve_torque_adapter_scale(
        args_cli.torque_adapter_preset or DEFAULT_TORQUE_ADAPTER_PRESET,
        args_cli.torque_adapter_scale,
    )
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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import kevin_integration.tasks  # noqa: F401
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def _tensor_to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(value)


def _debug_flag(value: torch.Tensor | str | bool | int | float) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(_tensor_to_float(value) > 0.5)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _format_joint_errors(value: torch.Tensor | None) -> str:
    if value is None:
        return "[]"
    values = value.detach().flatten().cpu().tolist()
    return "[" + ", ".join(f"{item:.4f}" for item in values) + "]"


def _format_debug_line(
    *,
    step: int,
    reward: torch.Tensor,
    dones: torch.Tensor,
    debug: dict[str, torch.Tensor | str],
    control_mode: str,
    position_delta_source: str,
    use_sd_observer: bool,
    use_sd_feedback: bool,
    use_fd_effort: bool,
) -> str:
    values = {
        "step_reward": _tensor_to_float(reward),
        "done_frac": _tensor_to_float(dones.float()),
    }
    for key in (
        "q_error_mean",
        "q_error_mean_weighted",
        "q_error_mean_unweighted",
        "episode_step",
        "episode_length_buf",
        "reset_buf_mean",
        "done_buf_mean",
        "time_out_frac",
        "terminated_frac",
        "target_changed_flag",
        "target_change_step",
        "target_hold_steps",
        "last_reset_step_global",
        "reset_due_to_timeout",
        "reset_due_to_termination",
        "reset_due_to_manual_target_change",
        "q_ref_delta_abs_max",
        "q_ref_target_delta_abs_max",
        "q_ref_ramp_progress",
        "q_ref_target_scale",
        "q_ref_ramp_time_s",
        "q_ref_curriculum_progress",
        "q_ref_jump_fraction",
        "joint1_q_actual",
        "joint1_q_ref",
        "joint1_q_error_signed",
        "joint1_q_error_abs",
        "joint1_q_ref_delta_abs",
        "joint1_q_tracking_contribution",
        "is_settle_phase",
        "settle_tracking_reward",
        "joint1_settle_tracking_reward",
        "joint1_error_excess",
        "joint1_settle_error_deadband",
        "rew_settle_tracking",
        "rew_joint1_settle_tracking",
        "settle_ramp_progress_threshold",
        "joint5_coupling_gain",
        "joint5_coupling_bias",
        "joint5_coupling_source_joint",
        "joint5_coupling_target_joint",
        "joint4_q_ref",
        "joint5_q_ref",
        "joint5_coupled_ref",
        "joint5_coupling_error",
        "joint5_coupling_error_abs_max",
        "joint5_actual",
        "joint5_qdot",
        "joint5_tracking_reward_weight",
        "joint5_target_amplitude",
        "qdot_mean",
        "qdot_abs_max",
        "action_mean",
        "action_abs_mean",
        "action_abs_max",
        "joint1_action_abs",
        "action_delta_mean",
        "action_saturation_fraction",
        "qddot_cmd_mean",
        "joint1_qddot_cmd_abs",
        "joint1_valve_cmd_abs",
        "joint1_fd_tau_raw_abs",
        "joint1_tau_fd_adapted_abs",
        "joint1_tau_applied_abs",
        "joint1_torque_abs",
        "z_arm_mean",
        "z_arm_std",
        "valve_mean",
        "valve_max",
        "fd_simscape_mean",
        "fd_simscape_max",
        "tau_fd_adapted_mean",
        "tau_fd_adapted_max",
        "tau_applied_raw_mean",
        "tau_applied_raw_max",
        "tau_isaac_filtered_mean",
        "tau_isaac_filtered_max",
        "torque_raw_mean",
        "torque_raw_max",
        "torque_mean",
        "torque_max",
        "torque_adapter_clamp_fraction",
        "torque_clamp_fraction",
        "torque_rate_limited_fraction",
        "torque_rate_limit",
        "torque_lowpass_alpha",
        "fd_residual_alpha",
        "max_abs_torque",
        "qddot_cmd_scale",
        "rew_q_tracking",
        "rew_action_saturation",
        "rew_action_magnitude",
        "rew_action_rate",
        "rew_progress",
        "rew_settle_tracking",
        "rew_joint1_settle_tracking",
        "settle_ramp_progress_threshold",
        "joint1_settle_error_deadband",
        "action_saturation_threshold",
        "progress_ema_alpha",
        "sd_feedback_beta",
        "sd_dP_mean",
        "sd_dP_abs_max",
        "sd_fnet_mean",
        "sd_fnet_abs_max",
        "dP_mean",
        "dP_abs_max",
        "fnet_mean",
        "fnet_abs_max",
        "sd_feedback_dP_delta_mean",
        "sd_feedback_dP_delta_max",
        "sd_feedback_fnet_delta_mean",
        "sd_feedback_fnet_delta_max",
        "sd_feedback_rate_limited_fraction",
        "sd_feedback_clamped_fraction",
        "sd_output_nan_or_inf",
        "sd_output_clamped_fraction",
        "sd_feedback_nan_or_inf",
        "rom_dt",
        "rom_valve_cmd_mean",
        "rom_valve_cmd_abs_max",
        "rom_dP_mean",
        "rom_P_L_mean",
        "rom_P_L_abs_max",
        "rom_xdot_pred_mean",
        "rom_xdot_pred_abs_max",
        "rom_dx_pred_mean",
        "rom_dx_pred_abs_max",
        "rom_target_min",
        "rom_target_max",
        "rom_target_clamp_fraction",
        "rom_pressure_clamp_fraction",
        "rom_velocity_clamp_fraction",
        "rom_dx_clamp_fraction",
        "rom_nan_guard_triggered",
        "rom_output_zeroed_fraction",
        "rom_actuator_pos_error_mean",
        "rom_actuator_pos_error_abs_max",
        "reward_total",
        "q_tracking",
        "q_tracking_reward_total",
        "qdot_penalty",
        "progress_reward",
        "is_settle_phase",
        "settle_tracking_reward",
        "joint1_settle_tracking_reward",
        "joint1_error_excess",
        "action_smoothness_penalty",
        "action_rate_penalty",
        "action_magnitude_penalty",
        "action_saturation_penalty",
        "valve_effort_penalty",
        "torque_effort_penalty",
        "alive",
        "termination_penalty",
    ):
        if key in debug:
            values[key] = _tensor_to_float(debug[key])

    ordered_keys = (
        "step_reward",
        "reward_total",
        "q_tracking",
        "q_tracking_reward_total",
        "qdot_penalty",
        "progress_reward",
        "action_smoothness_penalty",
        "action_rate_penalty",
        "action_magnitude_penalty",
        "action_saturation_penalty",
        "valve_effort_penalty",
        "torque_effort_penalty",
        "alive",
        "termination_penalty",
        "q_error_mean",
        "q_error_mean_weighted",
        "q_error_mean_unweighted",
        "episode_step",
        "episode_length_buf",
        "reset_buf_mean",
        "done_buf_mean",
        "time_out_frac",
        "terminated_frac",
        "target_changed_flag",
        "target_change_step",
        "target_hold_steps",
        "last_reset_step_global",
        "reset_due_to_timeout",
        "reset_due_to_termination",
        "reset_due_to_manual_target_change",
        "q_ref_delta_abs_max",
        "q_ref_target_delta_abs_max",
        "q_ref_ramp_progress",
        "q_ref_target_scale",
        "q_ref_ramp_time_s",
        "q_ref_curriculum_progress",
        "q_ref_jump_fraction",
        "joint1_q_actual",
        "joint1_q_ref",
        "joint1_q_error_signed",
        "joint1_q_error_abs",
        "joint1_q_ref_delta_abs",
        "joint1_q_tracking_contribution",
        "is_settle_phase",
        "settle_tracking_reward",
        "joint1_settle_tracking_reward",
        "joint1_error_excess",
        "joint1_settle_error_deadband",
        "rew_settle_tracking",
        "rew_joint1_settle_tracking",
        "settle_ramp_progress_threshold",
        "joint5_coupling_gain",
        "joint5_coupling_bias",
        "joint5_coupling_source_joint",
        "joint5_coupling_target_joint",
        "joint4_q_ref",
        "joint5_q_ref",
        "joint5_coupled_ref",
        "joint5_coupling_error",
        "joint5_coupling_error_abs_max",
        "joint5_actual",
        "joint5_qdot",
        "joint5_tracking_reward_weight",
        "joint5_target_amplitude",
        "qdot_mean",
        "qdot_abs_max",
        "action_mean",
        "action_abs_mean",
        "action_abs_max",
        "joint1_action_abs",
        "action_delta_mean",
        "action_saturation_fraction",
        "qddot_cmd_mean",
        "joint1_qddot_cmd_abs",
        "joint1_valve_cmd_abs",
        "joint1_fd_tau_raw_abs",
        "joint1_tau_fd_adapted_abs",
        "joint1_tau_applied_abs",
        "joint1_torque_abs",
        "z_arm_mean",
        "z_arm_std",
        "valve_mean",
        "valve_max",
        "fd_simscape_mean",
        "fd_simscape_max",
        "tau_fd_adapted_mean",
        "tau_fd_adapted_max",
        "tau_applied_raw_mean",
        "tau_applied_raw_max",
        "tau_isaac_filtered_mean",
        "tau_isaac_filtered_max",
        "torque_raw_mean",
        "torque_raw_max",
        "torque_mean",
        "torque_max",
        "torque_adapter_clamp_fraction",
        "torque_clamp_fraction",
        "torque_rate_limited_fraction",
        "torque_rate_limit",
        "torque_lowpass_alpha",
        "fd_residual_alpha",
        "max_abs_torque",
        "qddot_cmd_scale",
        "rew_q_tracking",
        "rew_action_saturation",
        "rew_action_magnitude",
        "rew_action_rate",
        "rew_progress",
        "rew_settle_tracking",
        "rew_joint1_settle_tracking",
        "settle_ramp_progress_threshold",
        "joint1_settle_error_deadband",
        "action_saturation_threshold",
        "progress_ema_alpha",
        "sd_feedback_beta",
        "sd_dP_mean",
        "sd_dP_abs_max",
        "sd_fnet_mean",
        "sd_fnet_abs_max",
        "dP_mean",
        "dP_abs_max",
        "fnet_mean",
        "fnet_abs_max",
        "sd_feedback_dP_delta_mean",
        "sd_feedback_dP_delta_max",
        "sd_feedback_fnet_delta_mean",
        "sd_feedback_fnet_delta_max",
        "sd_feedback_rate_limited_fraction",
        "sd_feedback_clamped_fraction",
        "sd_output_nan_or_inf",
        "sd_output_clamped_fraction",
        "sd_feedback_nan_or_inf",
        "rom_dt",
        "rom_valve_cmd_mean",
        "rom_valve_cmd_abs_max",
        "rom_dP_mean",
        "rom_P_L_mean",
        "rom_P_L_abs_max",
        "rom_xdot_pred_mean",
        "rom_xdot_pred_abs_max",
        "rom_dx_pred_mean",
        "rom_dx_pred_abs_max",
        "rom_target_min",
        "rom_target_max",
        "rom_target_clamp_fraction",
        "rom_pressure_clamp_fraction",
        "rom_velocity_clamp_fraction",
        "rom_dx_clamp_fraction",
        "rom_nan_guard_triggered",
        "rom_output_zeroed_fraction",
        "rom_actuator_pos_error_mean",
        "rom_actuator_pos_error_abs_max",
        "done_frac",
    )
    mode = debug.get("control_mode", control_mode)
    delta_source = debug.get("position_delta_source", position_delta_source)
    sd_observer = debug.get("use_sd_observer", str(use_sd_observer))
    sd_feedback = debug.get("use_sd_feedback", str(use_sd_feedback))
    sd_feedback_blend = debug.get("use_sd_feedback_blend", "")
    sd_feedback_rate_limit = debug.get("use_sd_feedback_rate_limit", "")
    fd_effort = debug.get("use_fd_effort", str(use_fd_effort))
    rom_enabled = debug.get("rom_enabled", debug.get("use_reduced_hydraulic_rom", ""))
    rom_output_mode = debug.get("rom_output_mode", "")
    rom_use_sd_feedback = debug.get("rom_use_sd_feedback", "")
    rom_use_id_valve_cmd = debug.get("rom_use_id_valve_cmd", "")
    adapter_mode = debug.get("torque_adapter_mode", "")
    adapter_preset = debug.get("torque_adapter_preset", "")
    adapter_scale = debug.get("torque_adapter_scale", "")
    residual_enabled = debug.get("use_fd_residual_alpha", "")
    effective_scale = debug.get("effective_torque_adapter_scale", "")
    joint5_coupling = debug.get("use_joint5_coupling", "")
    terms = [
        f"step={step}",
        f"control_mode={mode}",
        f"position_delta_source={delta_source}",
        f"use_sd_observer={sd_observer}",
        f"use_sd_feedback={sd_feedback}",
        f"use_fd_effort={fd_effort}",
    ]
    if sd_feedback_blend:
        terms.append(f"use_sd_feedback_blend={sd_feedback_blend}")
    if sd_feedback_rate_limit:
        terms.append(f"use_sd_feedback_rate_limit={sd_feedback_rate_limit}")
    if rom_enabled:
        terms.append(f"rom_enabled={rom_enabled}")
    if rom_output_mode:
        terms.append(f"rom_output_mode={rom_output_mode}")
    if rom_use_sd_feedback:
        terms.append(f"rom_use_sd_feedback={rom_use_sd_feedback}")
    if rom_use_id_valve_cmd:
        terms.append(f"rom_use_id_valve_cmd={rom_use_id_valve_cmd}")
    if adapter_mode:
        terms.append(f"torque_adapter_mode={adapter_mode}")
    if adapter_preset:
        terms.append(f"torque_adapter_preset={adapter_preset}")
    if adapter_scale:
        terms.append(f"torque_adapter_scale={adapter_scale}")
    if residual_enabled:
        terms.append(f"use_fd_residual_alpha={residual_enabled}")
    if effective_scale:
        terms.append(f"effective_torque_adapter_scale={effective_scale}")
    if joint5_coupling:
        terms.append(f"use_joint5_coupling={joint5_coupling}")
    terms.extend(f"{key}={values[key]:.4f}" for key in ordered_keys if key in values)
    if values.get("torque_clamp_fraction", 0.0) > 0.5:
        terms.append("[WARN] final torque still too clamped")
    if values.get("torque_rate_limited_fraction", 0.0) > 0.8:
        terms.append("[WARN] torque still rate-limited")
    if values.get("q_ref_delta_abs_max", 0.0) > 0.02:
        terms.append("[WARN] q_ref_commanded changed too fast")
    if values.get("sd_feedback_rate_limited_fraction", 0.0) > 0.5:
        terms.append("[WARN] SD feedback is mostly rate-limited")
    if values.get("sd_feedback_clamped_fraction", 0.0) > 0.1:
        terms.append("[WARN] SD feedback is being clamped")
    if values.get("sd_output_nan_or_inf", 0.0) > 0.5 or values.get("sd_feedback_nan_or_inf", 0.0) > 0.5:
        terms.append("[WARN] SD produced invalid values and was sanitized")
    if values.get("sd_output_clamped_fraction", 0.0) > 0.5:
        terms.append("[WARN] SD observer output is mostly clamped")
    if _debug_flag(sd_feedback):
        terms.append("[WARN] SD feedback is actively modifying dP/Fnet")
    if _debug_flag(rom_enabled):
        terms.append("[INFO] ROM is commanding prismatic cylinder position targets")
    if values.get("rom_target_clamp_fraction", 0.0) > 0.5:
        terms.append("[WARN] ROM cylinder targets are mostly clamped")
    if values.get("rom_pressure_clamp_fraction", 0.0) > 0.5:
        terms.append("[WARN] ROM pressure state is mostly clamped")
    if values.get("rom_nan_guard_triggered", 0.0) > 0.5:
        terms.append("[WARN] ROM produced invalid values and was sanitized")
    if values.get("rom_output_zeroed_fraction", 0.0) > 0.99 and _debug_flag(rom_enabled):
        terms.append("[WARN] ROM dx outputs are all zero")
    if _debug_flag(joint5_coupling):
        coupling_error = values.get("joint5_coupling_error_abs_max", abs(values.get("joint5_coupling_error", 0.0)))
        if coupling_error > 1.0e-4:
            terms.append("[WARN] joint 5 reference is not matching the deterministic joint-4 coupling")
        if abs(values.get("joint5_tracking_reward_weight", 0.0)) > 1.0e-8:
            terms.append("[WARN] joint 5 is coupled/passive but has nonzero tracking reward weight")
    if (
        values.get("is_settle_phase", 0.0) > 0.5
        and values.get("joint1_q_error_abs", 0.0) > 0.03
        and values.get("joint1_action_abs", 0.0) < 0.2
    ):
        terms.append("[WARN] joint 1 has large settle error but small action command")
    tau_applied_raw_max = values.get("tau_applied_raw_max", 0.0)
    max_abs_torque = max(values.get("max_abs_torque", 0.0), 0.0)
    if max_abs_torque > 0.0 and tau_applied_raw_max > 2.0 * max_abs_torque:
        terms.append("[WARN] residual alpha still too high")
    if tau_applied_raw_max < 0.1:
        terms.append("[WARN] residual alpha may be too low")
    terms.append(f"q_actual={_format_joint_errors(debug.get('q_actual'))}")
    terms.append(f"q_ref={_format_joint_errors(debug.get('q_ref'))}")
    terms.append(f"q_ref_commanded={_format_joint_errors(debug.get('q_ref_commanded'))}")
    terms.append(f"q_ref_target={_format_joint_errors(debug.get('q_ref_target'))}")
    terms.append(f"q_ref_start={_format_joint_errors(debug.get('q_ref_start'))}")
    terms.append(f"q_ref_prev={_format_joint_errors(debug.get('q_ref_prev'))}")
    terms.append(f"q_ref_sampled_on_reset={_format_joint_errors(debug.get('q_ref_sampled_on_reset'))}")
    terms.append(f"q_error_signed_per_joint={_format_joint_errors(debug.get('q_error_signed_per_joint'))}")
    terms.append(f"q_error_abs_per_joint={_format_joint_errors(debug.get('q_error_abs_per_joint'))}")
    terms.append(f"q_ref_delta_per_joint={_format_joint_errors(debug.get('q_ref_delta_per_joint'))}")
    terms.append(f"q_ref_delta_abs_per_joint={_format_joint_errors(debug.get('q_ref_delta_abs_per_joint'))}")
    terms.append(f"q_error_weighted_per_joint={_format_joint_errors(debug.get('q_error_weighted_per_joint'))}")
    terms.append(f"q_tracking_reward_per_joint={_format_joint_errors(debug.get('q_tracking_reward_per_joint'))}")
    terms.append(f"q_tracking_reward_joint_weights={_format_joint_errors(debug.get('q_tracking_reward_joint_weights'))}")
    terms.append(f"q_tracking_debug_joint_weights={_format_joint_errors(debug.get('q_tracking_debug_joint_weights'))}")
    terms.append(f"qddot_cmd_abs_per_joint={_format_joint_errors(debug.get('qddot_cmd_abs_per_joint'))}")
    terms.append(f"rom_cylinder_pos_target={_format_joint_errors(debug.get('rom_cylinder_pos_target'))}")
    terms.append(f"rom_cylinder_actual_pos={_format_joint_errors(debug.get('rom_cylinder_actual_pos'))}")
    terms.append(f"rom_cylinder_pos_error={_format_joint_errors(debug.get('rom_cylinder_pos_error'))}")
    terms.append(f"rom_xdot_pred={_format_joint_errors(debug.get('rom_xdot_pred'))}")
    terms.append(f"rom_dx_pred={_format_joint_errors(debug.get('rom_dx_pred'))}")
    terms.append(f"rom_pressure={_format_joint_errors(debug.get('rom_pressure'))}")
    return " | ".join(terms)


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if args_cli.control_mode is not None:
        env_cfg.control_mode = args_cli.control_mode
    if args_cli.position_delta_source is not None:
        env_cfg.position_delta_source = args_cli.position_delta_source
    if args_cli.use_fd_effort is not None:
        env_cfg.use_fd_effort = args_cli.use_fd_effort
    if args_cli.use_sd_observer is not None:
        env_cfg.use_sd_observer = args_cli.use_sd_observer
    if args_cli.use_sd_feedback is not None:
        env_cfg.use_sd_feedback = args_cli.use_sd_feedback
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
    if args_cli.max_abs_torque is not None:
        env_cfg.max_abs_torque = args_cli.max_abs_torque
    if args_cli.torque_ramp_steps is not None:
        env_cfg.torque_ramp_steps = args_cli.torque_ramp_steps
    if args_cli.torque_safety_scale is not None:
        env_cfg.torque_safety_scale = args_cli.torque_safety_scale
    if args_cli.torque_adapter_mode is not None:
        env_cfg.torque_adapter_mode = args_cli.torque_adapter_mode
    if args_cli.torque_adapter_preset is not None:
        env_cfg.torque_adapter_preset = args_cli.torque_adapter_preset
        env_cfg.torque_adapter_scale = resolve_torque_adapter_scale(args_cli.torque_adapter_preset)
    if args_cli.torque_adapter_scale is not None:
        env_cfg.torque_adapter_scale = args_cli.torque_adapter_scale
    if args_cli.fd_torque_scale is not None:
        env_cfg.fd_torque_scale = args_cli.fd_torque_scale
    if args_cli.use_fd_residual_alpha is not None:
        env_cfg.use_fd_residual_alpha = args_cli.use_fd_residual_alpha
    if args_cli.fd_residual_alpha is not None:
        env_cfg.fd_residual_alpha = args_cli.fd_residual_alpha
    if args_cli.torque_rate_limit is not None:
        env_cfg.torque_rate_limit = args_cli.torque_rate_limit
    if args_cli.torque_lowpass_alpha is not None:
        env_cfg.torque_lowpass_alpha = args_cli.torque_lowpass_alpha
    if args_cli.action_to_qddot_scale is not None:
        env_cfg.action_to_qddot_scale = args_cli.action_to_qddot_scale
    if args_cli.qddot_cmd_clip is not None:
        env_cfg.qddot_cmd_clip = args_cli.qddot_cmd_clip
    if args_cli.q_ref_ramp_time_s is not None:
        env_cfg.q_ref_ramp_time_s = args_cli.q_ref_ramp_time_s
    if args_cli.q_ref_target_scale is not None:
        env_cfg.q_ref_target_scale_initial = args_cli.q_ref_target_scale
        env_cfg.q_ref_target_scale_final = args_cli.q_ref_target_scale
        env_cfg.use_q_ref_curriculum = False
    if args_cli.disable_q_ref_ramp:
        env_cfg.use_q_ref_ramp = False
    if args_cli.disable_q_ref_curriculum:
        env_cfg.use_q_ref_curriculum = False
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args_cli.print_interval <= 0:
        raise ValueError("--print_interval must be positive.")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_reward_terms = True
    _apply_env_overrides(env_cfg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        policy = None
        policy_device = agent_cfg.device
        if args_cli.checkpoint is not None:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
            runner.load(args_cli.checkpoint)
            policy = runner.get_inference_policy(device=policy_device)
        else:
            print("[INFO] No checkpoint provided. Running zero-action baseline.")

        obs = env.get_observations()
        if policy is not None:
            obs = obs.to(policy_device)

        for step in range(1, args_cli.steps + 1):
            with torch.inference_mode():
                if policy is None:
                    actions = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)
                else:
                    actions = policy(obs).to(env.device)
                obs, reward, dones, _ = env.step(actions)
                if policy is not None:
                    obs = obs.to(policy_device)

            if step == 1 or step % args_cli.print_interval == 0:
                debug = getattr(env.unwrapped, "_reward_debug", {})
                print(
                    _format_debug_line(
                        step=step,
                        reward=reward,
                        dones=dones,
                        debug=debug,
                        control_mode=env.unwrapped.cfg.control_mode,
                        position_delta_source=env.unwrapped.cfg.position_delta_source,
                        use_sd_observer=env.unwrapped.cfg.use_sd_observer,
                        use_sd_feedback=env.unwrapped.cfg.use_sd_feedback,
                        use_fd_effort=env.unwrapped.cfg.use_fd_effort,
                    ),
                    flush=True,
                )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()  # type: ignore
    finally:
        simulation_app.close()
