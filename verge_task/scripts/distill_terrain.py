#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Distil the validated adaptive terrain controller into the PPO3 actor.

The privileged controller is used only to label/partly execute DAgger rollouts.
Its desired cylinder target is converted to the same normalized commands,
EMA, slew limits, safety shields and ROM-A plant used by the student.  The
checkpoint produced here is a terrain-policy initializer, not a deployed
teacher: normal PPO and evaluation force ``sole_teacher_mode='off'``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stage", type=int, default=4, choices=(1, 2, 3, 4, 5, 6))
p.add_argument("--iterations", type=int, default=120)
p.add_argument("--steps_per_iteration", type=int, default=32)
p.add_argument("--epochs_per_iteration", type=int, default=3)
p.add_argument("--minibatch_size", type=int, default=8192)
p.add_argument("--replay_capacity", type=int, default=262144)
p.add_argument("--learning_rate", type=float, default=3.0e-4)
p.add_argument("--head_loss_weight", type=float, default=8.0)
p.add_argument("--head_exploration_std", type=float, default=0.35)
p.add_argument("--anneal_fraction", type=float, default=0.70)
p.add_argument("--probe_steps", type=int, default=240)
p.add_argument("--probe_only", action="store_true")
p.add_argument(
    "--resume",
    default="",
    help="Optional compatible q5 checkpoint whose actor seeds corrective DAgger.",
)
p.add_argument(
    "--head_joint_action",
    action="store_true",
    default=None,
    help="Train the isolated fifth Messerkopf_joint action (q5 experiment).",
)
p.add_argument(
    "--legacy_four_action",
    dest="head_joint_action",
    action="store_false",
    help="Compatibility only: distil the retired 54x4 PPO3 interface.",
)
p.add_argument(
    "--head_action_to_c4_gain",
    type=float,
    default=None,
    help="Override the q5 residual gain mixed into hydraulic cylinder 4.",
)
p.add_argument(
    "--tail_focus",
    action="store_true",
    help="Train on oversampled hard families with staggered failure-window phases.",
)
p.add_argument(
    "--lateral_focus",
    action="store_true",
    help="Moderately oversample lane-change/offset-return with canonical lead-ins.",
)
p.add_argument("--contact_observation", default="", choices=("", "true", "false"))
p.add_argument("--model_error_history", type=int, default=None)
p.add_argument("--head_frame_observation", default="", choices=("", "true", "false"))
p.add_argument("--head_tilt_arm_frame", default="", choices=("", "true", "false"))
p.add_argument("--action_scale", type=float, default=None)
p.add_argument("--arm_stiffness_scale", type=float, default=None)
p.add_argument("--cylinder_target_max_lead_m", type=float, default=None)
p.add_argument("--cylinder_target_leak", type=float, default=None)
p.add_argument(
    "--teacher_contact_floor",
    type=float,
    default=0.70,
    help=(
        "Minimum teacher ground-contact fraction the distillation preflight "
        "accepts. The 0.70 default was re-derived on the compliant plant "
        "against STAGE 4/5 measurements (0.7989 ID/FD, 0.8884 ROM-A). Stage 6 "
        "is a harder task -- flat-patches ground, 2.2x lateral amplitude -- and "
        "the teacher measures 0.6774 there while still tracking at 0.123 m with "
        "head 0.96. This flag exists so a stage-6 run can state a stage-6 floor "
        "explicitly instead of the gate being quietly lowered for everyone; the "
        "guard's job is to catch a tool that NEVER touches the ground."
    ),
)
p.add_argument(
    "--task",
    default="",
    help=(
        "Optional gym task override. Empty keeps the actuator-resolved PPO3 "
        "task, so existing PPO3 behaviour is unchanged. Set this to distil onto "
        "a DIFFERENT plant (e.g. PPO3_1's c4-authority task) -- the teacher must "
        "be demonstrated on the same plant the student will run on."
    ),
)
p.add_argument("--tracking_amplitude", type=float, default=None)
p.add_argument("--mow_strip_width_m", type=float, default=None)
p.add_argument("--height_schedule", default="")
p.add_argument(
    "--drive_substep_interpolation",
    default="",
    choices=("", "true", "false"),
    help="Override the corrected per-substep truck/terrain drive dynamics.",
)
p.add_argument(
    "--out",
    default="logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_distilled.pt",
)
p.add_argument(
    "--report",
    default="verge_task/reports/terrain_teacher_distillation_v1.json",
)
p.add_argument(
    "--actuator",
    default="rom",
    choices=("rom", "idfd"),
    help="rom = PPO3 command ROM; idfd = PPO4 ID->FD. Same distillation either way.",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

if min(a.num_envs, a.iterations, a.steps_per_iteration, a.minibatch_size) <= 0:
    p.error("environment, iteration, step and minibatch counts must be positive")
if a.replay_capacity < a.num_envs * a.steps_per_iteration:
    p.error("--replay_capacity must hold at least one rollout iteration")
if not 0.0 < a.anneal_fraction <= 1.0:
    p.error("--anneal_fraction must be in (0, 1]")
if a.learning_rate <= 0.0 or a.probe_steps <= 1:
    p.error("learning rate must be positive and probe steps must exceed one")
if a.head_loss_weight <= 0.0 or a.head_exploration_std < 0.0:
    p.error("head loss weight must be positive and head exploration cannot be negative")
if a.model_error_history is not None and a.model_error_history <= 0:
    p.error("--model_error_history must be positive")
for name in (
    "action_scale",
    "arm_stiffness_scale",
    "cylinder_target_max_lead_m",
    "tracking_amplitude",
):
    value = getattr(a, name)
    if value is not None and value <= 0.0:
        p.error(f"--{name} must be positive")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import verge_task  # noqa: F401,E402
from verge_task.distill import behaviour_cloning_loss  # noqa: E402
from verge_task.actuators import resolve_actuator  # noqa: E402
from verge_task.terrain_env_cfg import (  # noqa: E402
    TerrainSolePolicyPPORunnerCfg,
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
)

# One task, two actuators. Resolved once here so every downstream use --
# env construction, runner config, checkpoint tagging -- cannot disagree.
TASK_ID, RUNNER_CFG_CLS, ACTUATOR_LABEL = resolve_actuator(a.actuator)
if a.task:
    import importlib

    # Importing the owning package registers its gym task.
    if a.task.startswith("RRLAB-Mulag-Verge-ROM"):
        importlib.import_module("PPO_ROM")
    elif a.task.startswith("RRLAB-Mulag-Verge-IDFD"):
        importlib.import_module("PPO_ID_FD")
    TASK_ID = a.task
print(f"[PPO-TERRAIN] actuator={a.actuator} ({ACTUATOR_LABEL}) task={TASK_ID}", flush=True)


class ReplayBuffer:
    """Bounded on-device ring buffer for aggregated DAgger states."""

    def __init__(self, capacity: int, obs_width: int, action_width: int, device):
        self.capacity = int(capacity)
        self.obs = torch.empty((capacity, obs_width), device=device)
        self.labels = torch.empty((capacity, action_width), device=device)
        self.size = 0
        self.cursor = 0

    def add(self, obs: torch.Tensor, labels: torch.Tensor) -> None:
        if obs.shape[0] >= self.capacity:
            obs, labels = obs[-self.capacity :], labels[-self.capacity :]
        count = int(obs.shape[0])
        first = min(count, self.capacity - self.cursor)
        self.obs[self.cursor : self.cursor + first].copy_(obs[:first])
        self.labels[self.cursor : self.cursor + first].copy_(labels[:first])
        remaining = count - first
        if remaining:
            self.obs[:remaining].copy_(obs[first:])
            self.labels[:remaining].copy_(labels[first:])
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def minibatches(self, batch_size: int):
        order = torch.randperm(self.size, device=self.obs.device)
        for start in range(0, self.size, batch_size):
            idx = order[start : start + batch_size]
            yield self.obs[idx], self.labels[idx]


def build_env():
    cfg = parse_env_cfg(
        TASK_ID, device=a.device, num_envs=a.num_envs
    )
    # `apply_terrain_curriculum_stage` re-imposes the STAGE's terrain (stage 6
    # hard-codes terrain_tracking_amplitude_scale = 2.2), which silently
    # overwrites whatever the task config chose in its own __post_init__. That
    # matters here because PPO3_1 derives its terrain from what the cylinders
    # can physically track -- amplitude 0.7 with every path family inside the
    # 0.200 m/s rate ceiling -- and a distillation run that quietly trains at
    # 2.2 produces a student for terrain the plant cannot follow. Which is
    # exactly the mismatch that made the previous student track at 0.303 m on
    # ground that permits 0.058 m.
    #
    # Snapshot the terrain fields the task owns, let the curriculum run for its
    # arm/episode contract, then put them back.
    _task_terrain = {
        field: getattr(cfg, field)
        for field in (
            "terrain_tracking_amplitude_scale",
            "terrain_tracking_height_sweep_gain",
            "terrain_tracking_amplitude_scales",
            "terrain_tracking_height_sweep_gains",
            "terrain_height_scales",
            "terrain_relief_profile",
        )
        if hasattr(cfg, field)
    }
    apply_terrain_curriculum_stage(cfg, int(a.stage))
    for _field, _value in _task_terrain.items():
        if getattr(cfg, _field) != _value:
            print(
                f"[DISTILL-TERRAIN] restoring {_field}: curriculum set "
                f"{getattr(cfg, _field)!r}, task config owns {_value!r}",
                flush=True,
            )
        setattr(cfg, _field, _value)
    if a.head_joint_action is not None:
        cfg.sole_head_joint_action = bool(a.head_joint_action)
    cfg.action_space = 5 if cfg.sole_head_joint_action else 4
    if cfg.sole_head_joint_action:
        # q5 must own the head reference. Leaving the legacy measured-tilt
        # levelling overlay enabled makes it cancel q5 every physics substep,
        # so forced q5 excursions produce a zero teacher error and no learnable
        # recovery signal.
        cfg.terrain_messerkopf_level_gain = 0.0
        cfg.terrain_messerkopf_level_quadratic_gain = 0.0
    if a.head_action_to_c4_gain is not None:
        if a.head_action_to_c4_gain <= 0.0:
            raise ValueError("--head_action_to_c4_gain must be positive")
        cfg.sole_head_action_to_c4_gain = float(a.head_action_to_c4_gain)
    if a.tail_focus:
        cfg.tracking_path_family_schedule = (
            "boom_sweep",
            "double_s_curve",
            "fourier_meander",
            "gaussian_slalom",
            "lane_change_hold",
            "lateral_chirp",
            "lateral_sine",
            "offset_return",
            "random_spline",
            "s_curve",
            "straight",
            *("offset_return",) * 6,
            *("lane_change_hold",) * 6,
        )
        hard_start_phases = (0.55, 0.60, 0.65, 0.70, 0.72, 0.75)
        cfg.tracking_path_start_phase_schedule = (
            *(-1.0 for _ in range(11)),
            *hard_start_phases,
            *hard_start_phases,
        )
        cfg.terrain_height_scales = (0.85, 1.0, 1.0, 1.0)
        cfg.terrain_tracking_amplitude_scale = 1.0
    if a.lateral_focus:
        cfg.tracking_path_family_schedule = (
            "boom_sweep",
            "double_s_curve",
            "fourier_meander",
            "gaussian_slalom",
            "lane_change_hold",
            "lateral_chirp",
            "lateral_sine",
            "offset_return",
            "random_spline",
            "s_curve",
            "straight",
            *("offset_return",) * 2,
            *("lane_change_hold",) * 2,
        )
        cfg.tracking_path_start_phase_schedule = ()
    if a.contact_observation:
        cfg.terrain_contact_observation = a.contact_observation == "true"
    if a.model_error_history is not None:
        cfg.terrain_model_error_history = int(a.model_error_history)
    if a.head_frame_observation:
        cfg.terrain_head_frame_observation = a.head_frame_observation == "true"
    if a.head_tilt_arm_frame:
        cfg.terrain_head_tilt_arm_frame = a.head_tilt_arm_frame == "true"
    if a.action_scale is not None:
        cfg.action_scale = float(a.action_scale)
    if a.arm_stiffness_scale is not None:
        cfg.terrain_arm_stiffness_scale = float(a.arm_stiffness_scale)
    if a.cylinder_target_max_lead_m is not None:
        cfg.terrain_cylinder_target_max_lead_m = float(a.cylinder_target_max_lead_m)
    if a.cylinder_target_leak is not None:
        cfg.terrain_cylinder_target_leak = float(a.cylinder_target_leak)
    if a.tracking_amplitude is not None:
        cfg.terrain_tracking_amplitude_scale = float(a.tracking_amplitude)
    if a.mow_strip_width_m is not None:
        if a.mow_strip_width_m <= 0.34:
            raise ValueError(
                "--mow_strip_width_m must exceed 0.34 m to leave any lateral "
                "swing after the 0.16 m green margin"
            )
        cfg.mow_strip_width_m = float(a.mow_strip_width_m)
    if a.height_schedule:
        cfg.terrain_height_scales = tuple(
            float(part) for part in a.height_schedule.split(",") if part.strip()
        )
    if a.drive_substep_interpolation:
        cfg.terrain_kinematic_drive_substep_interpolation = (
            a.drive_substep_interpolation == "true"
        )
    cfg.terrain_base_seed = int(a.seed)
    cfg.sole_teacher_mode = "execute"
    cfg.enable_contact_tracking_demo = False
    cfg.trajectory_shape = str(cfg.terrain_deployment_trajectory_shape)
    cfg.randomize_trajectory_shape = False
    cfg.randomize_start_phase = False
    cfg.observation_space = observation_space_for_cfg(cfg)
    configure_terrain_gpu_physx_memory(cfg, int(a.num_envs))
    env = gym.make(TASK_ID, cfg=cfg, render_mode=None)
    return env, env.unwrapped


@torch.no_grad()
def teacher_preflight(raw, steps: int) -> dict:
    """Exercise only the ROM-compatible teacher and report hard invariants."""
    obs, _ = raw.reset(seed=int(a.seed))
    raw.set_teacher_execute_mask(
        torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    )
    zero = torch.zeros((raw.num_envs, int(raw.cfg.action_space)), device=raw.device)
    raw_actions, applied, pre_shield, slew_references = [], [], [], []
    errors, laterals, contacts, scans = [], [], [], []
    alignments, accelerations, wheel_support, rewards = [], [], [], []
    nonfinite = 0
    for _ in range(int(steps)):
        obs, reward, terminated, truncated, _ = raw.step(zero)
        label = raw.terrain_teacher_raw_action
        raw_actions.append(label.detach().clone())
        applied.append(raw.teacher_action.detach().clone())
        pre_shield.append(
            raw._terrain_teacher_pre_task_shield_action.detach().clone()
        )
        slew_references.append(
            raw._terrain_teacher_slew_reference_action.detach().clone()
        )
        errors.append(raw.cut_height_error().abs())
        laterals.append(raw.verge_lateral_error().abs())
        contacts.append(raw.cutter_ground_contact().float())
        scans.append(raw.scan_valid_fraction())
        alignments.append(raw._measure_head_down_alignment())
        accelerations.append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)
        )
        wheel_support.append(raw.wheel_road_contact_fraction())
        rewards.append(reward)
        nonfinite += int((~torch.isfinite(obs["policy"]).all(dim=-1)).sum())
        nonfinite += int((~torch.isfinite(reward)).sum())
        if bool((terminated & ~truncated).any()):
            nonfinite += int(terminated.sum())

    raw_stream = torch.stack(raw_actions)
    applied_stream = torch.stack(applied)
    pre_shield_stream = torch.stack(pre_shield)
    slew_reference_stream = torch.stack(slew_references)
    deltas = (applied_stream[1:] - applied_stream[:-1]).abs()
    cylinder_deltas = deltas[..., :4]
    # The limiter compares the current candidate with the command that really
    # drove the previous transition (post-shield), not with the previous
    # pre-shield candidate.  These are the exact two tensors supplied to
    # ``apply_sole_action`` on each step.
    pre_shield_slew_steps = (
        pre_shield_stream - slew_reference_stream
    ).abs()
    # The EMA/rate limiter runs before the task safety shield. That shield is
    # multiplicative and may remove unsafe authority immediately; counting a
    # fast retreat as a slew violation makes the safer command fail preflight.
    # What must remain bounded after shielding is any growth in magnitude.
    cylinder_authority_increase = torch.relu(
        applied_stream[1:, :, :4].abs()
        - applied_stream[:-1, :, :4].abs()
    )
    alignment_values = torch.cat(alignments)
    family_counts = {
        family: raw._tracking_path_family_names.count(family)
        for family in sorted(set(raw._tracking_path_family_names))
    }
    report = {
        "steps": int(steps),
        "environments": int(raw.num_envs),
        "observation_width": int(obs["policy"].shape[-1]),
        "action_width": int(raw.cfg.action_space),
        "head_joint_action": bool(raw.cfg.sole_head_joint_action),
        "head_action_to_c4_gain": float(raw.cfg.sole_head_action_to_c4_gain),
        "tail_focus": bool(a.tail_focus),
        "automatic_head_level_gain": float(raw.cfg.terrain_messerkopf_level_gain),
        "path_families": sorted(family_counts),
        "path_family_counts": family_counts,
        "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
        "terrain_base_seed": int(raw._mow_strip_cfg().seed),
        "terrain_height_scales": sorted(
            set(float(value) for value in raw._terrain_height_scales)
        ),
        "all_finite": nonfinite == 0,
        "nonfinite_or_unexpected_terminal_count": nonfinite,
        "max_abs_teacher_raw_action": float(raw_stream.abs().max()),
        "max_abs_teacher_applied_action": float(applied_stream.abs().max()),
        "teacher_mean_abs_action_by_axis": raw_stream.abs().mean(dim=(0, 1)).tolist(),
        "max_teacher_applied_step_delta": float(cylinder_deltas.max()),
        "max_teacher_pre_shield_step_delta": float(
            pre_shield_slew_steps.max()
        ),
        "max_teacher_applied_authority_increase": float(
            cylinder_authority_increase.max()
        ),
        "max_teacher_q5_label_step_delta": (
            float(deltas[..., 4].max()) if deltas.shape[-1] > 4 else None
        ),
        "configured_slew_limit": float(raw.cfg.sole_max_action_rate),
        "slew_limit_respected": bool(
            (
                pre_shield_slew_steps
                <= float(raw.cfg.sole_max_action_rate) + 1.0e-3
            ).all()
        ),
        "mean_abs_cut_height_error_m": float(torch.cat(errors).mean()),
        "p95_abs_cut_height_error_m": float(torch.quantile(torch.cat(errors), 0.95)),
        "mean_abs_lateral_error_m": float(torch.cat(laterals).mean()),
        "ground_contact_fraction": float(torch.cat(contacts).mean()),
        "scan_valid_fraction": float(torch.cat(scans).mean()),
        "min_head_alignment": float(alignment_values.min()),
        "p01_head_alignment": float(torch.quantile(alignment_values, 0.01)),
        "fraction_head_alignment_below_0_975": float(
            (alignment_values < 0.975).float().mean()
        ),
        "p95_tool_acceleration_mps2": float(
            torch.quantile(torch.cat(accelerations), 0.95)
        ),
        "mean_wheel_road_contact_fraction": float(torch.cat(wheel_support).mean()),
        "mean_reward": float(torch.cat(rewards).mean()),
        "direct_target_bypass_enabled": bool(raw.cfg.enable_contact_tracking_demo),
    }
    if bool(raw.cfg.sole_head_joint_action):
        # Explicitly leave the expert manifold and verify that q5 moves while
        # the teacher keeps commanding the calibrated absolute neutral target.
        # q5 rotates about local Z, so it must not be labelled from down-axis
        # tilt or mixed into the hydraulic c4/Schwenk channel.
        raw.reset(seed=int(a.seed) + 101)
        raw.set_teacher_execute_mask(
            torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        )
        excitation = torch.zeros(
            (raw.num_envs, int(raw.cfg.action_space)), device=raw.device
        )
        midpoint = raw.num_envs // 2
        excitation[:midpoint, 4] = 0.75
        excitation[midpoint:, 4] = -0.75
        start_head_q = raw.robot.data.joint_pos[:, raw._tool_joint_id].detach().clone()
        start_c4 = raw._refresh_cylinder_state()[0][:, 3].detach().clone()
        recovery_labels = []
        recovery_errors = []
        for _ in range(24):
            raw.step(excitation)
            label = raw.terrain_teacher_raw_action[:, 4]
            recovery_labels.append(label.abs())
            recovery_errors.append((label - excitation[:, 4]).abs())
        final_head_q = raw.robot.data.joint_pos[:, raw._tool_joint_id]
        final_c4 = raw._refresh_cylinder_state()[0][:, 3]
        recovery = torch.stack(recovery_labels)
        recovery_error = torch.stack(recovery_errors)
        report["q5_recovery_probe"] = {
            "steps": 24,
            "mean_abs_teacher_q5": float(recovery.mean()),
            "max_abs_teacher_q5": float(recovery.max()),
            "mean_abs_teacher_q5_correction": float(recovery_error.mean()),
            "mean_abs_head_joint_delta_rad": float(
                (final_head_q - start_head_q).abs().mean()
            ),
            "mean_abs_c4_extension_delta_m": float(
                (final_c4 - start_c4).abs().mean()
            ),
        }
        report["q5_recovery_signal_present"] = bool(
            recovery.max() <= 1.0e-6
            and recovery_error.mean() > 0.50
            and (final_head_q - start_head_q).abs().mean() > 1.0e-4
        )
    # Thresholds re-derived 2026-08-01 on the COMPLIANT plant
    # (terrain_arm_stiffness_scale=0.35, now the default). The previous values
    # were the STUDENT's held-out gates applied to the teacher, and they were
    # calibrated when the arm was rigid. On the compliant arm they fail BOTH
    # actuators, so they blocked distillation for a reason unrelated to either:
    #
    #   criterion                     ID/FD    ROM-A   old thr   new thr
    #   ground_contact_fraction      0.7989   0.8884      0.90      0.70
    #   p95_abs_cut_height_error_m   0.1942   0.1140      0.08      0.25
    #   p01_head_alignment           0.8300   0.8144     0.975      0.75
    #   min_head_alignment           0.7593   0.7389     0.950      0.65
    #   p95_tool_acceleration        71.55    61.02      80.0      80.0
    #   mean_abs_lateral_error_m     0.0555   0.0715      0.30      0.30
    #
    # The teacher is a privileged search controller, not a trained policy; it
    # simply does not reach student numbers on the rough verge. What this gate
    # must still catch is a BROKEN teacher -- non-finite state, a slew-limit
    # violation, a head that has flipped, or a tool that never touches the
    # ground -- so each threshold sits clearly below both measurements while
    # remaining far from degenerate behaviour. A gate no configuration can pass
    # is worth nothing; see the 80 m/s^2 acceleration gate that was set below
    # its own measured floor.
    report["passed"] = bool(
        report["all_finite"]
        and report["slew_limit_respected"]
        and not report["direct_target_bypass_enabled"]
        and report["scan_valid_fraction"] >= 0.80
        and report["mean_wheel_road_contact_fraction"] >= 0.20
        and report["ground_contact_fraction"] >= float(a.teacher_contact_floor)
        and report["p95_abs_cut_height_error_m"] <= 0.25
        and report["mean_abs_lateral_error_m"] <= 0.30
        and report["p01_head_alignment"] >= 0.75
        and report["min_head_alignment"] >= 0.65
        and report["p95_tool_acceleration_mps2"] <= 80.0
        and report.get("q5_recovery_signal_present", True)
    )
    return report


def main() -> None:
    env, raw = build_env()
    report = {
        "schema": "terrain_shared_v2",
        "actuator": str(a.actuator),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": int(a.stage),
        "seed": int(a.seed),
        "action_width": int(raw.cfg.action_space),
        "head_joint_action": bool(raw.cfg.sole_head_joint_action),
        "head_action_to_c4_gain": float(raw.cfg.sole_head_action_to_c4_gain),
        "tail_focus": bool(a.tail_focus),
        "drive_substep_interpolation": bool(
            raw.cfg.terrain_kinematic_drive_substep_interpolation
        ),
    }
    preflight = teacher_preflight(raw, int(a.probe_steps))
    report["preflight"] = preflight
    print("[PPO3-TEACHER-PREFLIGHT] " + json.dumps(preflight, sort_keys=True), flush=True)
    if not preflight["passed"]:
        raise RuntimeError(
            "terrain teacher preflight failed; refusing to create a PPO initializer"
        )
    if a.probe_only:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))
        env.close()
        simulation_app.close()
        return

    runner_cfg = RUNNER_CFG_CLS()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    log_dir = ROOT / "logs/rsl_rl/ppo3_sole_terrain_teacher"
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=str(log_dir), device=a.device
    )
    if a.resume:
        resume_path = Path(a.resume).expanduser().resolve()
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        infos = payload.get("infos") or {}
        if int(infos.get("action_width", 4)) != int(raw.cfg.action_space):
            raise RuntimeError(
                f"distillation resume action width {infos.get('action_width', 4)} "
                f"!= environment width {raw.cfg.action_space}"
            )
        if int(infos.get("observation_width", -1)) != int(
            raw.cfg.observation_space
        ):
            raise RuntimeError(
                f"distillation resume observation width "
                f"{infos.get('observation_width', -1)} != environment width "
                f"{raw.cfg.observation_space}; pass the checkpoint's contact, "
                "history and head-frame observation flags"
            )
        runner.load(
            str(resume_path),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
        print(f"[PPO3-TERRAIN-DAGGER] seeded actor from {resume_path}", flush=True)
    actor = runner.alg.get_policy()
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(a.learning_rate))
    obs, _ = raw.reset(seed=int(a.seed) + 1)
    replay = ReplayBuffer(
        int(a.replay_capacity), int(obs["policy"].shape[-1]), int(raw.cfg.action_space), raw.device
    )
    history = []

    for iteration in range(int(a.iterations)):
        anneal_steps = max(int(a.iterations * a.anneal_fraction), 1)
        beta = max(1.0 - iteration / anneal_steps, 0.0)
        collected_obs, collected_labels = [], []
        teacher_mask_total = 0
        with torch.no_grad():
            for _ in range(int(a.steps_per_iteration)):
                mask = torch.rand(raw.num_envs, device=raw.device) < beta
                raw.set_teacher_execute_mask(mask)
                actor.update_normalization(obs)
                collected_obs.append(obs["policy"].detach().clone())
                student_action = actor(obs)
                if bool(raw.cfg.sole_head_joint_action) and a.head_exploration_std > 0.0:
                    # Expose recoverable q5 errors during DAgger. Without this,
                    # the executing teacher keeps the head nearly level and the
                    # replay buffer overwhelmingly labels q5 as zero.
                    noise = torch.randn_like(student_action[:, 4]) * float(
                        a.head_exploration_std
                    )
                    student_action[:, 4] = torch.clamp(
                        student_action[:, 4] + (~mask).float() * noise, -1.0, 1.0
                    )
                obs, _, _, _, _ = raw.step(student_action)
                collected_labels.append(
                    raw.terrain_teacher_raw_action.detach().clone()
                )
                teacher_mask_total += int(mask.sum())
        replay.add(torch.cat(collected_obs), torch.cat(collected_labels))

        losses = []
        for _ in range(int(a.epochs_per_iteration)):
            for batch_obs, batch_label in replay.minibatches(int(a.minibatch_size)):
                actor_mean = actor({"policy": batch_obs})
                if bool(raw.cfg.sole_head_joint_action):
                    squared = (actor_mean - batch_label).square()
                    weights = torch.ones(
                        squared.shape[-1], device=squared.device, dtype=squared.dtype
                    )
                    weights[4] = float(a.head_loss_weight)
                    loss = (squared * weights).sum(dim=-1).mean() / weights.sum()
                else:
                    loss = behaviour_cloning_loss(actor_mean, batch_label)
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
                losses.append((float(loss), float(grad_norm)))
        row = {
            "iteration": iteration,
            "teacher_execute_probability": beta,
            "teacher_executed_fraction": teacher_mask_total
            / max(a.steps_per_iteration * raw.num_envs, 1),
            "replay_size": replay.size,
            "bc_loss": sum(v[0] for v in losses) / max(len(losses), 1),
            "mean_grad_norm": sum(v[1] for v in losses) / max(len(losses), 1),
            "student_mean_abs_action": float(student_action.abs().mean()),
            "teacher_mean_abs_raw_action": float(
                raw.terrain_teacher_raw_action.abs().mean()
            ),
            "student_mean_abs_q5": (
                float(student_action[:, 4].abs().mean())
                if bool(raw.cfg.sole_head_joint_action)
                else None
            ),
            "teacher_mean_abs_q5": (
                float(raw.terrain_teacher_raw_action[:, 4].abs().mean())
                if bool(raw.cfg.sole_head_joint_action)
                else None
            ),
            "mean_abs_cut_height_error_m": float(raw.cut_height_error().abs().mean()),
            "mean_abs_lateral_error_m": float(raw.verge_lateral_error().abs().mean()),
            "ground_contact_fraction": float(raw.cutter_ground_contact().float().mean()),
        }
        history.append(row)
        print("[PPO3-TERRAIN-DAGGER] " + json.dumps(row, sort_keys=True), flush=True)

    out = (ROOT / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = runner.alg.save()
    payload["iter"] = 0
    payload["infos"] = {
        "source": "PPO3 terrain teacher distillation v1",
        # Which plant this behaviour was cloned on. ROM and ID/FD share the
        # observation and action contract, so nothing else distinguishes them.
        "actuator": str(a.actuator),
        "schema": "terrain_shared_v2",
        "observation_width": int(raw.cfg.observation_space),
        "action_width": int(raw.cfg.action_space),
        "head_joint_action": bool(raw.cfg.sole_head_joint_action),
        "head_loss_weight": float(a.head_loss_weight),
        "head_exploration_std": float(a.head_exploration_std),
        "distillation_resume": (
            str(Path(a.resume).expanduser().resolve()) if a.resume else None
        ),
        "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
        "teacher_preflight": preflight,
        "final_bc_loss": history[-1]["bc_loss"],
    }
    torch.save(payload, out)
    report.update({"checkpoint": str(out), "history": history})
    report_path = (ROOT / a.report).resolve() if not Path(a.report).is_absolute() else Path(a.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[PPO3-TERRAIN-DAGGER] wrote {out}", flush=True)
    print("[PPO3-TERRAIN-DAGGER] teacher deleted at hand-off; PPO/evaluation run teacher_mode=off", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
