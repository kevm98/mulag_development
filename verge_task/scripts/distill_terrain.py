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
import copy
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
p.add_argument(
    "--teacher_warmup_iterations",
    type=int,
    default=0,
    help="Keep teacher execution at 100%% before beginning DAgger handoff.",
)
p.add_argument(
    "--teacher_execute_floor",
    type=float,
    default=0.0,
    help=(
        "Minimum probability of executing the teacher after handoff. A nonzero "
        "floor keeps expert support in the replay distribution while the "
        "student learns off-distribution recovery states."
    ),
)
p.add_argument("--probe_steps", type=int, default=240)
p.add_argument(
    "--student_probe_steps",
    type=int,
    default=0,
    help=(
        "Run the distilled actor without teacher execution for this many "
        "steps before writing the report. Use the final 26 m horizon (2808 "
        "steps) for a deployable warm-checkpoint check."
    ),
)
p.add_argument(
    "--validation_only",
    action="store_true",
    help=(
        "Load --resume and validate its actor teacher-off without running any "
        "DAgger update; optionally write an unchanged checkpoint with the "
        "validation report in its metadata."
    ),
)
p.add_argument("--probe_only", action="store_true")
p.add_argument(
    "--obs_normalizer",
    choices=("freeze", "adapt"),
    default="freeze",
    help=(
        "Keep a resumed actor's observation statistics fixed during DAgger, "
        "or update them from rollout observations. Freeze is the stable "
        "warm-checkpoint default."
    ),
)
p.add_argument(
    "--prior_anchor_weight",
    type=float,
    default=0.25,
    help=(
        "MSE weight to a frozen copy of the resumed warm actor during DAgger. "
        "This keeps corrective updates local while labels are collected off "
        "the distilled policy's original state distribution."
    ),
)
p.add_argument(
    "--teacher_label_mode",
    choices=("raw", "invert_applied"),
    default="raw",
    help=(
        "Use the teacher's raw command sent through the shared action chain "
        "(the production label), or explicitly use the legacy inverse of the "
        "post-shield applied command for diagnosis."
    ),
)
p.add_argument(
    "--student_error_recovery_threshold",
    type=float,
    default=0.10,
    help=(
        "Reject a DAgger optimizer update when the sampled student action "
        "error exceeds this value, then collect teacher-only recovery batches."
    ),
)
p.add_argument(
    "--recovery_teacher_iterations",
    type=int,
    default=2,
    help=(
        "Number of teacher-only, actor-frozen batches after a rejected "
        "student update."
    ),
)
p.add_argument(
    "--recovery_execute_floor",
    type=float,
    default=0.90,
    help=(
        "Teacher execution floor after the first rejected handoff batch. "
        "This reopens mixed control gradually instead of repeating the same "
        "late student transition."
    ),
)
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
p.add_argument(
    "--dagger_phase_coverage",
    action="store_true",
    help=(
        "Spread DAgger environments across four terrain start phases. This "
        "decorrelates terrain sections without changing the canonical arm "
        "deployment; evaluation keeps the canonical terrain phase."
    ),
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
    choices=("rom", "idfd", "id_aam_fd"),
    help=(
        "rom = PPO3 command ROM; idfd = PPO4 ID->FD; id_aam_fd = causal "
        "AAM -> ID adapter -> FD. Same distillation either way."
    ),
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
if not 0.0 <= a.teacher_execute_floor <= 1.0:
    p.error("--teacher_execute_floor must be in [0, 1]")
if not 0 <= a.teacher_warmup_iterations < a.iterations:
    p.error("--teacher_warmup_iterations must be in [0, iterations)")
if a.learning_rate <= 0.0 or a.probe_steps <= 1:
    p.error("learning rate must be positive and probe steps must exceed one")
if a.student_probe_steps < 0:
    p.error("--student_probe_steps must be non-negative")
if a.validation_only and (not a.resume or a.student_probe_steps <= 0):
    p.error("--validation_only requires --resume and --student_probe_steps > 0")
if a.prior_anchor_weight < 0.0:
    p.error("--prior_anchor_weight must be non-negative")
if a.student_error_recovery_threshold < 0.0:
    p.error("--student_error_recovery_threshold must be non-negative")
if a.recovery_teacher_iterations < 0:
    p.error("--recovery_teacher_iterations must be non-negative")
if not 0.0 <= a.recovery_execute_floor <= 1.0:
    p.error("--recovery_execute_floor must be in [0, 1]")
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
from verge_task.distill import (  # noqa: E402
    behaviour_cloning_loss,
    teacher_labels,
)
from verge_task.actuators import resolve_actuator  # noqa: E402
from verge_task.terrain_env_cfg import (  # noqa: E402
    TerrainSolePolicyPPORunnerCfg,
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
)
from verge_task.terrain import TRACKING_PATH_FAMILIES  # noqa: E402

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
        self.valid = torch.zeros((capacity,), dtype=torch.bool, device=device)
        self.size = 0
        self.cursor = 0

    def add(self, obs: torch.Tensor, labels: torch.Tensor) -> None:
        if obs.shape[0] >= self.capacity:
            obs, labels = obs[-self.capacity :], labels[-self.capacity :]
        count = int(obs.shape[0])
        first = min(count, self.capacity - self.cursor)
        self.obs[self.cursor : self.cursor + first].copy_(obs[:first])
        self.labels[self.cursor : self.cursor + first].copy_(labels[:first])
        self.valid[self.cursor : self.cursor + first] = True
        remaining = count - first
        if remaining:
            self.obs[:remaining].copy_(obs[first:])
            self.labels[:remaining].copy_(labels[first:])
            self.valid[:remaining] = True
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def invalidate_recent(self, count: int) -> None:
        """Quarantine the newest samples after a rejected covariate-shift batch."""
        count = min(max(int(count), 0), self.size)
        if not count:
            return
        start = (self.cursor - count) % self.capacity
        first = min(count, self.capacity - start)
        self.valid[start : start + first] = False
        remaining = count - first
        if remaining:
            self.valid[:remaining] = False

    @property
    def active_size(self) -> int:
        return int(self.valid[: self.size].sum().item())

    def minibatches(self, batch_size: int):
        order = torch.nonzero(self.valid, as_tuple=False).flatten()
        order = order[torch.randperm(order.numel(), device=self.obs.device)]
        for start in range(0, order.numel(), batch_size):
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
    if a.dagger_phase_coverage:
        base_families = tuple(
            getattr(cfg, "tracking_path_family_schedule", ())
            or TRACKING_PATH_FAMILIES
        )
        phases = (0.0, 0.25, 0.50, 0.75)
        cfg.tracking_path_family_schedule = tuple(
            family for family in base_families for _ in phases
        )
        cfg.tracking_path_start_phase_schedule = tuple(
            phase for _ in base_families for phase in phases
        )
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
    # The wide terrain phase curriculum is safe to randomize during DAgger;
    # arbitrary arm-trajectory starts are not: the privileged teacher's own
    # preflight is calibrated from the canonical deployed arm pose.
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
    # Distillation may begin only from a teacher that meets the same observable
    # quality envelope required of the student.  A looser teacher preflight
    # previously admitted 11 cm cut error and 0.23 contact, making a promotable
    # student impossible regardless of optimizer quality.
    report["passed"] = bool(
        report["all_finite"]
        and report["slew_limit_respected"]
        and not report["direct_target_bypass_enabled"]
        and report["scan_valid_fraction"] >= 0.80
        and report["mean_wheel_road_contact_fraction"] >= 0.20
        and report["ground_contact_fraction"] >= float(a.teacher_contact_floor)
        and report["mean_abs_cut_height_error_m"] <= 0.04
        and report["mean_abs_lateral_error_m"] <= 0.30
        and report["p01_head_alignment"] >= 0.90
        and report["min_head_alignment"] >= 0.65
        and report["p95_tool_acceleration_mps2"] <= 120.0
        and report.get("q5_recovery_signal_present", True)
    )
    return report


@torch.no_grad()
def safety_component_masks(raw) -> dict[str, torch.Tensor]:
    """Return the same component masks used by the raw safety predicate."""
    cfg = raw.cfg
    cylinder_q = raw.robot.data.joint_pos[:, raw._cylinder_joint_ids]
    limits = raw.robot.data.soft_joint_pos_limits[:, raw._cylinder_joint_ids]
    margin = float(getattr(cfg, "cylinder_limit_margin_m", 0.002))
    components = {
        "cylinder_limit": (
            (cylinder_q < limits[..., 0] - margin)
            | (cylinder_q > limits[..., 1] + margin)
        ).any(dim=-1),
        "nonfinite": raw._terrain_nonfinite_mask(),
        "truck_drift": (
            raw._truck_post_constraint_drift_xy
            > float(getattr(cfg, "truck_drift_termination_m", 0.02))
        )
        | (
            raw._truck_drift_xy
            > float(getattr(cfg, "truck_raw_reaction_termination_m", 0.05))
        ),
        "head_alignment": raw._measure_head_down_alignment()
        < float(getattr(cfg, "head_down_termination_alignment", 0.98)),
        "clearance": raw.cut_height_error().abs()
        > float(getattr(cfg, "cutter_clearance_hard_band_m", 0.20)),
    }
    warmup = (raw._preposition_counter > 0) | (raw._settle_counter > 0)
    components = {name: value & ~warmup for name, value in components.items()}
    components["union"] = torch.stack(tuple(components.values()), dim=0).any(dim=0)
    return components


@torch.no_grad()
def student_validation(raw, actor, steps: int) -> dict:
    """Validate the saved actor on the full horizon with teacher execution off."""
    obs, _ = raw.reset(seed=int(a.seed) + 1001)
    raw.set_teacher_execute_mask(
        torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    )
    errors, laterals, contacts, alignments, accelerations, headrooms = [], [], [], [], [], []
    termination_count = 0
    safety_violation_count = 0
    safety_component_counts = {
        "cylinder_limit": 0,
        "nonfinite": 0,
        "truck_drift": 0,
        "head_alignment": 0,
        "clearance": 0,
    }
    max_truck_travel_m = 0.0
    nonfinite = 0
    for step_index in range(int(steps)):
        action = actor(obs)
        obs, reward, terminated, truncated, _ = raw.step(action)
        termination_count += int((terminated & ~truncated).sum())
        nonfinite += int((~torch.isfinite(obs["policy"]).all(dim=-1)).sum())
        nonfinite += int((~torch.isfinite(reward)).sum())
        nonfinite += int((~torch.isfinite(action).all(dim=-1)).sum())
        errors.append(raw.cut_height_error().abs())
        laterals.append(raw.verge_lateral_error().abs())
        contacts.append(raw.cutter_ground_contact().float())
        alignments.append(raw._measure_head_down_alignment())
        accelerations.append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)
        )
        measured_x, _ = raw._refresh_cylinder_state()
        headroom_low, headroom_high = raw._cylinder_headroom(measured_x)
        headrooms.append(torch.minimum(headroom_low, headroom_high).amin(dim=-1))
        safety_components = safety_component_masks(raw)
        safety_violation_count += int(safety_components["union"].sum())
        for name in safety_component_counts:
            safety_component_counts[name] += int(safety_components[name].sum())
        max_truck_travel_m = max(max_truck_travel_m, float(raw._truck_travel_m.max()))
        if step_index == 0 or (step_index + 1) % 256 == 0:
            print(
                "[PPO3-STUDENT-VALIDATION-PROGRESS] "
                + json.dumps(
                    {
                        "step": step_index + 1,
                        "steps": int(steps),
                        "max_truck_travel_m": max_truck_travel_m,
                        "mean_abs_cut_height_error_m": float(errors[-1].mean()),
                        "mean_abs_lateral_error_m": float(laterals[-1].mean()),
                        "ground_contact_fraction": float(contacts[-1].mean()),
                        "episode_termination_count": termination_count,
                        "safety_violation_count": safety_violation_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    error_steps = torch.stack(errors)
    lateral_steps = torch.stack(laterals)
    contact_steps = torch.stack(contacts)
    alignment_values = torch.cat(alignments)
    headroom_values = torch.stack(headrooms)
    family_rows = []
    for family in sorted(set(raw._tracking_path_family_names)):
        family_mask = torch.as_tensor(
            [name == family for name in raw._tracking_path_family_names],
            dtype=torch.bool,
            device=raw.device,
        )
        family_rows.append(
            {
                "family": family,
                "mean_abs_lateral_error_m": float(lateral_steps[:, family_mask].mean()),
                "ground_contact_fraction": float(contact_steps[:, family_mask].mean()),
            }
        )
    result = {
        "steps": int(steps),
        "num_envs": int(raw.num_envs),
        "episode_length_s": float(raw.cfg.episode_length_s),
        "mow_strip_length_m": float(raw.cfg.mow_strip_length_m),
        "max_truck_travel_m": max_truck_travel_m,
        "all_finite": nonfinite == 0,
        "nonfinite_count": nonfinite,
        "mean_abs_cut_height_error_m": float(error_steps.mean()),
        "worst_family_mean_abs_lateral_error_m": max(
            row["mean_abs_lateral_error_m"] for row in family_rows
        ),
        "worst_family_ground_contact_fraction": min(
            row["ground_contact_fraction"] for row in family_rows
        ),
        "p01_head_alignment": float(torch.quantile(alignment_values, 0.01)),
        "p95_tool_acceleration_mps2": float(
            torch.quantile(torch.cat(accelerations), 0.95)
        ),
        "p01_cylinder_headroom_m": float(
            torch.quantile(headroom_values.reshape(-1), 0.01)
        ),
        # Keep the established evaluation meaning: this is the raw terrain
        # safety predicate rate, not the synchronized reset rate. Expose the
        # latter separately so the two diagnostics cannot be confused again.
        "safety_termination_rate": safety_violation_count
        / max(int(steps) * raw.num_envs, 1),
        "safety_violation_rate": safety_violation_count
        / max(int(steps) * raw.num_envs, 1),
        "episode_termination_rate": termination_count
        / max(int(steps) * raw.num_envs, 1),
        "safety_component_rates": {
            name: count / max(int(steps) * raw.num_envs, 1)
            for name, count in safety_component_counts.items()
        },
        "path_family_results": family_rows,
    }
    result["passed"] = bool(
        result["all_finite"]
        and result["max_truck_travel_m"] >= float(raw.cfg.mow_strip_length_m) - 0.05
        and result["mean_abs_cut_height_error_m"] <= 0.04
        and result["worst_family_mean_abs_lateral_error_m"] <= 0.30
        and result["worst_family_ground_contact_fraction"] >= 0.70
        and result["p01_head_alignment"] >= 0.90
        and result["p01_cylinder_headroom_m"] >= 0.005
        and result["p95_tool_acceleration_mps2"] <= 120.0
        and result["safety_violation_rate"] <= 0.08
    )
    return result


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
        "teacher_warmup_iterations": int(a.teacher_warmup_iterations),
        "teacher_execute_floor": float(a.teacher_execute_floor),
        "obs_normalizer": str(a.obs_normalizer),
        "prior_anchor_weight": float(a.prior_anchor_weight),
        "teacher_label_mode": str(a.teacher_label_mode),
        "validation_only": bool(a.validation_only),
        "student_error_recovery_threshold": float(a.student_error_recovery_threshold),
        "recovery_teacher_iterations": int(a.recovery_teacher_iterations),
        "recovery_execute_floor": float(a.recovery_execute_floor),
        "dagger_phase_coverage": bool(a.dagger_phase_coverage),
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
    if a.validation_only:
        student_probe = student_validation(raw, actor, int(a.student_probe_steps))
        report["student_validation"] = student_probe
        print(
            "[PPO3-STUDENT-VALIDATION] "
            + json.dumps(student_probe, sort_keys=True),
            flush=True,
        )
        out = (ROOT / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload["iter"] = 0
        payload["infos"] = dict(payload.get("infos") or {})
        payload["infos"].update(
            {
                "validation_only_source": str(resume_path),
                "validation_only": True,
                "student_validation": student_probe,
                "actuator": str(a.actuator),
                "schema": "terrain_shared_v2",
                "observation_width": int(raw.cfg.observation_space),
                "action_width": int(raw.cfg.action_space),
            }
        )
        torch.save(payload, out)
        report.update(
            {
                "checkpoint": str(out),
                "history": [],
            }
        )
        report_path = (
            (ROOT / a.report).resolve()
            if not Path(a.report).is_absolute()
            else Path(a.report)
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[PPO3-TERRAIN-DAGGER] wrote validation checkpoint {out}", flush=True)
        env.close()
        simulation_app.close()
        return
    prior_actor = copy.deepcopy(actor).to(a.device)
    prior_actor.eval()
    for parameter in prior_actor.parameters():
        parameter.requires_grad_(False)
    print(
        "[PPO3-TERRAIN-DAGGER] frozen warm actor anchor installed: "
        f"weight={float(a.prior_anchor_weight):g}",
        flush=True,
    )
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(a.learning_rate))
    obs, _ = raw.reset(seed=int(a.seed) + 1)
    replay = ReplayBuffer(
        int(a.replay_capacity), int(obs["policy"].shape[-1]), int(raw.cfg.action_space), raw.device
    )
    history = []
    recovery_remaining = 0
    handoff_floor = float(a.teacher_execute_floor)
    handoff_steps = max(
        int((a.iterations - a.teacher_warmup_iterations) * a.anneal_fraction),
        1,
    )

    for iteration in range(int(a.iterations)):
        handoff_iteration = iteration - int(a.teacher_warmup_iterations)
        beta = (
            1.0
            if handoff_iteration <= 0
            else max(1.0 - handoff_iteration / handoff_steps, 0.0)
        )
        beta = max(beta, handoff_floor)
        recovery_active = recovery_remaining > 0
        if recovery_active:
            beta = 1.0
            recovery_remaining -= 1
        collected_obs, collected_labels = [], []
        teacher_mask_total = 0
        unreachable_label_total = 0
        label_total = 0
        student_errors, student_laterals, student_contacts = [], [], []
        student_action_label_errors = []
        teacher_label_actions = []
        with torch.no_grad():
            for _ in range(int(a.steps_per_iteration)):
                mask = torch.rand(raw.num_envs, device=raw.device) < beta
                raw.set_teacher_execute_mask(mask)
                if a.obs_normalizer == "adapt":
                    actor.update_normalization(obs)
                collected_obs.append(obs["policy"].detach().clone())
                student_action = actor(obs)
                previous_applied = raw._sole_applied.detach().clone()
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
                teacher_applied = raw.teacher_action.detach().clone()
                if a.teacher_label_mode == "raw":
                    hydraulic_label = raw.terrain_teacher_raw_action[:, :4].detach().clone()
                    unreachable = torch.zeros_like(hydraulic_label, dtype=torch.bool)
                else:
                    hydraulic_label, unreachable = teacher_labels(
                        teacher_action=teacher_applied[:, :4],
                        previous_applied_action=previous_applied[:, :4],
                        filter_beta=float(getattr(raw.cfg, "sole_action_filter", 0.75)),
                    )
                if teacher_applied.shape[-1] > 4:
                    hydraulic_label = torch.cat(
                        [
                            hydraulic_label,
                            raw.terrain_teacher_raw_action[:, 4:].detach().clone(),
                        ],
                        dim=-1,
                    )
                collected_labels.append(hydraulic_label)
                teacher_label_actions.append(hydraulic_label[:, :4].detach().clone())
                unreachable_label_total += int(unreachable.sum())
                label_total += int(unreachable.numel())
                student_envs = ~mask
                if bool(student_envs.any()):
                    student_action_label_errors.append(
                        (student_action[student_envs, :4] - hydraulic_label[student_envs, :4]).abs()
                    )
                    student_errors.append(raw.cut_height_error()[student_envs].abs())
                    student_laterals.append(raw.verge_lateral_error()[student_envs].abs())
                    student_contacts.append(raw.cutter_ground_contact()[student_envs].float())
                teacher_mask_total += int(mask.sum())
        replay.add(torch.cat(collected_obs), torch.cat(collected_labels))

        actor_before_update = copy.deepcopy(actor.state_dict())
        optimizer_before_update = copy.deepcopy(optimizer.state_dict())
        losses = []
        anchor_losses = []
        anchor_action_errors = []
        if not recovery_active:
            for _ in range(int(a.epochs_per_iteration)):
                for batch_obs, batch_label in replay.minibatches(int(a.minibatch_size)):
                    actor_mean = actor({"policy": batch_obs})
                    with torch.no_grad():
                        prior_mean = prior_actor({"policy": batch_obs})
                    if bool(raw.cfg.sole_head_joint_action):
                        squared = (actor_mean - batch_label).square()
                        weights = torch.ones(
                            squared.shape[-1], device=squared.device, dtype=squared.dtype
                        )
                        weights[4] = float(a.head_loss_weight)
                        loss = (squared * weights).sum(dim=-1).mean() / weights.sum()
                    else:
                        loss = behaviour_cloning_loss(actor_mean, batch_label)
                    anchor_loss = (actor_mean - prior_mean).square().mean()
                    loss = loss + float(a.prior_anchor_weight) * anchor_loss
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optimizer.step()
                    losses.append((float(loss), float(grad_norm)))
                    anchor_losses.append(float(anchor_loss.detach()))
                    anchor_action_errors.append(
                        (actor_mean.detach() - prior_mean).abs().mean(dim=0)
                    )
        row = {
            "iteration": iteration,
            "teacher_warmup_active": iteration < int(a.teacher_warmup_iterations),
            "teacher_execute_probability": beta,
            "teacher_executed_fraction": teacher_mask_total
            / max(a.steps_per_iteration * raw.num_envs, 1),
            "student_control_fraction": 1.0
            - teacher_mask_total / max(a.steps_per_iteration * raw.num_envs, 1),
            "replay_size": replay.active_size,
            "unreachable_label_fraction": unreachable_label_total
            / max(label_total, 1),
            "bc_loss": sum(v[0] for v in losses) / max(len(losses), 1),
            "prior_anchor_loss": sum(anchor_losses) / max(len(anchor_losses), 1),
            "teacher_label_mean_abs_action": float(
                torch.cat(teacher_label_actions).abs().mean()
            ),
            "teacher_label_mean_abs_action_by_axis": torch.cat(
                teacher_label_actions
            ).abs().mean(dim=0).tolist(),
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
        if student_errors:
            row.update(
                {
                    "student_control_mean_abs_action_label_error": float(
                        torch.cat(student_action_label_errors).mean()
                    ),
                    "student_control_mean_abs_action_label_error_by_axis": torch.cat(
                        student_action_label_errors
                    ).mean(dim=0).tolist(),
                    "student_control_mean_abs_cut_height_error_m": float(
                        torch.cat(student_errors).mean()
                    ),
                    "student_control_mean_abs_lateral_error_m": float(
                        torch.cat(student_laterals).mean()
                    ),
                    "student_control_ground_contact_fraction": float(
                        torch.cat(student_contacts).mean()
                    ),
                }
            )
        if anchor_action_errors:
            row["prior_anchor_mean_abs_action_drift_by_axis"] = torch.stack(
                anchor_action_errors
            ).mean(dim=0).tolist()

        student_error = float(
            row.get("student_control_mean_abs_action_label_error", 0.0)
        )
        recovery_triggered = bool(
            bool(student_errors)
            and student_error > float(a.student_error_recovery_threshold)
        )
        if recovery_triggered:
            actor.load_state_dict(actor_before_update)
            optimizer.load_state_dict(optimizer_before_update)
            replay.invalidate_recent(a.steps_per_iteration * raw.num_envs)
            recovery_remaining = max(
                recovery_remaining, int(a.recovery_teacher_iterations)
            )
            handoff_floor = max(handoff_floor, float(a.recovery_execute_floor))
        row["student_error_recovery_active"] = recovery_active
        row["student_error_recovery_triggered"] = recovery_triggered
        row["recovery_teacher_iterations_remaining"] = recovery_remaining

        # A large student/teacher disagreement is an unsafe covariate-shift
        # batch, not a useful gradient target for the actor that produced it.
        # Restore the pre-update actor, quarantine that batch, and spend the
        # next batches collecting teacher-only recovery data while the actor
        # stays frozen. Reopen mixed control only at the higher recovery floor.
        update_rejected = recovery_triggered
        row["dagger_update_rejected"] = update_rejected
        history.append(row)
        print("[PPO3-TERRAIN-DAGGER] " + json.dumps(row, sort_keys=True), flush=True)

    student_probe = None
    if int(a.student_probe_steps) > 0:
        student_probe = student_validation(raw, actor, int(a.student_probe_steps))
        report["student_validation"] = student_probe
        print(
            "[PPO3-STUDENT-VALIDATION] "
            + json.dumps(student_probe, sort_keys=True),
            flush=True,
        )

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
        "obs_normalizer": str(a.obs_normalizer),
        "prior_anchor_weight": float(a.prior_anchor_weight),
        "teacher_label_mode": str(a.teacher_label_mode),
        "head_exploration_std": float(a.head_exploration_std),
        "distillation_resume": (
            str(Path(a.resume).expanduser().resolve()) if a.resume else None
        ),
        "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
        "teacher_preflight": preflight,
        "student_validation": student_probe,
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
