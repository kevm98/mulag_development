"""Configuration for the PPO3 sole-controller environment.

Inherits PPO2's residual configuration for everything that is genuinely shared
-- the plant (ROM-A), the trajectory library, the deployment/preposition
contract, the safety gates -- and overrides what changes when the policy stops
being a corrector.  Fields whose names begin with ``sole_`` are new; the
``dls_*`` fields are inherited but now govern the reset-only deployment hold
rather than anything inside a scored episode.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from PPO2_ROM_PhysX.policy.env_cfg import (
    ResidualPolicyEnvCfg,
    ResidualPolicyPPORunnerCfg,
)
from PPO2_ROM_PhysX.runtime.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg_PLAY
from PPO3_SOLE_PhysX.observations import sole_observation_dim


def apply_sole_curriculum_stage(cfg, stage: int) -> None:
    """Authority and speed schedule for the sole controller.

    PPO2's curriculum grew the residual from 1 mm to 4 mm around a baseline that
    was already tracking.  PPO3 has no baseline, so the schedule ramps physical
    authority (``action_scale``, metres of cylinder travel per unit action) and
    path speed together.

    ``sole_max_action_rate`` is deliberately held CONSTANT at 0.25 across every
    stage and is not a curriculum knob.  Measured 2026-07-28
    (``reports/reachability_stage1.json``): the DLS teacher's own command stream
    has a structural per-step bound of ``dls_action_smoothing * 2 * max_action``
    = 0.2, and the probe measures 0.2077.  An earlier schedule ramped this limit
    0.10 -> 0.25, which put stage 1 at half the teacher's envelope and made
    0.43 % of the distillation labels unreachable by construction -- a floor on
    imitation error that no amount of training removes.  The smoothness the tight
    limit was meant to buy is already bought by ``action_scale``, which is what
    actually bounds physical motion.

    Trajectory and clearance semantics are held constant across all four
    stages.  PPO2's stages 1-3 used the legacy ground arc with absolute
    clearance (``clearance_relative_to_reference = False``) and a 0.15 m band in
    order to reproduce an existing checkpoint's exact conditions.  PPO3 has no
    checkpoint to reproduce, and inheriting those settings made stage 1
    unusable: measured 2026-07-28, the DLS teacher itself was terminated roughly
    every 37 steps, 100 % on clearance, peaking at 0.195 m against the 0.150 m
    band while the truck sat at a ride height of 0.149 m.  That is the
    frame-inconsistency documented in ``ResidualPolicyEnv._clearance_signal``,
    where ride height leaks one-for-one into the reported clearance -- not a
    real cutter excursion.  Every stage therefore uses the reference-relative
    signal and the 0.20 m band.
    """
    stage = int(stage)
    if stage not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"curriculum stage must be 1-6; got {stage}")
    # Stage 6 shares stage 5's arm contract exactly -- same authority, filter,
    # slew and clearance band. Only the GROUND and the tracking path differ, so
    # a stage-6 result isolates those and nothing else.
    if stage == 6:
        stage = 5

    # Shared across all stages. See the note above.
    cfg.trajectory_shape = "line"
    cfg.randomize_trajectory_shape = True
    cfg.seed_lead_time_s = 0.25
    cfg.hydraulic_target_filter_beta = 0.5
    cfg.command_rom_direction_guard = True
    cfg.clearance_relative_to_reference = True
    cfg.cutter_clearance_hard_band_m = 0.20
    cfg.sole_max_action_rate = 0.25

    if stage == 1:
        cfg.path_speed_mps = 0.10
        cfg.action_scale = 0.003
        cfg.sole_action_filter = 0.85
    elif stage == 2:
        cfg.path_speed_mps = 0.13
        cfg.action_scale = 0.004
        cfg.sole_action_filter = 0.80
    elif stage == 3:
        cfg.path_speed_mps = 0.15
        cfg.action_scale = 0.004
        cfg.sole_action_filter = 0.75
    elif stage in (4, 5):
        # Stage 5 is stage 4's arm contract on rougher ground. The action scale,
        # rate limit and filter must stay identical or a stage-5 result would be
        # measuring a different controller, not a harder verge.
        cfg.path_speed_mps = 0.18
        cfg.action_scale = 0.004
        cfg.sole_action_filter = 0.70
    cfg.curriculum_stage = stage
    cfg.force_zero_policy = False


@configclass
class SolePolicyEnvCfg(ResidualPolicyEnvCfg):
    """PPO3: the policy is the only controller in the scored loop.

    The EE Jacobian cascade and the cylinder DLS controller are still
    constructed, because reset-time deployment and ROM priming use them, but
    neither is consulted between ``_pre_physics_step`` and the plant.
    """

    # Four outputs preserve the checkpoint/action interface. Tasks may remove
    # authority from an evidence-backed safety axis and give it a static target;
    # every remaining channel is policy-owned with no teacher fallback.
    action_space = 4

    # Metres of USD cylinder travel commanded per unit action. This is the
    # policy's entire authority budget; the DLS used to cap itself at
    # max_desired_delta_m = 0.004 per step and that is the stage-3/4 value.
    action_scale: float = 0.004

    # --- Sole-controller action chain ---
    sole_action_filter: float = 0.75
    """EMA on the raw action. Replaces CylinderDLSConfig.action_smoothing."""

    sole_max_action_rate: float = 0.20
    """Max change in normalized action per control step. Replaces the DLS slew."""

    sole_limit_shield_margin_m: float = 0.02
    """Extension headroom over which authority fades toward a limit. 0 disables.

    The DLS clamped its setpoint into the learned range unconditionally. This is
    the softer equivalent: it only ever scales the policy's command down, and
    only in the direction that consumes headroom.
    """

    # --- Observation ---
    sole_ik_hint: bool = False
    """Append the cascade's cylinder error to the observation (ablation only).

    False is the sole-controller contract: no inverse-kinematics solution
    reaches the policy. True quantifies what the Jacobian was worth by giving
    the policy the signal PPO2 had at slice 16:20, and must be reported as an
    ablation, never as the headline result.
    """

    # --- Distillation teacher ---
    sole_teacher_mode: str = "off"
    """"off" | "label" | "execute". The DLS teacher, for the distillation phase.

    "off" is the sole-controller contract and the only mode in which a result
    may be reported: the teacher code in ``_pre_physics_step`` is unreachable.
    "label" computes the teacher's command after the student's command is final,
    for DAgger labels, without touching the plant. "execute" additionally lets
    the teacher drive the environments selected by ``set_teacher_execute_mask``,
    which is how the annealed hand-over works.

    ``scripts/evaluate.py`` refuses to run unless this is "off".
    """

    # --- Reward ---
    sole_limit_penalty_weight: float = -1.0
    sole_limit_margin_m: float = 0.02
    sole_action_authority: tuple = (1.0, 1.0, 1.0, 1.0)
    """Per-cylinder policy authority; zero is an evidence-backed safety lock."""

    sole_head_joint_action: bool = True
    """Drive ``Messerkopf_joint`` as a real 5th action instead of slaving it to c4.

    ``Messerkopf_joint`` carries no cylinder and is not hydraulic, but the
    inherited pipeline commands it as ``0.76 * q4 + b`` -- a fixed feedforward
    ratio with no feedback. Measured consequence: it misses that reference by
    0.2096 rad at p99 while its actuator is a rigid position lock (stiffness
    5e6, effort limit 20000), i.e. the coupling asks for a pose the joint cannot
    hold and the error shows up as head tilt. teacher_student drives the same
    joint independently.

    Bolting a bounded correction onto the feedforward was tried and does not
    generalize: opening the levelling clamp scored 17/17 on one mesh and 14/17
    on the held-out one. ``min_head_alignment`` fails on varied terrain in every
    coupled configuration measured, while p01 stays healthy at 0.976-0.988.

    This is part of the shared PPO3/PPO4 interface. Legacy four-action
    checkpoints must be migrated before they can be resumed.
    """
    sole_head_action_to_c4_gain: float = 0.4
    """Mix q5 into hydraulic cylinder 4 as a dedicated head-tilt residual.

    Runtime authority testing showed that moving ``Messerkopf_joint`` changes
    its joint coordinate but not the cutter down-axis. The tilt DOF is
    ``Messerkopf_Schwenk_joint``, actuated by cylinder 4. q5 therefore remains
    a distinct policy output but reaches the plant through the validated ID/FD
    cylinder path instead of pretending an alignment-invariant joint can level
    the cutter.
    """
    sole_head_action_scale_rad: float = 0.010
    """Radians of head-joint reference commanded per unit action per step.

    The cylinders get 0.004 m per step; this is the rotational counterpart.
    Integrated onto the reference and clamped to the joint's own soft limits, so
    the policy steers the head rather than being handed an absolute pose.
    """
    sole_head_correction_max_rad: float = 0.35
    """Bound on the policy's integrated head correction above the coupling.

    The coupling 0.76*q4 is kept as feedforward because it tracks the boom the
    head hangs from -- freezing the reference instead measured min_head_alignment
    0.8369 against 0.9296 for the coupled reference. This bounds how far the
    policy may depart from that feedforward, so the integrator cannot wind the
    head into a pose the joint will not hold.
    """
    sole_head_action_filter: float = 0.75
    """Low-pass on the head action, matching the cylinder action filter."""
    sole_head_teacher_band_rad: float = 0.03
    """Tilt at which the head teacher reaches full command.

    Normalizing by the per-step action scale (0.010 rad) instead made the
    teacher saturate at 0.57 deg and behave as bang-bang, measured
    min_head_alignment 0.9296 against a healthy p01 of 0.9858 -- the worst
    samples were overshoot, not lack of authority. A saturating demonstration is
    also the worst possible signal to distil from.
    """
    sole_head_teacher_damping: float = 0.0
    """Rate damping on the head teacher, in units of the proportional gain.

    The head hangs off a moving boom, so pure proportional control chases an
    accelerating reference and overshoots precisely at the worst-case samples
    the hard-min alignment gate reads.
    """
    sole_head_teacher_sign: float = 1.0
    """Measured head-action -> tilt polarity, confirmed by direct probe.

    +1 head action drove the joint to 0.34034 rad and the head-down tilt to
    -0.12261 rad. Copying c4's sign convention instead of measuring this made
    the head teacher worse than no head control at all.
    """
    sole_head_teacher_gain: float = 0.0
    """Proportional gain of the head channel's distillation teacher.

    The cylinder DLS teacher inverts the cylinder Jacobian and has no opinion
    about ``Messerkopf_joint`` -- c4 only orients the boom the head hangs from.
    So the head gets its own teacher: drive measured world-down tilt to zero.
    Padding the DLS output with zeros instead would teach the student never to
    move the joint at all.
    """
    sole_rate_penalty_weight: float = -0.10
    sole_acceleration_weight: float = -0.20
    sole_acceleration_scale_mps2: float = 30.0

    # PPO2 stage 4 ran with acceleration shaping off because the DLS bounded
    # tool acceleration structurally. Without it the p95 acceleration gate is
    # the binding constraint, so the guard is on by default here.
    residual_tracking_guard_weight: float = 0.0
    residual_tracking_guard_scale_m: float = 0.04

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = sole_observation_dim(ik_hint=bool(self.sole_ik_hint))
        # 4 cylinders, plus Messerkopf_joint when it is driven as a real joint.
        self.action_space = 5 if bool(self.sole_head_joint_action) else 4


@configclass
class SolePolicyEnvCfg_PLAY(SolePolicyEnvCfg, MulagUpperLevelEnvCfg_PLAY):
    """Single-environment config for play/evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False
        self.cutter_clearance_hard_band_m = 0.20


@configclass
class SolePolicyPPORunnerCfg(ResidualPolicyPPORunnerCfg):
    """PPO runner for the sole controller.

    Two departures from PPO2's runner, both consequences of removing the
    baseline.  ``init_std`` rises from 0.05 to 0.30: a residual policy explores
    around a controller that already tracks, whereas a zero action here means
    the arm does not move at all.  ``entropy_coef`` rises for the same reason --
    the early policy has to find the command directions before it can refine
    them.
    """

    experiment_name = "ppo3_sole_controller"
    num_steps_per_env = 128
    max_iterations = 4000
    save_interval = 100
    empirical_normalization = True

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.30),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
