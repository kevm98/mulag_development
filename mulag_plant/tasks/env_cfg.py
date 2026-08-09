from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from mulag_plant.models import COMMAND_ROM_CHECKPOINT
from mulag_plant.mdp.ee_observations import EE_TRACKING_OBS_DIM
from mulag_plant.mdp.observations import CYLINDER_OBS_DIM
from mulag_plant.runtime.mulag_upper_level_env_cfg import (
    MulagUpperLevelEnvCfg,
    MulagUpperLevelEnvCfg_PLAY,
    MulagUpperLevelPPORunnerCfg,
)


# Cylinder actuator effort headroom. See PPO2ROMPhysXEnvCfg.__post_init__ for the
# measurement that fixes this at 16.
CYLINDER_EFFORT_SCALE = 16.0


def _obs_dim_for_task(cfg) -> int:
    """Return observation dimension based on task_type."""
    task_type = str(getattr(cfg, "task_type", "cylinder")).lower()
    if task_type == "ee_tracking":
        return EE_TRACKING_OBS_DIM
    return CYLINDER_OBS_DIM


@configclass
class PPO2ROMPhysXEnvCfg(MulagUpperLevelEnvCfg):
    """Cylinder-space task using one command-conditioned hydraulic ROM.
    Supports task_type = 'cylinder' | 'ee_tracking'."""

    task_type: str = "cylinder"

    # The mowing control point is the cutter bottom, not the Messerkopf body
    # origin.  The parent config explicitly sets this to 0.0, so relying on a
    # getattr fallback silently disabled the measured -0.79 m offset.
    tool_tip_local_offset_z = -0.79

    action_space = 4
    episode_length_s = 12.0
    # action_scale 0.003 + leak 0.05 starved cylinder authority -> hold-still
    # collapse. Step-1 authority probe (leak=0, action_scale=0.02): every cylinder
    # reaches both directions well past the +-0.035 m goal range. Restore authority.
    action_scale = 0.02

    use_cylinder_id_pipeline = True
    use_sequence_cylinder_models = False
    cylinder_dynamics_model = "command_rom"
    cylinder_models_root = "disabled"
    id_cylinder_checkpoint = "disabled"
    fd_cylinder_checkpoint = "disabled"
    rom_cylinder_checkpoint = "disabled"
    base_rom_checkpoint = str(
        Path(__file__).resolve().parents[2]
        / "rom_model/base_run_001/best_base_rom.pt"
    )
    command_rom_checkpoint = str(COMMAND_ROM_CHECKPOINT)
    low_level_servo_mode = "none"
    # Retained shared-config field; inactive because low_level_servo_mode is none.
    fd_servo_usage = "prediction_only"
    fd_residual_over_zero_valve = False
    paired_id_fd_residual = False
    cylinder_idfd_reference_blend = 0.0
    cylinder_fd_residual_output_gain = [1.0, 1.0, 1.0, 1.0]
    # Build the ID's target features on the MODEL's sample time (the Simscape data is a
    # constant 0.05 s), not on the 1/60 s control step. See TRAINING.md.
    learned_model_dt_s = 0.05

    # The per-axis direction guard is REMOVED from the main experiment (kept as a config
    # switch for the ablation). The wiring audit showed that for a
    # single-cylinder command every off-diagonal has cmd_i = 0, so limit = ratio * 0 = 0 and
    # the guard zeroes ALL cross-axis response -- for any ratio. The composite is inherently
    # cross-coupled, so a per-axis rule cannot be applied to it. The ratio cap never binds
    # at all (1.5 / 2.0 / 3.0 give bit-identical matrices; the responses sit at 1-46% of
    # command, far below the 150% cap). Raising the ratio, as previously proposed, is a no-op.
    cylinder_fd_command_direction_guard = False
    cylinder_fd_command_max_ratio = 1.5
    cylinder_actuation_mode = "position"
    max_abs_valve = 1.0
    id_residual_over_zero_target = False
    cylinder_hydraulic_feature_mode = "fd_dp_state"
    command_rom_dp_abs_limit_pa = 1.5e8
    enable_mathbox_ground = False
    mowing_terrain_enable = False

    # Integrate the PhysX joint target in REFERENCE space (target += fd_delta)
    # instead of re-anchoring to the measured joint pos every step
    # (target = measured + fd_delta). The default (False) is the "tracking
    # ratchet": between steps gravity sags the arm, the measured pos drops, and
    # the target follows the sag down, so the small (1-4 mm) commanded delta is
    # washed out and the arm drifts to gravity equilibrium REGARDLESS of the
    # action sign (verified: constant action +1 and -1 both end at the same
    # pose). That removes the policy's authority -> no PPO gradient -> flat error,
    # zero success. Reference integration lets commanded deltas accumulate so the
    # action direction actually determines where the arm goes. _cylinder_pos_target
    # is re-seeded to the reset pose in _reset_idx, so integration starts clean.
    cylinder_target_integrates_from_previous_target = True

    # Leaky integration (anti-runaway). Pure reference integration is a velocity
    # integrator: target += fd_delta. To hold at a goal the policy must output
    # EXACTLY zero, so any small action bias integrates into runaway (verified:
    # deterministic policy starts at the goal then ramps away to 0.6-0.9 m). The
    # leak pulls the target gently back toward the measured joint pos each step:
    #   target = prev_target + fd_delta + leak * (measured_q - prev_target)
    # so a persistent bias reaches a BOUNDED steady offset (fd_delta / leak)
    # instead of running to the joint limit. leak=0 reproduces pure integration;
    # ~0.1 gives a stable regulator the policy can learn quickly. Only applied
    # when cylinder_target_integrates_from_previous_target is True.
    # ID and FD are residualized around neutral below, so the target integrator no
    # longer receives a state-driven zero-command bias.  The former leak now only
    # creates steady-state tracking error for the DLS setpoint controller.
    cylinder_target_leak = 0.0

    # Anti-windup on the reference integrator: how far the commanded PhysX target may
    # lead the measured cylinder. 0 disables. Tuned on the repaired plant -- see
    # the retained idfd_step_gate.py diagnostics.
    # 25 mm is needed only during the transient through a mechanically coupled configuration.
    # Goal-anchored integration prevents the target from winding past the requested goal.
    cylinder_target_max_lead_m = 0.025

    enable_scripted_vehicle_motion = True
    vehicle_speed_min_kmh = 3.0
    vehicle_speed_max_kmh = 8.0

    enable_startup_summary = False
    enable_verbose_startup = False
    enable_console_debug = False
    enable_sequence_tick_debug = False
    enable_reset_debug = False
    enable_pipeline_debug = False
    write_debug_csv = False
    log_target_delta_xdot_clipping = False
    pipeline_debug_prints = False

    # Cylinder tracking params
    cylinder_goal_delta_range_m = 0.035
    cylinder_tracking_std_m = 0.04
    cylinder_progress_weight = 4.0
    cylinder_action_rate_weight = 0.02
    cylinder_valve_weight = 0.01
    cylinder_fd_delta_weight = 0.1
    cylinder_success_tolerance_m = 0.01
    cylinder_success_hold_steps = 10
    cylinder_limit_margin_m = 0.002

    # EE tracking params
    ee_goal_delta_range_m = 0.15
    ee_tracking_std_m = 0.05
    ee_progress_weight = 10.0
    ee_action_rate_weight = 0.02
    ee_valve_weight = 0.01
    ee_fd_delta_weight = 0.1
    ee_success_tolerance_m = 0.02
    ee_success_hold_steps = 10

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.replicate_physics = True
        self.observation_space = _obs_dim_for_task(self)

        # Give the cylinder drives enough force to actually execute a position target.
        # As authored, stiffness (3e7-7.5e7) dwarfs effort_limit (5e4-1e5), so the
        # implicit PD clips to MAX FORCE beyond effort/stiffness = 0.7-10 mrad of error.
        # Every cylinder was therefore a saturated bang-bang actuator:
        #   * holding still at the reset pose already pinned cylinder 2 at its limit;
        #   * at the mowing pose cylinder 1 demanded 1.46e7 N.m against a 5e4 limit (293x)
        #     and d(measured)/d(target) measured ~0.00 out to +-50 mm, in BOTH directions.
        # The arm did not track a target, it sagged to a gravity/constraint equilibrium --
        # which was also observed in the superseded ground-map experiment: commanded and
        # settled cylinder positions differed by up to 0.25 m. No controller, learned or
        # classical, had authority on that plant. Raising effort alone fixed it.
        # Lowering stiffness instead makes it worse (step gain collapses to 0.0-0.47,
        # hold error 30-160 mm). At x16 the drive is a clean force-limited position servo:
        # step gain 0.83-1.02, settle 0.17-0.25 s, ripple <0.07 mm, ZERO saturation, and it
        # is the smallest multiplier that also reaches the extended (mowing) pose.
        for name in (
            "drehzapfen_cylinder",
            "ausleger_i_cylinder",
            "ausleger_ii_cylinder",
            "messerkopf_schwenk_cylinder",
            "messerkopf_passive_coupled",
        ):
            # NOTE: the parent's __post_init__ rebuilds robot_cfg.actuators from scratch,
            # so this must run after super().__post_init__(). The limit that PhysX actually
            # enforces is effort_limit_sim; effort_limit is None here.
            actuator = self.robot_cfg.actuators.get(name)
            if actuator is None or actuator.effort_limit_sim is None:
                raise RuntimeError(f"cylinder actuator {name!r} missing effort_limit_sim")
            actuator.effort_limit_sim = float(actuator.effort_limit_sim) * CYLINDER_EFFORT_SCALE

        # Make the wheels a VELOCITY drive so scripted driving can actually act.
        # The parent configures them as a position drive (stiffness 6e8 = parking
        # brake), so wheel velocity commands were fighting a stiff position hold:
        # measured 3.68 km/h commanded (~1.0 m/s, i.e. ~10 m over a 10 s run) but the
        # truck moved only 0.25 m. The parent's own comment says these gains "must be
        # switched to velocity-drive (stiffness 0, high damping) before wheel velocity
        # targets can act" -- that switch was never actually made.
        wheels = self.robot_cfg.actuators.get("mulag_wheels")
        if wheels is not None:
            wheels.stiffness = 0.0
            wheels.damping = 1.0e6

        # Ground the truck. The MULAG articulation root origin is authored ~1.11 m
        # ABOVE wheel contact, so the parent's __post_init__ spawn of z=0.05 does
        # not put "the wheels just above ground" — it buries the whole truck ~1.06 m
        # into the ground plane (z=0). That is the sunken-truck/sunken-tool symptom.
        # Measured settled height in this env (num_envs=1, ground-cuboid contact at
        # z=0, 250-step settle): root_pos_w.z == 1.2538, base_link +0.14, wheels
        # +0.73, tool +1.09 — the truck rests fully above ground. Spawn just above
        # that settled root height so the wheels drop cleanly onto the plane instead
        # of spawning penetrated and lurching up via depenetration every reset.
        self.robot_cfg.init_state.pos = (-25.4, 4.3, 1.26)


@configclass
class PPO2ROMPhysXEnvCfg_PLAY(PPO2ROMPhysXEnvCfg, MulagUpperLevelEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class PPO2ROMPhysXEETrackingEnvCfg(PPO2ROMPhysXEnvCfg):
    """EE-tracking PPO variant using the same command ROM pipeline."""
    task_type: str = "ee_tracking"
    episode_length_s = 12.0
    action_scale = 0.003          # reference-integration is velocity-style: smaller scale = less drift per unit action bias

    # Reward-shaping tweaks after the Stage-1 "point" run plateaued at ~0.24 m
    # error with 0% success: the exp(-err/std) reward was flat (~0) beyond
    # ~0.15 m, so there was no gradient pulling the EE closer once it stalled.
    ee_tracking_std_m = 0.12       # was 0.05: reward gradient out to ~0.2-0.3 m error
    ee_success_tolerance_m = 0.05  # was 0.02: 2 cm is below this pipeline's precision

    # Break the "hold still" collapse. Across every prior config the policy froze
    # (fd_delta~0) at ~0.12-0.14 m -- WORSE than the 0.08 m do-nothing baseline --
    # because motion was penalized while the tracking gradient near the goal was
    # weak, so "freeze" out-scored "servo". Remove the motion penalties and pull
    # harder toward the goal so active servoing wins.
    ee_fd_delta_weight = 0.0       # was 0.1: directly penalized moving
    ee_valve_weight = 0.0          # was 0.01: penalized the effort needed to hold/servo
    ee_progress_weight = 20.0      # was 10.0: reward closing distance harder

    # Isolate EE-tracking learning: freeze the truck. The goal is anchored in the
    # ROOT (truck) frame, so a driving truck (3-8 km/h + periodic speed resample)
    # makes the arm lag/sway in that frame -> a persistent goal-independent error
    # floor. Learn the arm's inverse map on a stationary base first; re-enable
    # driving as a later curriculum stage once tracking works.
    enable_scripted_vehicle_motion = False

    # Goal-size curriculum: start with small, near-linear goals so the policy can
    # actually close the gap (escapes the "hold still" optimum), then ramp
    # 0.05 -> 0.10 -> 0.15 across runs as success climbs.
    ee_goal_delta_range_m = 0.05   # was 0.15 (base); ramp up once success > 0

    # -- EE trajectory tracking (moving goal) --
    # "point" keeps the legacy static random-goal behavior (reach + hold one
    # point). The other modes move _ee_goal_local along a base-frame path so the
    # policy learns to FOLLOW a trajectory, not just reach a point. Nothing else
    # in the pipeline changes: the same 30-D observation (ee_error, ee_goal, ...)
    # and the same reward (tracking exp + progress) act on whatever the goal is.
    # The goal advances at a fixed path speed so it stays trackable by the arm.
    ee_trajectory_mode = "point"            # "point" | "line" | "curve" | "sweep"
    ee_trajectory_speed_mps = 0.03          # slow goal: easy to catch when warm-starting from a coarse point policy
    ee_trajectory_length_m = 0.30           # line length / sweep forward span (ping-pongs, stays bounded)
    ee_trajectory_sweep_amp_m = 0.15        # sweep lateral half-amplitude
    ee_trajectory_sweep_wavelength_m = 0.30 # sweep period in path length
    ee_trajectory_radius_m = 0.15           # curve arc radius


@configclass
class PPO2ROMPhysXEETrackingEnvCfg_PLAY(PPO2ROMPhysXEETrackingEnvCfg, MulagUpperLevelEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class PPO2ROMPhysXPPORunnerCfg(MulagUpperLevelPPORunnerCfg):
    experiment_name = "ppo2_rom_physx_cylinder_space"
    num_steps_per_env = 128
    max_iterations = 2000
    save_interval = 100
    empirical_normalization = True
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.20),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class PPO2ROMPhysXEETrackingPPORunnerCfg(MulagUpperLevelPPORunnerCfg):
    experiment_name = "ppo2_rom_physx_ee_tracking"
    num_steps_per_env = 128
    max_iterations = 2000
    save_interval = 100
    empirical_normalization = True
    # Bigger net for the 4-cylinder -> 3D-EE inverse map. LOW exploration is
    # essential here: with reference-space (velocity-style) integration, high
    # action noise random-walks the target AWAY from the goal, so the policy never
    # experiences being at the goal and learns to ramp away instead of holding.
    # NOTE: changing hidden_dims makes prior checkpoints incompatible -> train
    # fresh (no --checkpoint) when this changes.
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.15),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
