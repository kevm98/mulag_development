"""Configuration for the final 26 m wide PPO-ROM task."""

from __future__ import annotations

from isaaclab.utils import configclass
from verge_task.vehicle_speed import configure_path_horizon

from PPO_ROM.terrain_env_cfg import (
    TerrainC4AuthorityEnvCfg,
    TerrainC4AuthorityPPORunnerCfg,
)


@configclass
class Ppo31SoleEnvCfg(TerrainC4AuthorityEnvCfg):
    """Wide terrain, ROM plant, and optional final-reward overrides."""

    ppo31_cut_height: float | None = None
    ppo31_lateral: float | None = None
    ppo31_ground_contact: float | None = None
    ppo31_head_alignment: float | None = None
    ppo31_contact_loss: float | None = None
    ppo31_action_rate: float | None = None
    ppo31_tool_acceleration: float | None = None
    ppo31_contact_force: float | None = None
    ppo31_cylinder_headroom: float | None = None
    ppo31_lateral_scale_m: float | None = None
    ppo31_lateral_fine_scale_m: float | None = None
    ppo31_lateral_fine_fraction: float | None = None
    ppo31_cut_height_scale_m: float | None = None
    ppo31_cut_height_cap: float | None = None
    ppo31_head_threshold: float | None = None
    ppo31_head_scale: float | None = None
    # The headroom term is `weight * clamp(encroachment / scale, max=1)` where
    # encroachment is at most `margin`. With the shipped margin 0.02 and scale
    # 0.10 the ratio can never exceed 0.2, so a cylinder sitting DEAD on its
    # stop costs only -0.40 per step while 0.135 m of lateral error costs
    # -1.73. Measured 2026-08-15 on the AAM init: the policy rails cylinder 3
    # on hard terrain chunks because doing so is cheaper than tracking, which
    # is the reward working as written. Exposing both numbers so the ratio is
    # sweepable instead of frozen in the dataclass default.
    ppo31_headroom_scale_m: float | None = None
    ppo31_headroom_margin_m: float | None = None

    curriculum_stage: int = 6

    def __post_init__(self):
        super().__post_init__()
        dynamics_model = str(getattr(self, "cylinder_dynamics_model", "")).lower()
        # PPO-ID-FD subclasses this config and deliberately enters the parent
        # initializer with its validated fd actuator already selected.  It
        # performs its own strict ID-FD assertion after this call.
        if dynamics_model not in {"command_rom", "fd"}:
            raise RuntimeError(
                "PPO-ROM requires the command-ROM actuator: policy motion "
                "commands are translated to the ROM's internal valve."
            )
        if dynamics_model == "command_rom" and bool(getattr(self, "rom_use_id_valve_cmd", False)):
            raise RuntimeError(
                "PPO-ROM cannot accept an external ID valve command; "
                "use the desired cylinder motion interface."
            )
        from PPO_ROM.terrain_env_cfg import apply_c4_authority_stage

        apply_c4_authority_stage(self, 6)
        self.terrain_relief_profile = "flat_patches"
        self.terrain_tracking_height_sweep_gain = 0.5
        self.terrain_height_scales = (0.50, 0.70, 0.85, 1.00)
        self.terrain_tracking_amplitude_scale = 1.5
        self.terrain_tracking_amplitude_scales = (0.75, 1.20, 1.70, 2.20)
        self.terrain_tracking_height_sweep_gains = (0.25, 0.40, 0.55)
        self.vehicle_speed_min_kmh = 2.0
        self.vehicle_speed_max_kmh = 7.0
        self.vehicle_speed_per_episode = True
        self.ppo31_lateral_fine_scale_m = 0.30
        self.ppo31_lateral_fine_fraction = 0.70
        self.ppo31_lateral_scale_m = 3.00
        # The cutter starts at x=-25.40 in the truck deployment pose and moves
        # the same 26 m as the truck.  Anchoring the strip at -27.12 put the
        # cutter 1.72 m into the path at reset and 1.72 m beyond it at timeout.
        self.mow_strip_start_x_m = -25.40
        self.mow_strip_length_m = 26.0
        self.terrain_tracking_reference_x_m = -25.40
        self.tracking_path_phase_anchor_x_m = -25.40
        self.terrain_tracking_active_length_m = 26.0
        self.episode_length_s = 46.8
        configure_path_horizon(self)
        self.scene.env_spacing = self.mow_strip_length_m + 2.0
        self.terrain_tracking_band_owned_by_task = True
        self.cylinder_target_max_lead_m = 0.018
        # Calibrated on the full 26 m ID-FD and ROM task.  The previous -0.11
        # offset left the cutter above the contact pose, and force relief could
        # then erase the remaining preload by rising to -0.05.
        self.contact_demo_initial_vertical_offset_m = -0.55
        self.contact_demo_max_vertical_offset_m = 0.60
        self.contact_demo_max_upward_offset_m = -0.45
        self.terrain_teacher_max_raw_action = 0.80


@configclass
class Ppo31PPORunnerCfg(TerrainC4AuthorityPPORunnerCfg):
    """Isaac Lab's standard PPO settings, not the fine-tuning ones."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100
        self.experiment_name = "ppo3_1_sole"

        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.schedule = "adaptive"
        self.algorithm.entropy_coef = 0.005
        self.actor.distribution_cfg.init_std = 0.30
