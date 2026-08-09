"""Configuration for the independent PPO -> command ROM -> PhysX task."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from PPO1_ID_FD_PhysX.tasks.env_cfg import (
    IDFDCylinderSpaceEnvCfg,
    IDFDCylinderSpacePPORunnerCfg,
)


_ROM_ROOT = Path(__file__).resolve().parents[2]
_COMMAND_CHECKPOINT = _ROM_ROOT / "checkpoints/command_run_001/best_rom.pt"


@configclass
class ROMCylinderSpaceEnvCfg(IDFDCylinderSpaceEnvCfg):
    """Policy -> desired cylinder command -> internal-valve ROM -> PhysX targets."""

    cylinder_dynamics_model = "command_rom"
    command_rom_checkpoint = str(_COMMAND_CHECKPOINT)
    use_sequence_cylinder_models = False
    low_level_servo_mode = "none"

    # These switches belong only to the sibling ID/FD experiment and must be
    # inert in the ROM task.
    id_residual_over_zero_target = False
    fd_residual_over_zero_valve = False
    paired_id_fd_residual = False
    cylinder_idfd_reference_blend = 0.0

    # Carry the ROM's predicted pressure state forward between 50 ms ticks.
    cylinder_hydraulic_feature_mode = "fd_dp_state"
    learned_model_dt_s = 0.05
    cylinder_rom_delta_clip_m = 0.02
    cylinder_target_integrates_from_previous_target = True
    cylinder_target_max_lead_m = 0.025


@configclass
class ROMCylinderSpaceEnvCfg_PLAY(ROMCylinderSpaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class ROMCylinderSpacePPORunnerCfg(IDFDCylinderSpacePPORunnerCfg):
    experiment_name = "rom_cylinder_space"
