"""Plant and runner configuration for the sole-controller ROM task."""

from __future__ import annotations

from isaaclab.utils import configclass

from verge_task.terrain_env_cfg import (
    TerrainSolePolicyEnvCfg,
    TerrainSolePolicyPPORunnerCfg,
    apply_terrain_curriculum_stage,
)


def apply_c4_authority_stage(cfg, stage: int) -> None:
    """Apply a curriculum stage and refresh all per-axis authority values."""
    apply_terrain_curriculum_stage(cfg, stage)
    base_scale = max(float(cfg.action_scale), 1.0e-9)
    cfg.cylinder_target_max_lead_per_axis_m = (
        float(getattr(cfg, "cylinder_target_lead_c1_m", 0.05)),
        float(getattr(cfg, "cylinder_target_lead_c2_m", 0.05)),
        float(getattr(cfg, "cylinder_target_lead_c3_m", 0.05)),
        float(getattr(cfg, "cylinder_target_lead_c4_m", 0.020)),
    )
    cfg.cylinder_action_scale_per_axis_m = (
        base_scale,
        base_scale,
        base_scale,
        base_scale * float(getattr(cfg, "c4_action_scale_gain", 3.0)),
    )


@configclass
class TerrainC4AuthorityEnvCfg(TerrainSolePolicyEnvCfg):
    """PPO3 verge task with the c4 authority experiment enabled."""

    # 0.05 let the commanded target run 5 cm ahead of the measured extension,
    # which is enough for a cylinder to chase it into its own stop and have
    # nothing left.  Measured on model_7065 over meshes 1337/17/909: at 0.05
    # the headroom decayed along the path and pinned at exactly 0.0000 above
    # 4 km/h -- mesh 17's 0.53 m hill drove a cylinder to its limit and there
    # was no longer time to retract before the path ended, taking
    # `double_s_curve` cut error to 0.1326 m.  At 0.030 the same policy holds
    # 0.037-0.039 m of headroom FLAT from 2 to 7 km/h and clears all eight
    # gates at all six speeds; that cut error drops to 0.0040.
    #
    # Note this reverses the measurement quoted in the evaluator's
    # --cylinder_target_max_lead_per_axis_m help text (0.05 halving boom_sweep
    # lateral error).  That predates boom_sweep's re-authoring to a pure-XY
    # sweep, which removed the terrain-coupled lateral demand it was chosen
    # for.  c4 stays at 0.020: relaxing it collapses head alignment.
    cylinder_target_lead_c1_m: float = 0.030
    cylinder_target_lead_c2_m: float = 0.030
    cylinder_target_lead_c3_m: float = 0.030
    cylinder_target_lead_c4_m: float = 0.020
    c4_action_scale_gain: float = 1.0

    terrain_action_magnitude_weight: float = 0.0
    lateral_tracking_std_m: float = 0.30

    def __post_init__(self):
        super().__post_init__()
        self.sole_head_joint_action = False
        self.action_space = 4
        apply_c4_authority_stage(self, int(getattr(self, "curriculum_stage", 0) or 4))


@configclass
class TerrainC4AuthorityPPORunnerCfg(TerrainSolePolicyPPORunnerCfg):
    """PPO runner for ppo3_1; same architecture as PPO3's terrain runner."""

    experiment_name = "ppo3_1_sole_c4_authority"

    def __post_init__(self):
        super().__post_init__()
        self.actor.distribution_cfg.init_std = 0.10
        self.algorithm.entropy_coef = 0.001
