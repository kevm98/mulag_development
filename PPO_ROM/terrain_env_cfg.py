"""Configuration for PPO3-1, the c4-authority sole controller.

PPO3-1 inherits every field of ``TerrainSolePolicyEnvCfg`` -- the verge task,
66-value observation, rewards, safety logic, terrain, path families, gates and
ROM-A actuator. It changes exactly the two plant knobs that the PPO3/PPO4
handover identified as the untested decoupling axis (section 4):

    cylinder_target_max_lead_per_axis_m   c4 lead cap (metres)
    cylinder_action_scale_per_axis_m      c4 action scale (metres/unit action)

c1-c3 keep the shipped 0.018 m lead and 0.004 m action scale. c4 gets a larger
lead cap and a larger action scale so the policy can hold the head level while
c1-c3 hold the offset. ``apply_c4_authority_stage`` derives the per-axis values
from the scalar ``action_scale`` the shared curriculum assigns, so the stage
ramp for c1-c3 is untouched and c4 is exactly ``c4_action_scale_gain`` times
their authority at every stage.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from verge_task.terrain_env_cfg import (
    TerrainSolePolicyEnvCfg,
    TerrainSolePolicyEnvCfg_PLAY,
    TerrainSolePolicyPPORunnerCfg,
    apply_terrain_curriculum_stage,
)


def apply_c4_authority_stage(cfg, stage: int) -> None:
    """Apply the shared curriculum, then re-assert the per-axis c4 authority.

    ``apply_terrain_curriculum_stage`` assigns the scalar ``action_scale``
    (0.003 at stage 1, 0.004 from stage 2 on). The per-axis fields are derived
    from that scalar so the authority ramp for c1-c3 is identical to PPO3 and
    c4 is exactly ``c4_action_scale_gain`` times the c1-c3 value. Calling this
    after every curriculum application (including on stage transitions) is
    required: the shared wrapper overwrites ``action_scale`` but not the
    per-axis tuple, so a stale tuple from a previous stage would freeze c4's
    authority at the wrong value.
    """
    apply_terrain_curriculum_stage(cfg, stage)
    base_scale = max(float(cfg.action_scale), 1.0e-9)
    cfg.cylinder_target_max_lead_per_axis_m = (
        float(getattr(cfg, "cylinder_target_lead_c1_m", 0.018)),
        float(getattr(cfg, "cylinder_target_lead_c2_m", 0.018)),
        float(getattr(cfg, "cylinder_target_lead_c3_m", 0.018)),
        float(getattr(cfg, "cylinder_target_lead_c4_m", 0.30)),
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

    # The untested decoupling axis from the handover. c1-c3 keep the shipped
    # lead; c4 is the corrective head-pitch cylinder, and the handover measured
    # that clamping it makes head alignment monotonically worse. The default
    # c4 lead (0.30 m) is deliberately beyond the 0.20 m that was ever swept.
    cylinder_target_lead_c1_m: float = 0.018
    cylinder_target_lead_c2_m: float = 0.018
    cylinder_target_lead_c3_m: float = 0.018
    cylinder_target_lead_c4_m: float = 0.30
    # Per-axis action scale multiplier for c4. 3x gives the head-pitch cylinder
    # three times c1-c3's physical authority per unit policy action.
    c4_action_scale_gain: float = 3.0

    # Training fixes measured on this exact task (HANDOVER.md sections 2-4).
    # The action-magnitude penalty charges a sustained non-zero command every
    # step, and the three failing path families require exactly that; 0.0 is
    # the measured best value. Lateral std is kept at >= 0.30 (the shared
    # stage-4/5 value) so the lateral reward keeps a live gradient where the
    # failing families sit. c4 exploration is capped separately at resume time
    # (see scripts/train.py), never raised globally.
    terrain_action_magnitude_weight: float = 0.0
    lateral_tracking_std_m: float = 0.30

    def __post_init__(self):
        super().__post_init__()
        # The shared task is a 4-action sole controller here: no direct q5
        # Messerkopf joint, because the experiment is about the hydraulic
        # head-pitch cylinder's authority, not a fifth output.
        self.sole_head_joint_action = False
        self.action_space = 4
        # `curriculum_stage` is assigned BY apply_sole_curriculum_stage, so at
        # __post_init__ time it already exists and is 0 -- getattr's default
        # never fired and the stage guard rejected it. `or` picks the fallback
        # for 0/None while still honouring an explicitly requested stage.
        apply_c4_authority_stage(self, int(getattr(self, "curriculum_stage", 0) or 4))


@configclass
class TerrainC4AuthorityEnvCfg_PLAY(
    TerrainC4AuthorityEnvCfg, TerrainSolePolicyEnvCfg_PLAY
):
    """Single-environment config for play/evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False


@configclass
class TerrainC4AuthorityPPORunnerCfg(TerrainSolePolicyPPORunnerCfg):
    """PPO runner for ppo3_1; same architecture as PPO3's terrain runner."""

    experiment_name = "ppo3_1_sole_c4_authority"

    def __post_init__(self):
        super().__post_init__()
        # Same 0.10 init_std / 0.001 entropy as the shared terrain runner.
        self.actor.distribution_cfg.init_std = 0.10
        self.algorithm.entropy_coef = 0.001
