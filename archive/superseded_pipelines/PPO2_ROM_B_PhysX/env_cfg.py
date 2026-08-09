"""Configuration for robust PPO2 training on ROM-B."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from PPO2_ROM_PhysX.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyEnvCfg,
    FourCylinderResidualPolicyPPORunnerCfg,
)
from PPO2_ROM_PhysX.runtime.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg_PLAY


ROOT = Path(__file__).resolve().parents[1]


@configclass
class PPO2ROMBPolicyEnvCfg(FourCylinderResidualPolicyEnvCfg):
    observation_space = 42
    action_space = 4
    trajectory_library_path: str = str(
        ROOT / "PPO2_ROM_PhysX/models"
        / "plant_reachable_trajectory_library_v3_circle_c7_retime.json"
    )
    rom_b_mode: str = "random"
    rom_b_e_comp_range: tuple[float, float] = (-0.125, 0.020)
    rom_b_seal_range: tuple[float, float] = (0.0, 1.0)
    rom_b_health_scale: float = 1.0
    rom_b_randomization_scale: float = 0.0

    # Plant-lag observer. Default off so every existing artifact reproduces.
    # When on, the observation grows 42 -> 50 with two exponential estimates of
    # persistent cylinder goal error. No ROM-B factor is exposed: the policy
    # infers the plant from signals it already receives.
    rom_b_plant_observer: bool = False
    rom_b_plant_observer_fast_tau_s: float = 0.25
    rom_b_plant_observer_slow_tau_s: float = 2.0
    # Ablation intervention: "on" | "zero" | "shuffle". Evaluation-time only.
    rom_b_plant_observer_mode: str = "on"

    # Keep the inherited shaping local to the actual acceptance envelope. The
    # ROM-B environment adds bounded tail and hard-event penalties at the gates.
    residual_head_threshold: float = 0.985
    residual_head_penalty_weight: float = -2.0
    residual_ground_band_m: float = 0.180
    residual_ground_penalty_weight: float = -2.0
    residual_tool_acceleration_weight: float = 0.0
    rom_b_head_soft_alignment: float = 0.985
    rom_b_head_hard_alignment: float = 0.980
    rom_b_clearance_soft_m: float = 0.180
    rom_b_clearance_hard_m: float = 0.200
    rom_b_acceleration_soft_mps2: float = 35.0
    rom_b_acceleration_hard_mps2: float = 40.0
    rom_b_head_margin_weight: float = -4.0
    rom_b_clearance_margin_weight: float = -4.0
    rom_b_acceleration_margin_weight: float = -1.0
    # Truck reaction shaping. `truck_raw_reaction` is the dominant remaining
    # termination cause and the only gate with no approach gradient. The hard
    # value mirrors `truck_raw_reaction_termination_m` and does not move it; the
    # weight defaults to 0.0, so this term is inert until a run opts in via
    # `--truck_reaction_shaping`, which is recorded in the run contract.
    rom_b_truck_reaction_soft_m: float = 0.035
    rom_b_truck_reaction_hard_m: float = 0.050
    rom_b_truck_reaction_margin_weight: float = 0.0
    rom_b_hard_safety_event_weight: float = -20.0
    rom_b_acceleration_event_weight: float = -5.0
    # Clearance participation in the ROM-B reward. Defaults preserve the
    # historical behaviour exactly; nothing here changes unless a run opts in via
    # `--exclude_contaminated_clearance_reward`, which is recorded in the run
    # contract. The clearance measurement is known to carry a ~0.164 m frame
    # offset (reports/rom_b_frame_decomposition.md), so excluding it is an
    # experiment, not a fix, and must stay explicit and reproducible.
    rom_b_clearance_in_hard_safety_event: bool = True
    # PRIVILEGED ORACLE -- DIAGNOSTIC ONLY, NEVER CERTIFIABLE.
    #
    # Appends the true hidden factors to the actor and critic observation. This
    # deliberately violates the factor-blind contract, so a run with this enabled
    # can never be presented as a ROM-B robustness result.
    #
    # Purpose: bound what online adaptation could possibly buy. An estimator such
    # as AAM can only be worse than the truth, so the oracle's margin over
    # factor-blind ROM-B is a hard ceiling on AAM's margin over ROM-B. If the
    # oracle does not beat ROM-B, no adaptation scheme can.
    #
    # Exactly the five supported ROM-B factors are exposed.
    rom_b_privileged_factors: bool = False

    # Opt-in: use the frame-consistent clearance signal (orientation-aware cutter,
    # live-truck-pose reference) instead of the historical contaminated one.
    # Default False so every existing artifact reproduces exactly.
    clearance_frame_consistent: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = (
            42
            + (8 if self.rom_b_plant_observer else 0)
            + (5 if self.rom_b_privileged_factors else 0)
        )
        self.action_space = 4


@configclass
class PPO2ROMBPolicyEnvCfg_PLAY(
    PPO2ROMBPolicyEnvCfg, MulagUpperLevelEnvCfg_PLAY
):
    rom_b_mode: str = "nominal"

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False


@configclass
class PPO2ROMBPPORunnerCfg(FourCylinderResidualPolicyPPORunnerCfg):
    experiment_name = "ppo2_rom_b_robust"
    num_steps_per_env = 128
    max_iterations = 500
    save_interval = 10
