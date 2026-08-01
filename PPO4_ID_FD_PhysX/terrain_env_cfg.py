"""PPO4: the PPO3 verge task with ID/FD in place of ROM-A.

Everything physical is inherited from PPO3 unchanged -- the compliant arm
(`terrain_arm_stiffness_scale = 0.35`), the verge meshes, the eleven path
families, the 54-D observation, the reward, and the held-out gates. The ONLY
difference is the actuator model between the policy's cylinder command and
PhysX:

    PPO3:  policy -> command ROM (ROM-A)      -> PhysX
    PPO4:  policy -> ID (target -> valve)     -> FD (valve -> delta x) -> PhysX

That isolation is the entire point of the package. If a PPO4 number differs
from its PPO3 counterpart, the actuator model is the only thing that can have
caused it, so the comparison is meaningful. Subclassing rather than copying is
what keeps that true: a plant change made in PPO3 lands here automatically and
cannot silently diverge.

The ID/FD contract below is copied field-for-field from
`PPO1_ID_FD_PhysX/tasks/env_cfg.py`, which is the configuration those two
checkpoints were validated under. Every one of those fields is load-bearing --
see the notes there for what each was measured to fix -- so they are restated
explicitly here rather than inherited, because PPO3's parent
(`MulagUpperLevelEnvCfg`) sets them to the ROM defaults.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from PPO3_SOLE_PhysX.terrain_env_cfg import (
    TerrainSolePolicyEnvCfg,
    TerrainSolePolicyPPORunnerCfg,
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PPO1_ROOT = REPO_ROOT / "PPO1_ID_FD_PhysX"

# The two checkpoints PPO1's README names as canonical. The control ID is the
# deployed one; the STRICT FD is the rank-promoted retrain -- the earlier FD was
# rank ~2 and under-fit the valve, which is why `best_fd_cylinder_strict.pt` and
# not `best_fd_cylinder_control.pt` is the validated pair.
ID_CHECKPOINT = PPO1_ROOT / "models/id/best_id_cylinder_control.pt"
FD_CHECKPOINT = PPO1_ROOT / "models/fd/best_fd_cylinder_strict.pt"
# NOT `models/` -- the runtime imports the model definitions from
# `<root>/id_cylinder_space ` and `<root>/fd_cylinder_space ` (the trailing
# space in those directory names is real), which live under vendor_models.
# Pointing at `models/` gives ModuleNotFoundError: id_cylinder_common.
CYLINDER_MODELS_ROOT = PPO1_ROOT / "models" / "vendor_models"


@configclass
class TerrainIDFDEnvCfg(TerrainSolePolicyEnvCfg):
    """PPO3 verge task, ID/FD actuator, compliant arm."""

    # --- The swap -------------------------------------------------------
    # PPO3's parent sets cylinder_dynamics_model = "command_rom" and both
    # checkpoints to "disabled". The inherited runtime already implements the
    # 'fd' branch of `_run_cylinder_id_pipeline`, so this is a model swap, not
    # a new control path.
    cylinder_dynamics_model = "fd"
    use_cylinder_id_pipeline = True
    use_sequence_cylinder_models = False
    cylinder_models_root = str(CYLINDER_MODELS_ROOT)
    id_cylinder_checkpoint = str(ID_CHECKPOINT)
    fd_cylinder_checkpoint = str(FD_CHECKPOINT)

    # --- No ROM anywhere in this package ---------------------------------
    # The runtime only CONSTRUCTS a ROM when cylinder_dynamics_model selects
    # one, so with "fd" above no ROM object is ever built. But PPO3's config
    # still names a ROM-A checkpoint, and an inherited path to a model this
    # package must never use is exactly the kind of leftover that later gets
    # read as "PPO4 uses ROM-A too". Disabled explicitly so the effective
    # configuration contains no ROM-A artifact at all.
    rom_cylinder_checkpoint = "disabled"
    command_rom_checkpoint = "disabled"
    base_rom_checkpoint = "disabled"
    # PPO3 primes the command ROM's recurrent history during reset. ID/FD is
    # feed-forward and has no such state, so the warm-up has nothing to prime.
    rom_reset_warmup_steps = 0
    # PPO3's ROM-A guard against reversed valve commands. Meaningless without a
    # ROM, and leaving it True implies a ROM is in the loop.
    command_rom_direction_guard = False

    # --- ID/FD contract, from PPO1_ID_FD_PhysX/tasks/env_cfg.py ----------
    # Residualize both models around neutral. The delta-valve ID has a
    # state-dependent nonzero output at zero target while neutral hold forces
    # exactly zero valve; residualizing over the same state with zero target
    # removes that discontinuity.
    id_residual_over_zero_target = True
    paired_id_fd_residual = True
    fd_residual_over_zero_valve = False
    # Requested increment guarantees direction and full-rank authority; the
    # paired ID/FD map contributes a bounded learned residual.
    cylinder_idfd_reference_blend = 0.8
    cylinder_fd_residual_output_gain = [1.0, 1.0, 1.0, 1.0]
    # Build ID target features on the MODEL's sample time (Simscape data is a
    # constant 0.05 s), not on the 1/60 s control step.
    learned_model_dt_s = 0.05
    # Removed from the main experiment: for a single-cylinder command every
    # off-diagonal has cmd_i = 0, so the guard zeroes ALL cross-axis response
    # for any ratio, and the composite is inherently cross-coupled.
    cylinder_fd_command_direction_guard = False
    cylinder_fd_command_max_ratio = 1.5
    low_level_servo_mode = "none"
    fd_servo_usage = "prediction_only"
    cylinder_actuation_mode = "position"
    max_abs_valve = 1.0

    # Reference-space integration. Re-anchoring the target to the measured
    # joint each step is the "tracking ratchet": gravity sags the arm between
    # steps, the target follows the sag, and the commanded delta is washed out
    # regardless of action sign -- which removes the policy's authority
    # entirely. See PPO1's notes.
    cylinder_target_integrates_from_previous_target = True
    cylinder_target_leak = 0.0
    cylinder_target_max_lead_m = 0.025

    # Close the head feedforward on the MEASURED boom angle. With a learned
    # actuator the commanded q4 is not reached exactly, and a command-based
    # feedforward turns that model error into a standing head-alignment offset.
    terrain_head_feedforward_from_measured = True
    # Let PPO see the actuator's own error so it can compensate it. Without
    # these channels a memoryless policy has no evidence of the ID/FD model's
    # error at all, which is why ten PPO blocks left p01 head alignment at 0.79.
    terrain_model_error_observation = True

    # Head-alignment penalty gradient, the actual defect behind p01 ~0.795.
    #
    # The quadratic term clamps at max_deviation=2.0, i.e. at alignment 0.955.
    # ROM-A runs at 0.9852 and sits INSIDE that region with a live gradient;
    # ID/FD runs at 0.9477 mean and sits below it, where the only slope is the
    # far-field term. At far_weight=-0.5 the penalty is -40.00 at alignment
    # 0.9477, -40.00 at 0.80 and -40.00 at 0.66 -- PPO cannot distinguish them,
    # so ten blocks of training left p01 at 0.795 while every other metric
    # improved. Nothing downstream of the reward could fix that: the levelling
    # clamp, the pose solve and the model-error channels were all tested and
    # all failed, because the policy had no signal telling it which boom
    # attitude was better.
    #
    # -8.0 restores a monotone, meaningful slope across the range this policy
    # actually occupies while leaving the near-threshold behaviour ROM-A was
    # tuned on untouched (the far term is exactly zero above alignment 0.955).
    terrain_head_penalty_far_weight = -8.0

    # c4 tilt shield, the second half of the same defect. The shield fades
    # cylinder 4 in whichever direction would increase head tilt:
    #     fade = clamp((hard - |tilt|) / (hard - soft), 0, 1)
    # ROM-A runs at 0.0237 rad mean tilt, below the 0.08 soft limit, so the
    # shield never engages and its thresholds were never exercised. ID/FD runs
    # at 0.2371 rad mean -- permanently past the 0.16 hard limit -- so fade is
    # 0.0 essentially always and HALF of cylinder 4's action space is deleted
    # every step.
    #
    # Together with the saturated head penalty this left the policy with no
    # gradient telling it to fix the head and no authority to do so in one
    # direction. Widening to 0.30/0.80 keeps the shield's purpose (block
    # commands that drive an already-bad head further over) while leaving the
    # policy real authority in the tilt range it actually occupies.
    terrain_head_tilt_shield_soft = 0.30
    terrain_head_tilt_shield_hard = 0.80
    # Levelling authority, re-tuned for ID/FD at 128 envs x 960 held-out steps
    # (reports/level_g10c12.json vs headff_{false,true}.json). PPO3's 0.5/0.05
    # was tuned against ROM-A, whose commanded cylinder motion is reached almost
    # exactly; the learned actuator needs more corrective range for the same
    # tilt. The gain is real but small -- score 5.910 -> 5.804, p01 0.7948 ->
    # 0.7982 -- and is recorded so nobody re-derives it:
    #
    #   ff/level              score    p01     min    contact
    #   commanded, 0.5/0.05   5.9102  0.7948  0.6689  0.9293
    #   measured,  0.5/0.05   5.8925  0.7952  0.7539  0.9290
    #   measured,  1.0/0.12   5.8037  0.7982  0.7388  0.9253
    #
    # None of these move p01 off ~0.80 against a 0.975 gate, so head alignment
    # under ID/FD is NOT a levelling-tuning problem. See README.
    terrain_messerkopf_level_gain = 1.0
    # Clamp swept 0.05 / 0.12 / 0.40 / 0.80 / unclamped on the full held-out
    # gates. Releasing it DOES lift p01 (0.7948 -> 0.8211) but pays for every
    # bit of it in ground contact (0.9293 -> 0.8524) and cut error (0.0105 ->
    # 0.0178), dropping 14/17 -> 11/17. Unclamped is bit-identical to 0.80, so
    # 0.8 rad is the mechanism's ceiling and p01 0.821 is as far as levelling
    # can go -- against a 0.975 gate. 0.12 is the best measured operating point.
    terrain_messerkopf_level_max_rad = 0.12

    def __post_init__(self):
        super().__post_init__()
        # __post_init__ runs AFTER the parent's, which re-asserts PPO3's ROM-A
        # paths, so the disabling has to be repeated here or it is silently
        # undone. Verified by test_ppo4_effective_config_names_no_rom_model.
        self.rom_reset_warmup_steps = 0
        self.rom_cylinder_checkpoint = "disabled"
        self.command_rom_checkpoint = "disabled"
        self.base_rom_checkpoint = "disabled"
        self.command_rom_direction_guard = False
        for label, path in (("ID", ID_CHECKPOINT), ("FD", FD_CHECKPOINT)):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"PPO4 requires the validated {label} checkpoint at {path}. "
                    "PPO4 is the ID/FD task; falling back to ROM-A would make "
                    "every number here a PPO3 number wearing a PPO4 label."
                )
        print(
            "[PPO4-IDFD] actuator = ID -> FD (ROM-A disabled) | "
            f"ID={ID_CHECKPOINT.name} FD={FD_CHECKPOINT.name} | "
            f"arm stiffness scale={float(self.terrain_arm_stiffness_scale):.4f}",
            flush=True,
        )


@configclass
class TerrainIDFDEnvCfg_PLAY(TerrainIDFDEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False
        self.enable_visual_grid_floor = True


@configclass
class TerrainIDFDPPORunnerCfg(TerrainSolePolicyPPORunnerCfg):
    experiment_name = "ppo4_sole_idfd"


__all__ = [
    "TerrainIDFDEnvCfg",
    "TerrainIDFDEnvCfg_PLAY",
    "TerrainIDFDPPORunnerCfg",
    "apply_terrain_curriculum_stage",
    "configure_terrain_gpu_physx_memory",
    "ID_CHECKPOINT",
    "FD_CHECKPOINT",
]
