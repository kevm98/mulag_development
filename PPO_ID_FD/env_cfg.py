"""PPO4_1: the PPO3_1 verge task with ID/FD in place of ROM-A.

PPO4_1 : PPO3_1  ::  PPO4 : PPO3. Everything is inherited from
``Ppo31SoleEnvCfg`` -- the feasibility-derived terrain, the randomization band,
the nine-term reward, the c4 command authority, the eight gates, the 66-value
observation and the four-action contract. The ONLY difference is what converts a
cylinder command into motion:

    PPO3_1:  policy -> command ROM (ROM-A)                        -> PhysX
    PPO4_1:  policy -> ID (target -> valve) -> FD (valve -> dx)   -> PhysX

That isolation is the whole point of the package, and it is what makes the
comparison worth running: if a PPO4_1 number differs from its PPO3_1
counterpart, the actuator model is the only thing that can have caused it.
Subclassing rather than copying is what keeps that true -- a terrain or reward
change made in PPO3_1 lands here automatically and cannot silently diverge.

The ID/FD contract below is taken field-for-field from
``PPO4_ID_FD_PhysX/terrain_env_cfg.py``, which is the configuration those
checkpoints were validated under. It is restated rather than inherited because
PPO3_1's parent chain sets these to the ROM defaults.

WHAT IS DELIBERATELY *NOT* CHANGED
==================================
The c4 command authority stays exactly as PPO3_1 sets it -- per-axis lead caps
(0.018, 0.018, 0.018, 0.30) and action scales derived from the stage. PPO4 uses
a scalar ``cylinder_target_max_lead_m`` of 0.025; adopting that here would mean
PPO3_1 and PPO4_1 differed in BOTH the actuator and the command authority, and
any measured gap could be attributed to either. The per-axis tuple is applied
after this class by ``apply_c4_authority_stage`` and takes precedence, so the
scalar below only affects axes the tuple does not name.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from PPO_ROM.env_cfg import Ppo31PPORunnerCfg, Ppo31SoleEnvCfg

REPO_ROOT = Path(__file__).resolve().parents[1]
# assets/, not PPO1_ID_FD_PhysX/. These three artefacts are the only things the
# ID/FD pipeline needs from PPO1, so they live with the other runtime assets and
# PPO1 itself is archived. Copied, not moved from a live tree, so the paths
# cannot silently resolve to a stale training directory.
ASSETS = REPO_ROOT / "id_fd_model"

# The canonical pair from PPO1's README. The control ID is the deployed one; the
# STRICT FD is the rank-promoted retrain -- the earlier FD was rank ~2 and
# under-fit the valve, which is why `best_fd_cylinder_strict.pt` is the
# validated partner and not `best_fd_cylinder_control.pt`.
ID_CHECKPOINT = ASSETS / "best_id_cylinder_control.pt"
FD_CHECKPOINT = ASSETS / "best_fd_cylinder_strict.pt"
# NOT `models/` -- the runtime imports model definitions from
# `<root>/id_cylinder_space ` and `<root>/fd_cylinder_space ` (the trailing
# space in those directory names is real), which live under vendor_models.
CYLINDER_MODELS_ROOT = ASSETS / "vendor_models"


@configclass
class Ppo41IDFDEnvCfg(Ppo31SoleEnvCfg):
    """PPO3_1's verge task and terrain, driven through ID -> FD."""

    # --- the swap --------------------------------------------------------
    cylinder_dynamics_model = "fd"
    use_cylinder_id_pipeline = True
    use_sequence_cylinder_models = False
    cylinder_models_root = str(CYLINDER_MODELS_ROOT)
    id_cylinder_checkpoint = str(ID_CHECKPOINT)
    fd_cylinder_checkpoint = str(FD_CHECKPOINT)

    # --- no ROM anywhere in this package ---------------------------------
    # The runtime only CONSTRUCTS a ROM when cylinder_dynamics_model selects
    # one, so with "fd" above none is ever built. Disabled explicitly anyway:
    # an inherited path to a model this package must never use is exactly the
    # leftover that later gets read as "PPO4_1 uses ROM-A too".
    rom_cylinder_checkpoint = "disabled"
    command_rom_checkpoint = "disabled"
    base_rom_checkpoint = "disabled"
    rom_reset_warmup_steps = 0
    command_rom_direction_guard = False

    # --- ID/FD contract, from PPO1_ID_FD_PhysX/tasks/env_cfg.py ----------
    id_residual_over_zero_target = True
    paired_id_fd_residual = True
    fd_residual_over_zero_valve = False
    cylinder_idfd_reference_blend = 0.8
    cylinder_fd_residual_output_gain = [1.0, 1.0, 1.0, 1.0]
    # ID target features are built on the MODEL's sample time (Simscape data is
    # a constant 0.05 s), not on the control step.
    learned_model_dt_s = 0.05
    cylinder_fd_command_direction_guard = False
    cylinder_fd_command_max_ratio = 1.5
    low_level_servo_mode = "none"
    fd_servo_usage = "prediction_only"
    cylinder_actuation_mode = "position"
    max_abs_valve = 1.0

    # Reference-space integration. Re-anchoring the target to the measured
    # joint each step is the "tracking ratchet": gravity sags the arm between
    # steps, the target follows the sag, and the commanded delta is washed out
    # regardless of action sign, which removes the policy's authority entirely.
    # See PPO1's notes.
    cylinder_target_integrates_from_previous_target = True
    cylinder_target_leak = 0.0

    def __post_init__(self):
        super().__post_init__()
        # The parent's __post_init__ runs first and re-asserts PPO3_1's ROM-A
        # paths, so the disabling has to be repeated here or it is silently
        # undone -- the same class of override that had train.py re-imposing old
        # terrain height scales and the curriculum re-imposing amplitude 2.2.
        self.cylinder_dynamics_model = "fd"
        self.use_cylinder_id_pipeline = True
        self.rom_reset_warmup_steps = 0
        self.rom_cylinder_checkpoint = "disabled"
        self.command_rom_checkpoint = "disabled"
        self.base_rom_checkpoint = "disabled"
        self.command_rom_direction_guard = False
        self.cylinder_models_root = str(CYLINDER_MODELS_ROOT)
        self.id_cylinder_checkpoint = str(ID_CHECKPOINT)
        self.fd_cylinder_checkpoint = str(FD_CHECKPOINT)

        for label, path in (("ID", ID_CHECKPOINT), ("FD", FD_CHECKPOINT)):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"PPO4_1 requires the validated {label} checkpoint at "
                    f"{path}. This is the ID/FD task; falling back to ROM-A "
                    "would make every number here a PPO3_1 number wearing a "
                    "PPO4_1 label."
                )
        print(
            "[PPO4-1-IDFD] actuator = ID -> FD (ROM-A disabled) | "
            f"ID={ID_CHECKPOINT.name} FD={FD_CHECKPOINT.name} | "
            f"amplitude={float(self.terrain_tracking_amplitude_scale):.2f} "
            f"sweep_gain={float(self.terrain_tracking_height_sweep_gain):.2f}",
            flush=True,
        )


@configclass
class Ppo41IDFDEnvCfg_PLAY(Ppo41IDFDEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False
        self.enable_visual_grid_floor = True


@configclass
class Ppo41IDFDPPORunnerCfg(Ppo31PPORunnerCfg):
    """Identical PPO regime to PPO3_1; only the experiment name differs.

    Same learning rate, schedule, entropy, init_std and network sizes. If the
    two pipelines are to be compared, the optimiser must not be a variable.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "ppo4_1_sole_idfd"


__all__ = [
    "Ppo41IDFDEnvCfg",
    "Ppo41IDFDEnvCfg_PLAY",
    "Ppo41IDFDPPORunnerCfg",
    "ID_CHECKPOINT",
    "FD_CHECKPOINT",
]
