"""Configuration for policy -> online AAM -> ID adapter -> frozen FD -> PhysX."""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.utils import configclass

from PPO_ID_FD.env_cfg import (
    FD_CHECKPOINT,
    ID_CHECKPOINT,
    CYLINDER_MODELS_ROOT,
    Ppo41IDFDEnvCfg,
    Ppo41IDFDEnvCfg_PLAY,
    Ppo41IDFDPPORunnerCfg,
)

ROOT = Path(__file__).resolve().parents[1]
# Stage 31 artifacts are produced from the fresh final-ROM collection.  Keep
# the paths explicit so deployment cannot silently fall back to stale files.
# Causal history length in 20 Hz model ticks, and the AAM trained for it.
# L20 = 1.0 s is the promoted estimator horizon.  It improves the frozen-ROM
# validation/test RMSE over L5 (0.7966/0.7241 vs 0.8280/0.7428) and the CNN
# checkpoint is selected explicitly so an old flat-MLP artifact cannot be
# loaded by accident.  Override the lag only for a declared ablation.
AAM_HISTORY_LAG = int(os.environ.get("MULAG_AAM_HISTORY_LAG", "20"))
AAM_MODEL = ROOT / f"outputs/stage31_rom/aam_cnn/L{AAM_HISTORY_LAG}_d64"
TRUE_LATENT_MODEL = ROOT / "sd_model/stage31_frozen/z2_rom"
# The adapter is trained against the same latent map and history length as the
# selected AAM.  Keep an explicit override for declared legacy ablations, but
# do not silently pair the L20 CNN with the old L5 adapter.
ADAPTER_MODEL = Path(os.environ.get(
    "MULAG_AAM_ADAPTER_MODEL",
    str(ROOT / "outputs/stage31_rom/id_adapter_cnn_l20_s5/zhat_adapter.pt"),
))


@configclass
class PpoIDAAMFDEnvCfg(Ppo41IDFDEnvCfg):
    """PPO4_1 with the selected causal AAM/adapter deployment path."""

    online_aam_enabled: bool = True
    online_aam_policy_observation: bool = True
    online_aam_checkpoint: str = str(AAM_MODEL)
    online_aam_true_latent_checkpoint: str = str(TRUE_LATENT_MODEL)
    online_aam_adapter_checkpoint: str = str(ADAPTER_MODEL)
    online_aam_history_lag: int = AAM_HISTORY_LAG
    # The Isaac upper-level control loop is 60 Hz; AAM follows the 20 Hz ROM.
    online_aam_update_period_ticks: int = 3
    online_aam_max_inference_latency_ms: float = 50.0
    # Pressure-inclusive Stage-31 adapter.  The residual is bounded and the
    # paired ID-FD plant contract stays active at every scale.  0.25 is the
    # conservative starting point for the final paired sweep.
    online_aam_residual_scale: float = 0.25
    # L20 already supplies one second of history; a five-prediction smoother
    # avoids adding a second full second of adaptation latency.
    online_aam_smoother_window: int = 5
    # Mechanical c3 extension reserve.  The solver enforces this shortened
    # joint limit while the headroom gate continues to score the original
    # stroke, so the paired PPO ablations all use the same physical contract.
    c3_physical_extension_reserve_m: float = 0.008
    # Ablation only. 'estimated' is the deployed system and is
    # bit-identical to the behaviour before this field existed.
    online_aam_latent_mode: str = "estimated"
    # Latent exposed in the last two PPO observation columns. Promotion
    # zero/shuffled arms change only this field; the AAM-conditioned adapter
    # remains on the identical estimated latent in every PPO arm.
    online_aam_policy_latent_mode: str = "estimated"
    # Control tick at which latent_mode='frozen' latches the live estimate and
    # holds it. Only read when a latent mode is 'frozen'; 0 elsewhere. It must
    # be declared here rather than left to getattr, because override_cfg
    # refuses to create fields that do not exist -- an invented field would be
    # silently ignored and the arm would score identically to 'estimated'.
    online_aam_latent_freeze_tick: int = 0
    # Cutoff in Hz for policy_latent_mode='lpf', which band-limits the latent on
    # the POLICY path only while the adapter keeps the full-rate estimate.
    # 'estimated' is the unlimited-bandwidth end of that axis and 'frozen' the
    # zero-bandwidth end, so the three together vary the latent's temporal
    # variation while holding its content fixed.
    # Declared here rather than left to getattr for the same reason as the
    # freeze tick: override_cfg refuses to create fields that do not exist, so
    # an undeclared field would be ignored and the arm would silently score as
    # 'estimated'. 0.0 means unset and is an error only when the mode is 'lpf'.
    online_aam_policy_latent_lpf_hz: float = 0.0
    online_aam_fail_safe: bool = True

    # Keep the validated ID-FD plant contract intact.  AAM changes the ID
    # valve command before the same paired-zero FD residual and reference blend
    # are applied; disabling these terms would make the AAM task a different
    # actuator experiment and would invalidate the PPO-ID-FD comparison.
    id_residual_over_zero_target: bool = True
    paired_id_fd_residual: bool = True
    cylinder_idfd_reference_blend: float = 0.8
    observation_space: int = 68  # PPO4_1's 66 plus z_hat(2).

    # RMA Stage 1 upper bound. When true the policy reads the RAW 6-D
    # privileged factor vector (flow, compliance, seal, pressure, mass_scale,
    # com_z) in place of the 2-D frozen-encoder code, widening the observation
    # 66+2 -> 66+6. Privileged and simulation-only: it is a Stage 1 training
    # route, never a deployable arm. Set through
    # MULAG_AAM_PRIVILEGED_E=1 or the cfg field.
    online_aam_policy_privileged_e: bool = bool(
        int(os.environ.get("MULAG_AAM_PRIVILEGED_E", "0"))
    )

    # Gate 0c interface A (classic RMA): the policy emits the valve command
    # directly, pi(x, z) -> u, and the frozen ID leaves the command path.
    # Gate 0 measured that path 79.9% saturated at its preclip limit, which
    # makes any oracle-gap null measured through it uninterpretable. Off by
    # default: this changes the actuator contract and is not comparable to the
    # PPO-ID-FD baseline. Set with MULAG_DIRECT_VALVE=1.
    direct_valve_command: bool = bool(
        int(os.environ.get("MULAG_DIRECT_VALVE", "0"))
    )

    # RIF Stage 1: policy reads the per-env actuator context (alpha, delay)
    # instead of the 2-D AAM code, widening 66+2 -> 66+2 (same width, different
    # content). Set with MULAG_RIF_CONTEXT=1. Requires the plant perturbation,
    # which this flag enables with the screened delay-led ranges.
    online_aam_policy_rif_context: bool = bool(
        int(os.environ.get("MULAG_RIF_CONTEXT", "0"))
    )

    # Mixed-plant randomisation over the PLANT MODEL (command ROM vs ID->FD).
    # Set with MULAG_MIXED_PLANT=1. The configured command_rom_checkpoint
    # (command_run_001) is absent from this checkout, so the one ROM that IS
    # present is used; a missing file would otherwise fail at env construction
    # with a path error rather than anything about mixed plants.
    mixed_plant_enabled: bool = bool(int(os.environ.get("MULAG_MIXED_PLANT", "0")))

    # Wide payload randomisation, the measured adaptation axis. Set with
    # MULAG_PAYLOAD_RANGE="1.0,5.0". Payload 2-5x costs a fixed policy +24% to
    # +59% cut-height error (paired, CIs excluding zero); the manifest's own
    # 0.8-1.5 range costs it nothing measurable, which is why every previous
    # oracle gate over that range measured a null.
    payload_scale_range = (
        tuple(float(v) for v in os.environ["MULAG_PAYLOAD_RANGE"].split(","))
        if os.environ.get("MULAG_PAYLOAD_RANGE") else None
    )

    # NOT re-scaled. Setting ppo31_headroom_scale_m = 0.02 (five times stronger
    # at the stop) was measured 2026-08-15 over 300 iterations from this same
    # init: p01 cylinder_headroom went 0.00438 -> 0.00079 on mesh 1337 at
    # 2 km/h under combined_test. Fewer environments failed (4 -> 3) but the
    # ones that did failed harder, which is the wrong direction for a p01 gate.
    # The mispricing is real -- a cylinder dead on its stop costs -0.40/step
    # against -1.73/step for 0.135 m of lateral error -- but correcting it did
    # not help, so the default stands until something measures otherwise.

    def __post_init__(self):
        super().__post_init__()
        if (
            AAM_HISTORY_LAG != 20
            and "MULAG_AAM_ADAPTER_MODEL" not in os.environ
        ):
            raise ValueError(
                "MULAG_AAM_HISTORY_LAG changes the AAM window, but the default "
                "adapter is the promoted L20 adapter; set "
                "MULAG_AAM_ADAPTER_MODEL for a declared matching ablation"
            )
        self.online_aam_enabled = True
        # Distillation runs CONTEXT-FREE at 66 wide. The DLS teacher inverts
        # the cylinder Jacobian and is plant-blind, so its labels are identical
        # on every plant: a context slot present here is a channel whose value
        # never correlates with the target, which teaches the student the
        # channel predicts nothing. E4 measured that cost at 7.1% worse than a
        # constant. The context is introduced at Phase 1 iteration 0 instead,
        # through a 66->70 migration whose context columns are zeroed.
        self.online_aam_policy_observation = not bool(
            int(os.environ.get("MULAG_NO_CONTEXT_SLOT", "0"))
        )
        self.online_aam_history_lag = AAM_HISTORY_LAG
        self.online_aam_update_period_ticks = 3
        self.online_aam_residual_scale = 0.25
        self.online_aam_smoother_window = 5
        self.c3_physical_extension_reserve_m = 0.008
        self.id_residual_over_zero_target = True
        self.paired_id_fd_residual = True
        self.cylinder_idfd_reference_blend = 0.8
        # The privileged-e route replaces the 2 latent slots with 6 factor
        # channels, so the actor/critic input width must follow it.
        self.online_aam_policy_privileged_e = bool(
            int(os.environ.get("MULAG_AAM_PRIVILEGED_E", "0"))
        )
        self.direct_valve_command = bool(
            int(os.environ.get("MULAG_DIRECT_VALVE", "0"))
        )
        if self.direct_valve_command:
            # The residual has no anchor to correct once the ID is out of the
            # path; leaving it non-zero would add a bounded perturbation on top
            # of the policy's own command and misattribute the result.
            self.online_aam_residual_scale = 0.0
        self.online_aam_policy_rif_context = bool(
            int(os.environ.get("MULAG_RIF_CONTEXT", "0"))
        )
        # RMA Stage 1 four-wide context: (authority, lag tau, delay, mass_scale).
        # Distinct from the privileged-e route (six ROM factors, four of them
        # inert on FD) and from the RIF route (alpha, delay only).
        self.online_aam_policy_rma_context = bool(
            int(os.environ.get("MULAG_RMA_CONTEXT", "0"))
        )
        if self.online_aam_policy_rma_context and (
            self.online_aam_policy_privileged_e or self.online_aam_policy_rif_context
        ):
            raise ValueError(
                "MULAG_RMA_CONTEXT is mutually exclusive with MULAG_AAM_PRIVILEGED_E "
                "and MULAG_RIF_CONTEXT; enable exactly one context route"
            )
        if self.online_aam_policy_rif_context:
            if self.online_aam_policy_privileged_e:
                raise ValueError(
                    "MULAG_RIF_CONTEXT and MULAG_AAM_PRIVILEGED_E are two "
                    "different privileged channels; enable exactly one"
                )
            # The context is only informative if the axes are actually drawn
            # per environment. Delay leads: the measured dose-response is
            # +22% lateral error for delay 2->8 against 0.2% for alpha x0.6.
            self.plant_perturbation_enabled = True
            self.plant_perturbation_alpha_range = (0.8, 1.0)
            self.plant_perturbation_delay_steps_range = (0, 8)
        if os.environ.get("MULAG_DAMPING_RANGE"):
            self.damping_scale_range = tuple(
                float(v) for v in os.environ["MULAG_DAMPING_RANGE"].split(",")
            )
        if os.environ.get("MULAG_PAIR_PERIOD"):
            period = int(os.environ["MULAG_PAIR_PERIOD"])
            self.damping_scale_pair_period = period
            self.plant_perturbation_pair_period = period
            self.payload_scale_pair_period = period
        self.mixed_plant_enabled = bool(int(os.environ.get("MULAG_MIXED_PLANT", "0")))
        if os.environ.get("MULAG_MASS_STEP_TIME_S"):
            # Mid-episode plant change. This is what makes ONLINE adaptation
            # necessary rather than optional: a context drawn at reset is stale
            # after the step, so only an estimate from recent history can track
            # it. E4 measured the time-varying component at 13.6% against 0.43%
            # for static plant identity -- 32x larger.
            self.deployment_mass_step_time_s = float(
                os.environ["MULAG_MASS_STEP_TIME_S"]
            )
            self.deployment_mass_step_scale = float(
                os.environ.get("MULAG_MASS_STEP_SCALE", "2.0")
            )
            self.deployment_mass_step_jitter_s = float(
                os.environ.get("MULAG_MASS_STEP_JITTER_S", "0.0")
            )
            if os.environ.get("MULAG_MASS_BIMODAL"):
                self.deployment_mass_bimodal = tuple(
                    float(v) for v in os.environ["MULAG_MASS_BIMODAL"].split(",")
                )
            # The mass-step path lives behind the deployment-factor machinery.
            self.deployment_factor_enabled = True
        if os.environ.get("MULAG_PAYLOAD_RESAMPLE"):
            # Opt in to a fresh payload draw every episode. Off by default so the
            # payload is a fixed property of each environment, matching the
            # alpha/delay and damping draws and keeping runs reproducible.
            self.payload_scale_resample_per_episode = bool(
                int(os.environ["MULAG_PAYLOAD_RESAMPLE"])
            )
        if os.environ.get("MULAG_PAYLOAD_RANGE"):
            self.payload_scale_range = tuple(
                float(v) for v in os.environ["MULAG_PAYLOAD_RANGE"].split(",")
            )
            # A payload draw is only in the observation if the factor path runs.
            self.deployment_factor_enabled = True
        if self.mixed_plant_enabled:
            available_rom = ROOT / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"
            if not Path(self.command_rom_checkpoint).is_file() and available_rom.is_file():
                self.command_rom_checkpoint = str(available_rom)
        if not self.online_aam_policy_observation:
            self.observation_space = 66   # context-free, for distillation
        elif self.online_aam_policy_privileged_e:
            self.observation_space = 72
        elif self.online_aam_policy_rma_context:
            self.observation_space = 70   # 66 base + 4 RMA context
        else:
            self.observation_space = 68
        for label, path in (
            ("AAM", Path(self.online_aam_checkpoint) / "aam.pt"),
            ("AAM metrics", Path(self.online_aam_checkpoint) / "metrics.json"),
            ("true-latent encoder", Path(self.online_aam_true_latent_checkpoint) / "encoder.pt"),
            ("true-latent metrics", Path(self.online_aam_true_latent_checkpoint) / "metrics.json"),
            ("ID adapter", Path(self.online_aam_adapter_checkpoint)),
            ("ID", ID_CHECKPOINT), ("FD", FD_CHECKPOINT),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"PPO_AAM_FD requires {label} artifact at {path}")
        print(
            f"[PPO-ID-AAM-FD] Stage31 policy -> causal ROM-history AAM(L{AAM_HISTORY_LAG},20Hz) -> "
            "ID adapter -> paired ID/FD -> PhysX | obs=68 | AAM row=18 | "
            f"residual_scale={self.online_aam_residual_scale:.2f} | fallback=frozen ID",
            flush=True,
        )


@configclass
class PpoIDAAMFDEnvCfg_PLAY(PpoIDAAMFDEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False
        self.enable_visual_grid_floor = True


@configclass
class PpoIDAAMFDPPORunnerCfg(Ppo41IDFDPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "ppo_id_aam_fd"


__all__ = [
    "PpoIDAAMFDEnvCfg", "PpoIDAAMFDEnvCfg_PLAY", "PpoIDAAMFDPPORunnerCfg",
    "ID_CHECKPOINT", "FD_CHECKPOINT", "CYLINDER_MODELS_ROOT", "AAM_MODEL", "ADAPTER_MODEL",
    "TRUE_LATENT_MODEL",
]
