"""PPO3_1 configuration: Isaac Lab standard training regime, PPO3 physics.

THE REGIME IS THE POINT
=======================
Measured comparison that motivated this file:

    setting          Isaac Lab standard      old PPO3 runs
    learning rate    1.0e-3, ADAPTIVE        5.0e-7, FIXED
    iterations       1500-3000               60 per block
    mode             train from scratch      warm-start fine-tune
    reward terms     11                      53

Warm-start fine-tuning of a converged policy is the HARDEST reinforcement
learning regime -- entropy has collapsed, the critic is stale the moment the
reward changes, and the clipped objective pins the policy where it started. It
was the default here. Thirteen campaigns produced no improvement, and the
measured cause was mechanical: `distribution.std_param` was byte-identical
across every checkpoint (~0.4 mm of exploration per step at action_scale
0.0035), and entropy moved only in the 4th decimal over 60 iterations.

So PPO3_1 trains from scratch at standard hyperparameters. If from-scratch also
fails, that is real evidence about the TASK; nothing measured under the old
regime is, because every run was confounded by fine-tuning.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from PPO_ROM.terrain_env_cfg import (
    TerrainC4AuthorityEnvCfg,
    TerrainC4AuthorityPPORunnerCfg,
)


@configclass
class Ppo31SoleEnvCfg(TerrainC4AuthorityEnvCfg):
    """PPO3 terrain and plant, PPO3_1 reward weights.

    The inherited 53 reward fields are left in place but UNUSED: `Ppo31SoleEnv`
    never calls `compute_verge_reward`. They are not deleted because the same
    cfg class still drives the terrain, the observation width and the plant,
    and severing that would fork the physics -- the thing that works.
    """

    # Optional per-weight overrides, read by `Ppo31SoleEnv._reward_weights`.
    # Declared so `override_cfg` will set them: it refuses to invent fields,
    # precisely so a typo cannot silently produce a baseline-identical run.
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

    # Stage 6 explicitly: the parent applies the curriculum inside its own
    # __post_init__, so requesting the stage afterwards would be too late.
    curriculum_stage: int = 6

    def __post_init__(self):
        super().__post_init__()
        # Stage 6 is the tracking task: flat ground with discrete raised chunks,
        # 2.2x lateral amplitude. Inherited unchanged -- it is a good task.
        # The parent chain applies stage 5 inside its own __post_init__ and
        # ASSIGNS cfg.curriculum_stage while doing so, overwriting the field
        # default above before the c4 helper ever reads it (measured: preflight
        # P4 saw stage=5). Re-assert stage 6 here, which also re-derives the
        # per-axis c4 authority from stage 6's action_scale.
        from PPO_ROM.terrain_env_cfg import apply_c4_authority_stage

        apply_c4_authority_stage(self, 6)
        self.terrain_relief_profile = "flat_patches"
        # TERRAIN SET TO WHAT THE PLANT CAN ACTUALLY TRACK.
        #
        # The objective is line tracking; the terrain exists to make that
        # non-trivial, not to be the task. At the shipped settings it WAS the
        # task, and an impossible one: measured with the simulator's own path
        # generator (scripts/probe_path_feasibility.py), 9 of 11 families
        # demanded more lateral cutter speed than the cylinders can deliver.
        # The ceiling is 0.200 m/s (cylinder_action_scale 0.004 m per 0.02 s
        # control step, and that is optimistic -- it assumes a lateral Jacobian
        # of 1.0 and the whole budget spent sideways). Worst family demanded
        # 0.734 m/s, 3.7x over. The privileged DLS teacher fails those families
        # too, which is the tell that it is the plant and not the policy.
        #
        # Peak demanded lateral speed, worst of meshes 1337/17/909:
        #
        #   setting                       worst family      peak    over 0.150
        #   amp 2.2, sweep 1.5, h 1.00    boom_sweep       0.734      9 of 11
        #   amp 1.4, sweep 1.5, h 1.00    boom_sweep       0.475      8 of 11
        #   amp 0.9, sweep 0.8, h 0.50    boom_sweep       0.179      2 of 11
        #   amp 0.7, sweep 0.5, h 0.35    lateral_chirp    0.135      0 of 11
        #
        # The last row is chosen: every family sits at or under 75% of the
        # ceiling, so tracking is achievable everywhere rather than accumulating
        # error by construction. At the fitted error/demand ratio of 0.465 that
        # predicts ~0.06 m worst-family lateral error against a 0.300 gate.
        #
        # Three knobs, because they do different jobs. `amplitude_scale` sets
        # every family's lateral swing; `height_sweep_gain` is read ONLY by
        # boom_sweep, whose offset is coupled to terrain height and which is
        # therefore the family that goes infeasible first; `height_scales`
        # lowers the chunks, which eases contact AND feeds back into
        # boom_sweep's demand through that same coupling.
        #
        # Gate numbers from before this change are NOT comparable with numbers
        # after it -- the task is easier on purpose. Baselines must be re-measured.
        # LATERAL and CONTACT are driven by SEPARABLE knobs, which is what makes
        # both gates reachable at once:
        #   amplitude + sweep gain -> lateral rate demand -> the lateral gate
        #   height scales          -> how much ground there is to touch -> contact
        #
        # Measured on the DLS teacher, mesh 1337 (reports/frontier/):
        #   old terrain (amp 2.2, heights 0.35-1.15): lateral 0.3657 FAIL,
        #                                             contact 0.7514 pass
        #   amp 0.7, heights 0.25-0.40:               lateral 0.0576 pass,
        #                                             contact 0.5417 FAIL
        # Dropping the heights alongside the amplitude fixed tracking and broke
        # contact -- there was simply less terrain for the cutter to be in
        # contact with. Tracking now has 5x margin (0.058 against 0.300), so the
        # heights can go back up to buy contact back without endangering it:
        # boom_sweep is the only family coupled to terrain height and its demand
        # is 0.089 m/s at full height with the sweep gain at 0.5, well inside the
        # 0.200 m/s ceiling. Preflight P4 asserts that rather than trusting it.
        self.terrain_tracking_amplitude_scale = 0.7
        self.terrain_tracking_height_sweep_gain = 0.5
        self.terrain_height_scales = (0.50, 0.70, 0.85, 1.00)
        # DOMAIN RANDOMIZATION over the path parameters, so the policy is
        # trained on a BAND rather than a point. A policy generalises over the
        # distribution it saw and essentially nothing else -- measured here, a
        # student distilled at amplitude 2.2 tracked at 0.303 m on amplitude-0.7
        # terrain that permits 0.058 m, five times worse than the plant allows,
        # purely from the shift.
        #
        # Every value is inside the measured feasible band (amplitude is safe to
        # 0.75, tight at 0.9-1.0, infeasible at 1.1; sweep gain safe to 0.8).
        # Lengths 4 and 3 are coprime, so 12 consecutive environments cover all
        # 12 amplitude x gain pairs exactly once with stride 1 on both.
        self.terrain_tracking_amplitude_scales = (0.30, 0.45, 0.60, 0.75)
        self.terrain_tracking_height_sweep_gains = (0.25, 0.40, 0.55)
        # The shipped clamp. The relaxed 0.026 halved lateral error on the
        # pinned mesh and cost ~0.30 of head alignment on unseen ones.
        self.cylinder_target_max_lead_m = 0.018


@configclass
class Ppo31PPORunnerCfg(TerrainC4AuthorityPPORunnerCfg):
    """Isaac Lab's standard PPO settings, not the fine-tuning ones."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100
        self.experiment_name = "ppo3_1_sole"
        # `num_steps_per_env` is deliberately INHERITED (128). Isaac Lab's
        # locomotion default is 24 at 4096 envs; this task runs 256 envs, so 128
        # gives a comparable batch. Changing it to match a number from a
        # different vectorization width would be the blind copying this rebuild
        # exists to stop.

        # Adaptive LR at 1e-3: the Isaac Lab locomotion default. The old runs
        # used 5e-7 FIXED, which is 2000x lower with no schedule -- small enough
        # that 10x made no measurable difference to entropy.
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.schedule = "adaptive"
        # Exploration must not collapse. This is the single defect that pinned
        # every previous fine-tune to its warm start.
        self.algorithm.entropy_coef = 0.005
        self.actor.distribution_cfg.init_std = 0.30
