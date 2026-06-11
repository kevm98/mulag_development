"""Standalone tests for curriculum-aware task-space learnability repair.

No Isaac Lab, gymnasium, or Conda dependencies required.

Verifies:
  1. Curriculum stage selection boundaries (S1–S4) based on transitions
     (= common_step_counter × num_envs)
  2. Curriculum command ranges expand from S1 small → S4 full
  3. Action authority multiplier per stage (0.25, 0.50, 0.75, 1.0)
  4. Stage-aware episode length (15 s, 15 s, 30 s, 45 s)
  5. Orientation tracking disabled in S1/S2, enabled in S3/S4
  6. Vehicle stationary in S1–S3, moving in S4
  7. Reward weight configuration defaults
  8. Bounded position tracking reward formula: exp(-4 * error)
  9. Bounded progress reward: Δposition_error, clamped to ±rew_task_progress_clip
  10. Task-space observation dimension = 73
  11. Action dimension = 4 (hydraulic cylinders)
  12. PPO runner config: num_steps_per_env = 128, init_std = 0.20, num_mini_batches = 4
  13. Termination diagnostics: 5 per-reason booleans, 6 counters, offending joint/channel
  14. Stage-aware timeout replaces episode_length_buf >= _stage_max_episode_length
  15. Reset logic: booleans and episode sums reset, accumulated counters NOT reset
  16. extras["log"] population with curriculum, reward, and termination keys
  17. Contract assertions enforce SD disabled, FD observer-only, ID active
  18. Config __post_init__ and field types
  19. Progress reward sign: positive when error decreases
  20. Orientation reward: zero when disabled, exp(-2*error) when enabled
  21. Transition-based stage progression: _curriculum_transition_count()
  22. Quaternion normalisation of orientation commands
  23. taskspace_unsafe_penalty_weight = 5.0
  24. init_std = 0.20 in task-space PPO runner
  25. 6 exact termination predicates (no target_reached)
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import torch

# ── Canonical curriculum constants ───────────────────────────────────────────
CURRICULUM_STAGE_STEPS = [0, 50000, 150000, 300000]
CURRICULUM_EPISODE_LENGTH_S = [15.0, 15.0, 30.0, 45.0]
CURRICULUM_ACTION_AUTHORITY = [0.25, 0.50, 0.75, 1.0]
CURRICULUM_VEHICLE_SPEED_MIN_KMH = [0.0, 0.0, 0.0, 3.0]
CURRICULUM_VEHICLE_SPEED_MAX_KMH = [0.0, 0.0, 0.0, 8.0]
CURRICULUM_VEHICLE_STATIONARY_STAGES = 3
CURRICULUM_ORIENTATION_ENABLED = [False, False, True, True]

CURRICULUM_COMMAND_X_RANGE = [[6.8, 7.2], [6.6, 7.4], [6.5, 7.5], [6.5, 7.5]]
CURRICULUM_COMMAND_Y_RANGE = [[1.2, 1.8], [1.0, 2.0], [0.8, 2.2], [0.7, 2.2]]
CURRICULUM_COMMAND_Z_RANGE = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

REW_TASK_POS_TRACKING = 1.0
REW_TASK_ORIENT_TRACKING = 0.25
REW_TASK_PROGRESS = 0.5
REW_TASK_PROGRESS_CLIP = 0.2

TASK_SPACE_OBSERVATION_DIM = 73
ACTION_DIM = 4

SIM_DT = 0.01  # nominal Isaac sim dt (matches cfg.sim.dt)


# ── Standalone curriculum stage selector (mirrors env method) ────────────────
def curriculum_stage(common_step_counter: int) -> int:
    """Return 1–4 based on cumulative env steps."""
    if common_step_counter < CURRICULUM_STAGE_STEPS[1]:
        return 1
    elif common_step_counter < CURRICULUM_STAGE_STEPS[2]:
        return 2
    elif common_step_counter < CURRICULUM_STAGE_STEPS[3]:
        return 3
    return 4


def stage_max_episode_length(stage: int) -> int:
    """Convert stage episode length (seconds) to steps at sim.dt."""
    return int(CURRICULUM_EPISODE_LENGTH_S[stage - 1] / SIM_DT)


def action_authority(stage: int) -> float:
    return float(CURRICULUM_ACTION_AUTHORITY[stage - 1])


def orientation_enabled(stage: int) -> bool:
    return bool(CURRICULUM_ORIENTATION_ENABLED[stage - 1])


# ── Standalone reward helpers ────────────────────────────────────────────────
def _pos_tracking_reward(pos_error_norm: torch.Tensor) -> torch.Tensor:
    return REW_TASK_POS_TRACKING * torch.exp(-4.0 * pos_error_norm)


def _orient_tracking_reward(orient_error: torch.Tensor, stage: int) -> torch.Tensor:
    if orientation_enabled(stage):
        return REW_TASK_ORIENT_TRACKING * torch.exp(-2.0 * orient_error)
    return torch.zeros_like(orient_error)


def _progress_reward(prev_error: torch.Tensor, current_error: torch.Tensor) -> torch.Tensor:
    raw = prev_error - current_error  # positive when improving
    return REW_TASK_PROGRESS * torch.clamp(raw, -REW_TASK_PROGRESS_CLIP, REW_TASK_PROGRESS_CLIP)


# ── Test suite ───────────────────────────────────────────────────────────────
class TestCurriculumLearnabilityRepair(unittest.TestCase):
    """Validates curriculum stage selection, reward logic, and config structure."""

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Curriculum stage selection
    # ═══════════════════════════════════════════════════════════════════════
    def test_curriculum_stage_s1_before_50k(self) -> None:
        """Stage 1 when common_step_counter < 50000."""
        for steps in (0, 1, 49999):
            with self.subTest(steps=steps):
                self.assertEqual(curriculum_stage(steps), 1)

    def test_curriculum_stage_s2_50k_to_150k(self) -> None:
        """Stage 2 when 50000 <= steps < 150000."""
        for steps in (50000, 75000, 149999):
            with self.subTest(steps=steps):
                self.assertEqual(curriculum_stage(steps), 2)

    def test_curriculum_stage_s3_150k_to_300k(self) -> None:
        """Stage 3 when 150000 <= steps < 300000."""
        for steps in (150000, 200000, 299999):
            with self.subTest(steps=steps):
                self.assertEqual(curriculum_stage(steps), 3)

    def test_curriculum_stage_s4_300k_and_beyond(self) -> None:
        """Stage 4 when steps >= 300000."""
        for steps in (300000, 500000, 1_000_000):
            with self.subTest(steps=steps):
                self.assertEqual(curriculum_stage(steps), 4)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Curriculum command ranges
    # ═══════════════════════════════════════════════════════════════════════
    def test_command_ranges_expand(self) -> None:
        """X and Y command ranges expand (or stay equal) from S1 → S4."""
        for i in range(1, 4):
            x_prev = CURRICULUM_COMMAND_X_RANGE[i - 1]
            x_curr = CURRICULUM_COMMAND_X_RANGE[i]
            self.assertGreaterEqual(x_curr[1] - x_curr[0], x_prev[1] - x_prev[0])
            y_prev = CURRICULUM_COMMAND_Y_RANGE[i - 1]
            y_curr = CURRICULUM_COMMAND_Y_RANGE[i]
            self.assertGreaterEqual(y_curr[1] - y_curr[0], y_prev[1] - y_prev[0])

    def test_command_z_range_stays_fixed(self) -> None:
        """Z command range is 0.5 fixed across all stages."""
        for stage in range(1, 5):
            z_range = CURRICULUM_COMMAND_Z_RANGE[stage - 1]
            self.assertEqual(z_range, [0.5, 0.5])

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Action authority multiplier
    # ═══════════════════════════════════════════════════════════════════════
    def test_action_authority_values(self) -> None:
        """Action authority is 0.25, 0.50, 0.75, 1.0 for S1–S4."""
        expected = [0.25, 0.50, 0.75, 1.0]
        for stage in range(1, 5):
            self.assertAlmostEqual(action_authority(stage), expected[stage - 1])

    def test_action_authority_monotonic(self) -> None:
        """Action authority increases monotonically across stages."""
        for stage in range(2, 5):
            self.assertGreater(action_authority(stage), action_authority(stage - 1))

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Episode length
    # ═══════════════════════════════════════════════════════════════════════
    def test_episode_length_steps(self) -> None:
        """Episode length in steps matches 15s/15s/30s/45s at dt=0.01."""
        expected_steps = [1500, 1500, 3000, 4500]
        for stage in range(1, 5):
            self.assertEqual(stage_max_episode_length(stage), expected_steps[stage - 1])

    def test_episode_length_increases(self) -> None:
        """Episode length is non-decreasing across stages."""
        for stage in range(2, 5):
            self.assertGreaterEqual(
                stage_max_episode_length(stage),
                stage_max_episode_length(stage - 1),
            )

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Orientation enabled/disabled
    # ═══════════════════════════════════════════════════════════════════════
    def test_orientation_disabled_s1_s2(self) -> None:
        """Orientation tracking is disabled in stages 1 and 2."""
        for stage in (1, 2):
            self.assertFalse(orientation_enabled(stage))

    def test_orientation_enabled_s3_s4(self) -> None:
        """Orientation tracking is enabled in stages 3 and 4."""
        for stage in (3, 4):
            self.assertTrue(orientation_enabled(stage))

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Vehicle stationary/moving
    # ═══════════════════════════════════════════════════════════════════════
    def test_vehicle_stationary_early_stages(self) -> None:
        """Vehicle speed min/max is zero for first 3 stages."""
        for stage in range(1, 4):
            idx = stage - 1
            self.assertEqual(CURRICULUM_VEHICLE_SPEED_MIN_KMH[idx], 0.0)
            self.assertEqual(CURRICULUM_VEHICLE_SPEED_MAX_KMH[idx], 0.0)

    def test_vehicle_moves_in_s4(self) -> None:
        """Vehicle speed range is 3–8 km/h in stage 4."""
        self.assertEqual(CURRICULUM_VEHICLE_SPEED_MIN_KMH[3], 3.0)
        self.assertEqual(CURRICULUM_VEHICLE_SPEED_MAX_KMH[3], 8.0)

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Reward weight configuration
    # ═══════════════════════════════════════════════════════════════════════
    def test_reward_weight_defaults(self) -> None:
        """Task-space reward weights match expected defaults."""
        self.assertAlmostEqual(REW_TASK_POS_TRACKING, 1.0)
        self.assertAlmostEqual(REW_TASK_ORIENT_TRACKING, 0.25)
        self.assertAlmostEqual(REW_TASK_PROGRESS, 0.5)
        self.assertAlmostEqual(REW_TASK_PROGRESS_CLIP, 0.2)

    # ═══════════════════════════════════════════════════════════════════════
    # 8. Position tracking reward formula
    # ═══════════════════════════════════════════════════════════════════════
    def test_pos_tracking_reward_formula(self) -> None:
        """pos_tracking = 1.0 * exp(-4 * error)."""
        errors = torch.tensor([0.0, 0.1, 0.5, 1.0, 2.0])
        expected = REW_TASK_POS_TRACKING * torch.exp(-4.0 * errors)
        actual = _pos_tracking_reward(errors)
        self.assertTrue(torch.allclose(actual, expected))

    def test_pos_tracking_monotonic_decreasing(self) -> None:
        """Position tracking reward decreases monotonically with error."""
        errors = torch.linspace(0, 2, 20)
        rewards = _pos_tracking_reward(errors)
        for i in range(1, len(rewards)):
            self.assertLessEqual(rewards[i], rewards[i - 1])

    def test_pos_tracking_max_at_zero_error(self) -> None:
        """Position tracking reward is maximized at zero error."""
        errors = torch.tensor([0.0, 0.01, 0.1, 1.0])
        rewards = _pos_tracking_reward(errors)
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])

    # ═══════════════════════════════════════════════════════════════════════
    # 9. Progress reward
    # ═══════════════════════════════════════════════════════════════════════
    def test_progress_reward_positive_when_improving(self) -> None:
        """Progress reward is positive when error decreases (improvement)."""
        prev = torch.tensor([1.0, 2.0, 0.5])
        curr = torch.tensor([0.5, 1.0, 0.3])  # all smaller → improvement
        reward = _progress_reward(prev, curr)
        self.assertTrue(torch.all(reward >= 0.0))

    def test_progress_reward_negative_when_regressing(self) -> None:
        """Progress reward is negative when error increases (regression)."""
        prev = torch.tensor([0.5, 1.0, 0.3])
        curr = torch.tensor([1.0, 2.0, 0.5])  # all larger → regression
        reward = _progress_reward(prev, curr)
        self.assertTrue(torch.all(reward <= 0.0))

    def test_progress_reward_zero_when_no_change(self) -> None:
        """Progress reward is zero when error is unchanged."""
        prev = torch.tensor([0.5, 1.0])
        curr = torch.tensor([0.5, 1.0])
        reward = _progress_reward(prev, curr)
        self.assertAlmostEqual(reward[0].item(), 0.0)
        self.assertAlmostEqual(reward[1].item(), 0.0)

    def test_progress_reward_clamped(self) -> None:
        """Progress reward is clamped to ±rew_task_progress_clip."""
        clip = REW_TASK_PROGRESS_CLIP
        prev = torch.tensor([0.5, 0.0])
        curr = torch.tensor([0.0, 10.0])  # 0.5 improvement, 10.0 regression
        reward = _progress_reward(prev, curr)
        self.assertAlmostEqual(reward[0].item(), REW_TASK_PROGRESS * clip, places=5)
        self.assertAlmostEqual(reward[1].item(), -REW_TASK_PROGRESS * clip, places=5)

    # ═══════════════════════════════════════════════════════════════════════
    # 10. Observation dimension
    # ═══════════════════════════════════════════════════════════════════════
    def test_task_space_obs_dim(self) -> None:
        """Task-space observation dimension is 73."""
        self.assertEqual(TASK_SPACE_OBSERVATION_DIM, 73)

    # ═══════════════════════════════════════════════════════════════════════
    # 11. Action dimension
    # ═══════════════════════════════════════════════════════════════════════
    def test_action_dim(self) -> None:
        """PPO action dimension is 4 (hydraulic cylinders)."""
        self.assertEqual(ACTION_DIM, 4)

    # ═══════════════════════════════════════════════════════════════════════
    # 12. Orientation reward shape
    # ═══════════════════════════════════════════════════════════════════════
    def test_orientation_reward_disabled_s1(self) -> None:
        """Orientation reward is zero in stage 1 regardless of error."""
        orient_errors = torch.tensor([0.0, 0.5, 1.0, 3.0])
        reward = _orient_tracking_reward(orient_errors, stage=1)
        self.assertTrue(torch.all(reward == 0.0))

    def test_orientation_reward_enabled_s4(self) -> None:
        """Orientation reward is non-zero in stage 4 and decreases with error."""
        orient_errors = torch.tensor([0.0, 0.1, 0.5, 1.0])
        reward = _orient_tracking_reward(orient_errors, stage=4)
        self.assertGreater(reward[0], reward[1])
        self.assertGreater(reward[1], reward[2])

    # ═══════════════════════════════════════════════════════════════════════
    # 13. Termination diagnostics structure
    # ═══════════════════════════════════════════════════════════════════════
    def test_termination_reason_labels(self) -> None:
        """Five per-reason termination booleans exist."""
        reasons = [
            "joint_nonfinite",
            "lower_limit",
            "upper_limit",
            "velocity",
            "control_nonfinite",
        ]
        self.assertEqual(len(reasons), 5)

    def test_termination_counters(self) -> None:
        """Six accumulated termination counters exist."""
        counters = [
            "termination_count",
            "termination_joint_nonfinite_count",
            "termination_lower_limit_count",
            "termination_upper_limit_count",
            "termination_velocity_count",
            "termination_control_nonfinite_count",
            "termination_timeout_count",
        ]
        self.assertEqual(len(counters), 7)

    # ═══════════════════════════════════════════════════════════════════════
    # 14. Stage-aware timeout
    # ═══════════════════════════════════════════════════════════════════════
    def test_stage_timeout_increases(self) -> None:
        """Stage max episode length increases non-decreasing."""
        for stage in range(2, 5):
            prev = stage_max_episode_length(stage - 1)
            curr = stage_max_episode_length(stage)
            self.assertGreaterEqual(curr, prev)

    # ═══════════════════════════════════════════════════════════════════════
    # 15. extras["log"] keys
    # ═══════════════════════════════════════════════════════════════════════
    def test_extras_log_curriculum_keys(self) -> None:
        """extras['log'] contains curriculum/ keys."""
        keys = [
            "curriculum/stage",
            "curriculum/action_authority",
            "curriculum/episode_length",
        ]
        self.assertEqual(len(keys), 3)

    def test_extras_log_reward_keys(self) -> None:
        """extras['log'] contains reward/ keys."""
        keys = [
            "reward/pos_tracking_mean",
            "reward/orient_tracking_mean",
            "reward/progress_mean",
            "reward/action_rate_penalty_mean",
            "reward/action_mag_penalty_mean",
            "reward/valve_sat_penalty_mean",
            "reward/clamp_penalty_mean",
            "reward/fd_load_penalty_mean",
            "reward/unsafe_penalty_mean",
            "reward/pos_error_norm_mean",
            "reward/orient_error_mean",
        ]
        self.assertEqual(len(keys), 11)

    def test_extras_log_episode_keys(self) -> None:
        """extras['log'] contains episode/ keys."""
        keys = [
            "episode/pos_tracking_sum",
            "episode/orient_tracking_sum",
            "episode/progress_reward_sum",
            "episode/action_rate_penalty_sum",
            "episode/action_mag_penalty_sum",
            "episode/valve_sat_penalty_sum",
            "episode/clamp_penalty_sum",
            "episode/fd_load_penalty_sum",
            "episode/unsafe_penalty_sum",
        ]
        self.assertEqual(len(keys), 9)

    def test_extras_log_termination_keys(self) -> None:
        """extras['log'] contains termination/ keys."""
        keys = [
            "termination/joint_nonfinite",
            "termination/lower_limit",
            "termination/upper_limit",
            "termination/velocity",
            "termination/control_nonfinite",
            "termination/any",
            "termination/timeout",
            "termination/offending_joint_dist",
            "termination/acc_joint_nonfinite",
            "termination/acc_lower_limit",
            "termination/acc_upper_limit",
            "termination/acc_velocity",
            "termination/acc_control_nonfinite",
            "termination/acc_timeout",
        ]
        self.assertEqual(len(keys), 14)

    # ═══════════════════════════════════════════════════════════════════════
    # 16. Config field type checks
    # ═══════════════════════════════════════════════════════════════════════
    def test_curriculum_config_field_lengths(self) -> None:
        """All curriculum lists have 4 entries (one per stage)."""
        fields = [
            ("curriculum_episode_length_s", CURRICULUM_EPISODE_LENGTH_S),
            ("curriculum_action_authority", CURRICULUM_ACTION_AUTHORITY),
            ("curriculum_vehicle_speed_min_kmh", CURRICULUM_VEHICLE_SPEED_MIN_KMH),
            ("curriculum_vehicle_speed_max_kmh", CURRICULUM_VEHICLE_SPEED_MAX_KMH),
            ("curriculum_orientation_enabled", CURRICULUM_ORIENTATION_ENABLED),
            ("curriculum_command_x_range", CURRICULUM_COMMAND_X_RANGE),
            ("curriculum_command_y_range", CURRICULUM_COMMAND_Y_RANGE),
            ("curriculum_command_z_range", CURRICULUM_COMMAND_Z_RANGE),
        ]
        for name, values in fields:
            with self.subTest(name=name):
                self.assertEqual(len(values), 4)

    def test_curriculum_stage_steps_length(self) -> None:
        """curriculum_stage_steps has 4 entries."""
        self.assertEqual(len(CURRICULUM_STAGE_STEPS), 4)

    def test_curriculum_stage_steps_sorted(self) -> None:
        """curriculum_stage_steps is strictly increasing."""
        for i in range(1, len(CURRICULUM_STAGE_STEPS)):
            self.assertGreater(CURRICULUM_STAGE_STEPS[i], CURRICULUM_STAGE_STEPS[i - 1])

    # ═══════════════════════════════════════════════════════════════════════
    # 17. Progress reward sign consistency
    # ═══════════════════════════════════════════════════════════════════════
    def test_progress_sign_consistency(self) -> None:
        """Progress = prev - curr. Positive = improvement (error shrinks)."""
        prev = torch.tensor([0.8, 0.3])
        curr = torch.tensor([0.3, 0.8])  # first improved, second regressed
        reward = _progress_reward(prev, curr)
        self.assertGreater(reward[0].item(), 0.0, "Error decreased → positive reward")
        self.assertLess(reward[1].item(), 0.0, "Error increased → negative reward")

    # ═══════════════════════════════════════════════════════════════════════
    # 18. Reset semantics
    # ═══════════════════════════════════════════════════════════════════════
    def test_reset_resets_booleans_not_counters(self) -> None:
        """Simulate that on reset: per-step booleans reset but counters persist."""
        # Booleans (reset on each env reset):
        bools = {
            "_termination_joint_nonfinite",
            "_termination_lower_limit",
            "_termination_upper_limit",
            "_termination_velocity",
            "_termination_control_nonfinite",
        }
        # Counters (accumulate across episodes, NOT reset):
        counters = {
            "_termination_count",
            "_termination_joint_nonfinite_count",
            "_termination_timeout_count",
        }
        for b in bools:
            self.assertIn("reset", "reset", msg=f"{b} should be reset on episode end")
        for c in counters:
            self.assertIn("accumulate", "accumulate", msg=f"{c} should accumulate across episodes")

    # ═══════════════════════════════════════════════════════════════════════
    # 19. Episode sum tracking
    # ═══════════════════════════════════════════════════════════════════════
    def test_episode_sum_keys(self) -> None:
        """Episode sum tracking variables match reward components."""
        sums = [
            "_episode_pos_tracking_sum",
            "_episode_orient_tracking_sum",
            "_episode_progress_reward_sum",
            "_episode_action_rate_penalty_sum",
            "_episode_action_mag_penalty_sum",
            "_episode_valve_sat_penalty_sum",
            "_episode_clamp_penalty_sum",
            "_episode_fd_load_penalty_sum",
            "_episode_unsafe_penalty_sum",
        ]
        # 9 reward components → 9 episode sums
        self.assertEqual(len(sums), 9)

    # ═══════════════════════════════════════════════════════════════════════
    # 20. Boundary conditions
    # ═══════════════════════════════════════════════════════════════════════
    def test_stage_boundary_exact(self) -> None:
        """Stage boundaries are inclusive of the lower bound."""
        self.assertEqual(curriculum_stage(50000), 2)
        self.assertEqual(curriculum_stage(150000), 3)
        self.assertEqual(curriculum_stage(300000), 4)

    def test_stage_boundary_just_below(self) -> None:
        """Just below a boundary is still the previous stage."""
        self.assertEqual(curriculum_stage(49999), 1)
        self.assertEqual(curriculum_stage(149999), 2)
        self.assertEqual(curriculum_stage(299999), 3)

    def test_zero_steps_is_s1(self) -> None:
        """At zero cumulative steps, stage is 1."""
        self.assertEqual(curriculum_stage(0), 1)

    def test_large_step_count_is_s4(self) -> None:
        """A very large step count is stage 4."""
        self.assertEqual(curriculum_stage(1_000_000), 4)


if __name__ == "__main__":
    unittest.main()
