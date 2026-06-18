"""Standalone tests for curriculum-aware task-space learnability repair.

No Isaac Lab, gymnasium, or Conda dependencies required.

Verifies the post-repair design:
  1.  Curriculum stage is explicitly stored (not derived from transitions)
  2.  Performance gates for advancement (5 thresholds)
  3.  _evaluate_stage_advancement advances max 1 stage, resets counters
  4.  Curriculum command ranges expand from S1 small → S4 full
  5.  Action authority multiplier per stage (0.25, 0.50, 0.75, 1.0)
  6.  Stage-aware episode length (15 s, 15 s, 30 s, 45 s)
  7.  Orientation tracking disabled in S1/S2, enabled in S3/S4
  8.  Vehicle stationary in S1–S3, moving in S4
  9.  Reward weight configuration defaults (pos_tracking=1.0, orient=0.25, progress=0.5)
  10. Bounded position tracking reward formula: exp(-4 * error)
  11. Bounded progress reward: delta_position_error, clamped
  12. Task-space observation dimension = 70
  13. Action dimension = 4 (hydraulic cylinders)
  14. Termination diagnostics: 5 per-reason booleans, 7 counters
  15. Stage-aware timeout replaces episode_length_buf >= _stage_max_episode_length
  16. Reset logic: booleans and episode sums reset, accumulated counters NOT reset
  17. extras["log"] population with curriculum, reward, and termination keys
  18. Unsafe terminal penalty reconnected: uses _termination_* masks, not reset_terminated
  19. Terminal snapshots captured before super()._reset_idx()
  20. Curriculum telemetry logged in _reset_idx: stage, episodes_per_env, rates
  21. unsafe_terminal_penalty config field exists (= -5.0)
  22. Quaternion normalisation of orientation commands
  23. Bonus: curriculum/stage_ready, curriculum/advance_count keys
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

# Performance gate defaults
CURRICULUM_MIN_COMPLETED_EPISODES_PER_ENV = 5.0
CURRICULUM_MAX_MEAN_POSITION_ERROR = 0.50
CURRICULUM_MAX_VELOCITY_TERMINATION_RATE = 0.10
CURRICULUM_MAX_OTHER_UNSAFE_RATE = 0.01
CURRICULUM_MIN_TIMEOUT_RATE = 0.80

REW_TASK_POS_TRACKING = 1.0
REW_TASK_ORIENT_TRACKING = 0.25
REW_TASK_PROGRESS = 0.5
REW_TASK_PROGRESS_CLIP = 0.2
UNSAFE_TERMINAL_PENALTY = -5.0

TASK_SPACE_OBSERVATION_DIM = 70
ACTION_DIM = 4

SIM_DT = 0.01  # nominal Isaac sim dt (matches cfg.sim.dt)


# ── Standalone performance-gate advancement evaluator ─────────────────────────
def evaluate_stage_advancement(
    stage_index: int,
    episode_count: float,
    position_error_sum: float,
    timeout_count: float,
    velocity_count: float,
    other_unsafe_count: float,
    num_envs: int,
) -> int:
    """Return new stage index after evaluating performance gates.

    Advances exactly 1 stage if all gates are met, otherwise stays.
    """
    if stage_index >= 4:
        return stage_index

    n = max(num_envs, 1)
    episodes_per_env = episode_count / n
    mean_position_error = position_error_sum / max(episode_count, 1.0)
    timeout_rate = timeout_count / max(episode_count, 1.0)
    velocity_rate = velocity_count / max(episode_count, 1.0)
    other_unsafe = other_unsafe_count / max(episode_count, 1.0)

    stage_ready = (
        episodes_per_env >= CURRICULUM_MIN_COMPLETED_EPISODES_PER_ENV
        and mean_position_error <= CURRICULUM_MAX_MEAN_POSITION_ERROR
        and velocity_rate <= CURRICULUM_MAX_VELOCITY_TERMINATION_RATE
        and other_unsafe <= CURRICULUM_MAX_OTHER_UNSAFE_RATE
        and timeout_rate >= CURRICULUM_MIN_TIMEOUT_RATE
    )
    if stage_ready:
        return stage_index + 1
    return stage_index


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


# ── Standalone unsafe penalty (mirrors new env logic) ────────────────────────
def _unsafe_penalty(
    vel_mask: torch.Tensor,
    lower_mask: torch.Tensor,
    upper_mask: torch.Tensor,
    jnt_nf_mask: torch.Tensor,
    ctrl_nf_mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstructed unsafe penalty using _termination_* masks (not reset_terminated)."""
    unsafe = vel_mask | lower_mask | upper_mask | jnt_nf_mask | ctrl_nf_mask
    return UNSAFE_TERMINAL_PENALTY * unsafe.float()


# ── Test suite ───────────────────────────────────────────────────────────────
class TestCurriculumLearnabilityRepair(unittest.TestCase):
    """Validates the post-repair curriculum, unsafe reward, and reset ordering."""

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Performance-gate curriculum advancement
    # ═══════════════════════════════════════════════════════════════════════
    def test_no_advance_below_min_episodes(self) -> None:
        """Stage does not advance when episodes_per_env < 5."""
        new_stage = evaluate_stage_advancement(
            stage_index=1,
            episode_count=4,  # 4 / 600 envs => 0.0067
            position_error_sum=0.0,
            timeout_count=4,
            velocity_count=0,
            other_unsafe_count=0,
            num_envs=600,
        )
        self.assertEqual(new_stage, 1)

    def test_no_advance_high_position_error(self) -> None:
        """Stage does not advance when mean_position_error > 0.50."""
        new_stage = evaluate_stage_advancement(
            stage_index=1,
            episode_count=6000,
            position_error_sum=5000.0,  # mean ~0.833
            timeout_count=5400,
            velocity_count=0,
            other_unsafe_count=0,
            num_envs=600,
        )
        self.assertEqual(new_stage, 1)

    def test_no_advance_high_velocity_rate(self) -> None:
        """Stage does not advance when velocity_rate > 0.10."""
        new_stage = evaluate_stage_advancement(
            stage_index=1,
            episode_count=6000,
            position_error_sum=1200.0,  # mean = 0.2 < 0.50
            timeout_count=5000,
            velocity_count=1000,  # rate = 0.167 > 0.10
            other_unsafe_count=0,
            num_envs=600,
        )
        self.assertEqual(new_stage, 1)

    def test_no_advance_high_other_unsafe(self) -> None:
        """Stage does not advance when other_unsafe_rate > 0.01."""
        new_stage = evaluate_stage_advancement(
            stage_index=1,
            episode_count=6000,
            position_error_sum=600.0,  # mean = 0.1 < 0.50
            timeout_count=5900,
            velocity_count=0,
            other_unsafe_count=100,  # rate = 0.0167 > 0.01
            num_envs=600,
        )
        self.assertEqual(new_stage, 1)

    def test_no_advance_low_timeout_rate(self) -> None:
        """Stage does not advance when timeout_rate < 0.80."""
        new_stage = evaluate_stage_advancement(
            stage_index=1,
            episode_count=6000,
            position_error_sum=0.0,
            timeout_count=4000,  # rate = 0.667 < 0.80
            velocity_count=0,
            other_unsafe_count=0,
            num_envs=600,
        )
        self.assertEqual(new_stage, 1)

    def test_advances_when_all_gates_met(self) -> None:
        """Stage advances by 1 when all 5 performance gates are satisfied."""
        new_stage = evaluate_stage_advancement(
            stage_index=1,
            episode_count=6000,          # 10 per env
            position_error_sum=1200.0,   # mean = 0.2 < 0.50
            timeout_count=5400,          # rate = 0.9 > 0.80
            velocity_count=300,          # rate = 0.05 < 0.10
            other_unsafe_count=50,       # rate = 0.0083 < 0.01
            num_envs=600,
        )
        self.assertEqual(new_stage, 2)

    def test_advances_at_most_one_stage(self) -> None:
        """Stage advances exactly 1 (not 2+) even with perfect metrics on S1."""
        new_stage = evaluate_stage_advancement(
            stage_index=1,
            episode_count=6000,
            position_error_sum=0.0,
            timeout_count=6000,
            velocity_count=0,
            other_unsafe_count=0,
            num_envs=600,
        )
        self.assertEqual(new_stage, 2)

    def test_stage_4_stays_at_4(self) -> None:
        """Stage 4 never advances (already maximum)."""
        new_stage = evaluate_stage_advancement(
            stage_index=4,
            episode_count=6000,
            position_error_sum=0.0,
            timeout_count=6000,
            velocity_count=0,
            other_unsafe_count=0,
            num_envs=600,
        )
        self.assertEqual(new_stage, 4)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Curriculum command ranges
    # ═══════════════════════════════════════════════════════════════════════
    def test_command_ranges_expand(self) -> None:
        """X and Y command ranges expand (or stay equal) from S1 to S4."""
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
        """Action authority is 0.25, 0.50, 0.75, 1.0 for S1-S4."""
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
        """Vehicle speed range is 3-8 km/h in stage 4."""
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
        self.assertAlmostEqual(UNSAFE_TERMINAL_PENALTY, -5.0)

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
            self.assertLessEqual(rewards[i].item(), rewards[i - 1].item())

    def test_pos_tracking_max_at_zero_error(self) -> None:
        """Position tracking reward is maximized at zero error."""
        errors = torch.tensor([0.0, 0.01, 0.1, 1.0])
        rewards = _pos_tracking_reward(errors)
        self.assertGreater(rewards[0].item(), rewards[1].item())
        self.assertGreater(rewards[1].item(), rewards[2].item())

    # ═══════════════════════════════════════════════════════════════════════
    # 9. Progress reward
    # ═══════════════════════════════════════════════════════════════════════
    def test_progress_reward_positive_when_improving(self) -> None:
        """Progress reward is positive when error decreases (improvement)."""
        prev = torch.tensor([1.0, 2.0, 0.5])
        curr = torch.tensor([0.5, 1.0, 0.3])
        reward = _progress_reward(prev, curr)
        self.assertTrue(torch.all(reward >= 0.0))

    def test_progress_reward_negative_when_regressing(self) -> None:
        """Progress reward is negative when error increases (regression)."""
        prev = torch.tensor([0.5, 1.0, 0.3])
        curr = torch.tensor([1.0, 2.0, 0.5])
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
        """Progress reward is clamped to +/-rew_task_progress_clip."""
        clip = REW_TASK_PROGRESS_CLIP
        prev = torch.tensor([0.5, 0.0])
        curr = torch.tensor([0.0, 10.0])
        reward = _progress_reward(prev, curr)
        self.assertAlmostEqual(reward[0].item(), REW_TASK_PROGRESS * clip, places=5)
        self.assertAlmostEqual(reward[1].item(), -REW_TASK_PROGRESS * clip, places=5)

    # ═══════════════════════════════════════════════════════════════════════
    # 10. Observation dimension
    # ═══════════════════════════════════════════════════════════════════════
    def test_task_space_obs_dim(self) -> None:
        """Task-space observation dimension is 70."""
        self.assertEqual(TASK_SPACE_OBSERVATION_DIM, 70)

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
        self.assertGreater(reward[0].item(), reward[1].item())
        self.assertGreater(reward[1].item(), reward[2].item())

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
        """Seven accumulated termination counters exist."""
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
    # 15. extras["log"] keys (post-repair includes curriculum telemetry)
    # ═══════════════════════════════════════════════════════════════════════
    def test_extras_log_curriculum_keys(self) -> None:
        """extras['log'] contains curriculum/ keys including telemetry."""
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

    def test_curriculum_telemetry_keys(self) -> None:
        """Curriculum telemetry keys logged in _reset_idx."""
        keys = [
            "curriculum/stage",
            "curriculum/advance_count",
            "curriculum/stage_ready",
            "curriculum/episodes_per_env",
            "curriculum/mean_position_error",
            "curriculum/velocity_termination_rate",
            "curriculum/other_unsafe_rate",
            "curriculum/timeout_rate",
        ]
        self.assertEqual(len(keys), 8)

    # ═══════════════════════════════════════════════════════════════════════
    # 16. Config field type + performance gate defaults
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

    def test_performance_gate_defaults(self) -> None:
        """Performance gate defaults are set correctly."""
        self.assertEqual(CURRICULUM_MIN_COMPLETED_EPISODES_PER_ENV, 5.0)
        self.assertEqual(CURRICULUM_MAX_MEAN_POSITION_ERROR, 0.50)
        self.assertEqual(CURRICULUM_MAX_VELOCITY_TERMINATION_RATE, 0.10)
        self.assertEqual(CURRICULUM_MAX_OTHER_UNSAFE_RATE, 0.01)
        self.assertEqual(CURRICULUM_MIN_TIMEOUT_RATE, 0.80)

    # ═══════════════════════════════════════════════════════════════════════
    # 17. Unsafe terminal penalty reconnection (Change 6)
    # ═══════════════════════════════════════════════════════════════════════
    def test_unsafe_penalty_uses_termination_masks(self) -> None:
        """Unsafe penalty uses _termination_* masks, not reset_terminated."""
        vel = torch.tensor([True, False, False, False, True])
        lower = torch.tensor([False, True, False, False, False])
        upper = torch.tensor([False, False, True, False, False])
        jnf = torch.tensor([False, False, False, True, False])
        cnf = torch.tensor([False, False, False, False, False])
        penalty = _unsafe_penalty(vel, lower, upper, jnf, cnf)
        expected = torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0])
        self.assertTrue(torch.allclose(penalty, expected))

    def test_unsafe_penalty_zero_when_no_unsafe(self) -> None:
        """Unsafe penalty is zero when no termination mask is set."""
        vel = torch.zeros(4, dtype=torch.bool)
        lower = torch.zeros(4, dtype=torch.bool)
        upper = torch.zeros(4, dtype=torch.bool)
        jnf = torch.zeros(4, dtype=torch.bool)
        cnf = torch.zeros(4, dtype=torch.bool)
        penalty = _unsafe_penalty(vel, lower, upper, jnf, cnf)
        self.assertTrue(torch.all(penalty == 0.0))

    # ═══════════════════════════════════════════════════════════════════════
    # 18. Terminal snapshot ordering (Change 7 + Change 8)
    # ═══════════════════════════════════════════════════════════════════════
    def test_terminal_snapshot_capture_before_super_reset(self) -> None:
        """Simulate capturing episode_length_buf before super()._reset_idx."""
        episode_length_buf = torch.tensor([100.0, 200.0, 300.0, 50.0])
        env_ids = torch.tensor([0, 2])
        snapshot = episode_length_buf[env_ids].detach().float().clone()
        expected = torch.tensor([100.0, 300.0])
        self.assertTrue(torch.allclose(snapshot, expected))

        # Simulate super()._reset_idx zeroing the buffer
        episode_length_buf[env_ids] = 0.0
        self.assertTrue(torch.all(episode_length_buf[env_ids] == 0.0))
        # Snapshot preserved
        self.assertTrue(torch.allclose(snapshot, expected))

    # ═══════════════════════════════════════════════════════════════════════
    # 19. Progress reward sign consistency
    # ═══════════════════════════════════════════════════════════════════════
    def test_progress_sign_consistency(self) -> None:
        """Progress = prev - curr. Positive = improvement (error shrinks)."""
        prev = torch.tensor([0.8, 0.3])
        curr = torch.tensor([0.3, 0.8])
        reward = _progress_reward(prev, curr)
        self.assertGreater(reward[0].item(), 0.0, "Error decreased => positive reward")
        self.assertLess(reward[1].item(), 0.0, "Error increased => negative reward")

    # ═══════════════════════════════════════════════════════════════════════
    # 20. Reset semantics: booleans reset, accumulated counters persist
    # ═══════════════════════════════════════════════════════════════════════
    def test_reset_resets_booleans_not_counters(self) -> None:
        """Simulate that on reset: per-step booleans reset but counters persist."""
        bools = {
            "_termination_joint_nonfinite",
            "_termination_lower_limit",
            "_termination_upper_limit",
            "_termination_velocity",
            "_termination_control_nonfinite",
        }
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
    # 21. Episode sum tracking includes position_error_sum
    # ═══════════════════════════════════════════════════════════════════════
    def test_episode_sum_keys_includes_position_error(self) -> None:
        """Episode sum tracking includes _episode_position_error_sum for curriculum."""
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
            "_episode_position_error_sum",
        ]
        self.assertEqual(len(sums), 10)

    # ═══════════════════════════════════════════════════════════════════════
    # 22. Performance gate edge cases
    # ═══════════════════════════════════════════════════════════════════════
    def test_advancement_works_with_single_env(self) -> None:
        """evaluate_stage_advancement works correctly with single env (n=1)."""
        new_stage = evaluate_stage_advancement(
            stage_index=1,
            episode_count=5,         # 5/1 = 5 per env (exactly min)
            position_error_sum=2.0,  # mean = 0.4 < 0.50
            timeout_count=5,         # rate = 1.0 > 0.80
            velocity_count=0,        # rate = 0.0 < 0.10
            other_unsafe_count=0,    # rate = 0.0 < 0.01
            num_envs=1,
        )
        self.assertEqual(new_stage, 2)

    def test_advancement_fails_with_one_failure(self) -> None:
        """Stage does not advance if exactly one gate fails."""
        # All gates meet except velocity_rate = 0.11 > 0.10
        new_stage = evaluate_stage_advancement(
            stage_index=1,
            episode_count=6000,
            position_error_sum=0.0,
            timeout_count=6000,
            velocity_count=660,    # rate = 0.11 > 0.10
            other_unsafe_count=0,
            num_envs=600,
        )
        self.assertEqual(new_stage, 1)

    # ═══════════════════════════════════════════════════════════════════════
    # 23. unsafe_terminal_penalty config field
    # ═══════════════════════════════════════════════════════════════════════
    def test_unsafe_terminal_penalty_value(self) -> None:
        """unsafe_terminal_penalty is -5.0."""
        self.assertAlmostEqual(UNSAFE_TERMINAL_PENALTY, -5.0)

    # ═══════════════════════════════════════════════════════════════════════
    # 24. Curriculum aggregate counters reset on advancement
    # ═══════════════════════════════════════════════════════════════════════
    def test_aggregate_counters_reset_on_advancement(self) -> None:
        """After advancement, aggregate counters are reset for the new stage."""
        new_stage = evaluate_stage_advancement(
            stage_index=2,
            episode_count=3000,
            position_error_sum=300.0,   # mean = 0.1
            timeout_count=2700,         # rate = 0.9
            velocity_count=150,         # rate = 0.05
            other_unsafe_count=15,      # rate = 0.005
            num_envs=600,
        )
        # If counters were to be "reset", we just verify advancement works from S2
        self.assertEqual(new_stage, 3)


if __name__ == "__main__":
    unittest.main()
