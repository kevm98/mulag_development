"""Offline reward-balance tests for the fixed final-pose task repair."""

from __future__ import annotations

import math
import sys
import unittest
import ast
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CFG_SOURCE = (ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py").read_text(
    encoding="utf-8"
)
CFG_TREE = ast.parse(CFG_SOURCE)


def _literal_assignment(class_name: str, field: str):
    for node in ast.walk(CFG_TREE):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == field:
                            return ast.literal_eval(stmt.value)
    raise AssertionError(f"{class_name}.{field} not found")

from kevin_integration.rl.reward_terms import (
    avoidable_clamp_penalty,
    fd_load_excess_penalty,
    final_pose_within_tolerance,
    hold_progress_reward,
    near_target_angular_settling_reward,
    near_target_linear_settling_reward,
    normalized_task_space_potential,
    potential_progress_shaping,
    update_consecutive_success_counter,
    valve_saturation_penalty,
)


class TestRewardBalanceRepair(unittest.TestCase):
    def test_reward_contract_print_flag_is_separate_from_runtime_contract(self) -> None:
        source = (ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_runtime_contract_printed = False", source)
        self.assertIn("_reward_contract_printed = False", source)
        reward_printer = source[
            source.index("def _print_and_write_reward_contract_once") :
            source.index("def _record_rom_clamp_telemetry")
        ]
        self.assertIn("MulagAdaptiveRLEnv._reward_contract_printed", reward_printer)
        self.assertNotIn("MulagAdaptiveRLEnv._runtime_contract_printed", reward_printer)

    def test_reward_imbalance_arithmetic_matches_audit_example(self) -> None:
        positive = 0.1626 + 0.0198 + 0.0003
        negative = 0.1381 + 0.2106 + 0.0536 + 0.0018 + 0.0007
        self.assertAlmostEqual(positive, 0.1827, places=4)
        self.assertAlmostEqual(negative, 0.4048, places=4)
        self.assertAlmostEqual(negative / positive, 2.22, places=2)

    def test_clamp_occupancy_alone_is_zero_and_outward_request_is_bounded(self) -> None:
        boundary = torch.tensor([[True, False, False, False]])
        inward = torch.tensor([[False, False, False, False]])
        increment = torch.tensor([[0.004, 0.0, 0.0, 0.0]])
        penalty, activation, blocked = avoidable_clamp_penalty(
            boundary,
            inward,
            increment,
            increment_scale=0.008,
            weight=0.25,
        )
        self.assertEqual(float(penalty.item()), 0.0)
        self.assertEqual(float(activation.item()), 0.0)
        self.assertEqual(float(blocked.item()), 0.0)

        outward = torch.tensor([[True, False, False, False]])
        penalty, activation, blocked = avoidable_clamp_penalty(
            boundary,
            outward,
            increment,
            increment_scale=0.008,
            weight=0.25,
        )
        self.assertLess(float(penalty.item()), 0.0)
        self.assertGreater(float(activation.item()), 0.0)
        self.assertGreater(float(blocked.item()), 0.0)
        self.assertGreaterEqual(float(penalty.item()), -0.25)

    def test_valve_penalty_soft_threshold_and_bound(self) -> None:
        below = torch.tensor([[0.2, -0.5, 0.89, 0.0]])
        penalty, fraction = valve_saturation_penalty(below, sat_start=0.90, weight=0.15)
        self.assertEqual(float(penalty.item()), 0.0)
        self.assertEqual(float(fraction.item()), 0.0)

        saturated = torch.tensor([[0.95, -1.0, 0.0, 0.0]])
        penalty, fraction = valve_saturation_penalty(saturated, sat_start=0.90, weight=0.15)
        self.assertLess(float(penalty.item()), 0.0)
        self.assertGreater(float(fraction.item()), 0.0)
        self.assertGreaterEqual(float(penalty.item()), -0.15)

    def test_fd_load_excess_threshold_and_bound(self) -> None:
        below = torch.tensor([[0.5, -0.8, 1.0, 0.0]])
        penalty, fraction = fd_load_excess_penalty(below, soft_threshold=1.0, weight=0.01)
        self.assertEqual(float(penalty.item()), 0.0)
        self.assertEqual(float(fraction.item()), 0.0)

        above = torch.tensor([[1.2, -2.0, 0.0, 0.0]])
        penalty, fraction = fd_load_excess_penalty(above, soft_threshold=1.0, weight=0.01)
        self.assertLess(float(penalty.item()), 0.0)
        self.assertGreater(float(fraction.item()), 0.0)
        self.assertGreaterEqual(float(penalty.item()), -0.01)

    def test_potential_progress_toward_away_reset_and_static(self) -> None:
        far = normalized_task_space_potential(
            torch.tensor([0.6]),
            torch.tensor([math.radians(12.0)]),
            position_scale_m=1.0,
            orientation_scale_rad=math.pi,
            position_weight=1.0,
            orientation_weight=0.25,
        )
        near = normalized_task_space_potential(
            torch.tensor([0.5]),
            torch.tensor([math.radians(8.0)]),
            position_scale_m=1.0,
            orientation_scale_rad=math.pi,
            position_weight=1.0,
            orientation_weight=0.25,
        )
        toward = potential_progress_shaping(far, near, gamma=0.99, weight=0.5)
        away = potential_progress_shaping(near, far, gamma=0.99, weight=0.5)
        reset = potential_progress_shaping(far, near, gamma=0.99, weight=0.5, reset_mask=torch.tensor([True]))
        static = potential_progress_shaping(near, near, gamma=0.99, weight=0.5)
        self.assertGreater(float(toward.item()), 0.0)
        self.assertLess(float(away.item()), 0.0)
        self.assertEqual(float(reset.item()), 0.0)
        self.assertLess(abs(float(static.item())), 0.01)

    def test_hold_progress_monotonic_and_resets_when_broken(self) -> None:
        counters = torch.tensor([0, 10, 25, 50])
        valid = torch.tensor([True, True, True, True])
        rewards = hold_progress_reward(counters, required_hold_steps=50, weight=1.0, within_final_pose=valid)
        self.assertTrue(bool(torch.all(rewards[:-1] <= rewards[1:])))
        self.assertAlmostEqual(float(rewards[-1].item()), 1.0, places=6)
        broken = hold_progress_reward(
            torch.tensor([25]),
            required_hold_steps=50,
            weight=1.0,
            within_final_pose=torch.tensor([False]),
        )
        self.assertEqual(float(broken.item()), 0.0)

    def test_hold_progress_has_clear_return_advantage_for_full_hold(self) -> None:
        counters = torch.arange(1, 51)
        valid = torch.ones_like(counters, dtype=torch.bool)
        full_hold_return = torch.sum(
            hold_progress_reward(
                counters,
                required_hold_steps=50,
                weight=1.0,
                within_final_pose=valid,
            )
        )
        one_step_entries = torch.sum(
            hold_progress_reward(
                torch.ones(50, dtype=torch.long),
                required_hold_steps=50,
                weight=1.0,
                within_final_pose=valid,
            )
        )
        self.assertGreater(float(full_hold_return.item()), 10.0)
        self.assertGreater(
            float(full_hold_return.item()),
            500.0 * float(one_step_entries.item()),
        )

    def test_hold_progress_resets_when_any_strict_condition_fails(self) -> None:
        base = dict(
            position_tolerance_m=0.05,
            orientation_tolerance_rad=0.0872664626,
            linear_speed_tolerance_mps=0.05,
            angular_speed_tolerance_radps=0.10,
        )
        counter = torch.tensor([12, 12, 12, 12, 12], dtype=torch.long)
        within = final_pose_within_tolerance(
            torch.tensor([0.01, 0.06, 0.01, 0.01, 0.01]),
            torch.tensor([0.01, 0.01, 0.10, 0.01, 0.01]),
            torch.tensor([0.01, 0.01, 0.01, 0.06, 0.01]),
            torch.tensor([0.01, 0.01, 0.01, 0.01, 0.11]),
            **base,
        )
        updated = update_consecutive_success_counter(counter, within)
        self.assertTrue(torch.equal(updated, torch.tensor([13, 0, 0, 0, 0])))
        rewards = hold_progress_reward(
            updated,
            required_hold_steps=50,
            weight=1.0,
            within_final_pose=within,
        )
        self.assertEqual(float(torch.sum(rewards[1:]).item()), 0.0)

    def test_near_target_linear_settling_is_gated_and_speed_ordered(self) -> None:
        base = dict(
            position_tolerance_m=0.05,
            orientation_tolerance_rad=0.0872664626,
            linear_speed_tolerance_mps=0.05,
            position_gate_scale=1.5,
            orientation_gate_scale=1.5,
            speed_margin_mps=0.015,
            weight=0.35,
        )
        far = near_target_linear_settling_reward(
            torch.tensor([0.50]),
            torch.tensor([0.50]),
            torch.tensor([0.00]),
            **base,
        )
        near = near_target_linear_settling_reward(
            torch.tensor([0.02, 0.02, 0.02]),
            torch.tensor([0.02, 0.02, 0.02]),
            torch.tensor([0.052, 0.02, 0.00]),
            **base,
        )
        self.assertLess(abs(float(far.item())), 1.0e-8)
        self.assertLess(float(near[0].item()), 0.0)
        self.assertGreater(float(near[1].item()), 0.0)
        self.assertGreater(float(near[1].item()), float(near[0].item()))
        self.assertGreater(float(near[2].item()), float(near[1].item()))

    def test_near_target_angular_settling_is_gated_and_speed_ordered(self) -> None:
        base = dict(
            position_tolerance_m=0.05,
            orientation_tolerance_rad=0.0872664626,
            angular_speed_tolerance_radps=0.10,
            position_gate_scale=1.5,
            orientation_gate_scale=1.5,
            speed_margin_radps=0.05,
            weight=0.35,
        )
        far = near_target_angular_settling_reward(
            torch.tensor([0.50]),
            torch.tensor([0.50]),
            torch.tensor([0.00]),
            **base,
        )
        near = near_target_angular_settling_reward(
            torch.tensor([0.02, 0.02, 0.02]),
            torch.tensor([0.02, 0.02, 0.02]),
            torch.tensor([0.1074, 0.05, 0.00]),
            **base,
        )
        self.assertLess(abs(float(far.item())), 1.0e-8)
        self.assertLess(float(near[0].item()), 0.0)
        self.assertGreater(float(near[1].item()), float(near[0].item()))
        self.assertGreater(float(near[2].item()), float(near[1].item()))

    def test_reward_balance_weights_remain_bounded_after_hold_increase(self) -> None:
        class_name = "MulagAdaptiveRLJointReachEnvCfg"
        self.assertEqual(_literal_assignment(class_name, "rew_task_pos_tracking"), 1.0)
        self.assertEqual(_literal_assignment(class_name, "position_tracking_exponential_scale"), 2.0)
        self.assertEqual(_literal_assignment(class_name, "rew_task_orient_tracking"), 0.25)
        self.assertEqual(_literal_assignment(class_name, "rew_task_progress"), 0.5)
        self.assertEqual(_literal_assignment(class_name, "rew_task_progress_clip"), 0.2)
        self.assertEqual(_literal_assignment(class_name, "taskspace_progress_gamma"), 1.0)
        self.assertEqual(_literal_assignment(class_name, "rew_final_pose_settling"), 0.20)
        self.assertEqual(_literal_assignment(class_name, "rew_near_target_linear_settling"), 1.00)
        self.assertEqual(_literal_assignment(class_name, "rew_near_target_angular_settling"), 0.35)
        self.assertEqual(_literal_assignment(class_name, "near_target_linear_settling_speed_margin_mps"), 0.015)
        self.assertEqual(_literal_assignment(class_name, "near_target_linear_settling_sharpness"), 1.5)
        self.assertEqual(_literal_assignment(class_name, "near_target_angular_settling_speed_margin_radps"), 0.08)
        self.assertEqual(_literal_assignment(class_name, "rew_final_pose_hold_progress"), 1.00)

        dense_positive_bound = 1.0 + 0.25 + 1.0 * 0.2 + 0.20 + 1.00 + 0.35 + 1.00
        actuator_penalty_bound = 0.15 + 0.25 + 0.01
        velocity_penalty_bound = 0.15 + 0.08
        self.assertLessEqual(dense_positive_bound, 4.10)
        self.assertLessEqual(actuator_penalty_bound, 0.50)
        self.assertLessEqual(actuator_penalty_bound + velocity_penalty_bound, 0.65)


if __name__ == "__main__":
    unittest.main(verbosity=2)
