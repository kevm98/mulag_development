"""Evidence tests for the convergence plateau at ~6-9% success rate."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kevin_integration.rom_cylinders.valve_conditioned_causal_rom import EXPORT_DIR, ValveConditionedROMInference
from kevin_integration.rom_cylinders.rom_data_contract import (
    EXTENSION_SLICE, EXTENSION_VELOCITY_SLICE,
    VALVE_SLICE, DP_SLICE, JOINT_POS_SLICE, JOINT_VEL_SLICE,
    HISTORY_LENGTH, CORE_INPUT_FEATURE_NAMES,
)

CFG_PATH = ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py"
CFG_TREE = ast.parse(CFG_PATH.read_text(encoding="utf-8"))


def _literal(class_name: str, field: str):
    for node in ast.walk(CFG_TREE):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == field:
                            return ast.literal_eval(item.value)
    raise AssertionError(f"{class_name}.{field} not found")


class TestConvergencePlateau(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rom = ValveConditionedROMInference(EXPORT_DIR, device="cpu").model

    def test_position_tracking_reward_has_dense_plateau_signal_after_repair(self):
        """At the plateau error (0.66m), repaired tracking still gives usable signal."""
        mean_err = 0.66  # from curriculum/mean_position_error at iter 205
        target_err = 0.05  # strict position threshold
        scale = _literal("MulagAdaptiveRLJointReachEnvCfg", "position_tracking_exponential_scale")

        reward_at_mean = float(torch.exp(torch.tensor(-float(scale) * mean_err)))
        reward_at_target = float(torch.exp(torch.tensor(-float(scale) * target_err)))
        gradient_at_mean = float(scale) * reward_at_mean
        gradient_at_target = float(scale) * reward_at_target

        self.assertEqual(scale, 2.0)
        self.assertGreater(reward_at_mean, 0.25,
            f"Position tracking reward at {mean_err}m is {reward_at_mean:.4f} — "
            "should stay dense enough to aim from the plateau regime")

        grad_ratio = gradient_at_target / max(gradient_at_mean, 1e-8)
        self.assertLess(grad_ratio, 4.0,
            f"Gradient ratio target/mean = {grad_ratio:.1f} — "
            "the old exp(-4e) starvation has returned")

        self.assertLess(reward_at_target / max(reward_at_mean, 1e-8), 4.0,
            f"Reward drops {reward_at_target/reward_at_mean:.1f}x from target to mean")

    def test_stage1_targets_are_small_in_joint_space(self):
        """Curriculum stage 1 targets are ±0.008 rad — very small."""
        amplitude = _literal("MulagAdaptiveRLJointReachEnvCfg", "q_ref_target_amplitude")
        scales = _literal("MulagAdaptiveRLJointReachEnvCfg", "goal_curriculum_stage_scales")
        stage1_range = [a * scales[0] for a in amplitude]

        # Max joint displacement at stage 1
        self.assertLess(max(stage1_range), 0.01,
            f"Stage 1 max joint displacement {max(stage1_range):.4f} rad should be < 0.01")

        # But the policy mean error is 0.66m — much larger
        # This means the policy explores joint configurations far from the target
        # even though targets are close. The exploration noise (action std 0.18)
        # causes ~10x larger joint motion than the target displacement.

    def test_rom_feedforward_authority_ratio(self):
        """ROM feedforward restores a monotone, usable fraction of id_direct authority."""
        ff_gain = _literal("MulagAdaptiveRLJointReachEnvCfg", "final_causal_rom_valve_feedforward_gain")
        position_scale = _literal("MulagAdaptiveRLJointReachEnvCfg", "position_scale")

        rom_authority = ff_gain * position_scale
        direct_authority = position_scale  # 0.005 m/step

        self.assertAlmostEqual(rom_authority, 0.00325, places=5)
        self.assertGreaterEqual(rom_authority / direct_authority, 0.640)
        self.assertLessEqual(rom_authority / direct_authority, 0.70)

        # The ROM's own valve authority is weak and can have a small negative
        # slope near zero command.  The feedforward term provides the minimum
        # positive command slope needed by the pipeline stress test.

    def test_rom_latency_vs_id_direct(self):
        """ROM has 5x longer control latency than id_direct."""
        rom_dt = self.rom.dt  # 0.05s
        control_dt = 0.01

        ratio = rom_dt / control_dt
        self.assertEqual(ratio, 5.0)

        # Literature: Karafyllis & Krstic (2010) — sampled-data systems
        # with ZOH degrade proportionally to sampling period.
        # With plant time constant ~100ms (hydraulic cylinder),
        # sampling at 50ms gives ratio 0.5 — at stability boundary.
        plant_tc = 0.1  # estimated hydraulic cylinder time constant
        stability_ratio = rom_dt / plant_tc
        self.assertGreater(stability_ratio, 0.3,
            f"ROM sampling period / plant TC = {stability_ratio:.2f} — "
            "should exceed 0.3 (significant control degradation expected)")

    def test_late_action_std_schedule_reduces_plateau_noise(self):
        """Once any 50-step hold is demonstrated, the schedule tightens precision."""
        min_std = _literal("MulagHydraulicROMTaskSpacePPORunnerCfg", "action_std_schedule_min_std")
        max_std_final = _literal("MulagHydraulicROMTaskSpacePPORunnerCfg", "action_std_schedule_max_std_final")
        entropy_final = _literal("MulagHydraulicROMTaskSpacePPORunnerCfg", "action_std_schedule_entropy_final")

        self.assertEqual(min_std, 0.05)
        self.assertEqual(max_std_final, 0.15)
        self.assertEqual(entropy_final, 0.0005)
        self.assertLess(max_std_final, 0.18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
