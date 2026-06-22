"""Evidence tests for the current task-space learning failure modes.

These tests intentionally avoid IsaacLab startup.  They exercise the exported
ROM, the multirate wrapper, static config values, and saved TensorBoard
telemetry from the local failing task-space run.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kevin_integration.rom_cylinders.cylinder_coordinate_adapter import load_coordinate_mapping
from kevin_integration.rom_cylinders.multirate_rom_wrapper import MultirateROMConfig, MultirateROMWrapper
from kevin_integration.rom_cylinders.rom_data_contract import (
    CORE_INPUT_FEATURE_NAMES,
    DP_SLICE,
    EXTENSION_SLICE,
    EXTENSION_VELOCITY_SLICE,
    HISTORY_LENGTH,
    JOINT_POS_SLICE,
    VALVE_SLICE,
)
from kevin_integration.rom_cylinders.valve_conditioned_causal_rom import EXPORT_DIR, ValveConditionedROMInference


CFG_PATH = ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py"
CFG_TREE = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
LATEST_EVIDENCE_RUN = (
    ROOT
    / "logs/rsl_rl/kevin_hydraulic_rom_task_space/2026-06-21_16-54-47/"
    / "events.out.tfevents.1782053790.rrlab-Legion-5-Pro-16ARH7H.51491.0"
)


def _literal_assignment(class_name: str, field: str):
    for node in ast.walk(CFG_TREE):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == field:
                            return ast.literal_eval(item.value)
    raise AssertionError(f"{class_name}.{field} not found")


def _rom_history(rom, *, velocity: torch.Tensor, valve: float) -> torch.Tensor:
    history = torch.zeros((1, HISTORY_LENGTH, len(CORE_INPUT_FEATURE_NAMES)), dtype=torch.float32)
    history[:, :, EXTENSION_SLICE] = rom.extension_mean.reshape(1, 1, 4)
    history[:, :, DP_SLICE] = rom.training_mean_dp.reshape(1, 1, 4)
    history[:, :, JOINT_POS_SLICE] = rom.joint_pos_mean.reshape(1, 1, 5)
    history[:, :, EXTENSION_VELOCITY_SLICE] = velocity.reshape(1, 1, 4)
    history[:, :, VALVE_SLICE] = float(valve)
    return history


def _applied_isaac_delta_10ms(rom, mapping_sign: torch.Tensor, *, velocity: float, valve: float) -> torch.Tensor:
    history = _rom_history(rom, velocity=torch.full((4,), velocity), valve=valve)
    delta_50ms = rom(history)["final_delta_extension"][0]
    delta_10ms = delta_50ms / 5.0
    delta_10ms = torch.clamp(delta_10ms, -0.04 / 5.0, 0.04 / 5.0)
    delta_isaac = mapping_sign.float() * delta_10ms
    delta_isaac = torch.clamp(delta_isaac, -0.008, 0.008)
    return torch.clamp(delta_isaac, -0.80 * 0.01, 0.80 * 0.01)


class ValveEchoROM:
    """Predictor whose output follows the newest valve feature."""

    def __call__(self, history: torch.Tensor) -> torch.Tensor:
        valve_1 = history[:, -1, VALVE_SLICE.start : VALVE_SLICE.start + 1]
        return valve_1.repeat(1, 4) * 0.005


class TestLearningFailureDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inference = ValveConditionedROMInference(EXPORT_DIR, device="cpu")
        cls.rom = cls.inference.model
        cls.mapping = load_coordinate_mapping(
            ROOT / "kevin_integration/rom_cylinders/cylinder_coordinate_mapping.json",
            mode="nominal_sign_offset",
            allow_provisional_nominal_reference=True,
        )

    def test_rom_valve_authority_is_far_below_direct_id_position_scale(self) -> None:
        """Full valve through ROM is much weaker than direct ID position mode."""
        position_scale = _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "position_scale")
        h0 = _rom_history(self.rom, velocity=torch.full((4,), 0.02), valve=0.0)
        h1 = _rom_history(self.rom, velocity=torch.full((4,), 0.02), valve=1.0)
        delta = self.mapping.sign.float() * (
            (self.rom(h1)["final_delta_extension"][0] - self.rom(h0)["final_delta_extension"][0]) / 5.0
        )
        authority_fraction = delta / float(position_scale)

        self.assertLess(float(authority_fraction[0]), 0.01)
        self.assertLess(float(authority_fraction[1]), 0.01)
        self.assertLess(float(torch.max(authority_fraction)), 0.12)

    def test_physx_velocity_spike_saturates_rom_target_increment(self) -> None:
        """A 3 m/s velocity sample drives the applied ROM target to the clamp."""
        normal = _applied_isaac_delta_10ms(
            self.rom, self.mapping.sign, velocity=0.02, valve=1.0
        ).abs()
        spike = _applied_isaac_delta_10ms(
            self.rom, self.mapping.sign, velocity=3.0, valve=1.0
        ).abs()

        self.assertTrue(torch.allclose(spike, torch.full((4,), 0.008), atol=1.0e-9))
        self.assertGreater(float(spike[0] / normal[0]), 100.0)
        self.assertGreater(float(spike[1] / normal[1]), 50.0)

    def test_exported_rom_uses_current_sample_not_full_history(self) -> None:
        """Older history rows do not affect the exported ROM prediction."""
        current = _rom_history(self.rom, velocity=torch.full((4,), 0.02), valve=1.0)
        perturbed = current.clone()
        perturbed[:, :-1, :] = 1234.0

        out_current = self.rom(current)["final_delta_extension"]
        out_perturbed = self.rom(perturbed)["final_delta_extension"]
        self.assertTrue(torch.allclose(out_current, out_perturbed, atol=0.0, rtol=0.0))

    def test_multirate_wrapper_holds_stale_action_for_four_control_ticks(self) -> None:
        """After an update tick, action changes wait until the next 50 ms ROM tick."""
        wrapper = MultirateROMWrapper(
            ValveEchoROM(),
            num_envs=1,
            config=MultirateROMConfig(history_length=1),
            device="cpu",
        )

        obs_pos = torch.zeros((1, len(CORE_INPUT_FEATURE_NAMES)))
        obs_pos[:, VALVE_SLICE] = 1.0
        obs_neg = torch.zeros_like(obs_pos)
        obs_neg[:, VALVE_SLICE] = -1.0

        first = wrapper.step(obs_pos)
        held = [wrapper.step(obs_neg) for _ in range(4)]
        refreshed = wrapper.step(obs_neg)

        self.assertGreater(float(first[0, 0]), 0.0)
        for value in held:
            self.assertGreater(float(value[0, 0]), 0.0)
        self.assertLess(float(refreshed[0, 0]), 0.0)

    def test_current_config_has_velocity_and_feedforward_repair(self) -> None:
        self.assertGreaterEqual(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "final_causal_rom_valve_feedforward_gain"),
            0.640,
        )
        self.assertTrue(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "enable_validated_hydraulic_velocity_termination")
        )
        self.assertEqual(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "hydraulic_hard_velocity_limits_mps"),
            [0.5, 0.5, 0.5, 0.5],
        )

    @unittest.skipUnless(LATEST_EVIDENCE_RUN.exists(), "local TensorBoard evidence run is absent")
    def test_latest_run_has_low_success_high_spikes_and_lr_at_ceiling(self) -> None:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(LATEST_EVIDENCE_RUN), size_guidance={"scalars": 0})
        ea.Reload()

        def values(tag: str) -> list[float]:
            return [event.value for event in ea.Scalars(tag)]

        success = values("rolling_1000/success_rate")
        spikes = values("velocity_audit/physx_spike_count")
        velocity_done = values("termination/velocity")
        lr = values("Loss/learning_rate")

        self.assertLessEqual(max(success), 0.02)
        self.assertGreater(spikes[-1], 100_000.0)
        self.assertEqual(max(velocity_done), 0.0)
        self.assertGreaterEqual(max(lr), 0.009)


if __name__ == "__main__":
    unittest.main(verbosity=2)
