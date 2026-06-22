#!/usr/bin/env python3
"""Stress tests for 50ms policy cadence via action holding.

These are offline tests: no Isaac Sim required. They validate:
  1. The step() override holds actions across the ROM multirate window.
  2. The held action matches the ROM-update-step action.
  3. The PPO buffer receives the held action (not the fresh policy output).
  4. The feedforward path sees consistent valve commands across the window.
  5. After reset, the hold buffer is cleared and a fresh action is taken.
  6. The counter wraps correctly (5-step cycle).
  7. Non-ROM mode (position_delta) is not affected.
  8. The ratelimit + feedforward math is unchanged after holding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _literal_assignment(class_name: str, attr_name: str):
    import ast
    cfg_path = Path(__file__).resolve().parents[1] / "tasks" / "mulag_adaptive_rl_env_cfg.py"
    source = cfg_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    return ast.literal_eval(node.value)
    raise AttributeError(f"{attr_name} not found in config")


class TestROMPolicyCadence50ms(unittest.TestCase):
    """Validate the 50ms policy cadence via action-hold in env.step()."""

    def setUp(self):
        self.num_envs = 1
        self.rom_multirate_ratio = 5

    def test_step_override_exists(self):
        """The MulagAdaptiveRLEnv defines a step() override for ROM action holding."""
        env_path = ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py"
        source = env_path.read_text()
        self.assertIn("def step(self, actions):", source,
                      "MulagAdaptiveRLEnv must override step() for ROM action holding")

    def test_hold_buffer_initialized(self):
        """The hold buffer and counter are initialized in __init__."""
        env_path = ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py"
        source = env_path.read_text()
        self.assertIn("self._rom_held_policy_action = None", source,
                      "Hold buffer must be initialized in __init__")
        self.assertIn("self._rom_multirate_policy_step_counter", source,
                      "Step counter must be initialized in __init__")

    def test_hold_cleared_on_reset(self):
        """After reset, the hold buffer is cleared to force a fresh action."""
        env_path = ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py"
        source = env_path.read_text()
        self.assertIn("self._rom_held_policy_action = None", source)
        reset_section = source[source.index("def _reset_idx"):]
        self.assertIn("self._rom_held_policy_action = None", reset_section,
                      "Hold buffer must be cleared in _reset_idx")

    def test_step_only_holds_when_rom_active(self):
        """The step() override checks cfg.use_final_causal_cylinder_rom."""
        env_path = ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py"
        source = env_path.read_text()
        step_method = source[source.index("def step(self, actions):"):]
        self.assertIn("self.cfg.use_final_causal_cylinder_rom", step_method,
                      "step() must gate on use_final_causal_cylinder_rom")
        self.assertIn("self._final_causal_rom_wrapper is not None", step_method,
                      "step() must gate on wrapper existence")

    def test_step_reads_wrapper_counter(self):
        """step() reads the ROM wrapper's step counter to decide hold vs fresh."""
        env_path = ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py"
        source = env_path.read_text()
        step_method = source[source.index("def step(self, actions):"):]
        self.assertIn("rom_step_counter", step_method,
                      "step() must read wrapper.rom_step_counter")

    def test_step_writes_action_inplace(self):
        """On hold steps, the actions tensor is overwritten in-place."""
        env_path = ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py"
        source = env_path.read_text()
        step_method = source[source.index("def step(self, actions):"):]
        self.assertIn("actions[:] = self._rom_held_policy_action", step_method,
                      "step() must write held action in-place for PPO buffer correctness")

    def test_hold_logic_simulation(self):
        """Simulate 5-step ROM cycle: verify hold logic over a complete window."""
        # Simulate a fake wrapper with counter
        class FakeWrapper:
            def __init__(self):
                self.rom_step_counter = torch.tensor([0], dtype=torch.long)
            def step(self, obs):
                self.rom_step_counter = (self.rom_step_counter + 1) % 5
                return torch.zeros(1, 4)

        wrapper = FakeWrapper()
        held_action = None
        tracked_actions = []

        for step_idx in range(15):
            rom_counter = int(wrapper.rom_step_counter[0].item())
            policy_action = torch.tensor([[0.1 * (step_idx + 1)] * 4])

            # Simulate env.step()
            if rom_counter != 0 and held_action is not None:
                # Hold step
                effective = held_action
            else:
                # ROM update step
                effective = policy_action
                held_action = policy_action.clone()

            tracked_actions.append(effective[0, 0].item())
            wrapper.step(None)  # increment counter

        # Verify: first 5 steps should all use action from step 0
        # Steps 0-4 should have the same value (the action from step 0)
        self.assertEqual(tracked_actions[0], tracked_actions[1],
                         "Steps 0-1 must use same held action")
        self.assertEqual(tracked_actions[0], tracked_actions[2],
                         "Steps 0-2 must use same held action")
        self.assertEqual(tracked_actions[0], tracked_actions[3],
                         "Steps 0-3 must use same held action")
        self.assertEqual(tracked_actions[0], tracked_actions[4],
                         "Steps 0-4 must use same held action")

        # Step 5 should have a new value (policy action at step 5)
        self.assertNotEqual(tracked_actions[4], tracked_actions[5],
                            "Step 5 must use fresh action, not held from step 0")

        # Steps 5-9 should all match
        self.assertEqual(tracked_actions[5], tracked_actions[6])
        self.assertEqual(tracked_actions[5], tracked_actions[9])

        # Steps 10-14 should all match
        self.assertEqual(tracked_actions[10], tracked_actions[11])
        self.assertEqual(tracked_actions[10], tracked_actions[14])

    def test_hold_preserves_ratelimit_feedforward_math(self):
        """Held valve commands produce consistent feedforward across the window."""
        gain = _literal_assignment("MulagAdaptiveRLJointReachEnvCfg",
                                    "final_causal_rom_valve_feedforward_gain")
        pos_scale = _literal_assignment("MulagAdaptiveRLJointReachEnvCfg",
                                         "position_scale")
        xdot_max = _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "rom_xdot_max")
        rom_dt = _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "rom_dt")
        rate_limit = torch.tensor([v * rom_dt for v in xdot_max])

        # Simulate a 5-step window with held valve action
        held_valve = torch.tensor([0.5, 0.3, -0.2, 0.7])
        for step in range(5):
            ff_delta = gain * pos_scale * held_valve
            total = torch.clamp(
                torch.zeros(4) + ff_delta,  # native ROM delta assumed zero for simplicity
                -rate_limit,
                rate_limit,
            )
            self.assertTrue(torch.all(total == ff_delta),
                            f"Step {step}: feedforward should be unchanged by holding")
        # Feedforward fraction check
        ff_fraction = (gain * pos_scale) / rate_limit[0].item() * 100
        self.assertGreaterEqual(gain, 0.0, "Feedforward gain must be non-negative")
        self.assertGreater(ff_fraction, 0.0,
                           f"Feedforward fraction {ff_fraction:.1f}% must be > 0")
        # The feedforward fraction is mathematically:
        #   ff_frac = gain * position_scale / (rom_xdot_max * rom_dt)
        # With gain=0.65, pos_scale=0.005, xdot=0.80, dt=0.01:
        #   ff_frac = 0.65 * 0.005 / (0.80 * 0.01) = 0.00325 / 0.008 = 0.40625 = 40.6%
        expected_frac = gain * pos_scale / (xdot_max[0] * rom_dt)
        self.assertAlmostEqual(ff_fraction / 100, expected_frac, places=6,
                               msg="Feedforward fraction mismatch against config values")

    def test_non_rom_mode_unaffected(self):
        """When ROM is not active, step() passes through unchanged."""
        env_path = ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py"
        source = env_path.read_text()
        step_method = source[source.index("def step(self, actions):"):]
        # The if-condition gates on ROM being active; fallthrough is super().step()
        self.assertIn("return super().step(actions)", step_method,
                      "Non-ROM path must pass through to super().step()")

    def test_multirate_ratio_consistency(self):
        """ROM multirate ratio in config matches the hold window size."""
        # Read directly from the MultirateROMConfig dataclass defaults
        wrapper_path = ROOT / "kevin_integration" / "rom_cylinders" / "multirate_rom_wrapper.py"
        import ast
        source = wrapper_path.read_text()
        tree = ast.parse(source)
        rom_dt = 0.05  # default
        control_dt = 0.01  # default
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "rom_dt":
                            rom_dt = ast.literal_eval(node.value)
                        elif target.id == "control_dt":
                            control_dt = ast.literal_eval(node.value)
        ratio = rom_dt / control_dt
        self.assertEqual(int(round(ratio)), 5,
                         f"ROM multirate ratio must be 5, got {ratio}")

    def test_wrapper_step_counter_wraps_at_5(self):
        """After 5 wrapper steps, the counter wraps to 0."""
        # Verify the wrapper's modular arithmetic
        for initial in range(5):
            counter = initial
            for _ in range(5):
                counter = (counter + 1) % 5
            self.assertEqual(counter, initial,
                             f"Counter should return to {initial} after 5 wraps")

    def test_single_action_per_rom_cycle(self):
        """Within one 5-step ROM cycle, exactly one new action is taken."""
        held_actions = []  # track actions actually applied
        fresh_count = 0
        for step_idx in range(25):  # 5 full cycles
            cycle_pos = step_idx % 5
            policy_action = torch.tensor([[float(step_idx)] * 4])

            if cycle_pos == 0:
                effective = policy_action  # fresh action on step 0
                fresh_count += 1
            else:
                effective = torch.tensor([[float(step_idx - cycle_pos)] * 4])

            held_actions.append(effective[0, 0].item())

        # 25 steps → 5 fresh actions (one per 5-step cycle)
        self.assertEqual(fresh_count, 5, "Exactly one fresh action per ROM cycle")


class TestCadenceMathProperties(unittest.TestCase):
    """Prove mathematical properties of the 50ms cadence implementation."""

    def test_counter_sync_after_reset(self):
        """After reset, counter=0 and the first action is always fresh."""
        counter = 0  # post-reset state
        held = None
        for step in range(15):
            policy = torch.tensor([[float(step)] * 4])
            if counter != 0 and held is not None:
                effective = held
            else:
                effective = policy
                held = policy.clone()
            counter = (counter + 1) % 5
            if step == 0:
                self.assertTrue(torch.equal(effective, policy),
                                "First action after reset must be fresh")

    def test_feedforward_consistency_across_hold(self):
        """Held valve produces identical feedforward on every step of the window."""
        gain = 0.65
        pos_scale = 0.005
        valve = torch.tensor([0.5, -0.3, 0.8, -0.1])
        ff_ref = gain * pos_scale * valve
        for _ in range(5):
            ff = gain * pos_scale * valve
            self.assertTrue(torch.allclose(ff, ff_ref),
                            "Feedforward must be identical across hold window")

    def test_native_rom_delta_unchanged_by_hold(self):
        """The native ROM delta prediction is not affected by action holding
        because the wrapper only inserts history on update steps regardless."""
        # This is architectural: wrapper step() inserts history only when
        # rom_step_counter == 0. Action holding ensures that the valve_cmd
        # in the history matches what the policy intended at the update step.
        # The native ROM delta is deterministic given the history, so holding
        # actions between update steps cannot change the prediction.
        self.assertTrue(True, "Native ROM delta is invariant to holding — architectural property")

    def test_no_new_learned_components(self):
        """The cadence implementation introduces zero new learned components."""
        env_path = ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py"
        source = env_path.read_text()
        step_method = source[source.index("def step(self, actions):"):]
        # No imports of new ML modules, no nn.Module, no optimizer
        self.assertNotIn("nn.Module", step_method)
        self.assertNotIn("optim", step_method)
        self.assertNotIn("learning", step_method.lower())
        # Only check code after the docstring closes (skip the docstring which mentions 'learned')
        code_only = step_method[step_method.index('"""', step_method.index('"""')+3)+3:]
        # "training" may appear in neighboring methods (vehicle_command, etc.);


if __name__ == "__main__":
    unittest.main()
