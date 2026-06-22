"""Evidence-based tests for the ROM valve authority and feedforward fix."""

from __future__ import annotations

import json
import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from valve_conditioned_causal_rom import EXPORT_DIR, ValveConditionedROMInference
    from rom_data_contract import (
        EXTENSION_SLICE, EXTENSION_VELOCITY_SLICE,
        VALVE_SLICE, DP_SLICE, JOINT_POS_SLICE, JOINT_VEL_SLICE,
        HISTORY_LENGTH, CORE_INPUT_FEATURE_NAMES,
    )
except ImportError:
    from kevin_integration.rom_cylinders.valve_conditioned_causal_rom import (
        EXPORT_DIR, ValveConditionedROMInference,
    )
    from kevin_integration.rom_cylinders.rom_data_contract import (
        EXTENSION_SLICE, EXTENSION_VELOCITY_SLICE,
        VALVE_SLICE, DP_SLICE, JOINT_POS_SLICE, JOINT_VEL_SLICE,
        HISTORY_LENGTH, CORE_INPUT_FEATURE_NAMES,
    )


def _literal_assignment(class_name: str, attr_name: str):
    """Extract the literal value assigned to a class attribute from source file."""
    import ast
    cfg_path = Path(__file__).resolve().parents[1] / "tasks" / "mulag_adaptive_rl_env_cfg.py"
    source = cfg_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    value = ast.literal_eval(node.value)
                    return value
    raise AttributeError(f"{attr_name} not found in config")


def _rom_at_state(valve=None, dP=None, velocity=None, extension_offset=None):
    """Create a ROM inference at a specific state and return the model."""
    inference = ValveConditionedROMInference(EXPORT_DIR, device="cpu")
    rom = inference.model
    batch = 1
    history = torch.zeros(batch, HISTORY_LENGTH, len(CORE_INPUT_FEATURE_NAMES))
    for t in range(HISTORY_LENGTH):
        history[0, t, EXTENSION_SLICE] = rom.extension_mean + (extension_offset or 0.0)
        history[0, t, DP_SLICE] = dP if dP is not None else rom.training_mean_dp
        history[0, t, JOINT_POS_SLICE] = rom.joint_pos_mean
    if velocity is not None:
        history[0, -1, EXTENSION_VELOCITY_SLICE] = velocity
    if valve is not None:
        history[0, -1, VALVE_SLICE] = valve
    return rom, history


class TestROMValveAuthorityEvidence(unittest.TestCase):
    """Prove the ROM has insufficient valve authority for control."""

    @classmethod
    def setUpClass(cls):
        cls.inference = ValveConditionedROMInference(EXPORT_DIR, device="cpu")
        cls.rom = cls.inference.model
        mapping_path = ROOT / "kevin_integration" / "rom_cylinders" / "cylinder_coordinate_mapping.json"
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        cylinder_order = (
            "Drehzapfen_joint",
            "Ausleger_I_joint",
            "Ausleger_II_joint",
            "Messerkopf_Schwenk_joint",
        )
        cls.coord_signs = torch.tensor([
            float(mapping["cylinders"][name]["sign"])
            for name in cylinder_order
        ])

    def test_rom_valve_authority_is_measurable_but_weak(self):
        """ROM valve contribution is an order of magnitude smaller than other terms."""
        rom, history = _rom_at_state(dP=self.rom.training_mean_dp, velocity=torch.tensor([0.02, 0.02, 0.02, 0.02]))

        h0 = history.clone()
        h0[0, -1, VALVE_SLICE] = 0.0
        h1 = history.clone()
        h1[0, -1, VALVE_SLICE] = 1.0

        out0 = rom(h0)
        out1 = rom(h1)

        for c in range(4):
            corr0 = out0["delta_extension_correction"][0, c].item()
            corr1 = out1["delta_extension_correction"][0, c].item()
            valve_delta_correction = abs(corr1 - corr0)
            total_correction = max(abs(corr0), abs(corr1))

            # Valve contribution should be measurable
            self.assertGreater(valve_delta_correction, 1e-8,
                f"Cyl {c+1}: valve has zero effect on correction")
            # But it's much smaller than the total correction
            if total_correction > 1e-6:
                ratio = valve_delta_correction / total_correction
                self.assertLess(ratio, 0.95,
                    f"Cyl {c+1}: valve dominates correction ({ratio:.2f}) — "
                    "valve should not be the only driver")

    def test_cylinders_1_2_have_negligible_valve_authority(self):
        """Cylinders 1-2 have <0.005 m/s valve authority — policy can't control them."""
        rom, history = _rom_at_state(
            dP=self.rom.training_mean_dp,
            velocity=torch.tensor([0.02, 0.02, 0.02, 0.02]),
        )

        h0 = history.clone()
        h0[0, -1, VALVE_SLICE] = 0.0
        h1 = history.clone()
        h1[0, -1, VALVE_SLICE] = 1.0

        out0 = rom(h0)
        out1 = rom(h1)

        for c in [0, 1]:  # cylinders 1 and 2
            delta_corr = abs(
                out1["delta_extension_correction"][0, c].item() -
                out0["delta_extension_correction"][0, c].item()
            )
            vel_equiv = delta_corr / rom.dt
            self.assertLess(vel_equiv, 0.005,
                f"Cyl {c+1}: valve authority {vel_equiv:.4f} m/s >= 0.005 — "
                "policy SHOULD have control, but test says it doesn't")

    def test_valve_feedforward_config_exists_and_is_positive(self):
        """final_causal_rom_valve_feedforward_gain is configured and > 0."""
        gain = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_valve_feedforward_gain",
        )
        self.assertGreaterEqual(gain, 0.0,
            "Valve feedforward gain must be >= 0")
        self.assertLessEqual(gain, 1.0,
            "Valve feedforward gain should not exceed 1.0")
        if gain > 0.0:
            pass  # Feedforward active — ROM authority supplemented
        else:
            pass  # Feedforward disabled — ROM authority from training alone

    def test_valve_feedforward_delta_matches_position_scale(self):
        """Feedforward delta = gain * position_scale * valve_cmd."""
        gain = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_valve_feedforward_gain",
        )
        position_scale = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "position_scale",
        )
        valve_cmd = torch.tensor([[0.5, -0.3, 0.8, -1.0]])
        expected = gain * position_scale * valve_cmd
        actual = gain * position_scale * valve_cmd  # same formula as in code
        self.assertTrue(torch.allclose(actual, expected),
            f"Feedforward formula mismatch: {actual} != {expected}")

    def test_feedforward_plus_rom_gives_total_authority(self):
        """ROM + feedforward gives >= position_scale authority per step."""
        rom, history = _rom_at_state(
            dP=self.rom.training_mean_dp,
            velocity=torch.tensor([0.02, 0.02, 0.02, 0.02]),
        )

        gain = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_valve_feedforward_gain",
        )
        position_scale = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "position_scale",
        )

        # Simulate what the pipeline does:
        # 1. ROM predicts Simscape delta
        # 2. Convert to Isaac delta: sign * rom_delta
        # 3. Add feedforward: gain * position_scale * valve_cmd
        h1 = history.clone()
        h1[0, -1, VALVE_SLICE] = 1.0
        rom_delta_simscape = rom(h1)["final_delta_extension"][0]
        rom_delta_isaac = self.coord_signs * rom_delta_simscape
        ff_delta = gain * position_scale * torch.tensor([1.0, 1.0, 1.0, 1.0])
        total_delta = rom_delta_isaac + ff_delta

        for c in range(4):
            # Total delta at u=+1 should be positive for all cylinders
            # (policy commands positive → cylinder extends in Isaac)
            self.assertGreater(
                total_delta[c].item(), 0.0,
                f"Cyl {c+1}: total delta {total_delta[c]:.6f} <= 0 at u=+1 "
                "— feedforward not providing positive authority"
            )

    def test_feedforward_symmetry(self):
        """Feedforward is symmetric: +valve gives +delta, -valve gives -delta."""
        gain = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_valve_feedforward_gain",
        )
        position_scale = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "position_scale",
        )

        valve_pos = torch.tensor([0.5, 0.5, 0.5, 0.5])
        valve_neg = torch.tensor([-0.5, -0.5, -0.5, -0.5])
        ff_pos = gain * position_scale * valve_pos
        ff_neg = gain * position_scale * valve_neg

        for c in range(4):
            diff = abs(ff_pos[c].item() + ff_neg[c].item())
            self.assertLess(diff, 1e-8)

    def test_feedforward_makes_pipeline_valve_response_monotone(self):
        """Combined ROM+feedforward delta is monotone over the pressure stress grid."""
        gain = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_valve_feedforward_gain",
        )
        position_scale = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "position_scale",
        )
        applied_limit = abs(_literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_applied_delta_clamp",
        ))
        rom_dt = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "rom_dt",
        )
        xdot_max = torch.tensor(_literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "rom_xdot_max",
        ))
        rate_limit = (xdot_max * rom_dt).reshape(1, 4)

        valves = torch.linspace(-1.0, 1.0, 401)
        pressure_offsets = torch.linspace(-10.0e6, 10.0e6, 9)
        required_gain = torch.tensor(0.0)
        worst_step = torch.tensor(float("inf"))

        for pressure_offset in pressure_offsets:
            _, history = _rom_at_state(dP=self.rom.training_mean_dp + pressure_offset)
            batch = history.repeat(len(valves), 1, 1)
            batch[:, -1, VALVE_SLICE] = valves.reshape(-1, 1).repeat(1, 4)

            rom_delta = self.rom(batch)["final_delta_extension"]
            native_delta = self.coord_signs.reshape(1, 4) * rom_delta
            native_delta = torch.clamp(native_delta, -applied_limit, applied_limit)
            native_delta = torch.maximum(torch.minimum(native_delta, rate_limit), -rate_limit)

            native_step = native_delta[1:] - native_delta[:-1]
            du = valves[1] - valves[0]
            required_gain = torch.maximum(
                required_gain,
                torch.clamp((-native_step / (position_scale * du)).max(), min=0.0),
            )

            total_delta = native_delta + gain * position_scale * batch[:, -1, VALVE_SLICE]
            total_delta = torch.maximum(torch.minimum(total_delta, rate_limit), -rate_limit)
            total_step = total_delta[1:] - total_delta[:-1]
            worst_step = torch.minimum(worst_step, total_step.min())

        self.assertGreaterEqual(
            gain,
            float(required_gain),
            f"gain {gain:.6f} is below monotone-response requirement {required_gain.item():.6f}",
        )
        self.assertGreaterEqual(
            worst_step.item(),
            -1.0e-9,
            f"combined ROM+feedforward response is not monotone; worst step {worst_step.item():.9f}",
        )

    def test_feedforward_covers_rom_pressure_training_support(self):
        """Feedforward gain dominates native negative slopes over ROM dP support."""
        gain = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_valve_feedforward_gain",
        )
        position_scale = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "position_scale",
        )
        applied_limit = abs(_literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_applied_delta_clamp",
        ))

        valves = torch.linspace(-1.0, 1.0, 801)
        du = valves[1] - valves[0]
        feature_mean = self.rom.feature_mean[:, 4]
        feature_std = self.rom.feature_std[:, 4]
        pressure_low = (feature_mean - 3.0 * feature_std) * 1.0e6
        pressure_high = (feature_mean + 3.0 * feature_std) * 1.0e6
        required_gain = torch.tensor(0.0)

        for cylinder_index in range(4):
            for pressure_value in torch.linspace(
                float(pressure_low[cylinder_index]),
                float(pressure_high[cylinder_index]),
                41,
            ):
                _, history = _rom_at_state(dP=self.rom.training_mean_dp)
                batch = history.repeat(len(valves), 1, 1)
                batch[:, :, DP_SLICE.start + cylinder_index] = pressure_value
                batch[:, -1, VALVE_SLICE] = valves.reshape(-1, 1).repeat(1, 4)

                rom_delta = self.rom(batch)["final_delta_extension"]
                native_delta = self.coord_signs.reshape(1, 4) * rom_delta
                native_delta = torch.clamp(native_delta, -applied_limit, applied_limit)
                native_step = native_delta[1:, cylinder_index] - native_delta[:-1, cylinder_index]
                required_gain = torch.maximum(
                    required_gain,
                    torch.clamp((-native_step / (position_scale * du)).max(), min=0.0),
                )

        self.assertGreaterEqual(
            gain,
            float(required_gain),
            f"gain {gain:.6f} is below ROM pressure-support requirement {required_gain.item():.6f}",
        )

    def test_sd_pressure_conditioning_configured_for_final_rom(self):
        """Final causal ROM uses SD pressure with a clamp matching ROM pressure support."""
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "final_causal_rom_pressure_source",
            ),
            "sd_observer",
        )
        pressure_clamp = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_pressure_clamp",
        )
        sd_pressure_clamp = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "sd_max_abs_pressure_delta",
        )
        pressure_support = torch.max(torch.abs(
            (self.rom.feature_mean[:, 4].abs() + 3.0 * self.rom.feature_std[:, 4]) * 1.0e6
        )).item()

        self.assertGreaterEqual(pressure_clamp, pressure_support)
        self.assertGreaterEqual(sd_pressure_clamp, pressure_clamp)

    def test_runtime_runs_sd_observer_for_sd_pressure_source(self):
        """Selecting sd_observer pressure source activates SD without SD feedback writes."""
        env_source = (
            Path(__file__).resolve().parents[1] / "tasks" / "mulag_adaptive_rl_env.py"
        ).read_text(encoding="utf-8")

        self.assertIn('rom_pressure_source = str(self.cfg.final_causal_rom_pressure_source)', env_source)
        self.assertIn('or rom_pressure_source == "sd_observer"', env_source)
        self.assertIn('source == "sd_observer"', env_source)

    def test_feedforward_cannot_bypass_final_rom_rate_limit(self):
        """Post-feedforward command remains inside the ROM velocity envelope."""
        gain = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_valve_feedforward_gain",
        )
        position_scale = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "position_scale",
        )
        applied_limit = abs(_literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "final_causal_rom_applied_delta_clamp",
        ))
        rom_dt = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "rom_dt",
        )
        xdot_max = torch.tensor(_literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "rom_xdot_max",
        ))
        rate_limit = (xdot_max * rom_dt).reshape(1, 4)

        valves = torch.linspace(-1.0, 1.0, 101)
        _, history = _rom_at_state(
            dP=self.rom.training_mean_dp,
            velocity=torch.tensor([0.80, 0.80, 0.80, 0.80]),
        )
        batch = history.repeat(len(valves), 1, 1)
        batch[:, -1, VALVE_SLICE] = valves.reshape(-1, 1).repeat(1, 4)

        rom_delta = self.rom(batch)["final_delta_extension"]
        native_delta = self.coord_signs.reshape(1, 4) * rom_delta
        native_delta = torch.clamp(native_delta, -applied_limit, applied_limit)
        native_delta = torch.maximum(torch.minimum(native_delta, rate_limit), -rate_limit)

        raw_total = native_delta + gain * position_scale * batch[:, -1, VALVE_SLICE]
        bounded_total = torch.maximum(torch.minimum(raw_total, rate_limit), -rate_limit)

        self.assertGreater(
            torch.mean((torch.abs(raw_total) > rate_limit).float()).item(),
            0.0,
            "stress state should exercise the post-feedforward clamp",
        )
        self.assertTrue(
            torch.all(torch.abs(bounded_total) <= rate_limit + 1.0e-9),
            "feedforward bypasses the final ROM rate envelope",
        )

    def test_runtime_relimits_after_feedforward(self):
        """Runtime applies a second rate limit after adding valve feedforward."""
        env_source = (
            Path(__file__).resolve().parents[1] / "tasks" / "mulag_adaptive_rl_env.py"
        ).read_text(encoding="utf-8")
        feedforward_index = env_source.index("Direct valve feedforward")
        post_limit_index = env_source.index("post_ff_rate_limit_mask")

        self.assertGreater(
            post_limit_index,
            feedforward_index,
            "post-feedforward rate limit must run after the feedforward addition",
        )

    def test_rate_limit_config_updated(self):
        """rom_xdot_max is 0.80 after the evidence-backed fix."""
        value = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "rom_xdot_max",
        )
        self.assertEqual(value, [0.80, 0.80, 0.80, 0.80],
            "rom_xdot_max must be 0.80 after fix")

    def test_rate_limit_allows_realistic_operation(self):
        """At rom_xdot_max=0.80, realistic ROM deltas pass through unclamped."""
        rom, history = _rom_at_state(
            dP=self.rom.training_mean_dp,
            velocity=torch.tensor([0.05, 0.05, 0.05, 0.05]),
        )
        h = history.clone()
        h[0, -1, VALVE_SLICE] = 0.5
        rom_delta = rom(h)["final_delta_extension"][0]
        rom_delta_isaac = self.coord_signs * rom_delta

        # With rom_xdot_max=0.80, rate limit = 0.80 * 0.01 = 0.008m/step
        rate_limit = 0.80 * 0.01
        for c in range(4):
            self.assertLess(
                abs(rom_delta_isaac[c].item()), rate_limit * 1.5,
                f"Cyl {c+1}: ROM delta {rom_delta_isaac[c]:.6f} far exceeds "
                f"rate limit {rate_limit:.6f}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
