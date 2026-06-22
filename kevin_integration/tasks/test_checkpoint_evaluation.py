"""Offline tests for deterministic checkpoint discovery and ranking."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from packaging import version

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kevin_integration.scripts.evaluate_mulag_checkpoints import (
    CheckpointEvaluationResult,
    apply_evaluation_plant_mode,
    choose_best_checkpoint,
    discover_checkpoints,
    normalized_plant_modes,
    normalized_policy_modes,
    policy_action,
    prepare_rsl_rl_agent_cfg,
    ranking_key,
    result_summary,
    select_checkpoints,
    write_results,
)


def _result(
    iteration: int,
    *,
    success: float,
    hold_fraction: float = 0.0,
    hold_steps: float = 0.0,
    pos: float = 1.0,
    orient: float = 1.0,
    lin: float = 1.0,
    ang: float = 1.0,
    clamp: float = 1.0,
    valve: float = 1.0,
    reward: float = 0.0,
) -> CheckpointEvaluationResult:
    return CheckpointEvaluationResult(
        checkpoint_path=f"/tmp/model_{iteration}.pt",
        iteration=iteration,
        policy_mode="deterministic",
        plant_mode="canonical_rom",
        episodes=10,
        evaluation_seeds=[1, 2],
        strict_success_rate=success,
        hold_completion_fraction=hold_fraction,
        mean_max_hold_steps=hold_steps,
        mean_hold_counter_steps=hold_steps,
        final_position_error_mean=pos,
        final_orientation_error_mean=orient,
        final_linear_speed_mean=lin,
        final_angular_speed_mean=ang,
        clamp_activation_fraction_mean=clamp,
        valve_saturation_fraction_mean=valve,
        mean_reward=reward,
    )


class TestCheckpointEvaluation(unittest.TestCase):
    def test_current_distribution_schema_skips_deprecated_migration(self) -> None:
        class ModelCfg:
            hidden_dims = [128, 128]
            activation = "elu"
            distribution_cfg = object()

        class AgentCfg:
            actor = ModelCfg()
            critic = ModelCfg()

        calls = []

        def migration_fn(agent_cfg, installed_version):
            calls.append((agent_cfg, installed_version))
            raise AssertionError("current-schema configs must not be migrated")

        agent_cfg = AgentCfg()
        prepared = prepare_rsl_rl_agent_cfg(agent_cfg, "5.0.0", version, migration_fn)
        self.assertIs(prepared, agent_cfg)
        self.assertEqual(calls, [])

    def test_legacy_stochastic_schema_uses_deprecated_migration(self) -> None:
        class ModelCfg:
            distribution_cfg = None
            stochastic = True
            init_noise_std = 0.2

        class AgentCfg:
            actor = ModelCfg()
            critic = ModelCfg()

        calls = []

        def migration_fn(agent_cfg, installed_version):
            calls.append((agent_cfg, installed_version))
            return "migrated"

        self.assertEqual(prepare_rsl_rl_agent_cfg(AgentCfg(), "5.0.0", version, migration_fn), "migrated")
        self.assertEqual(len(calls), 1)

    def test_numeric_checkpoint_sorting_and_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("model_100.pt", "model_2.pt", "model_50.pt", "notes.txt"):
                (root / name).write_text("", encoding="utf-8")
            checkpoints = discover_checkpoints(root)
            self.assertEqual([item.iteration for item in checkpoints], [2, 50, 100])
            self.assertEqual([item.iteration for item in select_checkpoints(checkpoints, checkpoint_stride=2)], [2, 100])
            self.assertEqual(
                [item.iteration for item in select_checkpoints(checkpoints, explicit_checkpoints=["model_50.pt"])],
                [50],
            )

    def test_missing_checkpoint_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                discover_checkpoints(tmp)

    def test_success_rate_outranks_mean_reward(self) -> None:
        poor_success_high_reward = _result(10, success=0.1, reward=1000.0)
        better_success_low_reward = _result(20, success=0.2, reward=-10.0)
        best = choose_best_checkpoint([poor_success_high_reward, better_success_low_reward])
        self.assertEqual(best.iteration, 20)
        self.assertGreater(ranking_key(best), ranking_key(poor_success_high_reward))

    def test_lexicographic_tie_breaks_by_hold_then_errors(self) -> None:
        low_hold = _result(10, success=0.5, hold_fraction=0.1, hold_steps=5.0, pos=0.01)
        high_hold = _result(20, success=0.5, hold_fraction=0.2, hold_steps=10.0, pos=0.50)
        lower_position_error = _result(30, success=0.5, hold_fraction=0.2, hold_steps=10.0, pos=0.20)
        best = choose_best_checkpoint([low_hold, high_hold, lower_position_error])
        self.assertEqual(best.iteration, 30)

    def test_ranking_uses_max_hold_not_terminal_hold_counter(self) -> None:
        longer_max_hold_terminally_reset = _result(10, success=0.0, hold_fraction=0.70, hold_steps=35.0)
        longer_max_hold_terminally_reset.mean_hold_counter_steps = 0.0
        shorter_terminal_counter = _result(20, success=0.0, hold_fraction=0.40, hold_steps=20.0)
        shorter_terminal_counter.mean_hold_counter_steps = 20.0
        best = choose_best_checkpoint([longer_max_hold_terminally_reset, shorter_terminal_counter])
        self.assertEqual(best.iteration, 10)

    def test_result_summary_includes_modes_and_hold_metrics(self) -> None:
        result = _result(50, success=0.0, hold_fraction=0.7, hold_steps=35.0, lin=0.0631)
        result.mean_hold_counter_steps = 0.0
        text = result_summary(result)
        self.assertIn("iter=50", text)
        self.assertIn("policy=deterministic", text)
        self.assertIn("plant=canonical_rom", text)
        self.assertIn("hold=35.00", text)
        self.assertIn("hold_counter=0.00", text)

    def test_policy_action_selects_deterministic_or_stochastic_output(self) -> None:
        calls = []

        class Policy:
            def __call__(self, obs, **kwargs):
                calls.append(kwargs)
                return kwargs.get("stochastic_output", False)

        self.assertFalse(policy_action(Policy(), object(), policy_mode="deterministic"))
        self.assertTrue(policy_action(Policy(), object(), policy_mode="stochastic"))
        self.assertEqual(calls, [{}, {"stochastic_output": True}])

    def test_policy_and_plant_modes_are_validated(self) -> None:
        self.assertEqual(normalized_policy_modes(["deterministic", "stochastic"]), ["deterministic", "stochastic"])
        self.assertEqual(normalized_plant_modes(["canonical_rom", "id_direct"]), ["canonical_rom", "id_direct"])
        with self.assertRaises(ValueError):
            normalized_policy_modes(["sampled"])
        with self.assertRaises(ValueError):
            normalized_plant_modes(["fd_effort"])

    def test_id_direct_plant_ablation_disables_rom_but_keeps_id_valve_contract(self) -> None:
        class Cfg:
            low_level_control_mode = "position_delta"
            position_delta_source = "direct_action"
            use_reduced_hydraulic_rom = True
            use_final_causal_cylinder_rom = True
            rom_use_id_valve_cmd = False

        cfg = Cfg()
        apply_evaluation_plant_mode(cfg, "id_direct")
        self.assertEqual(cfg.low_level_control_mode, "position_delta")
        self.assertEqual(cfg.position_delta_source, "id_valve")
        self.assertFalse(cfg.use_reduced_hydraulic_rom)
        self.assertFalse(cfg.use_final_causal_cylinder_rom)
        self.assertTrue(cfg.rom_use_id_valve_cmd)

    def test_canonical_rom_plant_ablation_uses_final_rom_and_id_valve(self) -> None:
        class Cfg:
            low_level_control_mode = "effort"
            position_delta_source = "direct_action"
            use_reduced_hydraulic_rom = True
            use_final_causal_cylinder_rom = False
            rom_use_id_valve_cmd = False

        cfg = Cfg()
        apply_evaluation_plant_mode(cfg, "canonical_rom")
        self.assertEqual(cfg.low_level_control_mode, "position_delta")
        self.assertEqual(cfg.position_delta_source, "id_valve")
        self.assertFalse(cfg.use_reduced_hydraulic_rom)
        self.assertTrue(cfg.use_final_causal_cylinder_rom)
        self.assertTrue(cfg.rom_use_id_valve_cmd)

    def test_json_csv_and_best_metadata_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_results(tmp, [_result(1, success=0.0), _result(2, success=1.0)])
            for path in paths.values():
                self.assertTrue(path.is_file())
            best = json.loads(paths["best"].read_text(encoding="utf-8"))
            self.assertEqual(best["checkpoint_path"], "/tmp/model_2.pt")
            self.assertEqual(best["iteration"], 2)
            self.assertEqual(best["policy_mode"], "deterministic")
            self.assertEqual(best["plant_mode"], "canonical_rom")
            self.assertEqual(best["evaluation_seeds"], [1, 2])
            self.assertIn("ranking_explanation", best)
            self.assertIn("strict_success_rate", best["ranking_metrics"])
            self.assertIn("mean_hold_counter_steps", best["ranking_metrics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
