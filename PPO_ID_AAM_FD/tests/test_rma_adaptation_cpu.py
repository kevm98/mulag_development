"""CPU-only contract tests for the prepared RMA/A-RMA path.

These tests intentionally do not import IsaacLab, create a simulator, collect
rollouts, or run PPO.  They cover the offline pieces that must be true before
the future GPU runbook is allowed to execute.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from PPO_ID_AAM_FD.rma_adaptation import (
    AAMEstimator,
    AAMTrainer,
    ActuationConfig,
    ContextMode,
    DeploymentActuation,
    DeploymentContext,
    NativeRigidBodyWiring,
    OnPolicyHistoryRecorder,
    assemble_policy_observation,
    assert_no_context_leakage,
    causal_mismatch_features,
    delta_adapt,
    recovery_fraction,
    stress_thresholds,
    verify_native_mass_com_wiring,
)

ROOT = Path(__file__).resolve().parents[2]


def test_zero_change_identity_context_is_exact():
    wrapper = DeploymentActuation(ActuationConfig(max_delay_steps=4))
    raw = torch.tensor([[0.2, -0.3, 0.4, -0.5]])
    out = wrapper.step(raw, DeploymentContext())
    assert torch.equal(out, raw)


def test_positive_control_authority_point_one():
    wrapper = DeploymentActuation(ActuationConfig(max_delay_steps=4))
    raw = torch.ones(1, 4)
    out = wrapper.step(raw, DeploymentContext(actuator_authority_alpha=0.01))
    assert torch.allclose(out, torch.full_like(raw, 0.01))


def test_zero_authority_is_legal_and_safe():
    wrapper = DeploymentActuation(ActuationConfig(max_delay_steps=4))
    out = wrapper.step(torch.ones(1, 4), DeploymentContext(actuator_authority_alpha=0.0))
    assert torch.equal(out, torch.zeros(1, 4))


def test_pure_delay_fifo():
    wrapper = DeploymentActuation(ActuationConfig(max_delay_steps=3))
    context = DeploymentContext(actuator_delay_steps=2)
    values = [wrapper.step(torch.full((1, 4), float(i)), context)[0, 0].item() for i in range(4)]
    assert values == [0.0, 0.0, 0.0, 1.0]


def test_stable_lag_has_bounded_step_response():
    wrapper = DeploymentActuation(ActuationConfig(dt_s=0.01, max_delay_steps=2))
    out = wrapper.step(torch.ones(1, 4), DeploymentContext(hydraulic_response_time_tau=0.1))
    expected = 1.0 - torch.exp(torch.tensor(-0.1))
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-6)
    assert bool((out >= 0).all() and (out <= 1).all())


def test_authority_lag_delay_composition():
    wrapper = DeploymentActuation(ActuationConfig(dt_s=0.01, max_delay_steps=2))
    context = DeploymentContext(actuator_authority_alpha=0.5, hydraulic_response_time_tau=0.0, actuator_delay_steps=1)
    assert torch.equal(wrapper.step(torch.ones(1, 4), context), torch.zeros(1, 4))
    assert torch.allclose(wrapper.step(torch.zeros(1, 4), context), torch.full((1, 4), 0.5))


def test_context_and_normalization_are_four_wide():
    context = DeploymentContext(actuator_delay_steps=2).as_tensor()
    assert context.shape == (4,)
    from PPO_ID_AAM_FD.rma_adaptation.core import normalized_context
    assert normalized_context(context).shape == (4,)


def test_runtime_modes_have_explicit_routing():
    base = torch.zeros(2, 6)
    c = torch.ones(2, 4)
    assert assemble_policy_observation(base, ContextMode.BASELINE).shape == (2, 6)
    for mode, key in ((ContextMode.PRIVILEGED, "privileged"), (ContextMode.MEAN_CONTEXT, "mean_context"),
                      (ContextMode.SHUFFLED_CONTEXT, "shuffled_context"), (ContextMode.AAM_ESTIMATED, "estimated"),
                      (ContextMode.AAM_FROZEN, "frozen"), (ContextMode.AAM_ESTIMATED_FINETUNED, "estimated_finetuned")):
        assert assemble_policy_observation(base, mode, **{key: c}).shape == (2, 10)


def test_leakage_guard_rejects_context_labels():
    with pytest.raises(ValueError, match="leaks"):
        assert_no_context_leakage(("obs_t", "mass_scale"))
    assert_no_context_leakage(("obs_t", "action_t", "obs_tp1_minus_obs_t"))


def test_on_policy_recorder_has_causal_schema_only():
    recorder = OnPolicyHistoryRecorder(obs_dim=6, action_dim=4)
    recorder.append(torch.zeros(6), torch.ones(4), torch.ones(6), 1.0, False)
    assert set(recorder.tensors()) == {"obs_t", "action_t", "obs_tp1", "reward_t", "done_t"}
    assert recorder.estimator_features().shape == (1, 16)


def test_causal_features_do_not_need_privileged_context():
    features = causal_mismatch_features(torch.zeros(3, 5), torch.zeros(3, 4), torch.ones(3, 5))
    assert features.shape == (3, 14)


def test_aam_skeleton_cpu_synthetic_step():
    estimator = AAMEstimator(obs_dim=6, action_dim=4, hidden_dim=8)
    trainer = AAMTrainer(estimator)
    features = torch.randn(5, 16)
    target = torch.randn(5, 4)
    loss = trainer.train_step(features, target)
    assert loss >= 0.0 and all(parameter.device.type == "cpu" for parameter in estimator.parameters())


def test_future_metrics_are_declared():
    assert torch.equal(delta_adapt(torch.tensor([2.0]), torch.tensor([1.0])), torch.tensor([1.0]))
    assert recovery_fraction(torch.tensor([0.1, 0.8, 2.0]), 2.0, 1.0).item() == pytest.approx(2 / 3)
    assert stress_thresholds()["delay_onset_steps"] == 2.0


def test_native_mass_com_wiring_is_explicit():
    verify_native_mass_com_wiring(NativeRigidBodyWiring())
    with pytest.raises(ValueError, match="native PhysX"):
        verify_native_mass_com_wiring(NativeRigidBodyWiring(mass_scale_source="observation_only"))


def _load_migration_module():
    path = ROOT / "PPO_ID_AAM_FD/scripts/migrate_privileged_context_checkpoint.py"
    spec = importlib.util.spec_from_file_location("rma_migration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_surgery_expands_actor_and_critic_without_overwrite():
    module = _load_migration_module()
    source_path = ROOT / "logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    expanded = module.expand_checkpoint_payload(source)
    assert tuple(expanded["actor_state_dict"]["mlp.0.weight"].shape) == (256, 70)
    assert tuple(expanded["critic_state_dict"]["mlp.0.weight"].shape) == (256, 70)
    assert source["actor_state_dict"]["mlp.0.weight"].shape[1] == 66
    assert expanded["optimizer_state_dict"] == {}


def test_converted_actor_matches_original_for_random_context():
    module = _load_migration_module()
    source_path = ROOT / "logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    expanded = module.expand_checkpoint_payload(source)
    old = source["actor_state_dict"]
    new = expanded["actor_state_dict"]
    base = torch.randn(7, 66)
    context = torch.randn(7, 4)
    def forward(state, obs):
        mean = state["obs_normalizer._mean"]
        std = state["obs_normalizer._std"]
        x = (obs - mean) / std
        for index in (0, 2, 4):
            x = torch.nn.functional.elu(torch.nn.functional.linear(x, state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"]))
        return torch.nn.functional.linear(x, state["mlp.6.weight"], state["mlp.6.bias"])
    assert torch.allclose(forward(old, base), forward(new, torch.cat((base, context), dim=-1)), atol=1e-6)
