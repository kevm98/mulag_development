from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "verge_task/scripts/migrate_shared_policy.py"
SPEC = importlib.util.spec_from_file_location("migrate_shared_policy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _network_state(*, actor: bool) -> dict[str, torch.Tensor]:
    state = {
        "obs_normalizer._mean": torch.randn(1, 54),
        "obs_normalizer._var": torch.rand(1, 54) + 0.1,
        "obs_normalizer._std": torch.rand(1, 54) + 0.1,
        "obs_normalizer.count": torch.tensor(20.0),
        "mlp.0.weight": torch.randn(256, 54),
        "mlp.0.bias": torch.randn(256),
        "mlp.2.weight": torch.randn(128, 256),
        "mlp.2.bias": torch.randn(128),
        "mlp.4.weight": torch.randn(64, 128),
        "mlp.4.bias": torch.randn(64),
        "mlp.6.weight": torch.randn(4 if actor else 1, 64),
        "mlp.6.bias": torch.randn(4 if actor else 1),
    }
    if actor:
        state["distribution.std_param"] = torch.full((4,), 0.12)
    return state


def test_migration_preserves_old_policy_and_adds_neutral_channels():
    payload = {
        "actor_state_dict": _network_state(actor=True),
        "critic_state_dict": _network_state(actor=False),
        "optimizer_state_dict": {"state": {1: "legacy"}},
        "infos": {"observation_width": 54},
    }
    migrated = MODULE.migrate_checkpoint(payload, q5_std=0.04)
    old_actor = payload["actor_state_dict"]
    actor = migrated["actor_state_dict"]
    critic = migrated["critic_state_dict"]

    assert actor["mlp.0.weight"].shape == (256, 66)
    assert critic["mlp.0.weight"].shape == (256, 66)
    assert torch.equal(actor["mlp.0.weight"][:, :54], old_actor["mlp.0.weight"])
    assert torch.count_nonzero(actor["mlp.0.weight"][:, 54:]) == 0
    assert torch.equal(actor["mlp.6.weight"][:4], old_actor["mlp.6.weight"])
    assert torch.count_nonzero(actor["mlp.6.weight"][4]) == 0
    assert actor["mlp.6.bias"][4] == 0.0
    assert actor["distribution.std_param"].tolist()[-1] == pytest.approx(0.04)
    assert actor["obs_normalizer._mean"][:, 54:].eq(0.0).all()
    assert actor["obs_normalizer._var"][:, 54:].eq(1.0).all()
    assert actor["obs_normalizer._std"][:, 54:].eq(1.0).all()
    assert migrated["infos"]["schema"] == "terrain_shared_v2"
    assert migrated["infos"]["migration_requires_optimizer_reset"] is True
    assert payload["actor_state_dict"]["mlp.0.weight"].shape == (256, 54)


def test_migration_rejects_nonpositive_q5_std():
    payload = {
        "actor_state_dict": _network_state(actor=True),
        "critic_state_dict": _network_state(actor=False),
    }
    with pytest.raises(ValueError, match="q5_std must be positive"):
        MODULE.migrate_checkpoint(payload, q5_std=0.0)


def test_collapse_redundant_q5_preserves_combined_actor_output():
    actor = _network_state(actor=True)
    actor["mlp.6.weight"] = torch.randn(5, 64)
    actor["mlp.6.bias"] = torch.randn(5)
    actor["distribution.std_param"] = torch.tensor([0.11, 0.12, 0.13, 0.2, 0.3])
    payload = {
        "actor_state_dict": actor,
        "critic_state_dict": _network_state(actor=False),
        "optimizer_state_dict": {"state": {1: "stale"}},
        "infos": {"observation_width": 54, "action_width": 5},
    }
    hidden = torch.randn(17, 64)
    old_output = hidden @ actor["mlp.6.weight"].T + actor["mlp.6.bias"]

    migrated = MODULE.collapse_redundant_q5(
        payload, head_action_to_c4_gain=0.4
    )
    new_actor = migrated["actor_state_dict"]
    new_output = hidden @ new_actor["mlp.6.weight"].T + new_actor["mlp.6.bias"]

    assert torch.allclose(new_output[:, :3], old_output[:, :3])
    assert torch.allclose(
        new_output[:, 3], old_output[:, 3] + 0.4 * old_output[:, 4], atol=1.0e-5
    )
    assert new_actor["distribution.std_param"][3].item() == pytest.approx(
        (0.2**2 + (0.4 * 0.3) ** 2) ** 0.5
    )
    assert migrated["infos"]["action_width"] == 4
    assert migrated["infos"]["head_joint_action"] is False
    assert migrated["infos"]["migration_requires_optimizer_reset"] is True
    assert payload["actor_state_dict"]["mlp.6.weight"].shape[0] == 5


def test_collapse_redundant_q5_rejects_wrong_action_width():
    payload = {
        "actor_state_dict": _network_state(actor=True),
        "critic_state_dict": _network_state(actor=False),
    }
    with pytest.raises(ValueError, match="five-action"):
        MODULE.collapse_redundant_q5(payload)


def test_expand_direct_q5_preserves_four_outputs_and_neutralizes_new_contract():
    payload = {
        "actor_state_dict": _network_state(actor=True),
        "critic_state_dict": _network_state(actor=False),
        "optimizer_state_dict": {"state": {1: "stale"}},
        "infos": {"observation_width": 54, "action_width": 4},
    }
    migrated = MODULE.expand_direct_q5(
        payload,
        source_observations=54,
        target_observations=62,
        q5_std=0.04,
    )
    old = payload["actor_state_dict"]
    actor = migrated["actor_state_dict"]
    assert actor["mlp.0.weight"].shape == (256, 62)
    assert torch.equal(actor["mlp.0.weight"][:, :54], old["mlp.0.weight"])
    assert torch.count_nonzero(actor["mlp.0.weight"][:, 54:]) == 0
    assert torch.equal(actor["mlp.6.weight"][:4], old["mlp.6.weight"])
    assert torch.count_nonzero(actor["mlp.6.weight"][4]) == 0
    assert actor["mlp.6.bias"][4] == 0.0
    assert actor["distribution.std_param"][4].item() == pytest.approx(0.04)
    assert migrated["infos"]["head_control_mode"] == "direct_joint_position"
    assert migrated["infos"]["migration_requires_optimizer_reset"] is True
