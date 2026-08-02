from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "PPO3_SOLE_PhysX/scripts/migrate_shared_policy.py"
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
