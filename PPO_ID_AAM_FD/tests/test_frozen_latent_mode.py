"""`latent_mode='frozen'` is the control for the ONLINE adaptation claim.

It must supply a latent of the same content and magnitude as `estimated`,
differing in exactly one respect: it stops updating after the freeze tick.
If these properties do not hold, any "recovery came from online estimation"
conclusion drawn against this arm is unsound.

The tests exercise `_apply_latent_mode` directly against a lightweight stub so
they need no AAM checkpoint, no adapter and no simulator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PPO_ID_AAM_FD.online_runtime import OnlineAAMIDFDController  # noqa: E402


class _Stub:
    """Minimal object carrying only what _apply_latent_mode touches."""

    def __init__(self, num_envs: int, freeze_tick: int, mode: str = "frozen"):
        self.latent_mode = mode
        self.latent_freeze_tick = freeze_tick
        self.control_ticks = torch.zeros(num_envs, dtype=torch.long)
        self.z_frozen = torch.zeros((num_envs, 2))
        self.z_frozen_valid = torch.zeros(num_envs, dtype=torch.bool)
        self.true_z = torch.zeros((num_envs, 2))

    apply = OnlineAAMIDFDController._apply_latent_mode


def test_frozen_matches_estimated_before_the_latch():
    s = _Stub(num_envs=4, freeze_tick=10)
    s.control_ticks[:] = 3
    z = torch.randn(4, 2)
    assert torch.equal(s.apply(z), z), "before the freeze tick frozen must be a no-op"
    assert not bool(s.z_frozen_valid.any())


def test_frozen_latches_at_the_tick_and_holds():
    s = _Stub(num_envs=3, freeze_tick=5)
    s.control_ticks[:] = 5
    latched = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = s.apply(latched)
    assert torch.equal(out, latched)
    assert bool(s.z_frozen_valid.all())

    # a later, different estimate must be ignored
    s.control_ticks[:] = 40
    moved = latched + 100.0
    assert torch.equal(s.apply(moved), latched), "frozen must ignore later updates"


def test_frozen_latches_per_environment():
    """Environments reach the freeze tick independently."""
    s = _Stub(num_envs=2, freeze_tick=7)
    s.control_ticks = torch.tensor([7, 2])
    z0 = torch.tensor([[1.0, 1.0], [9.0, 9.0]])
    out = s.apply(z0)
    assert bool(s.z_frozen_valid[0]) and not bool(s.z_frozen_valid[1])
    assert torch.equal(out[0], z0[0])
    assert torch.equal(out[1], z0[1])       # env 1 still live

    s.control_ticks = torch.tensor([20, 7])
    z1 = torch.tensor([[2.0, 2.0], [8.0, 8.0]])
    out = s.apply(z1)
    assert torch.equal(out[0], z0[0]), "env 0 stays latched at its own value"
    assert torch.equal(out[1], z1[1]), "env 1 latches at its own tick"


def test_frozen_is_not_degenerate():
    """The latched value must be the real estimate, not zero or the mean.

    A frozen arm that silently published zeros would be a `zero` arm wearing a
    different name, and would make the online claim untestable.
    """
    s = _Stub(num_envs=4, freeze_tick=1)
    s.control_ticks[:] = 1
    z = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]])
    out = s.apply(z)
    assert torch.equal(out, z)
    assert out.abs().sum() > 0
    assert not torch.allclose(out, z.mean(dim=0, keepdim=True).expand_as(z))


def test_frozen_requires_a_freeze_tick():
    with pytest.raises(ValueError, match="latent_freeze_tick"):
        OnlineAAMIDFDController.__init__(
            object.__new__(OnlineAAMIDFDController),
            aam_checkpoint="unused", adapter_checkpoint="unused",
            num_envs=1, device="cpu", latent_mode="frozen",
            latent_freeze_tick=0,
        )


def test_other_modes_are_unchanged():
    """frozen must not perturb the existing arms."""
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    s = _Stub(num_envs=2, freeze_tick=5, mode="zero")
    assert torch.equal(s.apply(z), torch.zeros_like(z))
    s.latent_mode = "estimated"
    assert torch.equal(s.apply(z), z)
    s.latent_mode = "shuffled"
    assert torch.equal(s.apply(z), torch.roll(z, shifts=1, dims=0))
    s.latent_mode = "mean"
    assert torch.equal(s.apply(z), z.mean(dim=0, keepdim=True).expand_as(z))
