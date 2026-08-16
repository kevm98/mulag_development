"""ROM-side plant perturbation.

The command ROM does not consume a valve command -- it produces one -- so the
FD valve perturbation does not apply. Its per-env plant channel is
`system_factors` (num_envs, 7).

The channel order is defined in AAM_module/aam_data/master_schema.py. If these
two ever disagree, a "seal" step would silently perturb compliance instead and
the tracking experiment would be measuring the wrong factor while still
producing a plausible-looking figure. The order is therefore pinned against the
schema itself rather than duplicated by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "AAM_module"))

from PPO_ID_AAM_FD.plant_perturbation import (  # noqa: E402
    RomFactorPerturbation, RomFactorPerturbationCfg,
    SYSTEM_FACTOR_INDEX, SYSTEM_FACTOR_ORDER,
)


def make(**kw):
    return RomFactorPerturbation(
        RomFactorPerturbationCfg(**kw), num_envs=4, device="cpu", control_dt=1 / 60
    )


def test_channel_order_matches_the_schema():
    """Pin against master_schema, the authority for this layout."""
    from aam_data.master_schema import e_to_system_factors

    probe = {"compliance": 1.0, "flow": 2.0, "pressure": 3.0, "seal": 4.0,
             "mass_scale": 9.0, "com_z": 9.0}
    expected = e_to_system_factors(probe)
    assert len(expected) == 7
    for name, idxs in SYSTEM_FACTOR_INDEX.items():
        for i in idxs:
            assert expected[i] == probe[name], (
                f"channel {i} should carry {name!r}; schema says {expected[i]}"
            )
    assert list(SYSTEM_FACTOR_ORDER) == [
        "compliance", "flow", "pressure", "seal", "seal", "seal", "seal"
    ]


def test_seal_occupies_four_channels():
    assert len(SYSTEM_FACTOR_INDEX["seal"]) == 4
    p = make(enabled=True, ranges={"seal": (0.5, 0.5)})
    f = p.advance()
    assert torch.allclose(f[:, 3:7], torch.full((4, 4), 0.5))
    assert torch.allclose(f[:, 0:3], torch.zeros(4, 3)), "other channels untouched"


def test_disabled_returns_none():
    p = make(enabled=False, ranges={"seal": (1.0, 1.0)})
    assert p.advance() is None, "disabled must not drive the ROM channel"


def test_shape_is_the_rom_contract():
    p = make(enabled=True, ranges={"seal": (0.0, 1.0)})
    assert p.advance().shape == (4, 7)


def test_draw_is_per_env_and_deterministic():
    a = make(enabled=True, ranges={"seal": (0.0, 1.0)}, seed=3)
    b = make(enabled=True, ranges={"seal": (0.0, 1.0)}, seed=3)
    assert torch.allclose(a.base, b.base), "same seed must pair arms env-for-env"
    assert a.base[:, 3].std() > 0, "seal must vary across envs"
    c = make(enabled=True, ranges={"seal": (0.0, 1.0)}, seed=4)
    assert not torch.allclose(a.base, c.base)


def test_step_change_fires_once_and_holds():
    p = make(enabled=True, ranges={"seal": (0.1, 0.1)},
             step_time_s=0.05, step_values={"seal": 0.9})   # 3 steps at 60 Hz
    assert torch.allclose(p.advance()[:, 3], torch.full((4,), 0.1))
    assert torch.allclose(p.advance()[:, 3], torch.full((4,), 0.1))
    assert torch.allclose(p.advance()[:, 3], torch.full((4,), 0.9))
    assert bool(p.stepped.all())
    assert torch.allclose(p.advance()[:, 3], torch.full((4,), 0.9)), "must hold"


def test_step_only_touches_named_factors():
    p = make(enabled=True, ranges={"seal": (0.1, 0.1), "flow": (0.2, 0.2)},
             step_time_s=0.0, step_values={"seal": 0.9})
    f = p.advance()
    assert torch.allclose(f[:, 3:7], torch.full((4, 4), 0.9)), "seal stepped"
    assert torch.allclose(f[:, 1], torch.full((4,), 0.2)), "flow must be unchanged"


def test_reset_restores_baseline_and_rearms():
    p = make(enabled=True, ranges={"seal": (0.1, 0.1)},
             step_time_s=0.0, step_values={"seal": 0.9})
    p.advance()
    assert bool(p.stepped.all())
    p.reset()
    assert not bool(p.stepped.any())
    assert torch.allclose(p.factors, p.base)


def test_true_theta_reports_named_factors_and_is_a_copy():
    p = make(enabled=True, ranges={"seal": (0.3, 0.3), "flow": (0.4, 0.4)})
    p.advance()
    th = p.true_theta()
    assert torch.allclose(th["rom_seal"], torch.full((4,), 0.3))
    assert torch.allclose(th["rom_flow"], torch.full((4,), 0.4))
    th["rom_seal"][:] = 99.0
    assert not torch.allclose(p.factors[:, 3], torch.full((4,), 99.0))


def test_unknown_factor_names_are_rejected():
    with pytest.raises(ValueError, match="unknown ROM factor"):
        make(enabled=True, ranges={"mass_scale": (0.8, 1.2)})
    with pytest.raises(ValueError, match="unknown ROM factor"):
        make(enabled=True, step_values={"com_z": 0.1})


def test_mechanical_factors_are_not_in_this_channel():
    """mass_scale/com_z are rigid-body properties, not ROM system factors.

    Accepting them here would silently do nothing and look like a null result.
    """
    assert "mass_scale" not in SYSTEM_FACTOR_INDEX
    assert "com_z" not in SYSTEM_FACTOR_INDEX
