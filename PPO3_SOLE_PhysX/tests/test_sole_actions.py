"""Bounds contract for the sole-controller action chain.

Every bound the deleted DLS controller used to guarantee structurally is now a
claim made by ``actions.py``, so each one is tested against an adversarial
policy rather than a well-behaved one: the tests drive with saturated,
sign-flipping, and NaN actions, because those are what an untrained PPO emits.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from PPO3_SOLE_PhysX.actions import (
    apply_sole_action,
    directional_tilt_shield,
    limit_shield,
)


def _zeros(n: int = 4) -> torch.Tensor:
    return torch.zeros((n, 4))


def test_action_is_clamped_to_unit_box():
    out = apply_sole_action(
        policy_action=torch.tensor([[50.0, -50.0, 3.0, -3.0]]),
        previous_applied_action=torch.zeros((1, 4)),
        filter_beta=0.0,
        max_action_rate=10.0,
    )
    assert (out.raw.abs() <= 1.0 + 1e-7).all()
    assert (out.applied.abs() <= 1.0 + 1e-7).all()


def test_nan_and_inf_are_scrubbed():
    poisoned = torch.tensor([[float("nan"), float("inf"), float("-inf"), 0.5]])
    out = apply_sole_action(
        policy_action=poisoned,
        previous_applied_action=torch.zeros((1, 4)),
        filter_beta=0.0,
        max_action_rate=1.0,
    )
    assert torch.isfinite(out.raw).all()
    assert torch.isfinite(out.applied).all()
    assert float(out.applied[0, 0]) == 0.0


def test_slew_limit_binds_from_saturated_step():
    """A step from rest to full command must take 1/rate steps, not one."""
    rate = 0.2
    applied = torch.zeros((1, 4))
    full = torch.ones((1, 4))
    reached = None
    for step in range(1, 200):
        out = apply_sole_action(
            policy_action=full,
            previous_applied_action=applied,
            filter_beta=0.0,
            max_action_rate=rate,
        )
        delta = float((out.applied - applied).abs().max())
        assert delta <= rate + 1e-6, f"slew {delta} exceeded {rate} at step {step}"
        applied = out.applied
        if float(applied.min()) >= 1.0 - 1e-6:
            reached = step
            break
    assert reached is not None, "command never reached saturation"
    assert reached >= int(1.0 / rate), f"reached saturation too fast in {reached} steps"


def test_sign_flip_cannot_jump_the_slew_limit():
    """Winding the filter up at +1 must not buy a free jump to -1."""
    rate = 0.15
    applied = torch.ones((1, 4))
    out = apply_sole_action(
        policy_action=-torch.ones((1, 4)),
        previous_applied_action=applied,
        filter_beta=0.9,
        max_action_rate=rate,
    )
    assert float((out.applied - applied).abs().max()) <= rate + 1e-6


def test_shield_blocks_motion_into_a_limit_and_permits_motion_away():
    """At zero headroom the shield must be total in the consuming direction."""
    action = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    # cylinders 0,2 are at the high limit; 1,3 are at the low limit.
    headroom_high = torch.tensor([[0.0, 0.05, 0.0, 0.05]])
    headroom_low = torch.tensor([[0.05, 0.0, 0.05, 0.0]])
    shielded, gain = limit_shield(
        action,
        headroom_low_m=headroom_low,
        headroom_high_m=headroom_high,
        margin_m=0.02,
    )
    assert float(shielded[0, 0]) == 0.0, "extend into high limit was not blocked"
    assert float(shielded[0, 1]) == 0.0, "retract into low limit was not blocked"
    assert float(shielded[0, 2]) == 0.0
    assert float(shielded[0, 3]) == 0.0

    # Reverse the commands: every cylinder now moves away from its limit.
    away, gain_away = limit_shield(
        -action,
        headroom_low_m=headroom_low,
        headroom_high_m=headroom_high,
        margin_m=0.02,
    )
    assert torch.allclose(away, -action), "shield reduced motion away from a limit"
    assert torch.allclose(gain_away, torch.ones_like(gain_away))
    assert (gain <= 1.0 + 1e-7).all()


def test_shield_fades_linearly_across_the_margin():
    margin = 0.02
    headroom = torch.tensor([[0.02, 0.01, 0.005, 0.0]])
    shielded, _ = limit_shield(
        torch.ones((1, 4)),
        headroom_low_m=torch.full((1, 4), 1.0),
        headroom_high_m=headroom,
        margin_m=margin,
    )
    expected = torch.tensor([[1.0, 0.5, 0.25, 0.0]])
    assert torch.allclose(shielded, expected, atol=1e-6)


def test_shield_never_amplifies():
    """The shield is authority-reducing only; it must never exceed gain 1."""
    torch.manual_seed(0)
    action = 2.0 * torch.rand((64, 4)) - 1.0
    headroom_low = torch.rand((64, 4))
    headroom_high = torch.rand((64, 4))
    shielded, gain = limit_shield(
        action,
        headroom_low_m=headroom_low,
        headroom_high_m=headroom_high,
        margin_m=0.05,
    )
    assert (shielded.abs() <= action.abs() + 1e-7).all()
    assert (gain <= 1.0 + 1e-7).all()


def test_shield_respects_negative_action_to_extension_mapping():
    """This asset's positive USD action consumes LOW learned headroom."""
    action = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    low = torch.tensor([[0.0, 0.05, 0.0, 0.05]])
    high = torch.tensor([[0.05, 0.0, 0.05, 0.0]])
    shielded, _ = limit_shield(
        action,
        headroom_low_m=low,
        headroom_high_m=high,
        margin_m=0.02,
        action_to_extension_sign=-torch.ones(4),
    )
    assert torch.equal(shielded, torch.zeros_like(action))

    away, _ = limit_shield(
        -action,
        headroom_low_m=low,
        headroom_high_m=high,
        margin_m=0.02,
        action_to_extension_sign=-torch.ones(4),
    )
    assert torch.equal(away, -action)


def test_action_authority_lock_is_exact_under_adversarial_commands():
    out = apply_sole_action(
        policy_action=torch.tensor([[0.6, -0.7, 0.8, 1.0]]),
        previous_applied_action=torch.tensor([[0.1, -0.2, 0.3, -1.0]]),
        filter_beta=0.0,
        max_action_rate=10.0,
        action_authority=(1.0, 1.0, 1.0, 0.0),
    )
    assert torch.allclose(out.applied[:, :3], torch.tensor([[0.6, -0.7, 0.8]]))
    assert float(out.raw[0, 3]) == 0.0
    assert float(out.filtered[0, 3]) == 0.0
    assert float(out.applied[0, 3]) == 0.0


def test_directional_tilt_shield_blocks_only_worsening_head_motion():
    action = torch.tensor(
        [
            [0.1, 0.2, 0.3, -0.5],
            [0.1, 0.2, 0.3, 0.5],
            [0.1, 0.2, 0.3, -0.5],
            [0.1, 0.2, 0.3, 0.5],
        ]
    )
    # Positive c4 increases tilt. Rows: worsening mid-fade, corrective,
    # below-soft, and worsening beyond the hard limit.
    tilt = torch.tensor([-0.12, -0.12, -0.04, 0.20])
    shielded, gain = directional_tilt_shield(
        action,
        signed_tilt=tilt,
        action_channel=3,
        action_to_tilt_sign=1.0,
        soft_limit=0.08,
        hard_limit=0.16,
    )
    assert torch.allclose(gain[:, 3], torch.tensor([0.5, 1.0, 1.0, 0.0]))
    assert torch.allclose(shielded[:, :3], action[:, :3])
    assert torch.allclose(shielded[:, 3], action[:, 3] * gain[:, 3])


def test_directional_tilt_shield_rejects_invalid_contracts():
    for kwargs in (
        {"action_channel": 4},
        {"action_to_tilt_sign": 0.0},
        {"soft_limit": 0.2, "hard_limit": 0.1},
    ):
        config = {
            "action_channel": 3,
            "action_to_tilt_sign": 1.0,
            "soft_limit": 0.08,
            "hard_limit": 0.16,
            **kwargs,
        }
        try:
            directional_tilt_shield(
                torch.zeros((2, 4)), signed_tilt=torch.zeros(2), **config
            )
        except ValueError:
            continue
        raise AssertionError(f"invalid tilt shield config was accepted: {config}")


def test_zero_action_is_a_fixed_point():
    """Zero command from rest must leave the plant command at zero.

    PPO2 had a deterministic baseline to fall back on, so a zero action meant
    'do what the teacher says'. Here it means 'command no motion', and the
    stationary-hold contract depends on that being exact.
    """
    out = apply_sole_action(
        policy_action=torch.zeros((8, 4)),
        previous_applied_action=torch.zeros((8, 4)),
        filter_beta=0.75,
        max_action_rate=0.2,
        headroom_low_m=torch.full((8, 4), 0.001),
        headroom_high_m=torch.full((8, 4), 0.001),
        limit_shield_margin_m=0.02,
    )
    assert float(out.applied.abs().max()) == 0.0


def test_rejects_non_four_dimensional_action():
    for dim in (1, 3, 5):
        try:
            apply_sole_action(
                policy_action=torch.zeros((2, dim)),
                previous_applied_action=torch.zeros((2, dim)),
                filter_beta=0.5,
                max_action_rate=0.2,
            )
        except ValueError:
            continue
        raise AssertionError(f"action dim {dim} was accepted")




def test_chain_inversion_round_trips_through_the_action_chain():
    """invert_sole_chain must be an exact inverse in the unsaturated regime.

    This is the load-bearing claim of distillation: if the inversion is even
    slightly wrong, every behaviour-cloning label is biased and the student
    converges to a systematically attenuated version of the teacher.
    """
    from PPO3_SOLE_PhysX.actions import invert_sole_chain

    torch.manual_seed(0)
    beta = 0.75
    previous = 0.5 * (2.0 * torch.rand((32, 4)) - 1.0)
    teacher = previous + 0.05 * (2.0 * torch.rand((32, 4)) - 1.0)

    raw, unreachable = invert_sole_chain(
        target_applied=teacher, previous_applied_action=previous, filter_beta=beta
    )
    assert not bool(unreachable.any()), "small teacher steps should be reachable"

    out = apply_sole_action(
        policy_action=raw,
        previous_applied_action=previous,
        filter_beta=beta,
        max_action_rate=10.0,  # slew disabled so the filter inverse is isolated
    )
    assert torch.allclose(out.applied, teacher, atol=1e-5), (
        f"round trip error {float((out.applied - teacher).abs().max()):.2e}"
    )


def test_naive_label_would_be_attenuated_by_the_filter():
    """Guards the reason invert_sole_chain exists.

    Regressing the actor on the teacher command directly leaves the chain to
    attenuate it by (1 - beta). If this test ever starts passing with a small
    error, the filter is no longer doing anything and the inversion is moot.
    """
    beta = 0.75
    previous = torch.zeros((4, 4))
    teacher = torch.full((4, 4), 0.4)
    naive = apply_sole_action(
        policy_action=teacher,
        previous_applied_action=previous,
        filter_beta=beta,
        max_action_rate=10.0,
    )
    attenuation = float((naive.applied / teacher).mean())
    assert abs(attenuation - (1.0 - beta)) < 1e-5, (
        f"expected {(1.0 - beta):.2f}x attenuation from the naive label, "
        f"got {attenuation:.3f}"
    )


def test_unreachable_labels_are_flagged_not_silently_clamped():
    from PPO3_SOLE_PhysX.actions import invert_sole_chain

    raw, unreachable = invert_sole_chain(
        target_applied=torch.full((2, 4), 0.9),
        previous_applied_action=torch.full((2, 4), -0.9),
        filter_beta=0.9,
    )
    assert bool(unreachable.all()), "a huge one-step jump must be flagged"
    assert (raw.abs() <= 1.0 + 1e-7).all()


def test_chain_inversion_rejects_a_frozen_filter():
    from PPO3_SOLE_PhysX.actions import invert_sole_chain

    try:
        invert_sole_chain(
            target_applied=torch.zeros((2, 4)),
            previous_applied_action=torch.zeros((2, 4)),
            filter_beta=1.0,
        )
    except ValueError:
        return
    raise AssertionError("beta = 1.0 was accepted; the inverse does not exist")


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
