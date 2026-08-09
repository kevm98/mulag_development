"""Contracts for the PPO3_1 reward. Each test pins a failure that really happened."""

from __future__ import annotations

import pytest
import torch

from PPO_ROM.rewards import (
    RewardWeights,
    _bounded,
    _huber,
    compute_reward,
)


def _inputs(n=4, **over):
    base = dict(
        cut_height_error=torch.zeros(n),
        lateral_error=torch.zeros(n),
        head_down_alignment=torch.ones(n),
        cutter_contact=torch.ones(n, dtype=torch.bool),
        applied_action=torch.zeros((n, 5)),
        previous_applied_action=torch.zeros((n, 5)),
        tool_acceleration_norm=torch.zeros(n),
        contact_force_n=torch.zeros(n),
        cylinder_headroom_low_m=torch.ones((n, 4)),
        cylinder_headroom_high_m=torch.ones((n, 4)),
    )
    base.update(over)
    return base


def _lateral_grad(err: float) -> float:
    e = torch.tensor([err], requires_grad=True)
    compute_reward(**_inputs(1, lateral_error=e)).lateral.sum().backward()
    return float(e.grad.abs())


def test_lateral_gradient_survives_where_the_old_reward_died():
    """The whole reason for the rewrite.

    Measured on the old stack: `w * exp(-(err/std)^2)` at err = 0.50 m gives
    gradient -2.07 at sigma 0.30 and -0.004 at sigma 0.15 -- 520x weaker exactly
    at the errors the failing families sit at (0.30-0.56 m).

    THIS TEST USED TO PROBE (0.05, 0.20, 0.30) AND PASS. 0.30 was exactly the
    clamp boundary, where torch's clamp still passes gradient, so it verified
    the reward right up to the cliff and never stepped off it. The run then sat
    at 0.734 m for 888 iterations with an identically-zero lateral gradient.
    Probing past the operating range is the entire point, so it now does.
    """
    for err in (0.05, 0.20, 0.30, 0.50, 0.734, 1.00, 1.40):
        assert _lateral_grad(err) > 1.0, (err, _lateral_grad(err))


def test_lateral_gradient_is_sharper_inside_the_band_than_far_outside_it():
    """Two scales, and they must do different jobs.

    THIS TEST USED TO PROBE 0.40 m AS THE "FAR" REGIME and require the near
    gradient to be 5x it. That encoded the defect as a requirement: it asserted
    0.40 m -- above the 0.300 gate but squarely where the policy operates --
    belonged to the FLAT part of the curve. With the fine term saturating at
    0.15 m that is exactly what happened, lateral was worth a flat 4.0 per
    metre there, contact was worth 10.0, and PPO traded the line away while
    raising total reward (run warmup_prod, explained variance 0.955).

    The corrected contract splits at the operating band, not inside it:
    sharp across everything up to 2x the gate, shallower only beyond it.
    """
    inside_near, inside_far = _lateral_grad(0.05), _lateral_grad(0.45)
    outside_a, outside_b = _lateral_grad(0.80), _lateral_grad(1.20)
    assert inside_far > 2.0 * outside_a, (inside_far, outside_a)
    assert inside_near >= inside_far, (inside_near, inside_far)
    assert outside_a == pytest.approx(outside_b, rel=1e-6), (outside_a, outside_b)


def test_every_shaping_term_has_gradient_past_twice_its_gate():
    """The invariant that the 888-iteration run bought us.

    A term whose gradient dies before the policy reaches its acceptance
    threshold cannot ever drive the policy to pass. Checked here against the
    real gate values, and again on the live plant as preflight check S5.
    """
    from PPO_ROM.gates import GATES

    for gate_name, field, kwarg in (
        ("lateral", "lateral", "lateral_error"),
        ("cut_height", "cut_height", "cut_height_error"),
    ):
        probe = 2.0 * GATES[gate_name].threshold
        e = torch.tensor([probe], requires_grad=True)
        getattr(compute_reward(**_inputs(1, **{kwarg: e})), field).sum().backward()
        assert float(e.grad.abs()) > 1e-6, (gate_name, probe, float(e.grad.abs()))


def test_task_terms_are_non_negative_and_safety_terms_non_positive():
    """The sign convention that makes worst-case weighting safe to add.

    In the old stack a CVaR weight was applied to the POSITIVE contact reward,
    which paid the policy to be in the worst tail: contact quality 0.49 paid
    7.84 while 0.50 paid 2.00, and worst-family contact collapsed 0.7503 ->
    0.6208 in a single block.
    """
    terms = compute_reward(
        **_inputs(
            3,
            cut_height_error=torch.tensor([0.0, 0.2, 1.0]),
            lateral_error=torch.tensor([0.0, 0.4, 2.0]),
            head_down_alignment=torch.tensor([1.0, 0.9, -1.0]),
            cutter_contact=torch.tensor([True, False, False]),
            tool_acceleration_norm=torch.tensor([0.0, 100.0, 900.0]),
            contact_force_n=torch.tensor([0.0, 5e4, 5e5]),
        )
    )
    d = terms.as_dict()
    for name in ("cut_height", "lateral", "ground_contact"):
        assert (d[name] >= 0.0).all(), (name, d[name])
    for name in (
        "head_alignment", "contact_loss", "action_rate",
        "tool_acceleration", "contact_force", "cylinder_headroom",
    ):
        assert (d[name] <= 0.0).all(), (name, d[name])


def test_every_term_is_bounded_even_for_absurd_inputs():
    """A term that can grow without limit destabilises the critic."""
    w = RewardWeights()
    terms = compute_reward(
        **_inputs(
            2,
            cut_height_error=torch.tensor([1e3, -1e3]),
            lateral_error=torch.tensor([1e3, -1e3]),
            head_down_alignment=torch.tensor([-1e3, 1e3]),
            tool_acceleration_norm=torch.tensor([1e6, 1e6]),
            contact_force_n=torch.tensor([1e9, 1e9]),
            applied_action=torch.full((2, 5), 1e3),
            previous_applied_action=torch.full((2, 5), -1e3),
            cylinder_headroom_low_m=torch.full((2, 4), -1e3),
            cylinder_headroom_high_m=torch.full((2, 4), -1e3),
        )
    )
    for name, value in terms.as_dict().items():
        assert torch.isfinite(value).all(), name
        limit = abs(getattr(w, name)) + 1e-6
        assert (value.abs() <= limit).all(), (name, value, limit)
    assert torch.isfinite(terms.total).all()


def test_holding_a_steady_offset_is_free_but_jerking_is_not():
    """Why the action MAGNITUDE penalty is gone.

    lane_change_hold, offset_return and boom_sweep all require a sustained
    non-zero action. Charging `mean(action^2)` for it made the policy return to
    neutral, which is the opposite of the task -- thirteen campaigns read as
    "PPO cannot learn this".
    """
    held = torch.full((1, 5), 0.8)
    steady = compute_reward(
        **_inputs(1, applied_action=held, previous_applied_action=held)
    )
    jerked = compute_reward(
        **_inputs(1, applied_action=held, previous_applied_action=-held)
    )
    assert float(steady.action_rate) == 0.0
    assert float(jerked.action_rate) < 0.0


def test_head_penalty_is_the_same_order_as_the_task_not_triple_it():
    """The old head group (-20, -8 far, x2 tail) dwarfed lateral (3.0 + 3.0)."""
    w = RewardWeights()
    assert w.task_total() == 22.0
    assert abs(w.head_alignment) < w.task_total()
    # Tracking must not be outbiddable by planting the tool. NOTE this
    # weight-level check is necessary but NOT sufficient, and on its own it did
    # not prevent the trade: it held with room to spare at lateral 10.0 against
    # contact's 4.0 + 6.0 = 10.0, while the policy still gave up the line.
    #
    # Comparing weights compares the wrong quantity. Contact's weight IS its
    # marginal rate (the term is linear in the contact fraction), but lateral's
    # weight is split across scales and divided by each, so its marginal rate
    # was 10 * 0.6/1.50 = 4.0 per metre once the fine term saturated -- 2.5x
    # weaker than the number this assertion was reading. The gradient-level
    # check lives in test_lateral_outbids_contact_across_the_operating_band,
    # which is the one that actually binds.
    assert w.lateral >= w.ground_contact + abs(w.contact_loss)
    # Head is still the single largest penalty: it is a precondition, not a
    # competing objective.
    for name in ("contact_loss", "action_rate", "tool_acceleration",
                 "contact_force", "cylinder_headroom"):
        assert abs(w.head_alignment) > abs(getattr(w, name))


def test_ten_terms_not_fifty_three():
    """Isaac Lab's locomotion task uses 11 reward terms. The old PPO3 cfg had 53."""
    terms = compute_reward(**_inputs(1))
    assert len(terms.as_dict()) == 9  # nine shaped terms + termination handled by the env


def test_shapes_are_monotone_so_better_always_scores_better():
    for fn, scale in ((_bounded, 0.3), (_huber, 0.04)):
        x = torch.linspace(0.0, 5.0, 200)
        y = fn(x, scale, 1.0)
        assert (y.diff() >= -1e-9).all(), fn.__name__
        assert float(y.max()) <= 1.0 + 1e-9


def _contact_marginal_rate() -> float:
    """Reward gained per unit of ground-contact fraction.

    ``ground_contact`` pays ``+w.ground_contact * c`` and ``contact_loss``
    charges ``w.contact_loss * (1 - c)``, so the slope is their sum.
    """
    w = RewardWeights()
    on = compute_reward(**_inputs(1, cutter_contact=torch.ones(1, dtype=torch.bool)))
    off = compute_reward(**_inputs(1, cutter_contact=torch.zeros(1, dtype=torch.bool)))
    return float(
        (on.ground_contact + on.contact_loss) - (off.ground_contact + off.contact_loss)
    )


def test_lateral_outbids_contact_across_the_operating_band():
    """The failure a nonzero-gradient check cannot catch.

    ``preflight`` check S5 asserts the lateral gradient is not zero at 2x the
    gate, and it passed at 4.0 per metre -- while the policy still abandoned
    the line. Nonzero is not the requirement: the requirement is that tracking
    is worth MORE at the margin than the objective competing with it.

    Measured in run ``warmup_prod`` with a healthy critic (explained variance
    0.955), PPO traded lateral 0.189 -> 0.639 m for contact 0.595 -> 0.887 and
    raised total reward by 1.212. Contact's predicted gain, 10.0 * 0.292,
    equals the measured +2.921 -- the marginal rates, not noise, chose that.

    So: across the band the policy actually operates in, a metre of lateral
    error must cost more than a unit of contact fraction is worth.
    """
    contact_rate = _contact_marginal_rate()
    for error in (0.20, 0.30, 0.45, 0.55):
        assert _lateral_grad(error) > contact_rate, (
            f"at {error} m lateral is worth {_lateral_grad(error):.2f}/m against "
            f"contact's {contact_rate:.2f}; PPO will trade the line for contact"
        )


def test_lateral_fine_term_stays_alive_to_twice_the_gate():
    """The fine sub-term must obey the same rule as the coarse one.

    It saturated at 0.15 m -- half the 0.300 gate -- so beyond that only the
    coarse term contributed and the gradient fell to a flat 4.0 per metre.
    """
    from PPO_ROM.gates import GATES

    w = RewardWeights()
    assert w.lateral_fine_scale_m >= 2.0 * GATES["lateral"].threshold
