"""Safety-shaping contracts for the terrain policy."""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

from verge_task.terrain_rewards import (
    bounded_cost_progress,
    bounded_wrapped_yaw_reward,
    compute_verge_reward,
    lower_tail_weights,
    sanitize_terminal_reward,
    update_failure_memory,
)


def test_bounded_yaw_reward_has_far_field_gradient():
    error = torch.tensor([0.0, 1.0, 3.12], requires_grad=True)
    reward = bounded_wrapped_yaw_reward(error, weight=0.5)
    assert reward[0].item() == 0.5
    assert 0.0 < reward[2].item() < reward[1].item() < reward[0].item()
    reward.sum().backward()
    assert error.grad is not None
    assert error.grad[2].abs().item() > 0.1


def test_current_asset_yaw_target_is_calibrated_to_zero():
    source = (ROOT / "verge_task" / "terrain_env_cfg.py").read_text()
    assert "terrain_head_yaw_target_rad: float = 0.0" in source


def _terms(alignment: torch.Tensor, *, scale: float = 0.02, far_weight: float = 0.0):
    n = int(alignment.numel())
    zeros = torch.zeros((n, 4))
    return compute_verge_reward(
        cut_height=torch.full((n,), 0.05),
        target_cut_height_m=0.05,
        lateral_error=torch.zeros(n),
        applied_action=zeros,
        previous_applied_action=zeros,
        head_down_alignment=alignment,
        cutter_contact=torch.zeros(n, dtype=torch.bool),
        cylinder_headroom_low_m=torch.ones((n, 4)),
        cylinder_headroom_high_m=torch.ones((n, 4)),
        tool_acceleration_norm=torch.zeros(n),
        scan_valid_fraction=torch.ones(n),
        head_penalty_weight=-20.0,
        head_threshold=0.995,
        head_penalty_scale=scale,
        head_penalty_max_deviation=2.0,
        head_penalty_far_weight=far_weight,
    )


def test_head_barrier_matches_normalized_quadratic_contract():
    terms = _terms(torch.tensor([0.995, 0.990, 0.980, 0.970]))
    expected = torch.tensor([0.0, -1.25, -11.25, -31.25])
    assert torch.allclose(terms.head_penalty, expected, atol=1.0e-5)


def test_lower_tail_weights_only_the_worst_active_fraction():
    values = torch.tensor([0.99, 0.91, 0.95, 0.80, 0.70])
    active = torch.tensor([True, True, True, True, False])
    weights = lower_tail_weights(
        values, active_mask=active, fraction=0.25, multiplier=4.0
    )
    assert torch.equal(weights, torch.tensor([1.0, 1.0, 1.0, 4.0, 1.0]))


def test_failure_memory_retriggers_and_counts_down_only_while_active():
    memory = torch.tensor([0, 2, 2])
    updated = update_failure_memory(
        memory,
        trigger=torch.tensor([True, False, False]),
        active_mask=torch.tensor([True, True, False]),
        horizon_steps=3,
    )
    assert torch.equal(updated, torch.tensor([3, 1, 2]))


def test_head_barrier_is_zero_above_margin_and_finite_for_zero_scale():
    assert float(_terms(torch.tensor([1.0])).head_penalty[0]) == 0.0
    assert torch.isfinite(_terms(torch.tensor([0.98]), scale=0.0).reward).all()


def test_head_barrier_saturates_before_flipped_states_explode_value_targets():
    terms = _terms(torch.tensor([0.89, 0.0, -1.0]))
    assert torch.equal(terms.head_penalty, torch.full((3,), -80.0))
    assert torch.isfinite(terms.reward).all()


def test_head_far_field_keeps_a_gradient_out_of_the_saturated_plateau():
    """The clamped quadratic alone is flat below alignment 0.955.

    That plateau is what let a flipped head persist: every inverted pose scored
    -80, so PPO had no signal pointing back toward head-down. With the far-field
    weight the barrier must be strictly decreasing across the whole flipped range.
    """
    alignment = torch.tensor([0.955, 0.89, 0.5, 0.0, -0.5, -1.0])
    flat = _terms(alignment).head_penalty
    sloped = _terms(alignment, far_weight=-0.5).head_penalty

    assert torch.equal(flat, torch.full((6,), -80.0))
    assert (sloped.diff() < 0.0).all()
    # Recovering toward head-down is always worth strictly more reward.
    assert float(sloped[0]) == -80.0
    assert torch.isfinite(sloped).all()


def test_head_far_field_stays_bounded_and_inactive_above_the_clamp():
    """A fully inverted head must stay the same order as the terminal reward."""
    assert float(_terms(torch.tensor([-1.0]), far_weight=-0.5).head_penalty[0]) == -128.875
    # Above the clamp the far-field term contributes nothing, so the tuned
    # near-field quadratic is untouched.
    near = torch.tensor([0.995, 0.990, 0.980, 0.970])
    assert torch.allclose(
        _terms(near, far_weight=-0.5).head_penalty,
        _terms(near).head_penalty,
        atol=1.0e-5,
    )


def test_clearance_barrier_warns_before_strike_and_saturates_far_below_ground():
    heights = torch.tensor([0.03, 0.02, 0.01, 0.00, -0.02, -1.00])
    n = int(heights.numel())
    zeros = torch.zeros((n, 4))
    terms = compute_verge_reward(
        cut_height=heights,
        target_cut_height_m=0.05,
        lateral_error=torch.zeros(n),
        applied_action=zeros,
        previous_applied_action=zeros,
        head_down_alignment=torch.ones(n),
        cutter_contact=heights <= 0.0,
        cylinder_headroom_low_m=torch.ones((n, 4)),
        cylinder_headroom_high_m=torch.ones((n, 4)),
        tool_acceleration_norm=torch.zeros(n),
        scan_valid_fraction=torch.ones(n),
        clearance_barrier_weight=-2.0,
        clearance_margin_m=0.02,
        clearance_barrier_scale_m=0.02,
        clearance_barrier_max_deviation=5.0,
    )
    expected = torch.tensor([0.0, 0.0, -0.5, -2.0, -8.0, -50.0])
    assert torch.allclose(terms.clearance_barrier, expected, atol=1.0e-6)


def test_acceleration_penalty_keeps_gradient_through_observed_spikes_and_caps():
    acceleration = torch.tensor([0.0, 30.0, 60.0, 245.0, 300.0, 600.0])
    n = int(acceleration.numel())
    zeros = torch.zeros((n, 4))
    terms = compute_verge_reward(
        cut_height=torch.full((n,), 0.05),
        target_cut_height_m=0.05,
        lateral_error=torch.zeros(n),
        applied_action=zeros,
        previous_applied_action=zeros,
        head_down_alignment=torch.ones(n),
        cutter_contact=torch.zeros(n, dtype=torch.bool),
        cylinder_headroom_low_m=torch.ones((n, 4)),
        cylinder_headroom_high_m=torch.ones((n, 4)),
        tool_acceleration_norm=acceleration,
        scan_valid_fraction=torch.ones(n),
        acceleration_weight=-0.20,
        acceleration_scale_mps2=30.0,
        acceleration_max_ratio=10.0,
    )
    expected = -0.20 * torch.tensor(
        [0.0, 1.0, 4.0, (245.0 / 30.0) ** 2, 100.0, 100.0]
    )
    assert torch.allclose(terms.acceleration_penalty, expected, atol=1.0e-5)
    assert terms.acceleration_penalty[3] < terms.acceleration_penalty[2]
    assert terms.acceleration_penalty[4] == terms.acceleration_penalty[5]


def test_nonfinite_reward_is_replaced_without_changing_healthy_rows():
    reward = torch.tensor([1.25, float("nan"), float("inf"), -3.0])
    sanitized, bad = sanitize_terminal_reward(
        reward,
        invalid_mask=torch.tensor([False, False, False, True]),
        terminal_reward=-100.0,
    )
    assert torch.equal(bad, torch.tensor([False, True, True, True]))
    assert torch.equal(sanitized, torch.tensor([1.25, -100.0, -100.0, -100.0]))
    assert torch.isfinite(sanitized).all()


def test_bounded_cost_progress_has_direction_linear_region_and_hard_cap():
    previous = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    current = torch.tensor([0.5, 0.95, 1.0, 1.05, 2.0])
    progress = bounded_cost_progress(
        previous,
        current,
        scale=0.10,
        weight=0.5,
    )
    assert torch.allclose(
        progress,
        torch.tensor([0.5, 0.25, 0.0, -0.25, -0.5]),
        atol=1.0e-6,
    )


def test_bounded_cost_progress_zero_scale_stays_finite_and_reset_match_is_zero():
    progress = bounded_cost_progress(
        torch.tensor([0.2, 0.2, 0.2]),
        torch.tensor([0.1, 0.2, 0.3]),
        scale=0.0,
        weight=1.0,
    )
    assert torch.equal(progress, torch.tensor([1.0, 0.0, -1.0]))
    assert torch.isfinite(progress).all()


def test_bounded_cost_progress_rejects_shape_mismatch():
    try:
        bounded_cost_progress(
            torch.zeros(2),
            torch.zeros(3),
            scale=1.0,
            weight=1.0,
        )
    except ValueError as exc:
        assert "shapes differ" in str(exc)
    else:
        raise AssertionError("shape mismatch must raise")


def test_reward_splits_cleanly_into_gateable_task_and_always_on_safety():
    """The head-down latch zeroes task reward only; barriers must stay live.

    TerrainSolePolicyEnv rebuilds the shaped reward from these parts so a failed
    head can be denied tracking pay without also switching off the barriers that
    pull it back. If the split ever stops summing to ``reward`` the latch would
    silently change the reward scale for healthy envs too.
    """
    n = 6
    terms = compute_verge_reward(
        cut_height=torch.linspace(0.0, 0.1, n),
        target_cut_height_m=0.05,
        lateral_error=torch.linspace(-0.2, 0.2, n),
        applied_action=torch.full((n, 4), 0.3),
        previous_applied_action=torch.full((n, 4), 0.1),
        head_down_alignment=torch.tensor([1.0, 0.99, 0.98, 0.97, 0.9, -0.5]),
        cutter_contact=torch.ones(n, dtype=torch.bool),
        cylinder_headroom_low_m=torch.full((n, 4), 0.1),
        cylinder_headroom_high_m=torch.full((n, 4), 0.1),
        tool_acceleration_norm=torch.linspace(0.0, 60.0, n),
        scan_valid_fraction=torch.ones(n),
        head_penalty_weight=-20.0,
        head_threshold=0.995,
        head_penalty_scale=0.02,
        head_penalty_max_deviation=2.0,
        head_penalty_far_weight=-0.5,
    )
    task = terms.cut_tracking + terms.lateral + terms.contact_reward
    safety = (
        terms.smoothness
        + terms.head_penalty
        + terms.clearance_barrier
        + terms.limit_penalty
        + terms.acceleration_penalty
    )
    assert torch.allclose(task + safety, terms.reward, atol=1.0e-5)
    # Denying task pay must leave a strictly negative residue for a failed head,
    # so latching can never be worth more than tracking with the head down.
    assert float(safety[-1]) < 0.0
    assert float(task[-1]) > 0.0


def test_head_down_latch_releases_on_hysteresis_recovery():
    """A transient dip must cost its own duration, not the rest of the episode.

    Mirrors TerrainSolePolicyEnv._get_rewards' latch state machine. Five blocks
    on the rough verge collapsed with the monotone latch (v3 2-4, v4 1, v5 1);
    the release threshold sits above the latch threshold so an env still earns
    nothing while tilted or inside the hysteresis band.
    """
    required, release = 0.95, 0.97

    def paid_steps(trace, *, hysteresis):
        latched = torch.zeros(1, dtype=torch.bool)
        out = []
        for value in trace:
            alignment = torch.tensor([value])
            latched |= alignment < required
            if hysteresis and release > required:
                latched &= ~(latched & (alignment >= release))
            out.append(0 if bool(latched) else 1)
        return out

    dip = [0.99, 0.98, 0.96, 0.93, 0.96, 0.98, 0.99, 0.99, 0.99, 0.99]
    assert sum(paid_steps(dip, hysteresis=False)) == 3
    assert sum(paid_steps(dip, hysteresis=True)) == 8

    # Tilted steps and hysteresis-band steps still earn nothing.
    tilted = [0.99, 0.90, 0.90, 0.96, 0.96, 0.98]
    assert paid_steps(tilted, hysteresis=True) == [1, 0, 0, 0, 0, 1]


def test_latch_release_threshold_exceeds_required_alignment():
    cfg_source = (ROOT / "verge_task" / "terrain_env_cfg.py").read_text()
    assert "terrain_head_down_required_alignment: float = 0.95" in cfg_source
    assert "terrain_head_down_latch_release_alignment: float = 0.97" in cfg_source

    env_source = (ROOT / "verge_task" / "terrain_env.py").read_text()
    assert "terrain_head_down_latch_release_alignment" in env_source
    assert "safety/head_down_latch_released_fraction" in env_source


def test_messerkopf_level_correction_contract():
    """Pose-dependent head levelling: opt-in, bounded, idempotent, opposing.

    A fixed joint6_coupling ratio is wrong by a pose-varying amount and removing
    it is worse (held-out min head alignment 0.9628 coupled vs -0.6956 at
    stiffness 6000 and 0.6268 at 300000). The correction closes the loop around
    the ratio instead of replacing it.
    """
    import torch as _torch

    a, b, sign, bound = 0.76, 0.0, -1.0, 0.15

    def target(source_ref, tilt, gain):
        correction = _torch.clamp(
            _torch.tensor([sign * gain * tilt], dtype=_torch.float32), -bound, bound
        )
        return float(a * source_ref + b + correction.item())

    source, base = 0.5, 0.76 * 0.5

    # Recomputed from the coupling source, so repeated substeps never integrate.
    assert target(source, 0.10, 0.5) == target(source, 0.10, 0.5)
    # The correction opposes the measured tilt rather than amplifying it.
    assert target(source, 0.10, 0.5) < target(source, 0.0, 0.5) < target(source, -0.10, 0.5)
    # A wild tilt reading cannot sweep the joint across its travel.
    assert abs(target(source, 3.0, 0.5) - (base - bound)) < 1e-6
    # Gain zero must reproduce the inherited target bit-for-bit.
    assert target(source, 0.37, 0.0) == base


def test_messerkopf_level_production_contract():
    """The swept levelling contract that reaches 14/15 held-out gates.

    Measured on v5/model_best.pt, 128 envs x 960 steps, NO retraining
    (reports/heldout_level_*.json). Inherited head model scored 0.1130 at 13/15
    with p95_cut_height_error failing; this pair reaches 0.0823 at 14/15 with
    both head alignment and cut height passing:

        min_head_alignment      0.9628 -> 0.9559  (gate 0.950, still passing)
        p95_cut_height_error    0.0817 -> 0.0628  (gate 0.080, now passing)
        p95_tool_acceleration  130.94  -> 129.88  (gate 120, still failing)

    Nothing here is monotone -- gain 0.25 is worse than 0.0 and 0.5, clamp 0.04
    is worse than 0.05, action filter 0.85 and rate limit 0.15 both INCREASED
    acceleration. Re-sweep rather than bisect if any of it is changed.

    2026-07-31: clamp 0.10 / quadratic_gain 8.0 was TRIED AND REVERTED. The
    clamp really does starve the tail (at 0.323 rad of tilt the linear term
    alone demands 0.1615 rad, 3.2x past a 0.05 bound), and opening it scored
    17/17 on varied seed 909 -- but that was one mesh. Across three terrains:

        terrain        0.05/0.0            0.10/8.0
        held-out       16/17 align 0.9500  14/17 align 0.9158  accel 91.7 -> 127.4
        varied 17      16/17 align 0.9396  16/17 align 0.9111
        varied 909     14/17 align 0.9185  17/17 align 0.9530

    Sweeping on a single mesh bought that mesh at the others' expense. Any
    future head work must be scored on ALL THREE terrains before adoption.
    """
    cfg_source = (ROOT / "verge_task" / "terrain_env_cfg.py").read_text()
    assert "terrain_messerkopf_level_gain: float = 0.0" in cfg_source
    assert "terrain_messerkopf_level_max_rad: float = 0.05" in cfg_source
    assert "terrain_messerkopf_level_quadratic_gain: float = 0.0" in cfg_source
    assert "terrain_messerkopf_level_filter: float = 0.5" in cfg_source
    assert "terrain_action_filter: float = 0.55" in cfg_source

    env_source = (ROOT / "verge_task" / "terrain_env.py").read_text()
    assert "terrain_messerkopf_level_gain" in env_source
    # The head reference is now assembled from three contributions on one
    # joint: the coupling feedforward that tracks the boom, the measured-tilt
    # levelling correction, and the policy's own bounded correction. Measured
    # min_head_alignment for each variant: frozen reference 0.8369, coupling
    # without levelling 0.8662, coupling with levelling 0.9296.
    # The head reference must TRACK the boom (a frozen reference measured
    # min_head_alignment 0.8369). The boom angle is now taken either from the
    # commanded q4 or, under a learned actuator whose command is not reached,
    # from the measured joint -- so assert the tracking term, not one source.
    assert "a * source_angle" in env_source
    assert "source_angle = self._joint_ref[:, source]" in env_source
    assert "terrain_head_feedforward_from_measured" in env_source
    assert "self.robot.data.joint_pos[:, int(source_joint)]" in env_source
    assert "+ correction" in env_source
    assert "+ policy_correction" in env_source




def _chassis_terms(
    lateral_m: torch.Tensor,
    attitude_rad: torch.Tensor | None = None,
    *,
    weight: float = -0.25,
):
    n = int(lateral_m.numel())
    zeros = torch.zeros((n, 4))
    if attitude_rad is None:
        attitude_rad = torch.zeros(n)
    return compute_verge_reward(
        cut_height=torch.zeros(n),
        target_cut_height_m=0.0,
        lateral_error=torch.zeros(n),
        applied_action=zeros,
        previous_applied_action=zeros,
        head_down_alignment=torch.ones(n),
        cutter_contact=torch.ones(n, dtype=torch.bool),
        cylinder_headroom_low_m=torch.ones((n, 4)),
        cylinder_headroom_high_m=torch.ones((n, 4)),
        tool_acceleration_norm=torch.zeros(n),
        scan_valid_fraction=torch.ones(n),
        chassis_lateral_reaction_m=lateral_m,
        chassis_attitude_reaction_rad=attitude_rad,
        chassis_reaction_weight=weight,
        chassis_lateral_scale_m=0.020,
        chassis_attitude_scale_rad=0.004,
        chassis_reaction_max_ratio=5.0,
    )


def test_chassis_reaction_penalty_is_inert_at_zero_weight():
    """Default weight must reproduce the pre-existing reward bit-for-bit.

    The 15/15 checkpoint was scored without this term; a non-zero default would
    silently reinterpret every archived score.
    """
    lateral = torch.tensor([0.0, 0.055, 0.193])
    off = _chassis_terms(lateral, weight=0.0)
    assert torch.equal(off.chassis_reaction_penalty, torch.zeros(3))
    on = _chassis_terms(lateral)
    assert torch.equal(
        off.reward, on.reward - on.chassis_reaction_penalty
    )


def test_chassis_reaction_penalty_matches_normalized_quadratic_and_clamps():
    # 0 mm, 20 mm (ratio 1), the measured 54.77 mm p95 (ratio 2.7385), and
    # 193 mm which is past the 100 mm clamp.
    lateral = torch.tensor([0.0, 0.020, 0.05477, 0.193])
    penalty = _chassis_terms(lateral).chassis_reaction_penalty
    expected = -0.25 * torch.tensor([0.0, 1.0, 2.7385**2, 25.0])
    assert torch.allclose(penalty, expected, atol=1.0e-5)


def test_chassis_reaction_uses_worst_channel_not_the_sum():
    """Attitude alone must be able to drive the penalty.

    Summing the channels would let a policy trade 1 mm of slide against 1 mrad
    of roll, which are not comparable quantities.
    """
    lateral_only = _chassis_terms(
        torch.tensor([0.020]), torch.zeros(1)
    ).chassis_reaction_penalty
    attitude_only = _chassis_terms(
        torch.zeros(1), torch.tensor([0.004])
    ).chassis_reaction_penalty
    both = _chassis_terms(
        torch.tensor([0.020]), torch.tensor([0.004])
    ).chassis_reaction_penalty
    assert torch.allclose(lateral_only, attitude_only, atol=1.0e-6)
    assert torch.allclose(both, lateral_only, atol=1.0e-6)


def test_chassis_reaction_and_contact_production_contract():
    """Pin the retrained stage-5 objective.

    Both numbers were derived from measurement on the 15/15 checkpoint, not
    chosen: 54.77 mm / 0.629 deg p95 chassis reaction, and a 6.5 % skip rate
    that no physics change moved.
    """
    cfg_source = (ROOT / "verge_task" / "terrain_env_cfg.py").read_text()
    for contract in (
        "terrain_chassis_reaction_weight: float = 0.0",
        "terrain_chassis_lateral_scale_m: float = 0.020",
        "terrain_chassis_attitude_scale_rad: float = 0.004",
        "terrain_chassis_reaction_max_ratio: float = 5.0",
        "terrain_contact_reward_weight: float = 4.0",
        # True drives the measured chassis reaction to its zero-action floor
        # (17.72 mm / 0.206 deg vs floor 16.40 / 0.202) and cuts tool
        # acceleration 111.6 -> 91.7. Reaching the same sway number by training
        # cost head safety every time it was tried.
        "terrain_constrain_truck_per_substep: bool = True",
    ):
        assert contract in cfg_source

    # The reaction must be sampled before the fixture erases it, and forwarded.
    env_source = (ROOT / "verge_task" / "terrain_env.py").read_text()
    assert "_chassis_lateral_reaction_m" in env_source
    assert "terrain_chassis_reaction_weight" in env_source


def test_stage_curriculum_overwrites_lateral_std_so_overrides_must_be_reapplied():
    """The stage curriculum owns `lateral_tracking_std_m`; overrides come after.

    `apply_terrain_curriculum_stage` assigns this field itself, and it runs at
    startup AND on every stage transition. A CLI override applied once before
    the curriculum is therefore silently reverted mid-run -- the policy would
    train under the shipped 0.30 m reward while the operator believed the
    sharpened value was active, and the run would look like a null result. This
    pins the hazard so `apply_plant_overrides` keeps re-asserting the value.

    Checked at source level because `terrain_env_cfg` pulls in isaaclab ->
    pxr, which is unavailable in the CPU test environment.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cfg_source = (root / "verge_task/terrain_env_cfg.py").read_text()
    train_source = (root / "verge_task/scripts/train.py").read_text()

    # The curriculum really does assign the field, so the hazard is real.
    assert "cfg.lateral_tracking_std_m = 0.30" in cfg_source

    # ... and the override is re-asserted inside the function that runs after
    # EVERY curriculum application, not once at construction.
    overrides = train_source.split("def apply_plant_overrides(cfg) -> None:")[1]
    overrides = overrides.split("\ndef ")[0]
    for field in (
        "lateral_tracking_std_m",
        "lateral_tracking_weight",
        "dense_lateral_error_weight",
    ):
        assert field in overrides, f"{field} must be re-applied after the curriculum"


def test_worst_case_tail_weighting_defaults_to_off_and_targets_the_gate_metrics():
    """CVaR on contact and lateral must be opt-in and applied to the right terms.

    The gates read worst_family_ground_contact_fraction and
    worst_family_mean_abs_lateral_error_m -- MINIMA over 11 path families --
    while PPO maximizes the MEAN. Measured over two runs and ten blocks: mean
    contact rose 0.8165 -> 0.8701 while worst-family contact never left
    0.656-0.751. Up-weighting the worst environments is what puts the gate's
    own quantity into the gradient.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg_source = (root / "terrain_env_cfg.py").read_text()
    env_source = (root / "terrain_env.py").read_text()

    # Opt-in: the shipped reward is unchanged unless explicitly enabled.
    assert "terrain_contact_tail_fraction: float = 0.0" in cfg_source
    assert "terrain_contact_tail_multiplier: float = 1.0" in cfg_source
    assert "terrain_lateral_tail_fraction: float = 0.0" in cfg_source
    assert "terrain_lateral_tail_multiplier: float = 1.0" in cfg_source

    # Applied to the reward terms the two gates actually measure.
    assert "contact_tail_weights = lower_tail_weights(" in env_source
    assert "lateral_tail_weights = lower_tail_weights(" in env_source
    assert "dense_lateral * lateral_tail_weights" in env_source
    assert "contact_penalty * contact_tail_weights" in env_source
    # Error magnitudes must be NEGATED: lower_tail_weights selects the LOWEST
    # values, but a larger error is the WORSE environment.
    assert "-contact_gap.abs()" in env_source
    assert "-lateral_error.abs()" in env_source


def test_lower_tail_weights_selects_the_worst_environments():
    """Sanity-check the direction of the helper the CVaR terms rely on."""
    import torch

    from verge_task.terrain_rewards import lower_tail_weights

    errors = torch.tensor([0.05, 0.90, 0.10, 0.80])  # envs 1 and 3 are worst
    active = torch.ones(4, dtype=torch.bool)
    weights = lower_tail_weights(
        -errors.abs(), active_mask=active, fraction=0.5, multiplier=4.0
    )
    assert weights.tolist() == [1.0, 4.0, 1.0, 4.0]

    # Disabled by default.
    off = lower_tail_weights(
        -errors.abs(), active_mask=active, fraction=0.0, multiplier=1.0
    )
    assert off.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_tail_weights_are_never_applied_to_a_positive_reward_term():
    """Scaling a POSITIVE reward for the worst environments pays them to fail.

    Measured 2026-08-04: weighting `terms.contact_reward` (positive) by 4x for
    the worst 25 % made contact quality 0.49 pay 7.84 while 0.50 paid 2.00 --
    being worse was worth nearly four times more. One 60-iteration block
    collapsed worst-family contact 0.7503 -> 0.6208, and lane_change_hold, the
    family the term existed to rescue, became the worst at 0.621.

    The invariant: a tail multiplier may only ever scale a non-positive term,
    because multiplying a penalty by >= 1 can only lower the return.
    """
    from pathlib import Path

    env_source = (
        Path(__file__).resolve().parents[1] / "terrain_env.py"
    ).read_text()

    # The two positive task rewards must be summed unweighted.
    assert "terms.contact_reward * contact_tail_weights" not in env_source
    assert "terms.lateral * lateral_tail_weights" not in env_source

    # The contact group must be split at zero before weighting.
    assert "contact_penalty = torch.clamp(contact_terms.total, max=0.0)" in env_source
    assert "contact_bonus = contact_terms.total - contact_penalty" in env_source


def test_weighting_a_penalty_can_only_reduce_return():
    """The property that makes penalty-side weighting safe."""
    import torch

    total = torch.tensor([5.0, -3.0, 0.0, -0.5])
    penalty = torch.clamp(total, max=0.0)
    bonus = total - penalty
    for multiplier in (1.0, 2.0, 8.0):
        weighted = bonus + penalty * multiplier
        assert (weighted <= total + 1e-9).all(), multiplier
