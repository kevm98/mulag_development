"""Contracts for the contact-support state machine and reward group.

The state machine lives on ``TerrainSolePolicyEnv``, which cannot be imported
without Isaac Sim. It is written against a small, explicit surface -- the
contact sensor, the cutter height, the ground height, the warm-up counters and
the step counter -- so it is exercised here by binding the unbound methods to a
stub carrying exactly that surface. If a future change makes the machine reach
for something else, these tests fail with an ``AttributeError`` naming it,
which is the intended alarm rather than an inconvenience.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
TERRAIN_ENV = ROOT / "verge_task" / "terrain_env.py"

from verge_task.terrain_rewards import (
    airborne_task_reward_bound,
    compute_contact_safety_reward,
)
from verge_task.terrain_observations import (
    CONTACT_STATE_CHANNELS,
    CONTACT_STATE_CHANNEL_NAMES,
    MODEL_ERROR_CHANNELS,
    verge_observation_dim,
)


# ----------------------------------------------------------------------
# A stub that carries only what the state machine is allowed to touch.
# ----------------------------------------------------------------------

_MACHINE_METHODS = (
    "_ensure_contact_safety_buffers",
    "_reset_contact_safety",
    "_update_contact_safety",
    "contact_safety_terminal",
)


def _load_machine():
    """Extract the state-machine methods from terrain_env.py without Isaac.

    The module imports isaaclab at the top, so it cannot simply be imported.
    Compiling just the four method bodies keeps this a test of the SHIPPED
    source rather than of a copy that can silently drift from it.
    """
    tree = ast.parse(TERRAIN_ENV.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TerrainSolePolicyEnv"
    )
    wanted = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef,)) and node.name in _MACHINE_METHODS
    ]
    assert len(wanted) == len(_MACHINE_METHODS), (
        f"expected {_MACHINE_METHODS} on TerrainSolePolicyEnv, found "
        f"{[node.name for node in wanted]}"
    )
    for node in wanted:
        # Strip @torch.no_grad(); the helper is called directly here.
        node.decorator_list = []
    module = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"torch": torch}
    exec(compile(module, str(TERRAIN_ENV), "exec"), namespace)
    return namespace


MACHINE = _load_machine()


def _cfg(**overrides):
    base = dict(
        cutter_contact_force_threshold_n=10.0,
        cutter_contact_reference_height_m=-0.30,
        terrain_contact_acquire_steps=3,
        terrain_contact_loss_grace_steps=2,
        terrain_max_airborne_gap_m=0.005,
        terrain_takeoff_velocity_threshold_mps=0.02,
        terrain_hard_impact_force_n=30000.0,
        terrain_impact_ejection_force_n=30000.0,
        terrain_max_takeoff_events=0,
        terrain_terminate_on_contact_loss=False,
        terrain_terminate_on_hard_impact=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class Machine:
    """Stub environment exposing exactly the state machine's dependencies."""

    def __init__(self, *, num_envs: int = 1, warmup: int = 0, **cfg_overrides):
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = 1.0 / 60.0
        self.cfg = _cfg(**cfg_overrides)
        self.common_step_counter = 0
        self._preposition_counter = torch.full((num_envs,), int(warmup))
        self._settle_counter = torch.zeros(num_envs, dtype=torch.long)
        # Scoring is suppressed once the truck finishes its path, alongside
        # preposition and settle. Default False = the run is being scored,
        # which is the condition every test here exercises.
        self._truck_path_complete = torch.zeros(num_envs, dtype=torch.bool)
        self._contact_acquired = None
        # Driven by the test, read by the machine.
        self._force = torch.zeros(num_envs)
        self._cutter_z = torch.zeros(num_envs)
        self._ground_z = torch.full((num_envs,), -0.30)
        for name in _MACHINE_METHODS:
            setattr(self, name, MACHINE[name].__get__(self, Machine))

    # --- the measured surface ---
    def cutter_terrain_contact_force_n(self):
        return self._force

    def _cutter_world_z(self):
        return self._cutter_z

    def ground_height_under_cutter(self):
        return self._ground_z

    # --- test driver ---
    def advance(self, *, force, height_above_support=0.0, warmup=None):
        """Run one control step with the given contact force and geometry."""
        self.common_step_counter += 1
        if warmup is not None:
            self._preposition_counter = torch.full((self.num_envs,), int(warmup))
        self._force = torch.as_tensor(force, dtype=torch.float32).expand(
            self.num_envs
        ).clone()
        # ground + reference + requested gap; the machine subtracts the
        # reference itself, so this makes `height_above_support` the gap.
        self._cutter_z = (
            self._ground_z
            + (-0.30)
            + torch.as_tensor(
                height_above_support, dtype=torch.float32
            ).expand(self.num_envs)
        )
        self._update_contact_safety()


# ----------------------------------------------------------------------
# Acquisition hysteresis
# ----------------------------------------------------------------------


def test_support_is_acquired_only_after_the_configured_run_of_contact():
    m = Machine(terrain_contact_acquire_steps=3)
    for _ in range(2):
        m.advance(force=5000.0)
        assert not bool(m._contact_acquired[0])
    m.advance(force=5000.0)
    assert bool(m._contact_acquired[0])


def test_a_broken_run_restarts_acquisition():
    m = Machine(terrain_contact_acquire_steps=3)
    m.advance(force=5000.0)
    m.advance(force=5000.0)
    m.advance(force=0.0)
    m.advance(force=5000.0)
    m.advance(force=5000.0)
    assert not bool(m._contact_acquired[0]), (
        "two contact steps either side of a gap must not add up to acquisition"
    )
    m.advance(force=5000.0)
    assert bool(m._contact_acquired[0])


def test_support_is_not_acquired_during_preposition_or_settling():
    m = Machine(warmup=5, terrain_contact_acquire_steps=1)
    for _ in range(10):
        m.advance(force=50000.0, height_above_support=0.0)
    assert not bool(m._contact_acquired[0]), (
        "the reset ramp presses the cutter into the verge; that is not the "
        "policy acquiring support"
    )
    assert int(m._contact_loss_event_count[0]) == 0
    assert int(m._hard_impact_event_count[0]) == 0


def test_no_termination_during_preposition_or_settling():
    m = Machine(
        warmup=5,
        terrain_contact_acquire_steps=1,
        terrain_contact_loss_grace_steps=0,
        terrain_terminate_on_contact_loss=True,
        terrain_terminate_on_hard_impact=True,
    )
    for _ in range(10):
        m.advance(force=200000.0, height_above_support=0.5)
    assert not bool(m.contact_safety_terminal()[0])


# ----------------------------------------------------------------------
# Loss counting and the per-step cache
# ----------------------------------------------------------------------


def test_contact_loss_counts_one_event_per_transition():
    m = Machine(terrain_contact_acquire_steps=1)
    m.advance(force=5000.0)
    for _ in range(5):
        m.advance(force=0.0, height_above_support=0.05)
    assert int(m._contact_loss_event_count[0]) == 1, (
        "five airborne steps are one separation, not five"
    )
    assert int(m._max_consecutive_airborne_steps[0]) == 5


def test_repeated_calls_in_one_step_do_not_double_count():
    m = Machine(terrain_contact_acquire_steps=1)
    m.advance(force=5000.0)
    m.advance(force=0.0, height_above_support=0.05)
    before = int(m._contact_loss_event_count[0])
    # DirectRLEnv calls dones, rewards and observations in the same step.
    for _ in range(3):
        m._update_contact_safety()
    assert int(m._contact_loss_event_count[0]) == before == 1


def test_grace_window_keeps_support_across_an_isolated_missed_sample():
    m = Machine(terrain_contact_acquire_steps=1, terrain_contact_loss_grace_steps=2)
    m.advance(force=5000.0)
    m.advance(force=0.0, height_above_support=0.0)
    assert bool(m._contact_support[0]), "one missed sample must not drop support"
    m.advance(force=0.0, height_above_support=0.0)
    assert bool(m._contact_support[0])
    m.advance(force=0.0, height_above_support=0.0)
    assert not bool(m._contact_support[0])


# ----------------------------------------------------------------------
# Event detection
# ----------------------------------------------------------------------


def test_takeoff_requires_upward_motion_at_separation():
    rising = Machine(terrain_contact_acquire_steps=1)
    rising.advance(force=5000.0, height_above_support=0.0)
    rising.advance(force=0.0, height_above_support=0.02)
    assert bool(rising._takeoff_event[0])

    falling = Machine(terrain_contact_acquire_steps=1)
    falling.advance(force=5000.0, height_above_support=0.02)
    falling.advance(force=0.0, height_above_support=0.0)
    assert not bool(falling._takeoff_event[0]), (
        "ground falling away under a descending cutter is not a takeoff"
    )


def test_impact_ejection_requires_a_high_force_before_the_takeoff():
    ejected = Machine(terrain_contact_acquire_steps=1)
    ejected.advance(force=50000.0, height_above_support=0.0)
    ejected.advance(force=0.0, height_above_support=0.02)
    assert bool(ejected._impact_ejection_event[0])

    gentle = Machine(terrain_contact_acquire_steps=1)
    gentle.advance(force=2000.0, height_above_support=0.0)
    gentle.advance(force=0.0, height_above_support=0.02)
    assert bool(gentle._takeoff_event[0])
    assert not bool(gentle._impact_ejection_event[0]), (
        "a takeoff off a light contact is not an ejection"
    )


def test_hard_impact_is_detected_on_force_alone():
    m = Machine(terrain_hard_impact_force_n=30000.0)
    m.advance(force=29000.0)
    assert not bool(m._hard_impact_event[0])
    m.advance(force=31000.0)
    assert bool(m._hard_impact_event[0])
    assert int(m._hard_impact_event_count[0]) == 1


def test_event_counters_count_edges_not_samples():
    """"1559 hard impacts" was really a 5 % duty cycle wearing an event name.

    ``hard_impact`` and ``visible_lift`` are per-step LEVELS, because that is
    what the reward and the termination need. Accumulating a level gives a
    sample count, not an event count, and the first measured run reported
    exactly that.
    """
    m = Machine(terrain_hard_impact_force_n=30000.0)
    for _ in range(10):
        m.advance(force=50000.0)
    assert int(m._hard_impact_event_count[0]) == 1, "one sustained impact is one event"
    assert int(m._hard_impact_steps[0]) == 10, "and ten steps of duty"

    m.advance(force=1000.0)
    m.advance(force=50000.0)
    assert int(m._hard_impact_event_count[0]) == 2, "a new rise is a new event"
    assert int(m._hard_impact_steps[0]) == 11


def test_visible_lift_counter_counts_edges_too():
    m = Machine(
        terrain_contact_acquire_steps=1, terrain_contact_loss_grace_steps=0
    )
    m.advance(force=5000.0, height_above_support=0.0)
    for _ in range(6):
        m.advance(force=0.0, height_above_support=0.05)
    assert int(m._visible_lift_event_count[0]) == 1
    assert int(m._visible_lift_steps[0]) == 6


def test_visible_lift_needs_both_a_geometric_gap_and_a_real_loss():
    # Sensor-only miss: no gap. Must not be a visible lift at any grace.
    sensor = Machine(
        terrain_contact_acquire_steps=1, terrain_contact_loss_grace_steps=0
    )
    sensor.advance(force=5000.0, height_above_support=0.0)
    sensor.advance(force=0.0, height_above_support=0.0)
    assert not bool(sensor._visible_lift_event[0])
    assert float(sensor._airborne_gap_m[0]) == pytest.approx(0.0, abs=1e-6)

    lifted = Machine(
        terrain_contact_acquire_steps=1, terrain_contact_loss_grace_steps=0
    )
    lifted.advance(force=5000.0, height_above_support=0.0)
    lifted.advance(force=0.0, height_above_support=0.04)
    assert bool(lifted._visible_lift_event[0])
    assert float(lifted._max_airborne_gap_m[0]) == pytest.approx(0.04, abs=1e-5)


def test_a_gap_inside_the_grace_window_is_not_yet_a_lift():
    m = Machine(terrain_contact_acquire_steps=1, terrain_contact_loss_grace_steps=2)
    m.advance(force=5000.0, height_above_support=0.0)
    m.advance(force=0.0, height_above_support=0.04)
    assert not bool(m._visible_lift_event[0])
    m.advance(force=0.0, height_above_support=0.04)
    assert not bool(m._visible_lift_event[0])
    m.advance(force=0.0, height_above_support=0.04)
    assert bool(m._visible_lift_event[0])


# ----------------------------------------------------------------------
# Termination
# ----------------------------------------------------------------------


def test_airborne_gap_and_sustained_loss_terminate_when_enabled():
    m = Machine(
        terrain_contact_acquire_steps=1,
        terrain_contact_loss_grace_steps=1,
        terrain_terminate_on_contact_loss=True,
    )
    m.advance(force=5000.0, height_above_support=0.0)
    m.advance(force=0.0, height_above_support=0.04)
    assert not bool(m.contact_safety_terminal()[0]), "inside the grace window"
    m.advance(force=0.0, height_above_support=0.04)
    assert bool(m.contact_safety_terminal()[0])


def test_termination_is_off_by_default():
    m = Machine(terrain_contact_acquire_steps=1)
    m.advance(force=5000.0, height_above_support=0.0)
    for _ in range(20):
        m.advance(force=0.0, height_above_support=0.5)
    assert not bool(m.contact_safety_terminal()[0]), (
        "every stored PPO3/PPO4 report must stay reproducible by default"
    )


def test_hard_impact_termination_is_independent_of_contact_loss():
    m = Machine(
        terrain_contact_acquire_steps=1, terrain_terminate_on_hard_impact=True
    )
    m.advance(force=200000.0, height_above_support=0.0)
    assert bool(m.contact_safety_terminal()[0])


def test_repeated_ejection_termination():
    m = Machine(
        terrain_contact_acquire_steps=1,
        terrain_max_takeoff_events=2,
        terrain_terminate_on_contact_loss=False,
    )
    for _ in range(2):
        m.advance(force=50000.0, height_above_support=0.0)
        m.advance(force=0.0, height_above_support=0.02)
    assert int(m._impact_ejection_event_count[0]) == 2
    assert bool(m.contact_safety_terminal()[0])


# ----------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------


def test_reset_clears_every_contact_variable():
    m = Machine(num_envs=2, terrain_contact_acquire_steps=1)
    m.advance(force=50000.0, height_above_support=0.0)
    m.advance(force=0.0, height_above_support=0.10)
    assert bool(m._contact_acquired[0])

    m._reset_contact_safety(torch.tensor([0]))
    assert not bool(m._contact_acquired[0])
    assert int(m._contact_loss_event_count[0]) == 0
    assert int(m._hard_impact_event_count[0]) == 0
    assert int(m._max_consecutive_airborne_steps[0]) == 0
    assert float(m._max_airborne_gap_m[0]) == 0.0
    assert torch.isnan(m._previous_cutter_z[0]), (
        "seeding the previous height with 0.0 would differentiate the deployed "
        "cutter against the world origin and manufacture a takeoff"
    )
    # The second environment is untouched.
    assert bool(m._contact_acquired[1])


def test_the_first_step_after_reset_reports_no_vertical_velocity():
    m = Machine(terrain_contact_acquire_steps=1)
    m.advance(force=5000.0, height_above_support=0.0)
    m._reset_contact_safety(torch.tensor([0]))
    m.advance(force=5000.0, height_above_support=1.0)
    assert float(m._cutter_vertical_velocity_mps[0]) == pytest.approx(0.0)


def test_reset_does_not_invalidate_the_per_step_cache():
    """`_reset_idx` runs mid-step, between rewards and observations.

    Clearing the marker there would make the observation call perform a second
    update for that control step, for every environment rather than just the
    reset ones.
    """
    m = Machine(terrain_contact_acquire_steps=1)
    m.advance(force=5000.0, height_above_support=0.0)
    marker = m._contact_safety_step
    m._reset_contact_safety(torch.tensor([0]))
    assert m._contact_safety_step == marker, (
        "the reset invalidated the per-step cache; the observation call would "
        "run a second update for this step"
    )
    # And the observation-side call is then a no-op that reads cleared state.
    m._update_contact_safety()
    assert not bool(m._contact_acquired[0])
    assert int(m._contact_loss_event_count[0]) == 0


# ----------------------------------------------------------------------
# Reward group
# ----------------------------------------------------------------------


def _reward(**overrides):
    n = overrides.pop("n", 1)
    kwargs = dict(
        contact=torch.ones(n, dtype=torch.bool),
        contact_acquired=torch.ones(n, dtype=torch.bool),
        contact_force_n=torch.full((n,), 6000.0),
        airborne_gap_m=torch.zeros(n),
        cutter_vertical_velocity_mps=torch.zeros(n),
        hard_impact=torch.zeros(n, dtype=torch.bool),
        active=torch.ones(n, dtype=torch.bool),
    )
    kwargs.update(overrides)
    return compute_contact_safety_reward(**kwargs)


def test_losing_acquired_support_costs_more_than_any_airborne_task_reward():
    bound = airborne_task_reward_bound(
        cut_tracking_weight=8.0,
        lateral_weight=3.0,
        cut_progress_weight=1.0,
        lateral_progress_weight=0.5,
    )
    penalty = -float(
        _reward(contact=torch.zeros(1, dtype=torch.bool)).contact_loss[0]
    )
    assert penalty > bound, (
        f"contact-loss penalty {penalty} does not exceed the {bound} a floating "
        "environment can still collect; flying through a rise stays profitable"
    )


def test_stable_support_scores_better_than_a_violent_one():
    stable = float(_reward(contact_force_n=torch.full((1,), 6000.0)).total[0])
    violent = float(
        _reward(
            contact_force_n=torch.full((1,), 120000.0),
            hard_impact=torch.ones(1, dtype=torch.bool),
        ).total[0]
    )
    assert stable > violent, (
        "a binary contact bonus pays these identically; that is the defect"
    )


def test_force_reward_is_a_band_not_a_ramp():
    low = float(_reward(contact_force_n=torch.full((1,), 1000.0)).force_regulation[0])
    peak = float(_reward(contact_force_n=torch.full((1,), 6000.0)).force_regulation[0])
    high = float(_reward(contact_force_n=torch.full((1,), 40000.0)).force_regulation[0])
    assert peak > low and peak > high, "pressing harder must stop paying"


def test_force_reward_retains_gradient_at_the_measured_operating_point():
    # Measured stable-contact force is 18.9 kN mean. A band the policy sits
    # outside of is the reward-saturation defect the head penalty had.
    here = float(_reward(contact_force_n=torch.full((1,), 18900.0)).force_regulation[0])
    nearer = float(
        _reward(contact_force_n=torch.full((1,), 17900.0)).force_regulation[0]
    )
    assert nearer > here > 0.0, "no gradient where the policy actually lives"


def test_gap_penalty_is_continuous_in_the_gap():
    small = float(_reward(airborne_gap_m=torch.full((1,), 0.004)).airborne_gap[0])
    large = float(_reward(airborne_gap_m=torch.full((1,), 0.040)).airborne_gap[0])
    assert large < small <= 0.0, (
        "a binary term scores 4 mm and 40 mm identically and leaves no gradient "
        "back down"
    )


def test_takeoff_penalty_only_applies_under_weak_support():
    firm = float(
        _reward(
            contact_force_n=torch.full((1,), 6000.0),
            cutter_vertical_velocity_mps=torch.full((1,), 0.5),
        ).takeoff[0]
    )
    weak = float(
        _reward(
            contact=torch.zeros(1, dtype=torch.bool),
            cutter_vertical_velocity_mps=torch.full((1,), 0.5),
        ).takeoff[0]
    )
    assert firm == 0.0, "a supported cutter following a rising verge is the task"
    assert weak < 0.0


def test_downward_velocity_is_never_a_takeoff():
    assert float(
        _reward(
            contact=torch.zeros(1, dtype=torch.bool),
            cutter_vertical_velocity_mps=torch.full((1,), -2.0),
        ).takeoff[0]
    ) == 0.0


def test_every_term_is_bounded_and_finite_at_extremes():
    terms = _reward(
        n=1,
        contact=torch.zeros(1, dtype=torch.bool),
        contact_force_n=torch.full((1,), 1.0e9),
        airborne_gap_m=torch.full((1,), 1.0e3),
        cutter_vertical_velocity_mps=torch.full((1,), 1.0e3),
        hard_impact=torch.ones(1, dtype=torch.bool),
    )
    for name in (
        "total",
        "support",
        "contact_loss",
        "force_regulation",
        "excess_force",
        "hard_impact",
        "takeoff",
        "airborne_gap",
    ):
        value = getattr(terms, name)
        assert torch.isfinite(value).all(), f"{name} is non-finite at the extreme"
        assert float(value.abs().max()) < 1.0e4, f"{name} is unbounded"


def test_nonfinite_force_does_not_poison_the_reward():
    terms = _reward(contact_force_n=torch.full((1,), float("nan")))
    assert torch.isfinite(terms.total).all()


def test_the_whole_group_is_zero_during_warmup():
    terms = _reward(
        active=torch.zeros(1, dtype=torch.bool),
        contact=torch.zeros(1, dtype=torch.bool),
        airborne_gap_m=torch.full((1,), 0.5),
        hard_impact=torch.ones(1, dtype=torch.bool),
    )
    assert float(terms.total[0]) == 0.0


# ----------------------------------------------------------------------
# Observation width
# ----------------------------------------------------------------------


def test_reward_defaults_match_the_config_defaults():
    """One source of truth for every weight and threshold.

    ``compute_contact_safety_reward`` is unit-tested through its signature
    defaults and run in production through the config. If those two disagree,
    every test above is measuring a term the environment never computes.
    """
    import inspect
    import re

    signature = inspect.signature(compute_contact_safety_reward)
    cfg_source = (
        ROOT / "verge_task" / "terrain_env_cfg.py"
    ).read_text()
    pairs = {
        "support_weight": "terrain_contact_support_weight",
        "loss_weight": "terrain_contact_loss_weight",
        "force_weight": "terrain_contact_force_weight",
        "force_target_n": "terrain_contact_force_target_n",
        "force_std_n": "terrain_contact_force_std_n",
        "excess_weight": "terrain_contact_excess_weight",
        "force_soft_limit_n": "terrain_contact_force_soft_limit_n",
        "excess_max_ratio": "terrain_contact_excess_max_ratio",
        "hard_impact_weight": "terrain_contact_hard_impact_weight",
        "takeoff_weight": "terrain_contact_takeoff_weight",
        "takeoff_velocity_scale_mps": "terrain_contact_takeoff_velocity_scale_mps",
        "takeoff_max_ratio": "terrain_contact_takeoff_max_ratio",
        "weak_support_force_n": "terrain_contact_weak_support_force_n",
        "gap_weight": "terrain_contact_gap_weight",
        "gap_scale_m": "terrain_contact_gap_scale_m",
        "gap_max_ratio": "terrain_contact_gap_max_ratio",
    }
    for parameter, field in pairs.items():
        match = re.search(rf"^    {field}: float = ([-\d.e]+)$", cfg_source, re.M)
        assert match, f"{field} is not declared in terrain_env_cfg.py"
        assert float(match.group(1)) == pytest.approx(
            float(signature.parameters[parameter].default)
        ), f"{parameter} default disagrees with {field}"


def test_contact_channels_widen_the_observation_by_exactly_six():
    base = verge_observation_dim(model_error=True, model_error_history=1)
    with_contact = verge_observation_dim(
        model_error=True, model_error_history=1, contact_state=True
    )
    assert with_contact - base == CONTACT_STATE_CHANNELS == 6
    assert len(CONTACT_STATE_CHANNEL_NAMES) == CONTACT_STATE_CHANNELS


def test_model_error_history_is_counted_in_the_width():
    assert (
        verge_observation_dim(model_error=True, model_error_history=3)
        - verge_observation_dim(model_error=True, model_error_history=1)
        == 2 * MODEL_ERROR_CHANNELS
    )


def test_the_contact_safe_v1_width_is_ninety_six():
    """9 profile + 44 + 12x3 model error + 6 contact + 1 trailing."""
    assert (
        verge_observation_dim(
            model_error=True, model_error_history=3, contact_state=True
        )
        + 1
        == 96
    )


# ----------------------------------------------------------------------
# The contact machinery must not become a controller
# ----------------------------------------------------------------------

_ENV_SOURCE = TERRAIN_ENV.read_text()


def test_the_state_machine_writes_no_plant_command():
    """PPO stays the sole controller.

    A contact state machine that nudged a cylinder target, a valve command or
    a joint reference would be a force controller underneath the policy, which
    is exactly what this work is forbidden to add. Checked structurally on the
    four methods rather than trusted to review.
    """
    tree = ast.parse(_ENV_SOURCE)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TerrainSolePolicyEnv"
    )
    forbidden = (
        "_cylinder_pos_target",
        "_corrected_cylinder_goal",
        "_joint_ref",
        "_joint_vel_ref",
        "_current_valve_cmd",
        "_fd_delta_corrected",
        "_sole_applied",
        "_sole_raw",
        "_baseline_cylinder_goal",
    )
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in _MACHINE_METHODS:
            continue
        for inner in ast.walk(node):
            targets = []
            if isinstance(inner, ast.Assign):
                targets = inner.targets
            elif isinstance(inner, (ast.AugAssign, ast.AnnAssign)):
                targets = [inner.target]
            for target in targets:
                while isinstance(target, ast.Subscript):
                    target = target.value
                if isinstance(target, ast.Attribute) and target.attr in forbidden:
                    raise AssertionError(
                        f"{node.name} writes {target.attr}; the contact state "
                        "machine must observe, never command"
                    )


def test_dones_rewards_and_observations_all_share_one_update():
    """Three consumers, one transition.

    ``DirectRLEnv`` calls dones, rewards and observations in that order after a
    single ``common_step_counter`` increment. If any of them ran its own update
    the termination, the penalty and the observation would describe different
    steps.
    """
    tree = ast.parse(_ENV_SOURCE)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TerrainSolePolicyEnv"
    )
    methods = {
        node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
    }
    for name in ("_get_dones", "_get_rewards", "_contact_state_channels"):
        assert name in methods, f"{name} is missing"
        called = {
            inner.func.attr
            for inner in ast.walk(methods[name])
            if isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
        }
        assert called & {"_update_contact_safety", "contact_safety_terminal"}, (
            f"{name} does not go through the shared contact update"
        )
    assert "self._contact_safety_step = marker" in _ENV_SOURCE, (
        "the per-step cache marker is gone; events would be counted three times"
    )


def test_the_contact_loss_latch_is_the_affordable_form_of_termination():
    """Termination cannot do this job here, and that was MEASURED.

    With `terrain_terminate_on_contact_loss = True` at 256 environments, a
    per-step contact-termination rate of 0.0039 collapsed the mean episode
    length to 10-11 steps of a 960-step horizon, because resets are
    all-or-nothing. The latch gives the same incentive at zero reset cost.
    """
    m = Machine(
        terrain_contact_acquire_steps=1,
        terrain_contact_loss_grace_steps=0,
        terrain_contact_loss_latches_failure=True,
        terrain_contact_loss_latch_release_steps=0,
    )
    m.advance(force=5000.0, height_above_support=0.0)
    assert not bool(m._contact_loss_failed[0])
    m.advance(force=0.0, height_above_support=0.05)
    assert bool(m._contact_loss_failed[0]), "a visible lift must latch"
    # Regaining contact does not clear a monotone latch.
    for _ in range(50):
        m.advance(force=5000.0, height_above_support=0.0)
    assert bool(m._contact_loss_failed[0])


def test_the_latch_releases_after_sustained_recovery():
    m = Machine(
        terrain_contact_acquire_steps=1,
        terrain_contact_loss_grace_steps=0,
        terrain_contact_loss_latches_failure=True,
        terrain_contact_loss_latch_release_steps=5,
    )
    m.advance(force=5000.0, height_above_support=0.0)
    m.advance(force=0.0, height_above_support=0.05)
    assert bool(m._contact_loss_failed[0])
    for _ in range(4):
        m.advance(force=5000.0, height_above_support=0.0)
    assert bool(m._contact_loss_failed[0]), "released too early"
    m.advance(force=5000.0, height_above_support=0.0)
    assert not bool(m._contact_loss_failed[0])
    assert bool(m._contact_loss_latch_released[0])


def test_the_latch_is_off_by_default():
    m = Machine(terrain_contact_acquire_steps=1, terrain_contact_loss_grace_steps=0)
    m.advance(force=5000.0, height_above_support=0.0)
    m.advance(force=0.0, height_above_support=0.05)
    assert not bool(m._contact_loss_failed[0])


def test_reset_clears_the_latch():
    m = Machine(
        terrain_contact_acquire_steps=1,
        terrain_contact_loss_grace_steps=0,
        terrain_contact_loss_latches_failure=True,
    )
    m.advance(force=5000.0, height_above_support=0.0)
    m.advance(force=0.0, height_above_support=0.05)
    assert bool(m._contact_loss_failed[0])
    m._reset_contact_safety(torch.tensor([0]))
    assert not bool(m._contact_loss_failed[0])
    assert int(m._contact_recovery_steps[0]) == 0


def test_the_latch_zeroes_the_task_reward_in_get_rewards():
    """The latch must act where the head-down latch acts, on task reward only.

    Zeroing the safety group too would stop charging for the lift and stop
    paying for recovery, which is the opposite of what it is for.
    """
    body = _ENV_SOURCE[_ENV_SOURCE.index("def _get_rewards(self)") :]
    latch = body.index("terrain_contact_loss_latches_failure")
    assign = body.index("task_reward = torch.where(", latch)
    safety = body.index("safety_reward = (", latch)
    assert assign < safety, "the latch does not zero task reward before the sum"
    assert "self._contact_loss_failed" in body[latch : latch + 400]


def test_contact_termination_synchronizes_like_every_other_termination():
    """A partial reset does not merely reset less -- it CRASHES.

    ``_run_reset_preposition`` raises when handed fewer than num_envs
    environments, because the physical deployment ramp steps the whole scene.
    A contact termination that fired on a subset would take the run down with
    a RuntimeError instead of resetting those environments.
    """
    parent = (
        ROOT / "mulag_plant" / "policy" / "env.py"
    ).read_text()
    assert "Residual-policy resets are synchronized" in parent, (
        "the partial-reset guard this synchronization exists for is gone; "
        "re-derive whether contact terminations still have to synchronize"
    )
    tree = ast.parse(_ENV_SOURCE)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TerrainSolePolicyEnv"
    )
    dones = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_dones"
    )
    # Assert the BEHAVIOUR, not the spelling: some `.any()` reduction must
    # depend on contact_terminal. The production form combines it with the
    # nonfinite mask -- `(nonfinite | contact_terminal).any()` -- which
    # synchronizes exactly as required, so a literal "contact_terminal.any()"
    # substring check would fail on correct code.
    def _mentions(node, name: str) -> bool:
        return any(
            isinstance(n, ast.Name) and n.id == name for n in ast.walk(node)
        )

    any_reductions = [
        n for n in ast.walk(dones)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "any"
    ]
    assert any(_mentions(call.func.value, "contact_terminal") for call in any_reductions), (
        "_get_dones does not synchronize on a contact termination: no .any() "
        "reduction depends on contact_terminal"
    )
    source = ast.unparse(dones)
    assert "torch.ones_like(terminated)" in source


def test_reset_path_clears_contact_state():
    assert "self._reset_contact_safety(env_ids)" in _ENV_SOURCE, (
        "a new episode would inherit the previous one's support claim"
    )


def test_the_evaluator_measures_contact_loss_and_never_terminates_on_it():
    """A gate that shortens its own evidence cannot size an excursion."""
    source = (ROOT / "verge_task" / "scripts" / "evaluate.py").read_text()
    assert "env_cfg.terrain_terminate_on_contact_loss = False" in source
    assert "env_cfg.terrain_terminate_on_hard_impact = False" in source
    assert "env_cfg.enable_contact_tracking_demo = False" in source
    assert "assert_teacher_disabled(env_cfg" in source


def test_strict_gates_are_reported_separately_from_the_historical_seventeen():
    """Every stored report was scored against 17 gates.

    Folding nine more into GATE_SPECS would change the denominator of every
    historical "16/17" in this repository.
    """
    from verge_task.terrain_convergence import (
        CONTACT_SAFETY_GATE_SPECS,
        GATE_SPECS,
        active_gate_specs,
    )

    assert len(GATE_SPECS) == 17
    assert not set(GATE_SPECS) & set(CONTACT_SAFETY_GATE_SPECS)
    assert len(active_gate_specs()) == 17
    assert len(active_gate_specs(contact_safety=True)) == 17 + len(
        CONTACT_SAFETY_GATE_SPECS
    )


def test_promotion_requires_zero_lift_and_a_teacher_free_run():
    from verge_task.terrain_convergence import (
        contact_safety_promotion_blockers,
    )

    clean = {
        "visible_lift_event_count": 0,
        "impact_ejection_event_count": 0,
        "hard_impact_event_count": 0,
        "maximum_consecutive_airborne_steps": 0,
        "maximum_airborne_gap_m": 0.0,
        "post_acquisition_contact_fraction": 1.0,
        "contact_acquired_fraction": 1.0,
        "takeoff_event_count": 0,
        "maximum_contact_force_n": 12000.0,
        "teacher_mode": "off",
        "direct_target_contact_demo": False,
    }
    assert contact_safety_promotion_blockers(clean) == []

    # The pre-fix PPO3 measurement must be rejected on multiple grounds.
    measured = dict(clean)
    measured.update(
        {
            "visible_lift_event_count": 334,
            "impact_ejection_event_count": 201,
            "hard_impact_event_count": 286,
            "maximum_consecutive_airborne_steps": 45,
            "maximum_airborne_gap_m": 0.2966,
            "post_acquisition_contact_fraction": 0.9527,
            "takeoff_event_count": 314,
            "maximum_contact_force_n": 119278.0,
        }
    )
    blockers = contact_safety_promotion_blockers(measured)
    assert "visible_lift_events" in blockers
    assert "impact_ejection_events" in blockers
    assert "consecutive_airborne_steps" in blockers

    # A teacher-executed or demo-driven run is never promotable, whatever its
    # contact numbers say.
    assert contact_safety_promotion_blockers(
        {**clean, "teacher_mode": "execute"}
    ) == ["teacher_mode='execute' (must be 'off')"]
    assert contact_safety_promotion_blockers(
        {**clean, "direct_target_contact_demo": True}
    ) == ["direct target-writing contact demo was active"]


def test_exact_zero_gate_scores_event_counts_without_epsilon_explosion():
    """No 1e12 blow-up, and no unbounded raw count either.

    The original defect was dividing a count by epsilon (one lift scored 1e12).
    That was fixed with the raw excess, which then created the opposite
    problem: an unbounded count drowning the order-0-2 relative violations of
    every other gate. The contract now is log1p -- zero stays zero, ordering is
    preserved, and the magnitude stays comparable with the ratio gates.
    """
    import math

    from verge_task.terrain_convergence import GateSpec

    gate = GateSpec("events", 0.0, "max")
    assert gate.normalized_violation(0.0) == 0.0
    assert gate.normalized_violation(3.0) == math.log1p(3.0)
    # Strictly increasing, and nowhere near the epsilon explosion.
    assert gate.normalized_violation(3.0) > gate.normalized_violation(2.0)
    assert gate.normalized_violation(1.0e6) < 20.0


def test_supervisor_forwards_width_changing_flags_to_both_children():
    """A block and its evaluation must build the same network.

    ``--contact_observation`` and ``--model_error_history`` change the
    observation width. Forwarded to training only, every block would be scored
    by an evaluator constructing a differently shaped actor.
    """
    import importlib.util

    script = ROOT / "verge_task" / "scripts" / "train_until_converged.py"
    spec = importlib.util.spec_from_file_location("train_until_converged", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--resume", "/tmp/seed.pt",
            "--contact_observation", "true",
            "--contact_safety_reward", "true",
            "--model_error_history", "3",
            "--action_scale", "0.0035",
            "--cylinder_target_max_lead_m", "0.010",
            "--contact_focus",
            "--contact_stage", "B",
            "--strict_contact",
        ]
    )
    train = module.training_command(
        args,
        resume=Path("/tmp/seed.pt"),
        checkpoint_out=Path("/tmp/out.pt"),
        block_index=1,
    )
    evaluate = module.evaluation_command(
        args, checkpoint=Path("/tmp/out.pt"), report=Path("/tmp/r.json")
    )
    for flag, value in (
        ("--contact_observation", "true"),
        ("--contact_safety_reward", "true"),
        ("--model_error_history", "3"),
        ("--action_scale", "0.0035"),
        ("--cylinder_target_max_lead_m", "0.010"),
    ):
        forwarded = {}
        for name, command in (("train", train), ("evaluate", evaluate)):
            assert flag in command, f"{name} command is missing {flag}"
            forwarded[name] = command[command.index(flag) + 1]
        assert forwarded["train"] == forwarded["evaluate"], (
            f"{flag} differs between training and evaluation: {forwarded}"
        )
        # Compared numerically where it is a number: str(float("0.010")) is
        # "0.01", and a string comparison would fail on formatting rather than
        # on the thing that matters.
        try:
            assert float(forwarded["train"]) == pytest.approx(float(value))
        except ValueError:
            assert forwarded["train"] == value
    # Training-only: the evaluator measures lift and must never terminate on it.
    assert "--contact_focus" in train and "--contact_focus" not in evaluate
    assert "--contact_stage" in train and "--contact_stage" not in evaluate
    assert "--strict_contact" in evaluate and "--strict_contact" not in train


def test_supervisor_forwards_floating_cutter_plant_to_both_children():
    module = _supervisor()
    args = module.build_parser().parse_args(
        [
            "--resume", "/tmp/seed.pt",
            "--floating_cutter",
            "--floating_cutter_stiffness", "400000",
            "--floating_cutter_damping", "22000",
            "--floating_cutter_effort_limit", "25000",
        ]
    )
    train = module.training_command(
        args,
        resume=Path("/tmp/seed.pt"),
        checkpoint_out=Path("/tmp/out.pt"),
        block_index=1,
    )
    evaluate = module.evaluation_command(
        args, checkpoint=Path("/tmp/out.pt"), report=Path("/tmp/r.json")
    )
    for command in (train, evaluate):
        assert "--floating_cutter" in command
        assert command[command.index("--floating_cutter_stiffness") + 1] == "400000.0"
        assert command[command.index("--floating_cutter_damping") + 1] == "22000.0"
        assert command[command.index("--floating_cutter_effort_limit") + 1] == "25000.0"


def test_supervisor_forwards_adaptive_trajectory_contract_to_both_children():
    module = _supervisor()
    args = module.build_parser().parse_args(
        [
            "--resume", "/tmp/seed.pt",
            "--adaptive_trajectories",
            "--adaptive_episode_length", "40",
            "--adaptive_amplitude", "8",
            "--adaptive_frequency", "1.6",
            "--adaptive_frequency_jitter", "0.2",
            "--eval_steps", "2400",
        ]
    )
    train = module.training_command(
        args,
        resume=Path("/tmp/seed.pt"),
        checkpoint_out=Path("/tmp/out.pt"),
        block_index=1,
    )
    evaluate = module.evaluation_command(
        args, checkpoint=Path("/tmp/out.pt"), report=Path("/tmp/r.json")
    )
    for command in (train, evaluate):
        assert "--adaptive_trajectories" in command
        assert command[command.index("--adaptive_episode_length") + 1] == "40.0"
        assert command[command.index("--adaptive_amplitude") + 1] == "8.0"
        assert command[command.index("--adaptive_frequency") + 1] == "1.6"
        assert command[command.index("--adaptive_frequency_jitter") + 1] == "0.2"


def _supervisor():
    import importlib.util

    script = ROOT / "verge_task" / "scripts" / "train_until_converged.py"
    spec = importlib.util.spec_from_file_location("train_until_converged", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_migrated_checkpoint_resets_the_optimizer_on_EVERY_block(tmp_path):
    """Including block 1 of a later stage.

    `copy_checkpoint` keeps the checkpoint's infos, so if no block beats the
    initial score then model_best.pt is still the migrated seed -- and the next
    STAGE is a fresh supervisor run whose block 1 does not pass
    --reset_optimizer_first_block. Measured: PPO3 stage B1 kept the migrated
    seed as its best, and stage B2 block 1 died on "migrated shared-policy
    checkpoints require --reset_optimizer_on_resume". Restricting the check to
    block_index != 1 fixed only the earlier within-stage rollback case.
    """
    module = _supervisor()
    migrated = tmp_path / "seed.pt"
    torch.save({"infos": {"migration_requires_optimizer_reset": True}}, migrated)
    args = module.build_parser().parse_args(["--resume", str(migrated)])
    for block in (1, 2, 5):
        command = module.training_command(
            args,
            resume=migrated,
            checkpoint_out=tmp_path / "o.pt",
            block_index=block,
        )
        assert "--reset_optimizer_on_resume" in command, (
            f"block {block} resumes a migrated checkpoint without the flag"
        )
        assert command.count("--reset_optimizer_on_resume") == 1, (
            f"block {block} passes the flag twice"
        )


def test_rollback_to_a_migrated_seed_still_resets_the_optimizer(tmp_path):
    """A regressed block rolls back to model_best.pt -- which may BE the seed.

    Measured: block 1 regressed, block 2 rolled back to the migrated seed
    without --reset_optimizer_on_resume, and train.py refused it in 6.3
    seconds. The requirement is a property of the checkpoint, so it has to be
    read from the checkpoint rather than from the block index.
    """
    module = _supervisor()
    migrated = tmp_path / "seed.pt"
    torch.save({"infos": {"migration_requires_optimizer_reset": True}}, migrated)
    plain = tmp_path / "trained.pt"
    torch.save({"infos": {}}, plain)

    assert module.checkpoint_requires_optimizer_reset(migrated) is True
    assert module.checkpoint_requires_optimizer_reset(plain) is False

    args = module.build_parser().parse_args(["--resume", str(migrated)])
    block2 = module.training_command(
        args, resume=migrated, checkpoint_out=tmp_path / "o.pt", block_index=2
    )
    assert "--reset_optimizer_on_resume" in block2

    block2_plain = module.training_command(
        args, resume=plain, checkpoint_out=tmp_path / "o.pt", block_index=2
    )
    assert "--reset_optimizer_on_resume" not in block2_plain, (
        "resetting the optimizer on every block would discard the run's own "
        "learning at each rollback"
    )


def test_a_child_that_exits_zero_without_writing_is_a_failure(tmp_path):
    """Isaac can mask a non-zero exit; the artifact cannot lie.

    Measured: a training block died on a RuntimeError, printed the traceback,
    exited 0, and the supervisor evaluated the STALE model_latest.pt from the
    previous block -- reporting block 1's numbers as block 2's.
    """
    module = _supervisor()
    missing = tmp_path / "never_written.pt"
    with pytest.raises(RuntimeError, match="did not write"):
        module.run_child(
            ["true"], timeout_s=30, label="fake block", produces=missing
        )

    stale = tmp_path / "stale.pt"
    stale.write_bytes(b"x")
    import os
    import time

    old = time.time() - 3600
    os.utime(stale, (old, old))
    with pytest.raises(RuntimeError, match="stale artifact"):
        module.run_child(
            ["true"], timeout_s=30, label="fake block", produces=stale
        )

    fresh = tmp_path / "fresh.pt"
    module.run_child(
        ["touch", str(fresh)], timeout_s=30, label="fake block", produces=fresh
    )


def test_a_ground_contact_duty_cycle_cannot_replace_these_gates():
    """The measured pre-fix run PASSES `ground_contact` and must still fail.

    This is the whole reason the strict block exists: 0.9486 duty cycle on a
    policy that spent 45 consecutive steps and 297 mm off the verge.
    """
    from verge_task.terrain_convergence import GATE_SPECS

    assert GATE_SPECS["ground_contact"].passes(0.9486)


def test_zero_threshold_violation_is_comparable_to_relative_violations():
    """Event counts must not swamp the ratio gates in strict ranking.

    Measured motivation: the direct-q5 PPO3 best had
    hard_impact_event_count = 1043 while its worst_family_lateral violation was
    0.61. With a raw-count violation the strict supervisor score was ~1043 and
    a lateral improvement was invisible in checkpoint ranking.
    """
    from verge_task.terrain_convergence import GateSpec

    spec = GateSpec("hard_impact_event_count", 0.0, "max")

    # Passing is still exactly zero, and costs nothing.
    assert spec.passes(0.0)
    assert spec.normalized_violation(0.0) == 0.0
    assert not spec.passes(1.0)

    # Any violation still costs something...
    assert spec.normalized_violation(1.0) > 0.0
    # ...ordering is preserved...
    values = [1.0, 5.0, 50.0, 1043.0]
    violations = [spec.normalized_violation(v) for v in values]
    assert violations == sorted(violations)
    # ...but 1043 events no longer dwarf a ratio gate by three orders of
    # magnitude. A relative gate at twice its threshold contributes 1.0.
    assert violations[-1] < 10.0
    assert violations[0] < 1.0


def test_zero_threshold_ranking_no_longer_hides_a_lateral_gain():
    """A checkpoint that fixes lateral must be able to win on strict score."""
    from verge_task.terrain_convergence import GateSpec

    impacts = GateSpec("hard_impact_event_count", 0.0, "max")
    lateral = GateSpec("worst_family_mean_abs_lateral_error_m", 0.30, "max")

    # Candidate halves the lateral error at the cost of 40 more impacts.
    best = impacts.normalized_violation(1000.0) + lateral.normalized_violation(0.48)
    candidate = impacts.normalized_violation(1040.0) + lateral.normalized_violation(0.19)
    assert candidate < best, "large lateral gain must survive a small impact rise"
