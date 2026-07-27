"""Offline contracts for the audited ROM-B factor family."""

import re
from pathlib import Path

import pytest
import torch

from PPO2_ROM_B_PhysX.runtime import (
    FACTOR_NAMES,
    SUPPORTED_FACTOR_NAMES,
    ROMBCommandCylinderROMRuntime,
    clopper_pearson_upper,
    condition_bound,
    quasi_binomial_dispersion,
    seed_group_t_upper,
    summarize_acceptance,
    ROMBCurriculumController,
    ROMBCurriculumThresholds,
    ROMBFactorConfig,
    health_scale_to_factors,
    held_out_evaluation_seed,
    summarize_deterministic_probe,
)
from PPO2_ROM_B_PhysX.runtime.plant_observer import (
    PlantLagObserver,
    pad_state_dict_for_wider_observation,
)
from PPO2_ROM_B_PhysX.rewards import compute_rom_b_safety_reward
from PPO2_ROM_B_PhysX.runtime.evaluation import (
    _expand_fixed_factors,
    _PreTerminationRecorder,
    _summarize_path_profile,
)
from PPO2_ROM_PhysX.runtime.command_rom_runtime import CommandCylinderROMRuntime


ROOT = Path(__file__).resolve().parents[2]
ROM = (
    ROOT / "ROM-cylinders_pipeline/checkpoints"
    / "plant_reachable_v1_finetune_blend25/best_rom.pt"
)


def test_health_scale_endpoints_and_nominal():
    factors = health_scale_to_factors(torch.tensor([0.4, 0.7, 1.0, 1.2]))
    assert torch.allclose(
        factors[0], torch.tensor([-0.125, 1.0, 1.0, 1.0, 1.0])
    )
    assert torch.allclose(
        factors[1], torch.tensor([-0.0625, 0.5, 0.5, 0.5, 0.5])
    )
    assert torch.equal(factors[2], torch.zeros(5))
    assert torch.allclose(
        factors[3], torch.tensor([0.020, 0.0, 0.0, 0.0, 0.0])
    )


def test_health_scale_maps_to_supported_physical_multipliers():
    factors = health_scale_to_factors(torch.tensor([0.4, 0.7, 1.0, 1.2]))
    stiffness = torch.exp(factors[:, 0])
    leakage = (1.0 + 0.25 * factors[:, 1]).pow(3)

    assert torch.allclose(
        stiffness,
        torch.tensor([0.8824969, 0.9394131, 1.0, 1.0202013]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        leakage, torch.tensor([1.953125, 1.423828125, 1.0, 1.0])
    )


def test_random_factors_are_in_the_audited_uncertainty_set():
    torch.manual_seed(17)
    runtime = ROMBCommandCylinderROMRuntime(
        checkpoint=ROM,
        num_envs=1024,
        device="cpu",
        control_dt_s=1.0 / 60.0,
        factor_cfg=ROMBFactorConfig(mode="random"),
    )
    factors = runtime.true_factors
    assert factors.shape == (1024, 5)
    assert torch.all((-0.125 <= factors[:, 0]) & (factors[:, 0] <= 0.020))
    assert torch.all((0.0 <= factors[:, 1:]) & (factors[:, 1:] <= 1.0))
    assert torch.std(factors[:, 0]) > 0.0
    assert torch.all(torch.std(factors[:, 1:], dim=0) > 0.0)


def test_curriculum_scale_contracts_the_ranges_about_nominal():
    torch.manual_seed(19)
    runtime = ROMBCommandCylinderROMRuntime(
        checkpoint=ROM,
        num_envs=4096,
        device="cpu",
        control_dt_s=1.0 / 60.0,
        factor_cfg=ROMBFactorConfig(mode="random", randomization_scale=0.40),
    )
    factors = runtime.true_factors
    assert torch.all((-0.050 <= factors[:, 0]) & (factors[:, 0] <= 0.008))
    assert torch.all((0.0 <= factors[:, 1:]) & (factors[:, 1:] <= 0.40))

    runtime.set_randomization_scale(1.0)
    assert runtime.randomization_scale == 1.0
    assert torch.any(runtime.true_factors[:, 1:] > 0.40)

    runtime.set_randomization_scale(0.0)
    assert torch.equal(runtime.true_factors, torch.zeros_like(runtime.true_factors))


def _passing_curriculum_metrics() -> dict[str, float]:
    return {
        "max_shape_mean_error_m": 0.030,
        "max_shape_p95_error_m": 0.060,
        "max_shape_p95_acceleration_mps2": 30.0,
        "min_head_alignment": 0.990,
        "max_clearance_error_m": 0.150,
        "max_post_constraint_truck_drift_m": 0.0005,
        "max_termination_rate": 0.0,
    }


def test_curriculum_runs_nominal_mild_medium_then_full():
    curriculum = ROMBCurriculumController()
    assert (curriculum.stage, curriculum.scale) == (1, 0.0)

    for expected_stage, expected_scale in ((2, 0.40), (3, 0.70), (4, 1.00)):
        for _ in range(2):
            decision = curriculum.update(_passing_curriculum_metrics())
            assert not decision.advanced
        decision = curriculum.update(_passing_curriculum_metrics())
        assert decision.advanced
        assert (decision.stage, decision.scale) == (expected_stage, expected_scale)

    for _ in range(4):
        decision = curriculum.update(_passing_curriculum_metrics())
        assert not decision.complete
    decision = curriculum.update(_passing_curriculum_metrics())
    assert decision.complete


def test_curriculum_failure_resets_consecutive_passes():
    curriculum = ROMBCurriculumController()
    curriculum.update(_passing_curriculum_metrics())
    failing = _passing_curriculum_metrics()
    failing["min_head_alignment"] = 0.97
    decision = curriculum.update(failing)
    assert not decision.passed
    assert decision.failed_checks == ("head_alignment",)
    assert decision.consecutive_passes == 0
    assert not decision.advanced


def test_curriculum_progress_can_be_restored_after_an_interruption():
    curriculum = ROMBCurriculumController(stage=4)
    curriculum.restore_progress(stage_rounds=2, consecutive_passes=2)
    decision = curriculum.update(_passing_curriculum_metrics())
    assert decision.stage_rounds == 3
    assert decision.consecutive_passes == 3
    assert not decision.complete


def test_deterministic_probe_summary_counts_failures_and_incomplete_envs():
    values = torch.tensor([[0.01, 0.02, 0.03], [0.02, 0.03, float("nan")]])
    summary = summarize_deterministic_probe(
        shape="line",
        error=values,
        acceleration=values * 1000.0,
        alignment=1.0 - values,
        clearance=values * 2.0,
        drift=values * 0.01,
        completed=torch.tensor([True, True, False]),
        safety_failed=torch.tensor([False, True, False]),
        timed_out=torch.tensor([False, False, False]),
    )
    assert summary["evaluated_environments"] == 3
    assert summary["completed_environments"] == 2
    assert summary["completion_fraction"] == torch.tensor(2.0 / 3.0).item()
    assert summary["safety_termination_rate"] == torch.tensor(1.0 / 3.0).item()
    assert summary["max_termination_rate"] == torch.tensor(2.0 / 3.0).item()
    assert summary["mean_error_m"] == torch.tensor(0.02).item()


def test_fixed_probe_factors_expand_per_environment_and_reject_bad_shapes():
    template = torch.zeros(4, 5)
    fixed = health_scale_to_factors(0.4)[0]
    expanded = _expand_fixed_factors(fixed, template)
    assert expanded.shape == (4, 5)
    assert torch.equal(expanded[0], fixed)
    assert torch.equal(expanded[-1], fixed)
    with pytest.raises(ValueError, match="fixed_factors"):
        _expand_fixed_factors(torch.zeros(4, 4), template)


def test_held_out_evaluation_seeds_are_fixed_per_shape():
    seeds = [held_out_evaluation_seed(17, index) for index in range(5)]
    assert seeds == [100017, 100018, 100019, 100020, 100021]
    assert held_out_evaluation_seed(17, 2) == held_out_evaluation_seed(17, 2)


def test_rom_b_safety_reward_is_zero_inside_soft_margins():
    terms = compute_rom_b_safety_reward(
        head_alignment=torch.tensor([0.99]),
        clearance_error_m=torch.tensor([0.10]),
        acceleration_mps2=torch.tensor([30.0]),
        additional_safety_violation=torch.tensor([False]),
    )
    assert torch.equal(terms.reward, torch.zeros(1))
    assert not torch.any(terms.hard_safety_violation)


def test_rom_b_safety_reward_penalizes_margin_and_hard_gate_crossings():
    margin = compute_rom_b_safety_reward(
        head_alignment=torch.tensor([0.9825]),
        clearance_error_m=torch.tensor([0.19]),
        acceleration_mps2=torch.tensor([37.5]),
        additional_safety_violation=torch.tensor([False]),
    )
    assert torch.allclose(margin.reward, torch.tensor([-2.25]), atol=1.0e-5)
    assert not torch.any(margin.hard_safety_violation)

    hard = compute_rom_b_safety_reward(
        head_alignment=torch.tensor([0.97, 1.0]),
        clearance_error_m=torch.tensor([0.30, 0.0]),
        acceleration_mps2=torch.tensor([50.0, 0.0]),
        additional_safety_violation=torch.tensor([False, True]),
    )
    assert torch.all(hard.hard_safety_violation)
    assert torch.equal(hard.reward, torch.tensor([-34.0, -20.0]))


def test_nominal_runtime_is_bit_exact_over_multiple_rom_ticks():
    nominal = CommandCylinderROMRuntime(
        checkpoint=ROM, num_envs=4, device="cpu", control_dt_s=1.0 / 60.0
    )
    rom_b = ROMBCommandCylinderROMRuntime(
        checkpoint=ROM,
        num_envs=4,
        device="cpu",
        control_dt_s=1.0 / 60.0,
        factor_cfg=ROMBFactorConfig(mode="nominal"),
    )
    generator = torch.Generator().manual_seed(23)
    for _ in range(9):
        values = {
            "command": torch.randn(4, 8, generator=generator) * 0.01,
            "extension": torch.rand(4, 4, generator=generator) * 0.4 + 0.25,
            "extension_velocity": torch.randn(4, 4, generator=generator) * 0.05,
            "dP": torch.randn(4, 4, generator=generator) * 1.0e7,
        }
        expected = nominal.step(**values)
        actual = rom_b.step(**values)
        for field in expected.__dataclass_fields__:
            assert torch.equal(getattr(expected, field), getattr(actual, field))


def test_clopper_pearson_upper_matches_known_values():
    # 22/640 is the observed Monte Carlo line result.
    assert clopper_pearson_upper(22, 640) == pytest.approx(0.04872, abs=1e-5)
    assert clopper_pearson_upper(0, 640) == pytest.approx(0.00467, abs=1e-5)
    assert clopper_pearson_upper(5, 5) == 1.0
    with pytest.raises(ValueError):
        clopper_pearson_upper(6, 5)


def test_seed_group_t_upper_needs_at_least_two_groups():
    assert seed_group_t_upper([0.03]) is None
    # Monte Carlo line: 7, 0, 3, 7, 5 failures out of 128.
    rates = [7 / 128, 0 / 128, 3 / 128, 7 / 128, 5 / 128]
    assert seed_group_t_upper(rates) == pytest.approx(0.05647, abs=1e-5)


def test_acceptance_takes_the_larger_of_the_two_bounds():
    """Line: between-group spread binds and the pooled bound is too narrow."""

    bound = condition_bound(
        shape="line", failures_per_group=[7, 0, 3, 7, 5], replicas_per_group=128
    )
    assert bound.total_failures == 22 and bound.total_replicas == 640
    assert bound.rate == pytest.approx(0.034375)
    assert bound.clopper_pearson_upper == pytest.approx(0.04872, abs=1e-5)
    assert bound.seed_group_t_upper == pytest.approx(0.05647, abs=1e-5)
    assert bound.upper_bound == pytest.approx(0.05647, abs=1e-5)
    assert bound.binding_term == "seed_group_t"
    assert not bound.passed


def test_acceptance_keeps_binomial_uncertainty_when_groups_agree():
    """Circle: zero spread, so the t term degenerates and must not certify 0."""

    bound = condition_bound(
        shape="circle", failures_per_group=[0, 0, 0, 0, 0], replicas_per_group=128
    )
    assert bound.seed_group_t_upper == pytest.approx(0.0)
    # A finite test cannot establish a true rate of exactly zero.
    assert bound.upper_bound == pytest.approx(0.00467, abs=1e-5)
    assert bound.binding_term == "clopper_pearson"
    assert bound.upper_bound > 0.0
    assert bound.passed

    # The same protection with a non-zero but perfectly uniform result.
    uniform = condition_bound(
        shape="arc", failures_per_group=[4, 4, 4, 4, 4], replicas_per_group=128
    )
    assert uniform.seed_group_t_upper == pytest.approx(4 / 128)
    assert uniform.binding_term == "clopper_pearson"
    assert uniform.upper_bound > uniform.rate


def test_acceptance_falls_back_to_clopper_pearson_for_one_group():
    bound = condition_bound(
        shape="line", failures_per_group=[7], replicas_per_group=128
    )
    assert bound.seed_group_t_upper is None
    assert bound.binding_term == "clopper_pearson"
    assert bound.upper_bound == clopper_pearson_upper(7, 128)


def test_dispersion_is_reported_but_never_gates():
    phi = quasi_binomial_dispersion([7, 0, 3, 7, 5], 128)
    assert phi == pytest.approx(2.07, abs=0.01)
    assert quasi_binomial_dispersion([4, 4, 4, 4, 4], 128) == pytest.approx(0.0)
    assert quasi_binomial_dispersion([0, 0, 0], 128) is None
    assert quasi_binomial_dispersion([3], 128) is None
    # A heterogeneous condition still passes if both bounds clear the threshold.
    bound = condition_bound(
        shape="sine", failures_per_group=[0, 0, 0, 1, 0], replicas_per_group=128
    )
    assert bound.dispersion_phi is not None
    assert bound.passed


def test_worst_shape_controls_acceptance_and_never_averages_shapes():
    bounds = [
        condition_bound(
            shape="line", failures_per_group=[7, 0, 3, 7, 5], replicas_per_group=128
        ),
        condition_bound(
            shape="circle", failures_per_group=[0, 0, 0, 0, 0], replicas_per_group=128
        ),
        condition_bound(
            shape="square", failures_per_group=[2, 1, 4, 3, 4], replicas_per_group=128
        ),
    ]
    summary = summarize_acceptance(bounds)
    assert summary["worst_condition"]["shape"] == "line"
    assert not summary["passed"]
    assert [c["shape"] for c in summary["failed_conditions"]] == ["line"]
    # A passing circle must not soften the failing line.
    assert summary["worst_condition"]["upper_bound"] == pytest.approx(
        0.05647, abs=1e-5
    )


def test_acceptance_rejects_malformed_conditions():
    with pytest.raises(ValueError, match="at least one seed group"):
        condition_bound(shape="line", failures_per_group=[], replicas_per_group=128)
    with pytest.raises(ValueError, match="replicas_per_group"):
        condition_bound(shape="line", failures_per_group=[1], replicas_per_group=0)
    with pytest.raises(ValueError, match=r"\[0, replicas_per_group\]"):
        condition_bound(
            shape="line", failures_per_group=[200], replicas_per_group=128
        )


def test_unrotated_cutter_formula_is_wrong_when_the_head_points_down():
    """Pin the clearance-gate frame bug measured on 2026-07-24.

    `_compute_cutter_world_z` is `ee_origin_z + tool_tip_local_offset_z`, which
    ignores orientation. `_compute_cutter_pos_w` applies the same offset through
    the live EE quaternion. In the head-down pose the EE body Z axis points *down*
    in world, so a body-frame offset of -0.79 resolves to +0.79 in world and the
    two differ by 2*|offset| = 1.58 m.

    Measured live over 84 pre-termination samples: -1.5798 m, sd 0.0003.
    The trajectory library's `cutter_world_z` column was generated with the
    unrotated formula too, which is why the two errors largely cancel and the gate
    reads a plausible-looking residual instead of a 1.6 m absurdity. What does not
    cancel is the truck's ride height.
    """

    offset = -0.79
    # Head-down: body Z axis maps to world -Z. A 180 deg rotation about X.
    head_down = torch.tensor([0.0, 1.0, 0.0, 0.0])  # (w, x, y, z)

    def quat_apply(q, v):
        w, x, y, z = q
        u = torch.tensor([x, y, z])
        return (
            2.0 * torch.dot(u, v) * u
            + (w * w - torch.dot(u, u)) * v
            + 2.0 * w * torch.linalg.cross(u, v)
        )

    body_offset = torch.tensor([0.0, 0.0, offset])
    rotated_z = float(quat_apply(head_down, body_offset)[2])
    assert rotated_z == pytest.approx(+0.79, abs=1e-6)

    ee_origin_z = 0.8764  # measured live
    gate = ee_origin_z + offset
    rotated = ee_origin_z + rotated_z
    assert gate == pytest.approx(0.0864, abs=1e-4)
    assert rotated == pytest.approx(1.6664, abs=1e-4)
    assert gate - rotated == pytest.approx(2.0 * offset, abs=1e-4)
    assert gate - rotated == pytest.approx(-1.58, abs=1e-3)


def test_clearance_gate_leaks_truck_ride_height():
    """The residual that does not cancel is the truck's absolute world height.

    clearance = cutter_world_z - reference_world_z, where the reference is a fixed
    world number but the commanded geometry is authored in the truck frame. Any
    change in ride height therefore appears one-for-one as clearance error.
    """

    def reference_frame_term(truck_base_world_z, ref_cutter_truck_z, ref_world_z):
        return truck_base_world_z + ref_cutter_truck_z - ref_world_z

    cutter_definition = -1.5797  # measured, constant to sd 0.0003
    settled = reference_frame_term(0.1423, 1.55, 0.1036) + cutter_definition
    failing = reference_frame_term(0.2533, 1.55, 0.0600) + cutter_definition

    assert settled == pytest.approx(0.009, abs=0.002)
    assert failing == pytest.approx(0.163, abs=0.002)
    # A 0.111 m ride-height change moves the gate by essentially the same amount.
    assert failing - settled == pytest.approx(0.2533 - 0.1423, abs=0.05)


class _StubEnv:
    """Minimal stand-in exposing the attributes the recorder observes."""

    class _Cfg:
        clearance_relative_to_reference = True

    class _Robot:
        class _Data:
            pass

        def __init__(self, num_envs: int) -> None:
            self.data = self._Data()
            self.data.joint_pos = torch.zeros(num_envs, 4)
            self.data.joint_vel = torch.zeros(num_envs, 4)
            self.data.body_link_state_w = torch.zeros(num_envs, 2, 13)

    def __init__(self, num_envs: int = 3) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.cfg = self._Cfg()
        self.robot = self._Robot(num_envs)
        self._cylinder_joint_ids = [0, 1, 2, 3]
        self._ee_body_id = 0
        self._original_reference_world_z = torch.zeros(num_envs)
        self._original_reference_tool = torch.zeros(num_envs, 3)
        self._measured_tool_acceleration_truck = torch.zeros(num_envs, 3)
        self._residual_filtered = torch.zeros(num_envs, 4)
        self._residual_raw = torch.zeros(num_envs, 4)
        self._truck_drift_xy = torch.zeros(num_envs)
        self._truck_post_constraint_drift_xy = torch.zeros(num_envs)
        self._reference_path_s = torch.zeros(num_envs)
        self._traj_phase = torch.zeros(num_envs)
        self._scored_elapsed = torch.zeros(num_envs)
        self._baseline_cylinder_goal = torch.zeros(num_envs, 4)
        self._corrected_cylinder_goal = torch.zeros(num_envs, 4)
        self._current_valve_cmd = torch.zeros(num_envs, 8)
        self._fd_delta_corrected = torch.zeros(num_envs, 4)
        self.cutter_z = torch.zeros(num_envs)
        self.alignment = torch.ones(num_envs)

    def _compute_cutter_world_z(self):
        return self.cutter_z

    def _clearance_signal(self):
        return self.cutter_z

    def _truck_base_body_id(self):
        return 1

    def _compute_cutter_pos_w(self):
        return torch.zeros(self.num_envs, 3)

    def _measure_head_down_alignment(self):
        return self.alignment

    def _residual_scale_vector(self):
        return torch.tensor([0.002, 0.002, 0.002, 0.00025])


def test_pre_termination_recorder_keeps_a_chronological_window():
    raw = _StubEnv(num_envs=2)
    recorder = _PreTerminationRecorder(raw, window=3)
    tool = torch.zeros(2, 3)
    # Five observations into a three-deep ring: only 0.3, 0.4, 0.5 survive.
    for step in range(5):
        raw.cutter_z = torch.full((2,), 0.1 * (step + 1))
        raw._reference_path_s = torch.full((2,), float(step))
        recorder.observe(tool)
    causes = {
        "clearance": torch.tensor([True, False]),
        "head_alignment": torch.tensor([False, False]),
    }
    recorder.capture(
        env_ids=torch.tensor([0, 1]),
        causes=causes,
        simulator_step=5,
        scored_step=torch.tensor([4, 4]),
        factors=torch.zeros(2, 5),
    )
    assert len(recorder.records) == 2
    first, second = recorder.records
    assert first["termination_causes"] == ["clearance"]
    assert second["termination_causes"] == ["unclassified"]
    assert first["simulator_step"] == 5
    assert first["trace"]["path_s_m"] == [[2.0], [3.0], [4.0]]
    assert first["at_termination"]["clearance_signed_m"] == pytest.approx(0.5)
    assert len(first["at_termination"]["valve_command"]) == 8
    # Frame decomposition fields are captured so the clearance offset can be
    # split into reference disagreement versus genuine tracking error.
    for name in (
        "reference_world_z", "cutter_world_z", "cutter_pos_w_z",
        "ee_origin_world_z", "truck_base_world_z",
    ):
        assert name in first["at_termination"]
    assert len(first["at_termination"]["reference_cutter_truck"]) == 3
    assert len(first["at_termination"]["measured_cutter_truck"]) == 3


def test_path_profile_bins_scored_samples_by_path_position():
    steps, envs = 4, 1
    samples = {
        "path_s": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        "acceleration": torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        "clearance": torch.tensor([[0.10], [0.12], [0.14], [0.19]]),
        "alignment": torch.tensor([[0.999], [0.995], [0.990], [0.981]]),
        "error": torch.tensor([[0.01], [0.02], [0.03], [0.04]]),
    }
    assert samples["path_s"].shape == (steps, envs)
    profile = _summarize_path_profile(samples, bins=2)
    assert [entry["bin"] for entry in profile] == [0, 1]
    assert profile[0]["samples"] == 2 and profile[1]["samples"] == 2
    # The trailing half of the path carries the acceleration and clearance tail.
    assert profile[0]["max_acceleration_mps2"] == pytest.approx(20.0)
    assert profile[1]["max_acceleration_mps2"] == pytest.approx(40.0)
    assert profile[1]["max_clearance_error_m"] == pytest.approx(0.19)
    assert profile[1]["min_head_alignment"] == pytest.approx(0.981)


def test_path_profile_is_empty_without_finite_path_samples():
    samples = {
        name: torch.full((2, 1), float("nan"))
        for name in ("path_s", "acceleration", "clearance", "alignment", "error")
    }
    assert _summarize_path_profile(samples) == []


def test_clearance_and_head_gates_coincide_with_termination_thresholds():
    """Document why two promotion gates cannot fail independently.

    The environment terminates a replica the moment |clearance| exceeds
    `cutter_clearance_hard_band_m` or head alignment drops below
    `head_down_termination_alignment`. Those thresholds equal the
    `max_clearance_error_m` and `min_head_alignment` gates, so a violation is
    always re-expressed as a termination and never as a gate failure. Any
    change to one side must be mirrored deliberately on the other.
    """

    # Parsed rather than imported: the config module pulls in Isaac Lab, and
    # these contracts must stay checkable without a simulator.
    source = (
        ROOT / "PPO2_ROM_PhysX/policy_4cyl/env_cfg.py"
    ).read_text()
    defaults = dict(
        re.findall(
            r"^\s*(\w+):\s*(?:float|bool)\s*=\s*([\d.]+|True|False)\s*$",
            source,
            flags=re.MULTILINE,
        )
    )
    thresholds = ROMBCurriculumThresholds()
    assert float(defaults["cutter_clearance_hard_band_m"]) == (
        thresholds.max_clearance_error_m
    )
    assert float(defaults["head_down_termination_alignment"]) == (
        thresholds.min_head_alignment
    )
    assert defaults["clearance_relative_to_reference"] == "True"


def test_zero_structured_factors_match_direct_model_inference():
    runtime = CommandCylinderROMRuntime(
        checkpoint=ROM, num_envs=8, device="cpu", control_dt_s=1.0 / 60.0
    )
    history = runtime.model.input_mean.reshape(1, 1, 24).expand(8, 6, 24).clone()
    command = torch.zeros(8, 8)
    with torch.no_grad():
        expected = runtime.model(history, command)
        actual = runtime.model(history, command, rom_b_factors=torch.zeros(8, 5))
    for field in expected._fields:
        assert torch.equal(getattr(expected, field), getattr(actual, field))


def test_rom_b_model_boundary_rejects_the_removed_sixth_factor_column():
    runtime = CommandCylinderROMRuntime(
        checkpoint=ROM, num_envs=8, device="cpu", control_dt_s=1.0 / 60.0
    )
    history = runtime.model.input_mean.reshape(1, 1, 24).expand(8, 6, 24).clone()
    command = torch.zeros(8, 8)
    with pytest.raises(ValueError, match="shape"):
        runtime.model(history, command, rom_b_factors=torch.zeros(8, 6))


def test_degraded_corner_changes_pressure_and_motion_without_output_scaling():
    runtime = CommandCylinderROMRuntime(
        checkpoint=ROM, num_envs=8, device="cpu", control_dt_s=1.0 / 60.0
    )
    history = runtime.model.input_mean.reshape(1, 1, 24).expand(8, 6, 24).clone()
    command = torch.tensor(
        [[0.005, -0.005, 0.005, -0.005, 0.05, -0.05, 0.05, -0.05]]
    ).expand(8, -1)
    corner = torch.tensor([[-0.125, 1.0, 1.0, 1.0, 1.0]]).expand(8, -1)
    with torch.no_grad():
        nominal = runtime.model(history, command)
        degraded = runtime.model(history, command, rom_b_factors=corner)
    assert not torch.equal(degraded.delta_pressure, nominal.delta_pressure)
    assert not torch.equal(degraded.delta_velocity, nominal.delta_velocity)
    assert not torch.equal(degraded.delta_extension, nominal.delta_extension)


# ----------------------------------------------------------------------------
# Plant-lag observer: the mechanism that lets randomization teach the policy
# ----------------------------------------------------------------------------


def test_plant_observer_separates_persistent_lag_from_transients():
    """A sluggish plant shows up as a persistent slow-filter offset."""

    obs = PlantLagObserver(num_envs=2, device="cpu", control_dt_s=1 / 60)
    # env 0: healthy, zero goal error. env 1: leaky, a constant 5 mm shortfall.
    healthy, leaky = 0.0, 0.005
    for _ in range(600):  # 10 s, several slow time constants
        error = torch.tensor([[healthy] * 4, [leaky] * 4])
        features = obs.update(error)
    slow, transient = features[:, :4], features[:, 4:]
    # The slow channel converges on the persistent offset (scaled by 100).
    assert torch.allclose(slow[0], torch.zeros(4), atol=1e-6)
    assert torch.allclose(slow[1], torch.full((4,), 0.5), atol=0.02)
    # With a constant input both filters agree, so the transient term vanishes.
    assert torch.allclose(transient, torch.zeros(2, 4), atol=0.02)
    # The two plants are distinguishable, which is the whole point.
    assert (slow[1] - slow[0]).abs().min() > 0.4


def test_plant_observer_transient_channel_responds_to_motion():
    obs = PlantLagObserver(num_envs=1, device="cpu", control_dt_s=1 / 60)
    for _ in range(600):
        obs.update(torch.zeros(1, 4))
    step = obs.update(torch.full((1, 4), 0.01))
    # A sudden change moves the fast filter first, so fast - slow is non-zero.
    assert step[0, 4:].abs().min() > 0.0
    assert step[0, :4].abs().max() < step[0, 4:].abs().max()


def test_plant_observer_holds_value_while_inactive():
    obs = PlantLagObserver(num_envs=2, device="cpu", control_dt_s=1 / 60)
    for _ in range(200):
        obs.update(torch.full((2, 4), 0.004))
    before = obs.features().clone()
    inactive = torch.tensor([False, False])
    for _ in range(200):
        obs.update(torch.full((2, 4), -0.05), inactive)
    # Deployment and settling transients must not pollute the plant estimate.
    assert torch.allclose(obs.features(), before)


def test_plant_observer_resets_per_environment():
    obs = PlantLagObserver(num_envs=3, device="cpu", control_dt_s=1 / 60)
    for _ in range(200):
        obs.update(torch.full((3, 4), 0.004))
    obs.reset(torch.tensor([1]))
    features = obs.features()
    assert torch.allclose(features[1], torch.zeros(8))
    assert features[0].abs().max() > 0.0 and features[2].abs().max() > 0.0
    obs.reset()
    assert torch.allclose(obs.features(), torch.zeros(3, 8))


def test_plant_observer_rejects_bad_configuration():
    with pytest.raises(ValueError):
        PlantLagObserver(num_envs=0, device="cpu", control_dt_s=1 / 60)
    with pytest.raises(ValueError):
        PlantLagObserver(num_envs=1, device="cpu", control_dt_s=0.0)
    with pytest.raises(ValueError):  # fast must be shorter than slow
        PlantLagObserver(
            num_envs=1, device="cpu", control_dt_s=1 / 60,
            fast_tau_s=3.0, slow_tau_s=1.0,
        )
    obs = PlantLagObserver(num_envs=2, device="cpu", control_dt_s=1 / 60)
    with pytest.raises(ValueError, match="cylinder_goal_error"):
        obs.update(torch.zeros(2, 3))


def test_widened_checkpoint_is_numerically_identical_after_transfer():
    """Zero-padding means model_53 transfers exactly, not approximately."""

    checkpoint = {
        "mlp.0.weight": torch.randn(256, 42),
        "mlp.0.bias": torch.randn(256),
        "obs_normalizer._mean": torch.randn(1, 42),
        "obs_normalizer._var": torch.rand(1, 42) + 0.5,
        "obs_normalizer._std": torch.rand(1, 42) + 0.5,
    }
    target = {
        "mlp.0.weight": torch.zeros(256, 50),
        "mlp.0.bias": torch.zeros(256),
        "obs_normalizer._mean": torch.zeros(1, 50),
        "obs_normalizer._var": torch.ones(1, 50),
        "obs_normalizer._std": torch.ones(1, 50),
    }
    adapted, report = pad_state_dict_for_wider_observation(checkpoint, target)

    assert report["added_dims"] == 8
    assert len(report["padded"]) == 4  # bias is unchanged
    assert report["shape_mismatch"] == []
    for key, tensor in adapted.items():
        assert tensor.shape == target[key].shape
    # Original columns survive untouched.
    assert torch.equal(adapted["mlp.0.weight"][:, :42], checkpoint["mlp.0.weight"])
    # New input weights are zero, so the appended features cannot change output.
    assert torch.equal(adapted["mlp.0.weight"][:, 42:], torch.zeros(256, 8))
    # Variance-like buffers pad with ones, never zeros, so nothing divides by 0.
    assert torch.equal(adapted["obs_normalizer._var"][:, 42:], torch.ones(1, 8))
    assert torch.equal(adapted["obs_normalizer._std"][:, 42:], torch.ones(1, 8))
    assert torch.equal(adapted["obs_normalizer._mean"][:, 42:], torch.zeros(1, 8))

    # End to end: the widened layer reproduces the original output exactly.
    narrow_obs = torch.randn(5, 42)
    wide_obs = torch.cat((narrow_obs, torch.randn(5, 8)), dim=-1)
    original = narrow_obs @ checkpoint["mlp.0.weight"].T + checkpoint["mlp.0.bias"]
    widened = wide_obs @ adapted["mlp.0.weight"].T + adapted["mlp.0.bias"]
    assert torch.allclose(original, widened, atol=1e-6)


def test_widening_reports_genuinely_incompatible_shapes():
    adapted, report = pad_state_dict_for_wider_observation(
        {"mlp.0.weight": torch.randn(256, 30)},
        {"mlp.0.weight": torch.zeros(128, 50)},
    )
    assert report["padded"] == []
    assert len(report["shape_mismatch"]) == 1
    assert adapted["mlp.0.weight"].shape == (128, 50)


def test_clearance_reward_defaults_are_unchanged_and_exclusion_is_opt_in():
    """The contaminated clearance signal no longer shapes the ROM-B reward.

    Clearance is untrustworthy (frame offset ~0.164 m, see
    reports/rom_b_frame_decomposition.md). Head alignment is a dot product and
    acceleration a norm, so both stay frame-independent and keep their weights.
    """

    source = (ROOT / "PPO2_ROM_B_PhysX/env_cfg.py").read_text()
    weights = dict(
        re.findall(
            r"^\s*(\w+):\s*float\s*=\s*(-?[\d.]+)\s*$", source, flags=re.MULTILINE
        )
    )
    # Defaults are unchanged: exclusion is opt-in via
    # --exclude_contaminated_clearance_reward and recorded in the run contract.
    assert float(weights["rom_b_clearance_margin_weight"]) == -4.0
    assert float(weights["residual_ground_penalty_weight"]) == -2.0
    # Frame-independent terms are untouched.
    assert float(weights["rom_b_head_margin_weight"]) == -4.0
    assert float(weights["residual_head_penalty_weight"]) == -2.0
    assert float(weights["rom_b_acceleration_margin_weight"]) == -1.0
    assert float(weights["rom_b_acceleration_event_weight"]) == -5.0
    assert float(weights["rom_b_hard_safety_event_weight"]) == -20.0
    # Bands themselves are unchanged: no safety limit was relaxed.
    assert float(weights["rom_b_clearance_hard_m"]) == 0.200
    assert float(weights["rom_b_clearance_soft_m"]) == 0.180


def test_zero_clearance_weight_removes_clearance_shaping_only():
    at_limit = dict(
        head_alignment=torch.tensor([1.0]),
        clearance_error_m=torch.tensor([0.195]),
        acceleration_mps2=torch.tensor([0.0]),
        additional_safety_violation=torch.tensor([False]),
    )
    before = compute_rom_b_safety_reward(**at_limit, clearance_margin_weight=-4.0)
    after = compute_rom_b_safety_reward(**at_limit, clearance_margin_weight=0.0)
    assert float(before.reward) < 0.0
    assert float(after.reward) == 0.0

    # Head shaping is unaffected by the clearance weight.
    tilted = dict(
        head_alignment=torch.tensor([0.9825]),
        clearance_error_m=torch.tensor([0.0]),
        acceleration_mps2=torch.tensor([0.0]),
        additional_safety_violation=torch.tensor([False]),
    )
    assert float(
        compute_rom_b_safety_reward(**tilted, clearance_margin_weight=0.0).reward
    ) == pytest.approx(-1.0, abs=1e-5)


def test_clearance_can_be_removed_from_the_hard_safety_event():
    beyond = dict(
        head_alignment=torch.tensor([1.0]),
        clearance_error_m=torch.tensor([0.30]),
        acceleration_mps2=torch.tensor([0.0]),
        additional_safety_violation=torch.tensor([False]),
        clearance_margin_weight=0.0,
    )
    # Default keeps clearance in the -20 event, which is the residual contamination.
    included = compute_rom_b_safety_reward(**beyond)
    assert bool(included.hard_safety_violation[0])
    assert float(included.reward) == pytest.approx(-20.0)

    excluded = compute_rom_b_safety_reward(
        **beyond, clearance_in_hard_safety_event=False
    )
    assert not bool(excluded.hard_safety_violation[0])
    assert float(excluded.reward) == 0.0

    # Head and the additional-violation channel still fire either way.
    for kwargs in (
        {"head_alignment": torch.tensor([0.97])},
        {"additional_safety_violation": torch.tensor([True])},
    ):
        args = {**beyond, "clearance_in_hard_safety_event": False, **kwargs}
        assert bool(compute_rom_b_safety_reward(**args).hard_safety_violation[0])


def test_e_flow_is_absent_from_the_rom_b_factor_interface():
    """Measured inert on 2026-07-25, so it is not a ROM-B factor.

    Extension response to e_flow = +/-0.20 was 0.010-0.060 mm RMS against
    1.39-8.26 mm for e_comp and 6.28-39.0 mm for the seals; max sensitivity ratio
    1.46%, median 0.73%. Three cylinders sit at the model's 1e-9 flow-gain clamp.
    This is practical non-influence, not proven structural non-identifiability -
    The legacy hydraulic model adapter owns its structural zero compatibility
    slot; it must not leak into runtime state, configuration, or reports.
    """

    assert SUPPORTED_FACTOR_NAMES == (
        "e_comp", "d_seal_1", "d_seal_2", "d_seal_3", "d_seal_4"
    )
    assert FACTOR_NAMES == SUPPORTED_FACTOR_NAMES
    assert "e_flow" not in FACTOR_NAMES
    assert len(FACTOR_NAMES) == 5
    assert not hasattr(ROMBFactorConfig(), "e_flow_range")
    torch.manual_seed(3)
    runtime = ROMBCommandCylinderROMRuntime(
        checkpoint=ROM, num_envs=256, device="cpu", control_dt_s=1.0 / 60.0,
        factor_cfg=ROMBFactorConfig(mode="random"),
    )
    assert runtime.true_factors.shape == (256, 5)
    assert torch.std(runtime.true_factors[:, 0]) > 0.0
    assert torch.all(torch.std(runtime.true_factors[:, 1:], dim=0) > 0.0)

    for path in (
        "PPO2_ROM_B_PhysX/env_cfg.py",
        "PPO2_ROM_B_PhysX/scripts/train.py",
        "PPO2_ROM_B_PhysX/scripts/evaluate.py",
    ):
        assert "e_flow" not in (ROOT / path).read_text()


def test_h_curve_now_varies_compliance_and_leakage_only():
    factors = health_scale_to_factors(torch.linspace(0.4, 1.2, 9))
    assert factors.shape == (9, 5)
    assert torch.std(factors[:, 0]) > 0.0
    assert torch.std(factors[:, 1]) > 0.0


def test_clearance_frame_consistency_is_opt_in_everywhere():
    """The corrected clearance signal must never switch on silently.

    Verified live 2026-07-25 on MC line round 1, seed 300017:
      default path              -> term 0.05469, 7 failures (6 clearance, 1 truck)
                                   bit-exact against the 2026-07-24 audit
      --clearance_frame_consistent -> term 0.00000, 0 failures

    The whole failure mode was the frame offset. Because the two signals differ,
    the flag must stay opt-in in both the shared parent and ROM-B, or historical
    numbers silently stop being comparable.
    """

    for path, flag in (
        ("PPO2_ROM_PhysX/policy/env_cfg.py", "clearance_frame_consistent"),
        ("PPO2_ROM_B_PhysX/env_cfg.py", "clearance_frame_consistent"),
    ):
        source = (ROOT / path).read_text()
        match = re.search(rf"^\s*{flag}:\s*bool\s*=\s*(\w+)\s*$", source, re.MULTILINE)
        assert match is not None, f"{flag} missing from {path}"
        assert match.group(1) == "False", f"{flag} must default False in {path}"

    # Both clearance interventions exist and are mutually exclusive by design:
    # one hides the contaminated signal, the other corrects it.
    train = (ROOT / "PPO2_ROM_B_PhysX/scripts/train.py").read_text()
    assert "--exclude_contaminated_clearance_reward" in train
    assert "--clearance_frame_consistent" in train
    assert "Use one or the other." in train


def test_evaluator_records_the_same_clearance_signal_that_terminates():
    """The audit metric must come from `_clearance_signal`, not a private copy.

    Regression for a real defect. The evaluator recomputed clearance inline as
    `_compute_cutter_world_z()` minus the reference, which happened to equal the
    environment's signal only while `clearance_frame_consistent` was False. Under
    the corrected signal the two diverged, so the run of 2026-07-25_00-56-12
    terminated on the corrected signal but *reported* the contaminated one. The
    contaminated metric was no longer capped by termination and floated to
    0.2343 m, tripping the 0.200 m gate while `termination_cause_rates` recorded
    no clearance terminations at all -- an impossible pair that only a
    metric/termination mismatch can produce.

    Any future clearance signal change must flow through one function.
    """

    source = (ROOT / "PPO2_ROM_B_PhysX/runtime/evaluation.py").read_text()
    # Three consumers: the pre-termination trace, the per-step sample buffer and
    # the termination-cause attribution. All must share one definition.
    assert source.count("raw._clearance_signal()") == 3, (
        "every clearance consumer in evaluation.py must call raw._clearance_signal()"
    )
    # The contaminated reconstruction was `cutter_world_z - reference_world_z`.
    # `_compute_cutter_world_z` may still appear as a *labelled component* of the
    # frame-decomposition trace, but never subtracted to form a gate metric.
    assert "clearance = raw._compute_cutter_world_z()" not in source
    assert "- raw._original_reference_world_z" not in source

    reward_source = (ROOT / "PPO2_ROM_B_PhysX/env.py").read_text()
    reward_body = reward_source.split("def _get_rewards", 1)[1]
    assert "clearance_error = self._clearance_signal()" in reward_body
    assert "clearance_error = self._compute_cutter_world_z()" not in reward_body


def _safety(n: int = 4, **kw):
    base = dict(
        head_alignment=torch.ones(n),
        clearance_error_m=torch.zeros(n),
        acceleration_mps2=torch.zeros(n),
        additional_safety_violation=torch.zeros(n, dtype=torch.bool),
    )
    base.update(kw)
    return compute_rom_b_safety_reward(**base)


def test_truck_reaction_shaping_is_inert_by_default():
    """Zero weight must leave the reward bit-exact for every existing artifact."""

    reaction = torch.tensor([0.0, 0.02, 0.04, 0.06])
    without = _safety()
    with_signal = _safety(truck_reaction_m=reaction)
    assert torch.equal(without.reward, with_signal.reward)
    assert torch.all(with_signal.truck_reaction_margin_penalty == 0.0)


def test_truck_reaction_margin_gives_a_gradient_before_the_threshold():
    """The whole point: penalty must grow while approaching 0.05 m, not at it.

    `truck_raw_reaction` already terminates and already carries the hard-event
    penalty, but had no shoulder, so the policy received no signal distinguishing
    0.049 m from 0.001 m. Measured 2026-07-25 it is the dominant remaining
    termination cause (0.0143 vs 0.0050 for head alignment).
    """

    reaction = torch.tensor([0.010, 0.035, 0.0425, 0.050, 0.080])
    terms = _safety(
        n=5, truck_reaction_m=reaction, truck_reaction_margin_weight=-2.0
    )
    penalty = terms.truck_reaction_margin_penalty
    # Inert below the soft shoulder, saturated at and beyond the hard threshold.
    assert penalty[0] == 0.0
    assert penalty[1] == 0.0
    assert penalty[2] < 0.0
    assert torch.isclose(penalty[3], torch.tensor(-2.0))
    assert torch.isclose(penalty[4], torch.tensor(-2.0)), "must stay bounded"
    # Strictly increasing punishment across the shoulder.
    assert penalty[2] < penalty[1]
    assert penalty[3] < penalty[2]


def test_truck_reaction_shaping_does_not_move_the_termination_threshold():
    """The forbidden change is relaxing a gate; adding a shoulder is not that."""

    cfg = (ROOT / "PPO2_ROM_B_PhysX/env_cfg.py").read_text()
    hard = re.search(r"rom_b_truck_reaction_hard_m:\s*float\s*=\s*([\d.]+)", cfg)
    weight = re.search(
        r"rom_b_truck_reaction_margin_weight:\s*float\s*=\s*([\d.-]+)", cfg
    )
    assert hard is not None and weight is not None
    # Mirrors the parent's `truck_raw_reaction_termination_m`, never replaces it.
    assert float(hard.group(1)) == 0.05
    assert float(weight.group(1)) == 0.0, "must stay opt-in"
    parent = (ROOT / "PPO2_ROM_PhysX/policy/env_cfg.py").read_text()
    parent_hard = re.search(
        r"truck_raw_reaction_termination_m:\s*float\s*=\s*([\d.]+)", parent
    )
    if parent_hard is not None:
        assert float(parent_hard.group(1)) == 0.05

    train = (ROOT / "PPO2_ROM_B_PhysX/scripts/train.py").read_text()
    assert "--truck_reaction_shaping" in train
    assert "must be <= 0" in train
    assert '"truck_reaction_margin_weight": float(a.truck_reaction_shaping)' in train


def test_num_envs_is_capped_at_the_validated_scene_scale():
    """Guard against silently corrupt physics, not just against crashes.

    Verified 2026-07-25: `--num_envs 4096` overflows the PhysX GPU contact buffer
    at t=15 s, before any PPO update. PhysX drops contacts and keeps running, so
    the first curriculum probes still produced numbers -- acc95 48-54 m/s^2 vs
    31-40 in the 128-env audit, safety rate 0.13-0.15 vs 0.01-0.03, completion
    0.85-0.87 -- and only later died with NaN in cylinder_q. Plausible-looking
    output from broken dynamics is the reason this is a hard stop.
    """

    train = (ROOT / "PPO2_ROM_B_PhysX/scripts/train.py").read_text()
    cap = re.search(r"^VALIDATED_MAX_NUM_ENVS\s*=\s*(\d+)\s*$", train, re.MULTILINE)
    assert cap is not None, "the validated environment ceiling must be explicit"
    assert int(cap.group(1)) == 128, "every ROM-B artifact was produced at 128"
    assert "if a.num_envs > VALIDATED_MAX_NUM_ENVS:" in train
    assert "default=VALIDATED_MAX_NUM_ENVS" in train


def test_certification_requires_per_rollout_process_isolation():
    """A sequential audit is not certifiable and must say so in its own output.

    Measured 2026-07-25 on `model_53`. Re-running the six failing h conditions
    one process each: truck_raw_reaction failures 100 -> 14 while head_alignment
    held at 50 -> 47, and all six passed (worst bound 0.0352 vs the 0.05 gate).
    A 3-rollout smoke test shows the mechanism directly -- the first rollout is
    bit-identical between the two modes and every later one diverges further:

        isolated   0.02368, 0.023967, 0.025506
        sequential 0.02368, 0.025771, 0.029034

    So the report must record which mode produced it, and the sequential path
    must refuse to look certifiable.
    """

    source = (ROOT / "PPO2_ROM_B_PhysX/scripts/evaluate.py").read_text()
    assert "--isolate_conditions" in source
    assert '"rollout_process_isolation": bool(isolated)' in source
    assert 'f"mc:{shape_index}:{round_index}"' in source
    assert 'f"h:{h_index}:{shape_index}:{round_index}"' in source
    assert "certification; re-run with --isolate_conditions." in source
    # A dropped shard would shrink the denominator and flatter the bound, so a
    # failing child must be fatal rather than skipped.
    assert "raise RuntimeError(" in source
    assert "failed with code {result.returncode}" in source


def test_retiming_keeps_geometric_start_phase_distribution_fixed():
    source = (ROOT / "PPO2_ROM_PhysX/policy/env.py").read_text()
    assert 'row.get("timing_s_m", row["path_s_m"])' in source
    assert "geometry_start" in source
    assert "self._geometry_to_timing(" in source
    assert "self._reference_timing_s" in source


def test_evaluation_binds_report_to_trajectory_library_hash():
    source = (ROOT / "PPO2_ROM_B_PhysX/scripts/evaluate.py").read_text()
    assert '"trajectory_library": str(trajectory_library)' in source
    assert '"trajectory_library_sha256": _sha256(trajectory_library)' in source


def test_zero_residual_full_audit_is_explicitly_non_certifiable():
    source = (ROOT / "PPO2_ROM_B_PhysX/scripts/evaluate.py").read_text()
    assert '"--force_zero_policy"' in source
    assert 'force_zero_policy=bool(a.force_zero_policy)' in source
    assert 'and not bool(a.force_zero_policy)' in source
    assert '"forced_zero_residual" if a.force_zero_policy' in source


def test_privileged_oracle_is_opt_in_and_marked_non_certifiable():
    """The oracle breaks the factor-blind contract on purpose; it must say so.

    Rationale for its existence: an online estimator such as AAM can only be
    worse than the true factors, so the oracle's margin over factor-blind ROM-B
    upper-bounds AAM's margin over ROM-B. If the oracle does not beat ROM-B, no
    adaptation scheme can. That makes it the cheapest experiment to run before
    committing to an AAM comparison -- but it is a diagnostic, and an oracle run
    must never be reachable as a robustness result.
    """

    cfg = (ROOT / "PPO2_ROM_B_PhysX/env_cfg.py").read_text()
    match = re.search(
        r"^\s*rom_b_privileged_factors:\s*bool\s*=\s*(\w+)\s*$", cfg, re.MULTILINE
    )
    assert match is not None and match.group(1) == "False", "must default off"

    train_source = (ROOT / "PPO2_ROM_B_PhysX/scripts/train.py").read_text()
    eval_source = (ROOT / "PPO2_ROM_B_PhysX/scripts/evaluate.py").read_text()
    for path, source in (
        ("PPO2_ROM_B_PhysX/scripts/train.py", train_source),
        ("PPO2_ROM_B_PhysX/scripts/evaluate.py", eval_source),
    ):
        assert "--privileged_oracle" in source
    assert '"certifiable": not bool(a.privileged_oracle)' in train_source
    certifiable = eval_source.split('"certifiable": (', 1)[1].split("),", 1)[0]
    assert "bool(isolated)" in certifiable
    assert "not bool(a.privileged_oracle)" in certifiable
    assert "not bool(a.force_zero_policy)" in certifiable

    train = (ROOT / "PPO2_ROM_B_PhysX/scripts/train.py").read_text()
    # Separate log tree: an oracle checkpoint must not be reachable by anything
    # expecting a factor-blind policy.
    assert 'experiment_name = "ppo2_rom_b_privileged_oracle"' in train
    assert "if a.privileged_oracle and a.plant_observer:" in train
    # Fairness: the oracle must start where ROM-B started (PPO2_A, zero-padded
    # into the wider observation), not from scratch. `model_53` was itself a
    # fine-tune of PPO2_A, so a from-scratch oracle would differ in
    # initialization and effective budget as well as in observation content, and
    # the resulting gap would be uninterpretable.
    assert "if a.privileged_oracle or a.plant_observer:" in train
    assert "added_dims=added" in train


def test_privileged_observation_width_excludes_the_inert_factor():
    """The oracle consumes exactly the five public ROM-B factors."""

    # `PPO2_ROM_B_PhysX.env` imports Isaac Lab, so parse the module instead.
    import ast

    tree = ast.parse((ROOT / "PPO2_ROM_B_PhysX/env.py").read_text())
    consts: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    consts[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    names = consts["PRIVILEGED_FACTOR_NAMES"]

    from PPO2_ROM_B_PhysX.runtime.rom_b_runtime import (
        FACTOR_NAMES,
        SUPPORTED_FACTOR_NAMES,
    )

    assert tuple(names) == tuple(SUPPORTED_FACTOR_NAMES)
    assert tuple(FACTOR_NAMES) == tuple(SUPPORTED_FACTOR_NAMES)


def test_logs_and_contract_name_only_the_five_rom_b_factors():

    env = (ROOT / "PPO2_ROM_B_PhysX/env.py").read_text()
    assert "SUPPORTED_FACTOR_NAMES" in env
    assert "e_flow" not in env


def test_rom_b_and_rom_c_default_to_the_retimed_circle_library():
    expected = "plant_reachable_trajectory_library_v3_circle_c7_retime.json"
    paths = (
        ROOT / "PPO2_ROM_B_PhysX/env_cfg.py",
        ROOT / "PPO2_ROM_B_PhysX/scripts/train.py",
        ROOT / "PPO2_ROM_B_PhysX/scripts/evaluate.py",
        ROOT / "PPO2_ROM_B_PhysX/scripts/diagnose.py",
        ROOT / "PPO2_ROM_C_PhysX/scripts/train.py",
    )
    for path in paths:
        assert expected in path.read_text(), path
    env_cfg = (ROOT / "PPO2_ROM_PhysX/policy/env_cfg.py").read_text()
    assert "dls_velocity_filter_beta: float = 0.0" in env_cfg


def test_run_contract_reports_factor_blindness_truthfully():
    """The oracle must not record itself as factor-blind.

    These fields were hardcoded True, which was correct until
    --privileged_oracle existed. An oracle run would then have claimed a
    factor-blind actor while feeding the true factors to both networks -- the
    single most misleading thing a run contract could say.
    """

    train = (ROOT / "PPO2_ROM_B_PhysX/scripts/train.py").read_text()
    assert '"actor_factor_blind": not bool(a.privileged_oracle)' in train
    assert '"critic_factor_blind": not bool(a.privileged_oracle)' in train
    assert '"actor_factor_blind": True' not in train


def test_observer_intervention_modes_are_declared():
    source = (ROOT / "PPO2_ROM_B_PhysX/env_cfg.py").read_text()
    match = re.search(
        r'^\s*rom_b_plant_observer_mode:\s*str\s*=\s*"(\w+)"\s*$', source, re.MULTILINE
    )
    assert match is not None and match.group(1) == "on"
    env_source = (ROOT / "PPO2_ROM_B_PhysX/env.py").read_text()
    for mode in ("zero", "shuffle"):
        assert f'"{mode}"' in env_source
