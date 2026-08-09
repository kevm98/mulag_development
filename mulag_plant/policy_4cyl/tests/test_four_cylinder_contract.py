from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from mulag_plant.policy.actions import apply_residual_action
from mulag_plant.policy.observations import (
    FOUR_CYLINDER_RESIDUAL_OBS_DIM,
    build_residual_observation,
)
from mulag_plant.policy_4cyl.checkpoint_transfer import (
    assert_frozen_teacher_unchanged,
    configure_joint_finetune,
    expand_three_to_four_state_dict,
    expand_three_to_pitch_branch_state_dict,
    freeze_teacher_and_train_cylinder4_head,
    snapshot_frozen_teacher,
)
from mulag_plant.policy_4cyl.pitch_actor import (
    FrozenTeacherPitchActor,
)
from mulag_plant.policy_4cyl.promotion_probe import (
    compare_probe_results,
)


ROOT = Path(__file__).resolve().parents[3]


def _observation(previous_action: torch.Tensor) -> torch.Tensor:
    n = previous_action.shape[0]
    return build_residual_observation(
        tool_error_truck=torch.zeros(n, 3),
        measured_tool_velocity_truck=torch.zeros(n, 3),
        reference_tool_velocity_truck=torch.zeros(n, 3),
        reference_tool_acceleration_truck=torch.zeros(n, 3),
        measured_cylinder_extensions=torch.zeros(n, 4),
        cylinder_goal_error=torch.zeros(n, 4),
        measured_cylinder_velocities=torch.zeros(n, 4),
        previous_filtered_policy_action=previous_action,
        current_valve_commands=torch.zeros(n, 4),
        corrected_fd_delta=torch.zeros(n, 4),
        reference_preview_xy_truck=torch.zeros(n, 2),
        trajectory_direction=torch.zeros(n),
        cutter_bottom_world_clearance=torch.zeros(n),
        head_down_alignment=torch.ones(n),
        head_down_alignment_rate=torch.zeros(n),
    )


def test_four_action_observation_is_42_dimensional():
    obs = _observation(torch.zeros(5, 4))
    assert obs.shape == (5, FOUR_CYLINDER_RESIDUAL_OBS_DIM)
    assert FOUR_CYLINDER_RESIDUAL_OBS_DIM == 42


def test_cylinder4_authority_is_independently_bounded():
    baseline = torch.full((1, 4), 0.5)
    _, filtered, corrected = apply_residual_action(
        baseline_goal=baseline,
        previous_filtered_action=torch.zeros(1, 4),
        policy_action=torch.ones(1, 4),
        residual_scale_m=0.002,
        residual_scale_by_action_m=torch.tensor([0.002, 0.002, 0.002, 0.0005]),
        filter_beta=torch.zeros(4),
        learned_min=torch.zeros(1, 4),
        learned_max=torch.ones(1, 4),
    )
    assert torch.allclose(filtered, torch.ones_like(filtered))
    assert torch.allclose(
        corrected - baseline,
        torch.tensor([[0.002, 0.002, 0.002, 0.0005]]),
        atol=1.0e-7,
        rtol=1.0e-5,
    )


def test_zero_cylinder4_authority_reproduces_teacher_channel():
    baseline = torch.tensor([[0.3, 0.4, 0.5, 0.6]])
    _, _, corrected = apply_residual_action(
        baseline_goal=baseline,
        previous_filtered_action=torch.zeros(1, 4),
        policy_action=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        residual_scale_m=0.002,
        residual_scale_by_action_m=torch.tensor([0.002, 0.002, 0.002, 0.0]),
        filter_beta=torch.zeros(4),
        learned_min=torch.zeros(1, 4),
        learned_max=torch.ones(1, 4),
    )
    assert corrected[0, 3] == baseline[0, 3]


def test_model249_transfer_preserves_old_channels_and_zeros_new_output():
    checkpoint = (
        ROOT
        / "logs/rsl_rl/ppo2_rom_residual/2026-07-19_18-55-23/model_249.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload["actor_state_dict"]
    target = {key: value.clone() for key, value in source.items()}
    target["obs_normalizer._mean"] = torch.zeros(1, 42)
    target["obs_normalizer._var"] = torch.ones(1, 42)
    target["obs_normalizer._std"] = torch.ones(1, 42)
    target["mlp.0.weight"] = torch.zeros(source["mlp.0.weight"].shape[0], 42)
    target["mlp.6.weight"] = torch.zeros(4, source["mlp.6.weight"].shape[1])
    target["mlp.6.bias"] = torch.zeros(4)
    target["distribution.std_param"] = torch.zeros(4)
    expanded = expand_three_to_four_state_dict(source, target, actor=True)
    assert torch.equal(expanded["mlp.0.weight"][:, :27], source["mlp.0.weight"][:, :27])
    assert torch.count_nonzero(expanded["mlp.0.weight"][:, 27]) == 0
    assert torch.equal(expanded["mlp.0.weight"][:, 28:], source["mlp.0.weight"][:, 27:])
    assert torch.equal(expanded["mlp.6.weight"][:3], source["mlp.6.weight"])
    assert torch.count_nonzero(expanded["mlp.6.weight"][3]) == 0
    assert expanded["mlp.6.bias"][3] == 0.0
    assert torch.allclose(
        expanded["distribution.std_param"], torch.tensor([0.02, 0.02, 0.02, 0.01])
    )


class _Distribution(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.std_param = torch.nn.Parameter(torch.tensor([0.001, 0.001, 0.001, 0.08]))


class _Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(42, 8), torch.nn.ELU(), torch.nn.Linear(8, 4))
        self.distribution = _Distribution()
        self.register_buffer("normalizer_marker", torch.tensor([1.0]))

    def update_normalization(self, obs):
        self.normalizer_marker.add_(1.0)


class _Algorithm:
    def __init__(self):
        self.actor = _Actor()

    def get_policy(self):
        return self.actor


class _Runner:
    def __init__(self):
        self.alg = _Algorithm()


def test_only_cylinder4_actor_head_can_change():
    runner = _Runner()
    actor = runner.alg.get_policy()
    freeze_teacher_and_train_cylinder4_head(runner)
    frozen = snapshot_frozen_teacher(actor)
    actor.update_normalization(None)
    loss = actor.mlp(torch.ones(2, 42)).sum() + actor.distribution.std_param.sum()
    loss.backward()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    optimizer.step()
    assert_frozen_teacher_unchanged(actor, frozen)
    assert torch.count_nonzero(actor.mlp[-1].weight.grad[:3]) == 0
    assert torch.count_nonzero(actor.mlp[-1].weight.grad[3]) > 0
    assert actor.distribution.std_param.grad is None
    assert torch.equal(actor.normalizer_marker, torch.tensor([1.0]))


def _distribution_cfg():
    return {
        "class_name": "GaussianDistribution",
        "init_std": 0.02,
        "std_type": "scalar",
    }


def test_independent_pitch_branch_preserves_model249_outputs():
    checkpoint = (
        ROOT
        / "logs/rsl_rl/ppo2_rom_residual/2026-07-19_18-55-23/model_249.pt"
    )
    source = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )["actor_state_dict"]
    obs41 = torch.randn(8, 41)
    obs42 = torch.cat((obs41[:, :27], torch.zeros(8, 1), obs41[:, 27:]), dim=-1)
    old = MLPModel(
        TensorDict({"policy": obs41}, batch_size=[8]),
        {"actor": ["policy"]},
        "actor",
        3,
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution_cfg(),
    )
    old.load_state_dict(source, strict=True)
    new = FrozenTeacherPitchActor(
        TensorDict({"policy": obs42}, batch_size=[8]),
        {"actor": ["policy"]},
        "actor",
        4,
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution_cfg(),
    )
    expanded = expand_three_to_pitch_branch_state_dict(
        source, new.state_dict()
    )
    new.load_state_dict(expanded, strict=True)
    old_actions = old(TensorDict({"policy": obs41}, batch_size=[8]))
    new_actions = new(TensorDict({"policy": obs42}, batch_size=[8]))
    assert torch.allclose(new_actions[:, :3], old_actions, atol=1.0e-6, rtol=1.0e-6)
    assert torch.count_nonzero(new_actions[:, 3]) == 0


def test_independent_pitch_branch_is_only_trainable_actor_path():
    obs = torch.randn(8, 42)
    actor = FrozenTeacherPitchActor(
        TensorDict({"policy": obs}, batch_size=[8]),
        {"actor": ["policy"]},
        "actor",
        4,
        hidden_dims=[16, 8],
        pitch_hidden_dims=[8, 4],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution_cfg(),
    )
    runner = _Runner()
    runner.alg.actor = actor
    freeze_teacher_and_train_cylinder4_head(runner)
    frozen = snapshot_frozen_teacher(actor)
    before = {
        key: value.detach().clone()
        for key, value in actor.mlp.pitch_mlp.state_dict().items()
    }
    action = actor(TensorDict({"policy": obs}, batch_size=[8]))
    action[:, 3].sum().backward()
    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    optimizer.step()
    assert_frozen_teacher_unchanged(actor, frozen)
    assert any(
        not torch.equal(value, before[key])
        for key, value in actor.mlp.pitch_mlp.state_dict().items()
    )
    assert all(
        parameter.grad is None
        for parameter in actor.mlp.teacher_mlp.parameters()
    )


def test_joint_finetune_trains_teacher_and_pitch_but_not_std():
    obs = torch.randn(4, 42)
    actor = FrozenTeacherPitchActor(
        TensorDict({"policy": obs}, batch_size=[4]),
        {"actor": ["policy"]},
        "actor",
        4,
        hidden_dims=[16, 8],
        pitch_hidden_dims=[8, 4],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution_cfg(),
    )
    runner = _Runner()
    runner.alg.actor = actor
    result = configure_joint_finetune(runner)
    assert result["architecture"] == "joint_teacher_and_pitch_branch"
    assert all(parameter.requires_grad for parameter in actor.mlp.parameters())
    assert not actor.distribution.std_param.requires_grad


def _probe(completed: list[bool], errors: list[float]):
    n = len(completed)
    error = torch.tensor(errors).reshape(1, n).repeat(2, 1)
    return {
        "completed": torch.tensor(completed),
        "error": error,
        "acceleration": torch.ones(2, n),
        "alignment": torch.ones(2, n),
        "drift": torch.zeros(2, n),
    }


def test_promotion_probe_matches_only_jointly_completed_environments():
    baseline = {"line": _probe([True, True, True, False], [1.0, 1.0, 1.0, 1.0])}
    # Candidate env 3 is deliberately terrible but baseline env 3 did not
    # complete; env 1 is excluded because the candidate did not complete it.
    candidate = {"line": _probe([True, False, True, True], [0.9, 99.0, 0.9, 99.0])}
    result = compare_probe_results(
        baseline,
        candidate,
        min_completion_fraction=0.5,
        min_mean_improvement=0.05,
        min_p95_improvement=0.05,
        max_acceleration_ratio=1.05,
        max_shape_acceleration_ratio=1.10,
        max_shape_regression_m=0.0,
    )
    assert result["passed"]
    line = result["per_shape"]["line"]
    assert line["paired_completion_fraction"] == 0.5
    assert line["baseline"]["matched_envs"] == 2
    assert abs(line["candidate"]["mean_error_m"] - 0.9) < 1.0e-6


def test_promotion_probe_rejects_insufficient_matched_completion():
    baseline = {"line": _probe([True, True, True, False], [1.0] * 4)}
    candidate = {"line": _probe([True, False, True, True], [0.9] * 4)}
    result = compare_probe_results(
        baseline,
        candidate,
        min_completion_fraction=0.75,
        min_mean_improvement=0.05,
        min_p95_improvement=0.05,
        max_acceleration_ratio=1.05,
        max_shape_acceleration_ratio=1.10,
        max_shape_regression_m=0.0,
    )
    assert not result["passed"]
    assert not result["checks"]["completion_fraction_safe"]
