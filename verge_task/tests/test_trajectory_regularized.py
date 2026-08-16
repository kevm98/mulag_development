from __future__ import annotations

import copy

import torch

from verge_task.trajectory_regularized import (
    center_future_cost,
    diagonal_gaussian_cross_entropy,
    diagonal_gaussian_kl,
    restore_optimizer_group_state,
    strict_future_cost,
)


def test_restore_optimizer_group_state_keeps_critic_moments():
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(
        [{"params": actor.parameters()}, {"params": critic.parameters()}], lr=1e-3
    )
    x = torch.ones(2, 2)
    (actor(x).sum() + critic(x).sum()).backward()
    optimizer.step()
    actor_before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    optimizer.zero_grad()
    (2.0 * actor(x).sum() + 3.0 * critic(x).sum()).backward()
    optimizer.step()
    critic_after = {k: v.detach().clone() for k, v in critic.state_dict().items()}
    critic_moment_after = optimizer.state[next(iter(critic.parameters()))]["exp_avg"].clone()

    actor.load_state_dict(actor_before)
    restore_optimizer_group_state(optimizer, optimizer_before, 0)
    assert all(torch.equal(actor.state_dict()[k], v) for k, v in actor_before.items())
    assert all(torch.equal(critic.state_dict()[k], v) for k, v in critic_after.items())
    assert torch.equal(
        optimizer.state[next(iter(critic.parameters()))]["exp_avg"], critic_moment_after
    )


def test_forward_gaussian_kl_is_zero_only_for_matching_distributions():
    mean = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    std = torch.full_like(mean, 0.15)
    matching = diagonal_gaussian_kl(mean, std, mean, std)
    shifted = diagonal_gaussian_kl(mean, std, mean + 0.1, std)
    assert torch.allclose(matching, torch.zeros_like(matching), atol=1.0e-7)
    assert torch.all(shifted > 0.0)


def test_forward_cross_entropy_has_same_mean_gradient_as_forward_kl():
    teacher_mean = torch.tensor([[0.2, -0.1]])
    teacher_std = torch.full_like(teacher_mean, 0.15)
    student_mean = torch.tensor([[0.5, 0.4]], requires_grad=True)
    student_std = torch.full_like(student_mean, 0.15)
    ce = diagonal_gaussian_cross_entropy(
        teacher_mean, teacher_std, student_mean, student_std
    ).sum()
    ce_grad = torch.autograd.grad(ce, student_mean, retain_graph=True)[0]
    kl = diagonal_gaussian_kl(
        teacher_mean, teacher_std, student_mean, student_std
    ).sum()
    kl_grad = torch.autograd.grad(kl, student_mean)[0]
    assert torch.allclose(ce_grad, kl_grad)


def test_future_cost_excludes_current_and_resets_at_done():
    local = torch.tensor([[1.0], [2.0], [3.0], [10.0], [20.0]])
    dones = torch.tensor([[0], [0], [1], [0], [1]], dtype=torch.bool)
    future, valid = strict_future_cost(local, dones)
    assert torch.allclose(future[:, 0], torch.tensor([2.5, 3.0, 0.0, 20.0, 0.0]))
    assert torch.equal(valid[:, 0], torch.tensor([True, True, False, True, False]))


def test_future_cost_does_not_cross_rollout_edge_or_episode_boundary():
    local = torch.tensor([[4.0, 1.0], [8.0, 2.0], [16.0, 3.0]])
    dones = torch.tensor([[0, 1], [0, 0], [0, 0]], dtype=torch.bool)
    future, valid = strict_future_cost(local, dones)
    assert torch.allclose(future[:, 0], torch.tensor([12.0, 16.0, 0.0]))
    assert torch.allclose(future[:, 1], torch.tensor([0.0, 3.0, 0.0]))
    assert torch.equal(valid[:, 1], torch.tensor([False, True, False]))


def test_center_future_cost_is_zero_mean_only_on_valid_entries():
    future = torch.tensor([[1.0, 100.0], [3.0, 200.0]])
    valid = torch.tensor([[True, False], [True, False]])
    centered = center_future_cost(future, valid)
    assert torch.allclose(centered[valid].mean(), torch.tensor(0.0))
    assert torch.equal(centered[~valid], torch.zeros(2))
