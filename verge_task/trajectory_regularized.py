"""Reference-trajectory regularization for continuous-action PPO.

This module adapts Distributional DAgger/DistIL's future-disagreement credit
to the feed-forward Gaussian PPO used by the terrain task.  The frozen
reference is queried only on states visited by PPO.  It never supplies an
executed action, so PPO remains the sole plant controller.

The implementation intentionally has no replay buffer: actions, old policy
probabilities, reference disagreement and reward advantages all come from the
same on-policy rollout.  The future cost is strictly future-only (tau > t), is
cut at episode boundaries, and is truncated at the rollout boundary.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ReferenceTrajectoryPPOCfg:
    """Coefficients and trust-region limits for reference-anchored PPO."""

    local_coef: float = 0.05
    # Optional DistIL-style future disagreement is experimental here because
    # the frozen ROM reference is not a uniformly better teacher.  The
    # production default is therefore the plain fixed-reference KL anchor.
    future_coef: float = 0.0
    max_update_kl: float = 0.01
    max_reference_kl: float = 0.05
    eps: float = 1.0e-8

    def validate(self) -> None:
        if self.local_coef < 0.0 or self.future_coef < 0.0:
            raise ValueError("trajectory coefficients must be non-negative")
        if self.local_coef == 0.0 and self.future_coef == 0.0:
            raise ValueError("at least one trajectory coefficient must be positive")
        if self.max_update_kl <= 0.0:
            raise ValueError("max_update_kl must be positive")
        if self.max_reference_kl <= 0.0:
            raise ValueError("max_reference_kl must be positive")


def diagonal_gaussian_kl(
    mean_p: torch.Tensor,
    std_p: torch.Tensor,
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return per-sample ``KL(N_p || N_q)`` summed over action axes."""

    if mean_p.shape != mean_q.shape:
        raise ValueError(f"Gaussian mean shapes differ: {mean_p.shape} vs {mean_q.shape}")
    std_p = torch.broadcast_to(std_p, mean_p.shape).clamp_min(eps)
    std_q = torch.broadcast_to(std_q, mean_q.shape).clamp_min(eps)
    return (
        torch.log(std_q / std_p)
        + (std_p.square() + (mean_p - mean_q).square()) / (2.0 * std_q.square())
        - 0.5
    ).sum(dim=-1)


def diagonal_gaussian_cross_entropy(
    mean_p: torch.Tensor,
    std_p: torch.Tensor,
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return per-sample forward cross-entropy ``H(N_p, N_q)``."""

    std_p = torch.broadcast_to(std_p, mean_p.shape).clamp_min(eps)
    std_q = torch.broadcast_to(std_q, mean_q.shape).clamp_min(eps)
    return (
        0.5 * math.log(2.0 * math.pi)
        + torch.log(std_q)
        + (std_p.square() + (mean_p - mean_q).square()) / (2.0 * std_q.square())
    ).sum(dim=-1)


def strict_future_cost(
    local_cost: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average disagreement strictly after each action within the same episode.

    Args:
        local_cost: Tensor shaped ``[time, env]`` (a final singleton is allowed).
        dones: Done flags for the action at each ``[time, env]`` location.

    Returns:
        ``(future_cost, valid)``.  ``valid`` is false when no later state from
        the same episode is present in this rollout.  Current-step cost is never
        included, and costs beyond a done flag never leak into the prior episode.
    """

    if local_cost.ndim == 3 and local_cost.shape[-1] == 1:
        local_cost = local_cost.squeeze(-1)
    if dones.ndim == 3 and dones.shape[-1] == 1:
        dones = dones.squeeze(-1)
    if local_cost.ndim != 2 or dones.shape != local_cost.shape:
        raise ValueError(
            f"expected matching [time, env] costs/dones, got {local_cost.shape}/{dones.shape}"
        )

    future = torch.zeros_like(local_cost)
    valid = torch.zeros_like(local_cost, dtype=torch.bool)
    running_sum = torch.zeros(local_cost.shape[1], device=local_cost.device, dtype=local_cost.dtype)
    running_count = torch.zeros_like(running_sum)

    for step in range(local_cost.shape[0] - 1, -1, -1):
        ended = dones[step].bool()
        running_sum = torch.where(ended, torch.zeros_like(running_sum), running_sum)
        running_count = torch.where(ended, torch.zeros_like(running_count), running_count)
        has_future = running_count > 0
        future[step] = torch.where(
            has_future,
            running_sum / running_count.clamp_min(1.0),
            torch.zeros_like(running_sum),
        )
        valid[step] = has_future
        running_sum = running_sum + local_cost[step]
        running_count = running_count + 1.0

    return future, valid


def center_future_cost(
    future_cost: torch.Tensor,
    valid: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Apply a detached scalar policy-gradient baseline to valid future costs."""

    if future_cost.shape != valid.shape:
        raise ValueError("future_cost and valid must have the same shape")
    centered = torch.zeros_like(future_cost)
    values = future_cost[valid]
    if values.numel() == 0:
        return centered
    scale = values.std(unbiased=False)
    centered[valid] = (values - values.mean()) / scale.clamp_min(eps)
    return centered


def _normalizer_modules(model: nn.Module) -> list[nn.Module]:
    return [module for module in model.modules() if type(module).__name__ == "EmpiricalNormalization"]


def normalizer_snapshot(*models: nn.Module) -> dict[str, torch.Tensor]:
    """Clone every empirical-normalizer buffer for drift assertions."""

    result: dict[str, torch.Tensor] = {}
    for model_index, model in enumerate(models):
        for module_index, module in enumerate(_normalizer_modules(model)):
            for name, value in module.named_buffers(recurse=False):
                result[f"{model_index}.{module_index}.{name}"] = value.detach().clone()
    return result


def assert_normalizers_unchanged(
    before: dict[str, torch.Tensor],
    *models: nn.Module,
) -> None:
    """Raise immediately if a supposedly frozen input transform moved."""

    after = normalizer_snapshot(*models)
    if before.keys() != after.keys():
        raise RuntimeError("empirical-normalizer topology changed during training")
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if changed:
        raise RuntimeError(f"frozen empirical-normalizer buffers changed: {changed}")


def reseed_frozen_speed_normalizer(
    algorithm: Any,
    speed_min_kmh: float,
    speed_max_kmh: float,
    constant_channel_std: float = 1.0e-2,
) -> list[str]:
    """Repair a frozen normalizer inherited from a NARROWER speed range.

    A resumed normalizer is part of the policy, so it is frozen rather than
    re-estimated.  That contract breaks the moment a run randomizes an input
    the frozen statistics saw as a constant.  RSL-RL normalizes as
    ``(x - mean) / (std + eps)`` with ``eps = 1e-2``, so a channel frozen at
    ``std = 0`` is not merely stale -- it is multiplied by 100.

    Measured on ``model_6766.pt``, whose normalizer came from a fixed 2 km/h
    run (mean 0.5556 m/s, std 0.0) and was then reused for 2--7 km/h::

        2 km/h -> feature   +0.00     (the value the weights were trained on)
        3 km/h -> feature  +27.78
        7 km/h -> feature +138.89

    The first-layer contribution of that one channel reaches 163 against 99.6
    for the summed column norms of all 65 others at one sigma.  The policy
    passed every gate at 2.0 km/h and collapsed at every other speed -- head
    alignment inverted, a third of episodes terminating on safety -- while the
    privileged teacher tracked 3 km/h at 0.119 m on the same mesh.  It read as
    an RL failure for eleven hours; no amount of training can fix an input.

    Reseeding the speed channel alone is worth 8.5x on this task: scored at
    3 km/h on mesh 1337, ``model_6766.pt`` goes from 0.896 m worst-family
    lateral error and a 33.4% safety-violation rate to 0.106 m and 0.16%, with
    no weight changed -- past the teacher's own 0.119 m.

    Only channels that were *exactly* constant are floored.  A genuinely small
    standard deviation is a real statistic (``rom_delta_extension`` sits at
    5e-4) and raising it would rescale a working policy's inputs; a channel at
    exactly zero carries no information about its own spread, so it gets
    RSL-RL's own ``eps`` and a loud line in the log rather than an invented
    distribution.  Idempotent; returns a description of every change.
    """

    low = float(speed_min_kmh)
    high = float(speed_max_kmh)
    if not 0.0 < low <= high:
        raise ValueError(f"speed range must satisfy 0 < min <= max; got {low}, {high}")
    floor = float(constant_channel_std)
    if floor <= 0.0:
        raise ValueError(f"constant-channel std must be positive, got {floor}")

    from verge_task.terrain_observations import TRUCK_SPEED_OBS_INDEX

    # Uniform on [low, high] in m/s -- the contract the env samples under.
    speed_mean = 0.5 * (low + high) / 3.6
    speed_std = (high - low) / 3.6 / math.sqrt(12.0)

    changes: list[str] = []
    models = (algorithm.actor, algorithm.critic)
    for model_index, model in enumerate(models):
        for module in _normalizer_modules(model):
            mean = module._mean
            std = module._std
            var = module._var
            index = TRUCK_SPEED_OBS_INDEX
            if index >= mean.shape[-1]:
                raise RuntimeError(
                    f"observation width {mean.shape[-1]} cannot hold the truck "
                    f"speed channel at index {index}; the layout moved"
                )
            # Test COVERAGE, not just spread. A normalizer can have a perfectly
            # healthy standard deviation and still be centred on the wrong
            # speed: `promoted_policy.pt` carried mean 0.9671 m/s (3.5 km/h)
            # and std 0.2435, which passed a spread-only check while putting
            # 7 km/h at +3.85 sigma. Reseeding it to the sampled range moved
            # that to +1.69 sigma and took cut height 0.1084 -> 0.0298, safety
            # 0.2137 -> 0.0571 and ground contact 0.2066 -> 0.6168 with no
            # weight changed.
            #
            # Uniform on [low, high] puts its own extremes at +-1.73 sigma, so
            # requiring both ends within 2 sigma accepts a normalizer that
            # already covers the range and rejects one that does not, whether
            # the cause is a narrow spread or a shifted centre.
            frozen_mean = float(mean[0, index])
            frozen_scale = float(std[0, index]) + 1.0e-2  # RSL-RL's own eps
            worst_z = max(
                abs(low / 3.6 - frozen_mean) / frozen_scale,
                abs(high / 3.6 - frozen_mean) / frozen_scale,
            )
            if high > low and worst_z > 2.0:
                changes.append(
                    f"model{model_index}[{index}] truck_speed_mps "
                    f"mean {frozen_mean:.4f}->{speed_mean:.4f} "
                    f"std {float(std[0, index]):.4f}->{speed_std:.4f} "
                    f"(range hit {worst_z:.1f} sigma)"
                )
                mean[0, index] = speed_mean
                std[0, index] = speed_std
                var[0, index] = speed_std * speed_std
            # Any remaining exactly-constant channel is a latent copy of the
            # same bug: benign only for as long as it stays exactly constant.
            for other in (std[0] <= 0.0).nonzero().flatten().tolist():
                changes.append(
                    f"model{model_index}[{int(other)}] std "
                    f"0.0->{floor:.4f} (channel was constant while frozen)"
                )
                std[0, other] = floor
                var[0, other] = floor * floor
    return changes


def freeze_normalizers_across_train_mode(algorithm: Any) -> dict[str, torch.Tensor]:
    """Make RSL-RL's repeated ``train_mode`` calls preserve frozen statistics."""

    models = (algorithm.actor, algorithm.critic)
    modules = [module for model in models for module in _normalizer_modules(model)]
    for module in modules:
        module.eval()
    snapshot = normalizer_snapshot(*models)
    original_train_mode = algorithm.train_mode

    def train_mode_with_frozen_normalizers() -> None:
        original_train_mode()
        for module in modules:
            module.eval()
        assert_normalizers_unchanged(snapshot, *models)

    algorithm.train_mode = train_mode_with_frozen_normalizers
    return snapshot


def restore_optimizer_group_state(
    optimizer: torch.optim.Optimizer,
    snapshot: dict[str, Any],
    group_index: int,
) -> None:
    """Restore one parameter group's Adam state without rewinding other groups."""

    current = optimizer.state_dict()
    if len(current["param_groups"]) != len(snapshot["param_groups"]):
        raise RuntimeError("optimizer parameter-group topology changed during update")
    if not 0 <= group_index < len(current["param_groups"]):
        raise IndexError(f"optimizer group {group_index} does not exist")
    current_ids = current["param_groups"][group_index]["params"]
    snapshot_ids = snapshot["param_groups"][group_index]["params"]
    if len(current_ids) != len(snapshot_ids):
        raise RuntimeError("optimizer parameter topology changed during update")
    for current_id, snapshot_id in zip(current_ids, snapshot_ids, strict=True):
        if snapshot_id in snapshot["state"]:
            current["state"][current_id] = copy.deepcopy(snapshot["state"][snapshot_id])
        else:
            current["state"].pop(current_id, None)
    optimizer.load_state_dict(current)


class ReferenceTrajectoryPPOUpdate:
    """Drop-in feed-forward PPO update with local and future reference terms."""

    def __init__(self, algorithm: Any, reference_actor: nn.Module, cfg: ReferenceTrajectoryPPOCfg):
        cfg.validate()
        if algorithm.actor.is_recurrent or algorithm.critic.is_recurrent:
            raise ValueError("reference-trajectory PPO currently requires feed-forward actor/critic")
        if algorithm.rnd is not None or algorithm.symmetry is not None:
            raise ValueError("reference-trajectory PPO does not silently drop RND or symmetry losses")
        if algorithm.schedule != "fixed":
            raise ValueError("reference-trajectory PPO requires a fixed learning-rate schedule")
        self.alg = algorithm
        self.reference = reference_actor
        self.cfg = cfg

    @torch.no_grad()
    def _distribution_params(self, actor: nn.Module, observations: Any) -> tuple[torch.Tensor, torch.Tensor]:
        actor(observations, stochastic_output=True)
        return actor.distribution.mean.detach().clone(), actor.distribution.std.detach().clone()

    @torch.no_grad()
    def _full_kl(
        self,
        observations: Any,
        old_params: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        live_mean, live_std = self._distribution_params(self.alg.actor, observations)
        if old_params is None:
            ref_mean, ref_std = self._distribution_params(self.reference, observations)
            return diagonal_gaussian_kl(ref_mean, ref_std, live_mean, live_std, eps=self.cfg.eps)
        old_mean, old_std = old_params
        return diagonal_gaussian_kl(old_mean, old_std, live_mean, live_std, eps=self.cfg.eps)

    @torch.no_grad()
    def _future_targets(
        self,
        observations: Any,
        dones: torch.Tensor,
        time_steps: int,
        num_envs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference_mean, reference_std = self._distribution_params(self.reference, observations)
        live_mean, live_std = self._distribution_params(self.alg.actor, observations)
        local = diagonal_gaussian_kl(
            reference_mean,
            reference_std,
            live_mean,
            live_std,
            eps=self.cfg.eps,
        ).reshape(time_steps, num_envs)
        future, valid = strict_future_cost(local, dones.reshape(time_steps, num_envs))
        centered = center_future_cost(future, valid, eps=self.cfg.eps)
        return local.flatten(), future.flatten(), centered.flatten()

    def __call__(self) -> dict[str, float]:
        alg = self.alg
        storage = alg.storage
        if storage.step != storage.num_transitions_per_env:
            raise RuntimeError(
                f"incomplete PPO rollout: {storage.step}/{storage.num_transitions_per_env} transitions"
            )

        time_steps = int(storage.num_transitions_per_env)
        num_envs = int(storage.num_envs)
        batch_size = time_steps * num_envs
        mini_batch_size = batch_size // int(alg.num_mini_batches)
        if mini_batch_size <= 0 or mini_batch_size * int(alg.num_mini_batches) != batch_size:
            raise ValueError("rollout size must divide exactly by num_mini_batches")

        observations = storage.observations.flatten(0, 1)
        actions = storage.actions.flatten(0, 1)
        values_old = storage.values.flatten(0, 1)
        returns = storage.returns.flatten(0, 1)
        advantages = storage.advantages.flatten(0, 1)
        old_log_prob = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
        old_params = tuple(param.flatten(0, 1).detach().clone() for param in storage.distribution_params)

        actor_state = copy.deepcopy(alg.actor.state_dict())
        critic_state = copy.deepcopy(alg.critic.state_dict())
        optimizer_state = copy.deepcopy(alg.optimizer.state_dict())
        reference_kl_before_update = float(self._full_kl(observations).mean())

        sums = {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "reference_local_kl": 0.0,
            "reference_future_loss": 0.0,
            "clip_fraction": 0.0,
            "actor_grad_norm": 0.0,
        }
        updates = 0
        early_stopped = False
        raw_future = torch.zeros(batch_size, device=actions.device)
        future_valid = torch.zeros(batch_size, device=actions.device, dtype=torch.bool)

        for _epoch in range(int(alg.num_learning_epochs)):
            local_rollout, raw_future, future_adv = self._future_targets(
                observations,
                storage.dones,
                time_steps,
                num_envs,
            )
            # A zero raw future cost can be legitimate at the warm boundary.
            # Validity is determined by episode topology, not by cost magnitude.
            _, future_valid_2d = strict_future_cost(
                local_rollout.reshape(time_steps, num_envs),
                storage.dones.reshape(time_steps, num_envs),
            )
            future_valid = future_valid_2d.flatten()
            permutation = torch.randperm(batch_size, device=actions.device)

            for mini_batch in range(int(alg.num_mini_batches)):
                start = mini_batch * mini_batch_size
                idx = permutation[start : start + mini_batch_size]
                obs_batch = observations[idx]

                alg.actor(obs_batch, stochastic_output=True)
                live_mean = alg.actor.distribution.mean
                live_std = alg.actor.distribution.std
                actions_log_prob = alg.actor.get_output_log_prob(actions[idx])
                entropy = alg.actor.output_entropy
                new_params = (live_mean, live_std)

                with torch.no_grad():
                    old_to_new = diagonal_gaussian_kl(
                        old_params[0][idx],
                        old_params[1][idx],
                        live_mean,
                        live_std,
                        eps=self.cfg.eps,
                    ).mean()
                if not torch.isfinite(old_to_new) or float(old_to_new) > self.cfg.max_update_kl:
                    early_stopped = True
                    break

                ratio = torch.exp(actions_log_prob - old_log_prob[idx])
                advantage = advantages[idx].squeeze(-1)
                surrogate = -advantage * ratio
                surrogate_clipped = -advantage * torch.clamp(
                    ratio, 1.0 - alg.clip_param, 1.0 + alg.clip_param
                )
                surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()

                predicted_values = alg.critic(obs_batch)
                if alg.use_clipped_value_loss:
                    value_clipped = values_old[idx] + (predicted_values - values_old[idx]).clamp(
                        -alg.clip_param, alg.clip_param
                    )
                    value_loss = torch.maximum(
                        (predicted_values - returns[idx]).square(),
                        (value_clipped - returns[idx]).square(),
                    ).mean()
                else:
                    value_loss = (returns[idx] - predicted_values).square().mean()

                with torch.no_grad():
                    self.reference(obs_batch, stochastic_output=True)
                    reference_mean = self.reference.distribution.mean.detach().clone()
                    reference_std = self.reference.distribution.std.detach().clone()
                local_kl = diagonal_gaussian_kl(
                    reference_mean,
                    reference_std,
                    live_mean,
                    live_std,
                    eps=self.cfg.eps,
                ).mean()

                valid = future_valid[idx]
                if bool(valid.any()):
                    cost_advantage = future_adv[idx][valid]
                    future_ratio = ratio[valid]
                    future_ratio_clipped = torch.clamp(
                        future_ratio, 1.0 - alg.clip_param, 1.0 + alg.clip_param
                    )
                    future_loss = torch.maximum(
                        future_ratio * cost_advantage,
                        future_ratio_clipped * cost_advantage,
                    ).mean()
                else:
                    future_loss = live_mean.sum() * 0.0

                loss = (
                    surrogate_loss
                    + alg.value_loss_coef * value_loss
                    - alg.entropy_coef * entropy.mean()
                    + self.cfg.local_coef * local_kl
                    + self.cfg.future_coef * future_loss
                )
                if not torch.isfinite(loss):
                    early_stopped = True
                    break

                alg.optimizer.zero_grad()
                loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(alg.actor.parameters(), alg.max_grad_norm)
                nn.utils.clip_grad_norm_(alg.critic.parameters(), alg.max_grad_norm)
                alg.optimizer.step()

                updates += 1
                sums["value"] += float(value_loss.detach())
                sums["surrogate"] += float(surrogate_loss.detach())
                sums["entropy"] += float(entropy.mean().detach())
                sums["reference_local_kl"] += float(local_kl.detach())
                sums["reference_future_loss"] += float(future_loss.detach())
                sums["clip_fraction"] += float(
                    ((ratio - 1.0).abs() > alg.clip_param).float().mean().detach()
                )
                sums["actor_grad_norm"] += float(actor_grad_norm.detach())

            if early_stopped:
                break

        attempted_update_kl = float(self._full_kl(observations, old_params).mean())
        attempted_reference_kl = float(self._full_kl(observations).mean())
        reject_nonfinite = (
            updates == 0
            or not math.isfinite(attempted_update_kl)
            or not math.isfinite(attempted_reference_kl)
        )
        reject_update_kl = attempted_update_kl > self.cfg.max_update_kl
        reject_reference_kl = attempted_reference_kl > self.cfg.max_reference_kl
        reject = reject_nonfinite or reject_update_kl or reject_reference_kl
        critic_update_retained = False
        if reject_nonfinite:
            alg.actor.load_state_dict(actor_state)
            alg.critic.load_state_dict(critic_state)
            alg.optimizer.load_state_dict(optimizer_state)
        elif reject:
            # KL constraints apply to the policy, not the value function.  A
            # rejected actor step was previously rewinding valid critic fits on
            # the same on-policy rollout, wasting most batches at the boundary.
            alg.actor.load_state_dict(actor_state)
            restore_optimizer_group_state(alg.optimizer, optimizer_state, 0)
            critic_update_retained = updates > 0

        final_update_kl = float(self._full_kl(observations, old_params).mean())
        final_reference_kl = float(self._full_kl(observations).mean())
        storage.clear()

        denominator = max(updates, 1)
        result = {name: value / denominator for name, value in sums.items()}
        valid_future_values = raw_future[future_valid]
        result.update(
            {
                "policy_kl": final_update_kl,
                "policy_kl_attempted": attempted_update_kl,
                "reference_kl": final_reference_kl,
                "reference_kl_before_update": reference_kl_before_update,
                "reference_kl_attempted": attempted_reference_kl,
                "reference_future_cost_mean": (
                    float(valid_future_values.mean()) if valid_future_values.numel() else 0.0
                ),
                "reference_future_cost_std": (
                    float(valid_future_values.std(unbiased=False)) if valid_future_values.numel() else 0.0
                ),
                "reference_future_cost_p95": (
                    float(torch.quantile(valid_future_values, 0.95)) if valid_future_values.numel() else 0.0
                ),
                "reference_future_valid_fraction": float(future_valid.float().mean()),
                "update_rejected": float(reject),
                "critic_update_retained": float(critic_update_retained),
                "reject_nonfinite": float(reject_nonfinite),
                "reject_update_kl": float(reject_update_kl),
                "reject_reference_kl": float(reject_reference_kl),
                "kl_early_stop": float(early_stopped),
                "optimizer_minibatches": float(updates),
            }
        )
        return result


def install_reference_trajectory_ppo(
    algorithm: Any,
    reference_actor: nn.Module,
    cfg: ReferenceTrajectoryPPOCfg,
) -> ReferenceTrajectoryPPOUpdate:
    """Install and return the single-step reference-trajectory PPO updater."""

    updater = ReferenceTrajectoryPPOUpdate(algorithm, reference_actor, cfg)
    algorithm.update = updater
    return updater
