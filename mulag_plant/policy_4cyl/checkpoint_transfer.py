"""Transfer a 41-observation/3-action checkpoint into the 42/4 architecture."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Mapping

import torch


OLD_PREVIOUS_ACTION_END = 27
NEW_CYLINDER4_PREVIOUS_ACTION_INDEX = 27


def expand_three_to_four_state_dict(
    source: Mapping[str, torch.Tensor],
    target_template: Mapping[str, torch.Tensor],
    *,
    actor: bool,
    cylinder123_std: float = 0.02,
    cylinder4_std: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Expand observation input and, for the actor, add a zero fourth output."""
    target = {key: value.detach().clone() for key, value in target_template.items()}
    for key, source_value in source.items():
        if key not in target:
            continue
        destination = target[key]
        if tuple(source_value.shape) == tuple(destination.shape):
            destination.copy_(source_value)

    for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        old = source[key]
        new = target[key]
        if tuple(old.shape) != (1, 41) or tuple(new.shape) != (1, 42):
            raise ValueError(f"unexpected normalizer shapes for {key}: {old.shape} -> {new.shape}")
        new[:, :OLD_PREVIOUS_ACTION_END] = old[:, :OLD_PREVIOUS_ACTION_END]
        # Index 27 is the new previous cylinder-4 action. Keep the target
        # normalizer's safe zero-mean/unit-variance initialization.
        new[:, 28:] = old[:, OLD_PREVIOUS_ACTION_END:]

    old_input = source["mlp.0.weight"]
    new_input = target["mlp.0.weight"]
    if old_input.shape[1] != 41 or new_input.shape[1] != 42:
        raise ValueError(f"unexpected first-layer shapes: {old_input.shape} -> {new_input.shape}")
    new_input.zero_()
    new_input[:, :OLD_PREVIOUS_ACTION_END] = old_input[:, :OLD_PREVIOUS_ACTION_END]
    new_input[:, 28:] = old_input[:, OLD_PREVIOUS_ACTION_END:]

    if actor:
        old_output = source["mlp.6.weight"]
        new_output = target["mlp.6.weight"]
        old_bias = source["mlp.6.bias"]
        new_bias = target["mlp.6.bias"]
        if old_output.shape[0] != 3 or new_output.shape[0] != 4:
            raise ValueError(f"unexpected actor output shapes: {old_output.shape} -> {new_output.shape}")
        new_output.zero_()
        new_bias.zero_()
        new_output[:3] = old_output
        new_bias[:3] = old_bias
        target["distribution.std_param"][:] = torch.tensor(
            [cylinder123_std, cylinder123_std, cylinder123_std, cylinder4_std],
            dtype=target["distribution.std_param"].dtype,
            device=target["distribution.std_param"].device,
        )
    return target


def expand_three_to_pitch_branch_state_dict(
    source: Mapping[str, torch.Tensor],
    target_template: Mapping[str, torch.Tensor],
    *,
    cylinder123_std: float = 0.001,
    cylinder4_std: float = 0.08,
) -> dict[str, torch.Tensor]:
    """Load the old actor into the frozen teacher and zero the pitch output."""
    target = {key: value.detach().clone() for key, value in target_template.items()}

    for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        old = source[key]
        new = target[key]
        if tuple(old.shape) != (1, 41) or tuple(new.shape) != (1, 42):
            raise ValueError(f"unexpected normalizer shapes for {key}: {old.shape} -> {new.shape}")
        new[:, :OLD_PREVIOUS_ACTION_END] = old[:, :OLD_PREVIOUS_ACTION_END]
        new[:, 28:] = old[:, OLD_PREVIOUS_ACTION_END:]
    target["obs_normalizer.count"].copy_(source["obs_normalizer.count"])

    for source_key, source_value in source.items():
        if not source_key.startswith("mlp."):
            continue
        suffix = source_key.removeprefix("mlp.")
        target_key = f"mlp.teacher_mlp.{suffix}"
        if target_key not in target:
            raise KeyError(f"missing teacher parameter {target_key}")
        destination = target[target_key]
        if source_key == "mlp.0.weight":
            if tuple(source_value.shape)[1] != 41 or tuple(destination.shape)[1] != 42:
                raise ValueError(
                    f"unexpected teacher input shapes: {source_value.shape} -> {destination.shape}"
                )
            destination.zero_()
            destination[:, :OLD_PREVIOUS_ACTION_END] = source_value[:, :OLD_PREVIOUS_ACTION_END]
            destination[:, 28:] = source_value[:, OLD_PREVIOUS_ACTION_END:]
        elif tuple(source_value.shape) != tuple(destination.shape):
            raise ValueError(
                f"unexpected teacher parameter shape for {source_key}: "
                f"{source_value.shape} -> {destination.shape}"
            )
        else:
            destination.copy_(source_value)

    # The custom actor constructor zeros the pitch output layer. Keep all of
    # its branch initialization from the target template and set exploration
    # explicitly for the four physical action channels.
    target["distribution.std_param"][:] = torch.tensor(
        [cylinder123_std, cylinder123_std, cylinder123_std, cylinder4_std],
        dtype=target["distribution.std_param"].dtype,
        device=target["distribution.std_param"].device,
    )
    return target


def transfer_checkpoint_into_runner(
    runner,
    checkpoint: str | Path,
    *,
    cylinder123_std: float = 0.001,
    cylinder4_std: float = 0.08,
) -> dict[str, object]:
    """Load transferred actor/critic weights but never the old optimizer/iteration."""
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = runner.alg.get_policy()
    critic = runner.alg.critic
    actor_template = actor.state_dict()
    pitch_branch = "mlp.teacher_mlp.0.weight" in actor_template
    if pitch_branch:
        actor_state = expand_three_to_pitch_branch_state_dict(
            payload["actor_state_dict"],
            actor_template,
            cylinder123_std=cylinder123_std,
            cylinder4_std=cylinder4_std,
        )
    else:
        actor_state = expand_three_to_four_state_dict(
            payload["actor_state_dict"],
            actor_template,
            actor=True,
            cylinder123_std=cylinder123_std,
            cylinder4_std=cylinder4_std,
        )
    critic_state = expand_three_to_four_state_dict(
        payload["critic_state_dict"], critic.state_dict(), actor=False
    )
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    if pitch_branch:
        final_pitch_weight = next(
            value for key, value in reversed(tuple(actor_state.items()))
            if key.startswith("mlp.pitch_mlp.") and key.endswith(".weight")
        )
        final_pitch_bias = next(
            value for key, value in reversed(tuple(actor_state.items()))
            if key.startswith("mlp.pitch_mlp.") and key.endswith(".bias")
        )
        fourth_output_weight_max = float(final_pitch_weight.abs().max())
        fourth_output_bias = float(final_pitch_bias.reshape(-1)[0])
    else:
        fourth_output_weight_max = float(actor_state["mlp.6.weight"][3].abs().max())
        fourth_output_bias = float(actor_state["mlp.6.bias"][3])
    return {
        "source": str(path),
        "source_iteration": int(payload.get("iter", -1)),
        "architecture": "independent_pitch_branch" if pitch_branch else "shared_output_row",
        "fourth_output_weight_max": fourth_output_weight_max,
        "fourth_output_bias": fourth_output_bias,
        "policy_std": actor_state["distribution.std_param"].detach().cpu().tolist(),
    }


def freeze_teacher_and_train_cylinder4_head(runner) -> dict[str, object]:
    """Freeze the accepted policy and expose only actor output four to PPO.

    The critic remains fully trainable.  Gradient masks are used on the shared
    four-row output tensors so Adam cannot change cylinders 1-3, and actor
    normalization is frozen because changing its statistics would also change
    the accepted teacher's deterministic actions.
    """
    actor = runner.alg.get_policy()
    std = actor.distribution.std_param
    if tuple(std.shape) != (4,):
        raise ValueError(f"expected four policy standard deviations, got {std.shape}")

    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    pitch_branch = hasattr(actor.mlp, "pitch_mlp")
    if pitch_branch:
        for parameter in actor.mlp.pitch_mlp.parameters():
            parameter.requires_grad_(True)
    else:
        output = actor.mlp[-1]
        if not isinstance(output, torch.nn.Linear) or output.out_features != 4:
            raise ValueError("expected a four-output linear actor head")
        output.weight.requires_grad_(True)
        output.bias.requires_grad_(True)
    # Keep exploration explicitly scheduled by the trainer. Allowing PPO to
    # optimize it made the pilot grow from 0.08 to 0.098 and destabilized late
    # checkpoints even though deterministic cylinder-4 corrections stayed tiny.
    std.requires_grad_(False)

    if not pitch_branch:
        row_mask = torch.zeros_like(output.weight)
        row_mask[3] = 1.0
        bias_mask = torch.zeros_like(output.bias)
        bias_mask[3] = 1.0
        output.weight.register_hook(lambda gradient: gradient * row_mask)
        output.bias.register_hook(lambda gradient: gradient * bias_mask)

    # PPO otherwise updates this buffer after every environment step.  Keep it
    # bitwise fixed so the first three deterministic outputs remain identical.
    actor.update_normalization = MethodType(lambda self, obs: None, actor)
    trainable = (
        tuple(
            f"mlp.pitch_mlp.{name}"
            for name, _ in actor.mlp.pitch_mlp.named_parameters()
        )
        if pitch_branch
        else ("mlp.6.weight[3]", "mlp.6.bias[3]")
    )
    return {
        "architecture": "independent_pitch_branch" if pitch_branch else "shared_output_row",
        "trainable_actor_parameters": trainable,
        "actor_normalizer_frozen": True,
        "policy_std_schedule_owned_by_trainer": True,
    }


def configure_joint_finetune(runner) -> dict[str, object]:
    """Train all four action networks while keeping normalization/std controlled."""
    actor = runner.alg.get_policy()
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    actor.distribution.std_param.requires_grad_(False)
    actor.update_normalization = MethodType(lambda self, obs: None, actor)
    trainable = tuple(
        name for name, parameter in actor.named_parameters()
        if parameter.requires_grad
    )
    return {
        "architecture": "joint_teacher_and_pitch_branch",
        "trainable_actor_parameters": trainable,
        "actor_normalizer_frozen": True,
        "policy_std_schedule_owned_by_trainer": True,
    }


def snapshot_frozen_teacher(actor) -> dict[str, torch.Tensor]:
    """Capture actor state that must remain identical during cylinder-4 training."""
    state = actor.state_dict()
    if hasattr(actor.mlp, "pitch_mlp"):
        snapshot = {
            key: value.detach().clone()
            for key, value in state.items()
            if not key.startswith("mlp.pitch_mlp.")
            and key != "distribution.std_param"
        }
        snapshot["distribution.std_param[:3]"] = (
            state["distribution.std_param"][:3].detach().clone()
        )
        return snapshot
    output_prefix = f"mlp.{len(actor.mlp) - 1}"
    output_weight = f"{output_prefix}.weight"
    output_bias = f"{output_prefix}.bias"
    snapshot = {
        key: value.detach().clone()
        for key, value in state.items()
        if key not in {output_weight, output_bias, "distribution.std_param"}
    }
    snapshot[f"{output_weight}[:3]"] = state[output_weight][:3].detach().clone()
    snapshot[f"{output_bias}[:3]"] = state[output_bias][:3].detach().clone()
    snapshot["distribution.std_param[:3]"] = (
        state["distribution.std_param"][:3].detach().clone()
    )
    return snapshot


def assert_frozen_teacher_unchanged(
    actor, snapshot: Mapping[str, torch.Tensor]
) -> None:
    """Raise if PPO changed any frozen teacher weight, statistic, or std."""
    state = actor.state_dict()
    if hasattr(actor.mlp, "pitch_mlp"):
        current = {
            key: value
            for key, value in state.items()
            if not key.startswith("mlp.pitch_mlp.")
            and key != "distribution.std_param"
        }
        current["distribution.std_param[:3]"] = state[
            "distribution.std_param"
        ][:3]
        for key, expected in snapshot.items():
            if not torch.equal(current[key].detach(), expected):
                raise RuntimeError(f"frozen three-cylinder teacher changed at {key}")
        return
    output_prefix = f"mlp.{len(actor.mlp) - 1}"
    output_weight = f"{output_prefix}.weight"
    output_bias = f"{output_prefix}.bias"
    current = {
        key: value
        for key, value in state.items()
        if key not in {output_weight, output_bias, "distribution.std_param"}
    }
    current[f"{output_weight}[:3]"] = state[output_weight][:3]
    current[f"{output_bias}[:3]"] = state[output_bias][:3]
    current["distribution.std_param[:3]"] = state["distribution.std_param"][:3]
    for key, expected in snapshot.items():
        if not torch.equal(current[key].detach(), expected):
            raise RuntimeError(f"frozen three-cylinder teacher changed at {key}")
