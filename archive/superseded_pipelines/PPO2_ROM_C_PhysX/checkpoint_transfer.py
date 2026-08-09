"""Exact 42 -> 47 observation transfer for a locked ROM-B policy."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Mapping

import torch


BASE_OBS_DIM = 42
ADAPTIVE_OBS_DIM = 47
AAM_SLICE = slice(BASE_OBS_DIM, ADAPTIVE_OBS_DIM)


def _expanded_state(
    source: Mapping[str, torch.Tensor],
    template: Mapping[str, torch.Tensor],
    *,
    actor: bool,
) -> dict[str, torch.Tensor]:
    target = {key: value.detach().clone() for key, value in template.items()}
    for key, value in source.items():
        if key in target and tuple(value.shape) == tuple(target[key].shape):
            target[key].copy_(value)

    for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        old, new = source[key], target[key]
        if old.shape[-1] != BASE_OBS_DIM or new.shape[-1] != ADAPTIVE_OBS_DIM:
            raise ValueError(f"unexpected normalizer expansion {old.shape} -> {new.shape}")
        new[..., :BASE_OBS_DIM] = old
        # Constructor defaults (mean 0, variance/std 1) remain for AAM inputs.
    target["obs_normalizer.count"].copy_(source["obs_normalizer.count"])

    if actor:
        teacher_key = "mlp.teacher_mlp.0.weight"
        pitch_key = "mlp.pitch_mlp.0.weight"
        old_teacher, new_teacher = source[teacher_key], target[teacher_key]
        if old_teacher.shape[1] != BASE_OBS_DIM or new_teacher.shape[1] != ADAPTIVE_OBS_DIM:
            raise ValueError("unexpected ROM-B/ROM-C teacher input shapes")
        new_teacher.zero_()
        new_teacher[:, :BASE_OBS_DIM] = old_teacher
        old_pitch, new_pitch = source[pitch_key], target[pitch_key]
        if new_pitch.shape[1] != old_pitch.shape[1] + 5:
            raise ValueError("PPO2_C pitch branch must append exactly five AAM inputs")
        new_pitch.zero_()
        new_pitch[:, :old_pitch.shape[1]] = old_pitch
        # Keep the target's expanded index buffer; source has only 24 indices.
        target["mlp.pitch_obs_indices"] = template[
            "mlp.pitch_obs_indices"
        ].detach().clone()
    else:
        old_first, new_first = source["mlp.0.weight"], target["mlp.0.weight"]
        if old_first.shape[1] != BASE_OBS_DIM or new_first.shape[1] != ADAPTIVE_OBS_DIM:
            raise ValueError("unexpected ROM-B/ROM-C critic input shapes")
        new_first.zero_()
        new_first[:, :BASE_OBS_DIM] = old_first
    return target


def transfer_rom_b_checkpoint_into_runner(runner, checkpoint: str | Path) -> dict[str, object]:
    """Transfer ROM-B while preserving its zero-AAM deterministic mapping."""
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = runner.alg.get_policy()
    critic = runner.alg.critic
    actor_state = _expanded_state(
        payload["actor_state_dict"], actor.state_dict(), actor=True
    )
    critic_state = _expanded_state(
        payload["critic_state_dict"], critic.state_dict(), actor=False
    )
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    return {
        "source": str(path),
        "source_iteration": int(payload.get("iter", -1)),
        "observation_expansion": "42->47",
        "new_actor_columns_max_abs": max(
            float(actor_state["mlp.teacher_mlp.0.weight"][:, AAM_SLICE].abs().max()),
            float(actor_state["mlp.pitch_mlp.0.weight"][:, -5:].abs().max()),
        ),
        "optimizer": "fresh",
    }


def configure_aam_adapter_finetune(runner) -> dict[str, object]:
    """Allow only actor weights connected to the five AAM inputs to change."""
    actor = runner.alg.get_policy()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    teacher_first = actor.mlp.teacher_mlp[0]
    pitch_first = actor.mlp.pitch_mlp[0]
    if not isinstance(teacher_first, torch.nn.Linear) or not isinstance(
        pitch_first, torch.nn.Linear
    ):
        raise TypeError("PPO2_C actor branches must start with linear layers")
    teacher_first.weight.requires_grad_(True)
    pitch_first.weight.requires_grad_(True)
    teacher_mask = torch.zeros_like(teacher_first.weight)
    teacher_mask[:, AAM_SLICE] = 1.0
    pitch_mask = torch.zeros_like(pitch_first.weight)
    pitch_mask[:, -5:] = 1.0
    teacher_first.weight.register_hook(lambda gradient: gradient * teacher_mask)
    pitch_first.weight.register_hook(lambda gradient: gradient * pitch_mask)
    actor.distribution.std_param.requires_grad_(False)
    actor.update_normalization = MethodType(lambda self, obs: None, actor)
    return {
        "trainable_actor_parameters": (
            "mlp.teacher_mlp.0.weight[:,42:47]",
            "mlp.pitch_mlp.0.weight[:,-5:]",
        ),
        "base_policy_mapping_frozen": True,
        "actor_normalizer_frozen": True,
        "critic_trainable": True,
    }
