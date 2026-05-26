from __future__ import annotations

from typing import Iterable

import torch


def _check_last_dim(name: str, tensor: torch.Tensor, expected_dim: int) -> None:
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"{name} must have last dimension {expected_dim}, got {tensor.shape[-1]}")


def _check_dims(tensors: Iterable[tuple[str, torch.Tensor, int]]) -> None:
    for name, tensor, expected_dim in tensors:
        _check_last_dim(name, tensor, expected_dim)


def build_aam_input(
    q: torch.Tensor,
    qdot: torch.Tensor,
    dP: torch.Tensor,
    prev_valve_cmd: torch.Tensor,
) -> torch.Tensor:
    _check_dims(
        (
            ("q", q, 5),
            ("qdot", qdot, 5),
            ("dP", dP, 4),
            ("prev_valve_cmd", prev_valve_cmd, 4),
        )
    )
    return torch.cat([q, qdot, dP, prev_valve_cmd], dim=-1)


def build_rl_obs(
    q: torch.Tensor,
    qdot: torch.Tensor,
    q_ref: torch.Tensor,
    qdot_ref: torch.Tensor,
    prev_action: torch.Tensor,
    prev_valve_cmd: torch.Tensor,
    z_arm_hat: torch.Tensor,
) -> torch.Tensor:
    _check_dims(
        (
            ("q", q, 5),
            ("qdot", qdot, 5),
            ("q_ref", q_ref, 5),
            ("qdot_ref", qdot_ref, 5),
            ("prev_action", prev_action, 4),
            ("prev_valve_cmd", prev_valve_cmd, 4),
            ("z_arm_hat", z_arm_hat, 24),
        )
    )
    q_error = q_ref - q
    qdot_error = qdot_ref - qdot
    return torch.cat(
        [
            q,
            qdot,
            q_ref,
            qdot_ref,
            q_error,
            qdot_error,
            prev_action,
            prev_valve_cmd,
            z_arm_hat,
        ],
        dim=-1,
    )


def build_id_input(q: torch.Tensor, qdot: torch.Tensor, action_4d: torch.Tensor, dP: torch.Tensor) -> torch.Tensor:
    _check_dims((("q", q, 5), ("qdot", qdot, 5), ("action_4d", action_4d, 4), ("dP", dP, 4)))
    zero_tool_joint = torch.zeros_like(action_4d[..., :1])
    action_5d = torch.cat([action_4d, zero_tool_joint], dim=-1)
    return torch.cat([q, qdot, action_5d, dP], dim=-1)


def build_fd_input(q: torch.Tensor, qdot: torch.Tensor, dP: torch.Tensor, valve_cmd: torch.Tensor) -> torch.Tensor:
    _check_dims((("q", q, 5), ("qdot", qdot, 5), ("dP", dP, 4), ("valve_cmd", valve_cmd, 4)))
    return torch.cat([q, qdot, dP, valve_cmd], dim=-1)


def build_sd_input(
    q: torch.Tensor,
    qdot: torch.Tensor,
    dP: torch.Tensor,
    fnet: torch.Tensor,
    valve_cmd: torch.Tensor,
) -> torch.Tensor:
    _check_dims(
        (
            ("q", q, 5),
            ("qdot", qdot, 5),
            ("dP", dP, 4),
            ("fnet", fnet, 4),
            ("valve_cmd", valve_cmd, 4),
        )
    )
    return torch.cat([q, qdot, dP, fnet, valve_cmd], dim=-1)
