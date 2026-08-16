"""Shared scripted-truck speed and path-duration helpers."""

from __future__ import annotations

import torch


SPEED_SAMPLING_MODES = ("uniform", "active_time_balanced")


def map_unit_interval_to_speed(
    unit_samples: torch.Tensor,
    minimum_speed_kmh: float,
    maximum_speed_kmh: float,
    mode: str = "uniform",
) -> torch.Tensor:
    """Map uniform samples to constant episode speeds.

    ``active_time_balanced`` samples with density proportional to speed. Since
    a fixed-length path keeps speed ``v`` active for time proportional to
    ``1/v``, each speed band then contributes the same expected number of
    active PPO transitions before its replica parks.
    """
    low = float(minimum_speed_kmh)
    high = float(maximum_speed_kmh)
    if not 0.0 < low <= high:
        raise ValueError(f"speed range must satisfy 0 < min <= max; got {low}, {high}")
    mode = str(mode).strip().lower()
    if mode not in SPEED_SAMPLING_MODES:
        raise ValueError(
            f"speed sampling mode must be one of {SPEED_SAMPLING_MODES}; got {mode!r}"
        )
    samples = unit_samples.clamp(0.0, 1.0)
    if high == low:
        return torch.full_like(samples, low)
    if mode == "active_time_balanced":
        return torch.sqrt(samples * (high * high - low * low) + low * low)
    return low + samples * (high - low)


def episode_path_active_mask(
    preposition_counter: torch.Tensor,
    settle_counter: torch.Tensor,
    complete_mask: torch.Tensor,
    base_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return replicas whose current step belongs to the scored path.

    Faster replicas remain in the vectorized environment until the slowest
    one reaches the endpoint.  This mask makes that parked tail an absorbing,
    zero-reward state while also excluding deployment and settling.
    """
    active = (preposition_counter <= 0) & (settle_counter <= 0) & ~complete_mask
    if base_mask is not None:
        active &= base_mask
    return active

def path_duration_s(path_length_m: float, minimum_speed_kmh: float, margin_s: float = 0.5) -> float:
    """Return a safe horizon for the slowest constant episode speed."""
    length = float(path_length_m)
    speed = float(minimum_speed_kmh)
    if length <= 0.0:
        raise ValueError(f"path_length_m must be positive, got {length}")
    if speed <= 0.0:
        raise ValueError(f"minimum_speed_kmh must be positive, got {speed}")
    if float(margin_s) < 0.0:
        raise ValueError(f"margin_s must be non-negative, got {margin_s}")
    return length / (speed / 3.6) + float(margin_s)


def configure_path_horizon(cfg) -> float:
    """Extend a terrain config enough for its slowest sampled speed.

    The horizon is global because Isaac Lab vector environments share one
    timeout. Per-environment completion is handled by the terrain env; this
    helper only guarantees that the slowest replica can finish its path.
    """
    minimum_speed = float(getattr(cfg, "vehicle_speed_min_kmh", 0.0))
    path_length = float(
        getattr(cfg, "terrain_tracking_active_length_m", 0.0)
    )
    margin = float(getattr(cfg, "vehicle_path_completion_margin_s", 0.5))
    segment_duration = float(
        getattr(cfg, "vehicle_training_segment_duration_s", 0.0)
    )
    if segment_duration > 0.0:
        cfg.episode_length_s = segment_duration + margin
        return float(cfg.episode_length_s)
    if minimum_speed <= 0.0 or path_length <= 0.0:
        return float(getattr(cfg, "episode_length_s", 0.0))
    required = path_duration_s(path_length, minimum_speed, margin)
    cfg.episode_length_s = max(float(cfg.episode_length_s), required)
    return float(cfg.episode_length_s)


def advance_path_travel(
    travel_m: torch.Tensor,
    speed_kmh: torch.Tensor,
    active_mask: torch.Tensor,
    complete_mask: torch.Tensor,
    path_length_m: float | torch.Tensor,
    dt_s: float,
    tolerance_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance and clamp vectorized truck travel for one control step."""
    if isinstance(path_length_m, torch.Tensor):
        path_limit = path_length_m.to(device=travel_m.device, dtype=travel_m.dtype)
        if path_limit.ndim == 0:
            path_limit = path_limit.expand_as(travel_m)
        elif path_limit.shape != travel_m.shape:
            raise ValueError(
                f"path length shape {tuple(path_limit.shape)} must match travel "
                f"shape {tuple(travel_m.shape)}"
            )
    else:
        if float(path_length_m) < 0.0:
            raise ValueError(f"path_length_m must be non-negative, got {path_length_m}")
        path_limit = torch.full_like(travel_m, float(path_length_m))
    if float(dt_s) < 0.0:
        raise ValueError(f"dt_s must be non-negative, got {dt_s}")
    if float(tolerance_m) < 0.0:
        raise ValueError(f"tolerance_m must be non-negative, got {tolerance_m}")
    path_active = active_mask & ~complete_mask
    step = (speed_kmh / 3.6) * float(dt_s)
    proposed = travel_m + torch.where(path_active, step, torch.zeros_like(step))
    next_travel = torch.minimum(proposed, path_limit)
    advance = next_travel - travel_m
    newly_complete = path_active & (
        next_travel >= path_limit - float(tolerance_m)
    )
    return next_travel, advance, newly_complete


def fixed_duration_path_limits(
    speed_kmh: torch.Tensor,
    full_path_length_m: float,
    duration_s: float,
) -> torch.Tensor:
    """Distance each replica covers in an equal-duration training segment."""
    full_length = float(full_path_length_m)
    duration = float(duration_s)
    if full_length <= 0.0:
        raise ValueError(f"full_path_length_m must be positive, got {full_length}")
    if duration <= 0.0:
        raise ValueError(f"duration_s must be positive, got {duration}")
    return torch.clamp(speed_kmh / 3.6 * duration, min=0.0, max=full_length)
