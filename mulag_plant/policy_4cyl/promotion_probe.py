"""Per-environment paired promotion probes for four-cylinder policies."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


@torch.inference_mode()
def run_policy_probe(
    raw,
    env,
    policy,
    *,
    shape: str,
    seed: int,
    scored_steps: int,
) -> dict[str, torch.Tensor | int | str]:
    """Collect fixed-length histories independently for every environment."""
    synchronized_resets = bool(getattr(raw.cfg, "synchronize_resets", True))
    deferred_safety = bool(
        getattr(raw.cfg, "defer_safety_terminations", False)
    )
    raw.set_trajectory_shape(shape)
    raw.reset(seed=int(seed))
    obs = env.get_observations()
    num_envs = int(raw.num_envs)
    device = raw.device
    target = int(scored_steps)
    counts = torch.zeros(num_envs, dtype=torch.long, device=device)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    error = torch.full((target, num_envs), torch.nan, device=device)
    acceleration = torch.full_like(error, torch.nan)
    alignment = torch.full_like(error, torch.nan)
    drift = torch.full_like(error, torch.nan)

    # DirectRLEnv automatically resets every done environment inside step().
    # A physical reset includes a synchronized arm deployment, so a partial
    # reset is both invalid and would overwrite the failed replica's history.
    # Capture the real per-environment done mask, but return an all-false mask
    # to DirectRLEnv so no reset occurs during this short, fixed probe.
    original_get_dones = raw._get_dones
    captured_dones = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _capture_without_reset():
        nonlocal captured_dones
        terminated, truncated = original_get_dones()
        captured_dones = terminated.bool() | truncated.bool()
        return torch.zeros_like(terminated), torch.zeros_like(truncated)

    raw.cfg.synchronize_resets = False
    raw.cfg.defer_safety_terminations = False
    raw._get_dones = _capture_without_reset
    total_steps = 0
    try:
        for _ in range(target + 4000):
            obs, _, _, _ = env.step(policy(obs))
            total_steps += 1
            alive &= ~captured_dones
            active = (
                alive
                & (counts < target)
                & (raw._preposition_counter <= 0)
                & (raw._settle_counter <= 0)
            )
            ids = torch.nonzero(active, as_tuple=False).flatten()
            if ids.numel() > 0:
                rows = counts[ids]
                tool = raw._compute_ee_pos_local()
                error[rows, ids] = torch.linalg.norm(
                    raw._original_reference_tool[ids] - tool[ids], dim=-1
                )
                acceleration[rows, ids] = torch.linalg.norm(
                    raw._measured_tool_acceleration_truck[ids], dim=-1
                )
                alignment[rows, ids] = raw._measure_head_down_alignment()[ids]
                drift[rows, ids] = raw._truck_post_constraint_drift_xy[ids]
                counts[ids] += 1
            if bool(torch.all((counts >= target) | ~alive)):
                break
    finally:
        raw._get_dones = original_get_dones
        raw.cfg.synchronize_resets = synchronized_resets
        raw.cfg.defer_safety_terminations = deferred_safety
    return {
        "shape": str(shape),
        "seed": int(seed),
        "scored_steps": target,
        "total_steps": total_steps,
        "counts": counts.detach().cpu(),
        "completed": (counts >= target).detach().cpu(),
        "error": error.detach().cpu(),
        "acceleration": acceleration.detach().cpu(),
        "alignment": alignment.detach().cpu(),
        "drift": drift.detach().cpu(),
    }


def _summary(
    probe: Mapping[str, torch.Tensor | int | str],
    matched: torch.Tensor,
) -> dict[str, float | int | bool]:
    count = int(torch.count_nonzero(matched))
    total = int(matched.numel())
    if count == 0:
        return {
            "valid": False,
            "matched_envs": 0,
            "total_envs": total,
            "completion_fraction": 0.0,
            "mean_error_m": float("inf"),
            "p95_error_m": float("inf"),
            "max_error_m": float("inf"),
            "p95_acceleration_mps2": float("inf"),
            "min_head_alignment": -1.0,
            "max_truck_drift_m": float("inf"),
        }
    error = probe["error"][:, matched]  # type: ignore[index]
    acceleration = probe["acceleration"][:, matched]  # type: ignore[index]
    alignment = probe["alignment"][:, matched]  # type: ignore[index]
    drift = probe["drift"][:, matched]  # type: ignore[index]
    return {
        "valid": True,
        "matched_envs": count,
        "total_envs": total,
        "completion_fraction": count / total,
        "mean_error_m": float(torch.mean(error)),
        "p95_error_m": float(torch.quantile(error.flatten(), 0.95)),
        "max_error_m": float(torch.max(error)),
        "p95_acceleration_mps2": float(
            torch.quantile(acceleration.flatten(), 0.95)
        ),
        "min_head_alignment": float(torch.min(alignment)),
        "max_truck_drift_m": float(torch.max(drift)),
    }


def compare_probe_results(
    baseline_runs: Mapping[str, Mapping[str, torch.Tensor | int | str]],
    candidate_runs: Mapping[str, Mapping[str, torch.Tensor | int | str]],
    *,
    min_completion_fraction: float,
    min_mean_improvement: float,
    min_p95_improvement: float,
    max_acceleration_ratio: float,
    max_shape_acceleration_ratio: float,
    max_shape_regression_m: float,
) -> dict[str, object]:
    """Compare policies only on environment indices that completed both runs."""
    if tuple(baseline_runs) != tuple(candidate_runs):
        raise ValueError("baseline and candidate shapes differ")
    per_shape: dict[str, dict[str, object]] = {}
    for shape in baseline_runs:
        baseline_run = baseline_runs[shape]
        candidate_run = candidate_runs[shape]
        baseline_completed = baseline_run["completed"]  # type: ignore[assignment]
        candidate_completed = candidate_run["completed"]  # type: ignore[assignment]
        matched = baseline_completed.bool() & candidate_completed.bool()
        baseline = _summary(baseline_run, matched)
        candidate = _summary(candidate_run, matched)
        fraction = float(baseline["completion_fraction"])
        per_shape[shape] = {
            "paired_completion_fraction": fraction,
            "baseline_completion_fraction": float(
                torch.mean(baseline_completed.float())
            ),
            "candidate_completion_fraction": float(
                torch.mean(candidate_completed.float())
            ),
            "baseline": baseline,
            "candidate": candidate,
        }

    def average(policy: str, key: str) -> float:
        return sum(
            float(values[policy][key]) for values in per_shape.values()  # type: ignore[index]
        ) / len(per_shape)

    baseline_mean = average("baseline", "mean_error_m")
    candidate_mean = average("candidate", "mean_error_m")
    baseline_p95 = average("baseline", "p95_error_m")
    candidate_p95 = average("candidate", "p95_error_m")
    baseline_max = average("baseline", "max_error_m")
    candidate_max = average("candidate", "max_error_m")
    baseline_acceleration = average("baseline", "p95_acceleration_mps2")
    candidate_acceleration = average("candidate", "p95_acceleration_mps2")
    checks = {
        "completion_fraction_safe": all(
            float(values["paired_completion_fraction"])
            >= float(min_completion_fraction)
            for values in per_shape.values()
        ),
        "mean_error_improved": candidate_mean
        <= baseline_mean * (1.0 - float(min_mean_improvement)),
        "p95_error_improved": candidate_p95
        <= baseline_p95 * (1.0 - float(min_p95_improvement)),
        "aggregate_acceleration_safe": candidate_acceleration
        <= baseline_acceleration * float(max_acceleration_ratio),
        "per_shape_error_safe": all(
            float(values["candidate"]["mean_error_m"])  # type: ignore[index]
            <= float(values["baseline"]["mean_error_m"])  # type: ignore[index]
            + float(max_shape_regression_m)
            for values in per_shape.values()
        ),
        "per_shape_acceleration_safe": all(
            float(values["candidate"]["p95_acceleration_mps2"])  # type: ignore[index]
            <= float(values["baseline"]["p95_acceleration_mps2"])  # type: ignore[index]
            * float(max_shape_acceleration_ratio)
            for values in per_shape.values()
        ),
        "alignment_safe": min(
            float(values["candidate"]["min_head_alignment"])  # type: ignore[index]
            for values in per_shape.values()
        ) >= 0.98,
        "truck_drift_safe": max(
            float(values["candidate"]["max_truck_drift_m"])  # type: ignore[index]
            for values in per_shape.values()
        ) <= 0.001,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "aggregate": {
            "baseline_mean_error_m": baseline_mean,
            "candidate_mean_error_m": candidate_mean,
            "baseline_p95_error_m": baseline_p95,
            "candidate_p95_error_m": candidate_p95,
            "baseline_max_error_m": baseline_max,
            "candidate_max_error_m": candidate_max,
            "baseline_p95_acceleration_mps2": baseline_acceleration,
            "candidate_p95_acceleration_mps2": candidate_acceleration,
        },
        "per_shape": per_shape,
    }


@torch.inference_mode()
def paired_promotion_probe(
    raw,
    env,
    baseline_policy,
    candidate_policy,
    *,
    shapes: Sequence[str],
    seed: int,
    scored_steps: int,
    min_completion_fraction: float,
    min_mean_improvement: float,
    min_p95_improvement: float,
    max_acceleration_ratio: float,
    max_shape_acceleration_ratio: float,
    max_shape_regression_m: float,
) -> dict[str, object]:
    baseline_runs = {}
    candidate_runs = {}
    for index, shape in enumerate(shapes):
        shape_seed = int(seed) + index
        baseline_runs[shape] = run_policy_probe(
            raw, env, baseline_policy, shape=shape,
            seed=shape_seed, scored_steps=scored_steps,
        )
        candidate_runs[shape] = run_policy_probe(
            raw, env, candidate_policy, shape=shape,
            seed=shape_seed, scored_steps=scored_steps,
        )
    return compare_probe_results(
        baseline_runs,
        candidate_runs,
        min_completion_fraction=min_completion_fraction,
        min_mean_improvement=min_mean_improvement,
        min_p95_improvement=min_p95_improvement,
        max_acceleration_ratio=max_acceleration_ratio,
        max_shape_acceleration_ratio=max_shape_acceleration_ratio,
        max_shape_regression_m=max_shape_regression_m,
    )
