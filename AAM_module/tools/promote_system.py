#!/usr/bin/env python3
"""Promotion gate for the online adaptive system.

Promotion is intentionally stricter than the task's binary safety gates.  A
candidate is promoted only when all three independent claims hold:

1. the privileged and estimated SD latents improve paired dynamics rollouts
   over both zero and shuffled controls;
2. the estimated latent is not statistically worse than the true latent and
   retains a positive fraction of its advantage;
3. the PPO estimated-latent reports beat both zero and shuffled reports on
   paired continuous metrics.

The PPO unit is one matched report per mesh/speed pair.  Reports should be
collected with identical checkpoint, plant reserve, speed, seed, terrain and
episode settings; mismatches are rejected rather than silently pooled.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CHANNELS = ("extension", "velocity")
PPO_METRICS = {
    "lateral": ("worst_family_mean_abs_lateral_error_m", "lower"),
    "cut_height": ("mean_abs_cut_height_error_m", "lower"),
    "impacts": ("hard_impact_step_fraction", "lower"),
    "acceleration": ("p95_tool_acceleration_mps2", "lower"),
}
# Headroom is enforced by the identical physical reserve in every paired arm.
# It is therefore a safety constraint, not a metric on which an adaptive
# policy can be required to show strict superiority (all arms may legitimately
# sit at the same solver-enforced floor).
PPO_SAFETY_FLOORS = {
    "headroom": ("p01_cylinder_headroom_m", 0.005),
}
CI_SEED = 171
CI_BOOTSTRAPS = 10000


def bootstrap_ci(values: np.ndarray, seed: int = CI_SEED) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, values.size, size=(CI_BOOTSTRAPS, values.size))
    means = values[samples].mean(axis=1)
    return [float(v) for v in np.percentile(means, (2.5, 97.5))]


def _stats(improvements: list[float]) -> dict:
    values = np.asarray(improvements, dtype=np.float64)
    ci = bootstrap_ci(values)
    return {
        "n_pairs": int(values.size),
        "mean_improvement": float(values.mean()) if values.size else None,
        "bootstrap_95ci": ci,
        "pass": bool(ci is not None and ci[0] > 0.0),
    }


def _noninferior_stats(improvements: list[float], margin: float) -> dict:
    """One-sided non-inferiority test for candidate-minus-reference gains.

    Positive values mean the estimated latent is better than the true latent.
    Non-inferiority therefore requires the lower confidence bound to remain
    above ``-margin``. Merely having an interval that contains zero is not a
    non-inferiority test and can accept arbitrarily large uncertainty.
    """
    values = np.asarray(improvements, dtype=np.float64)
    ci = bootstrap_ci(values)
    return {
        "n_pairs": int(values.size),
        "mean_improvement": float(values.mean()) if values.size else None,
        "bootstrap_95ci": ci,
        "noninferiority_margin": float(margin),
        "pass": bool(ci is not None and ci[0] >= -float(margin)),
        "criterion": "lower 95% bootstrap bound >= -declared margin",
    }


def _sd_mode_map(report: dict, mode: str, horizon: str) -> dict[int, dict[str, float]]:
    try:
        rows = report["per_run"][mode]
    except KeyError as exc:
        raise ValueError(f"SD rollout is missing mode {mode!r}") from exc
    return {
        int(run_id): {
            channel: float(values["rollout_rmse"][horizon][channel])
            for channel in CHANNELS
        }
        for run_id, values in rows.items()
    }


def _sd_compare(
    candidate: dict[int, dict[str, float]],
    control: dict[int, dict[str, float]],
    run_ids: list[int],
) -> dict[str, dict]:
    # Positive improvement means candidate RMSE is lower than control RMSE.
    return {
        channel: _stats(
            [control[run_id][channel] - candidate[run_id][channel] for run_id in run_ids]
        )
        for channel in CHANNELS
    }


def sd_and_aam_gate(
    path: Path,
    horizon: int,
    *,
    minimum_retention: float = 0.8,
    noninferiority_margin_fraction: float = 0.2,
) -> dict:
    report = json.loads(path.read_text())
    h = str(int(horizon))
    true_map = _sd_mode_map(report, "true", h)
    zhat_map = _sd_mode_map(report, "zhat", h)
    zero_map = _sd_mode_map(report, "zero", h)
    shuffled_map = _sd_mode_map(report, "shuffled", h)
    common = sorted(set(true_map) & set(zhat_map) & set(zero_map) & set(shuffled_map))
    if len(common) < 3:
        raise ValueError(f"SD promotion gate needs at least 3 common runs, got {len(common)}")

    true_zero = _sd_compare(true_map, zero_map, common)
    true_shuffled = _sd_compare(true_map, shuffled_map, common)
    zhat_zero = _sd_compare(zhat_map, zero_map, common)
    zhat_shuffled = _sd_compare(zhat_map, shuffled_map, common)

    retention_by_control = {"zero": {}, "shuffled": {}}
    noninferiority_margins = {}
    for channel in CHANNELS:
        true_zero_advantage = np.mean(
            [zero_map[run_id][channel] - true_map[run_id][channel] for run_id in common]
        )
        estimated_zero_advantage = np.mean(
            [zero_map[run_id][channel] - zhat_map[run_id][channel] for run_id in common]
        )
        true_shuffled_advantage = np.mean(
            [shuffled_map[run_id][channel] - true_map[run_id][channel] for run_id in common]
        )
        estimated_shuffled_advantage = np.mean(
            [shuffled_map[run_id][channel] - zhat_map[run_id][channel] for run_id in common]
        )
        retention_by_control["zero"][channel] = (
            float(estimated_zero_advantage / true_zero_advantage)
            if true_zero_advantage > 1.0e-12 else None
        )
        retention_by_control["shuffled"][channel] = (
            float(estimated_shuffled_advantage / true_shuffled_advantage)
            if true_shuffled_advantage > 1.0e-12 else None
        )
        weakest_true_advantage = min(true_zero_advantage, true_shuffled_advantage)
        noninferiority_margins[channel] = max(
            0.0, float(noninferiority_margin_fraction) * float(weakest_true_advantage)
        )

    # Positive improvement means the estimated RMSE is lower than the true
    # RMSE. The allowed degradation is declared as a fraction of the weaker
    # privileged advantage over zero/shuffled controls.
    zhat_true = {
        channel: _noninferior_stats(
            [true_map[run_id][channel] - zhat_map[run_id][channel] for run_id in common],
            noninferiority_margins[channel],
        )
        for channel in CHANNELS
    }

    checks = {
        "true_beats_zero": all(true_zero[c]["pass"] for c in CHANNELS),
        "true_beats_shuffled": all(true_shuffled[c]["pass"] for c in CHANNELS),
        "estimated_beats_zero": all(zhat_zero[c]["pass"] for c in CHANNELS),
        "estimated_beats_shuffled": all(zhat_shuffled[c]["pass"] for c in CHANNELS),
        "estimated_not_worse_than_true": all(zhat_true[c]["pass"] for c in CHANNELS),
        "minimum_retention": all(
            retention_by_control[control][channel] is not None
            and retention_by_control[control][channel] >= minimum_retention
            for control in ("zero", "shuffled")
            for channel in CHANNELS
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "source": str(path.resolve()),
        "horizon_ticks": int(horizon),
        "n_runs": len(common),
        "checks": checks,
        "minimum_retention_required": float(minimum_retention),
        "noninferiority_margin_fraction": float(noninferiority_margin_fraction),
        "retention_fraction": retention_by_control,
        "true_vs_zero": true_zero,
        "true_vs_shuffled": true_shuffled,
        "estimated_vs_zero": zhat_zero,
        "estimated_vs_shuffled": zhat_shuffled,
        "estimated_minus_true": zhat_true,
    }


def _pair_key(path: Path, mesh: str) -> str:
    stem = re.sub(r"(^|_)(estimated|zero|shuffled)(?=_|$)", r"\1", path.stem)
    return f"{stem.strip('_')}::mesh={mesh}"


def _ppo_rows(paths: list[Path], mode: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in paths:
        report = json.loads(path.read_text())
        reported_mode = report.get("online_aam_latent_mode")
        if reported_mode != mode:
            raise ValueError(
                f"{path}: report latent mode {reported_mode!r} does not match "
                f"the declared {mode!r} arm"
            )
        if report.get("online_aam_controller_latent_mode") != "estimated":
            raise ValueError(
                f"{path}: PPO latent ablations require the controller/adapter "
                "to remain in estimated mode"
            )
        if not bool(report.get("z_hat_is_live")):
            raise ValueError(
                f"{path}: z_hat_is_live={report.get('z_hat_is_live')!r} is "
                "false; the shared AAM/adapter route was not active"
            )
        if not bool(report.get("deployment_factor_enabled")):
            raise ValueError(
                f"{path}: deployment factors are disabled; a nominal rollout "
                "cannot establish online latent adaptation"
            )
        if report.get("factor_domain") == "nominal":
            raise ValueError(
                f"{path}: factor_domain='nominal' cannot establish online "
                "latent adaptation"
            )
        for required in (
            "c3_physical_extension_reserve_m",
            "online_aam_residual_scale",
            "vehicle_speed_min_kmh",
            "vehicle_speed_max_kmh",
            "factor_domain",
            "factor_key",
            "deployment_factor_enabled",
            "deployment_factor_manifest",
            "online_aam_checkpoint",
            "online_aam_adapter_checkpoint",
            "online_aam_history_lag",
            "online_aam_smoother_window",
            "online_aam_controller_latent_mode",
            "evaluation_seed",
        ):
            value = report.get(required)
            if required != "factor_key" and value is None:
                raise ValueError(
                    f"{path}: merged report is missing promotion contract field "
                    f"{required!r}; rerun the per-mesh merge"
                )
        for mesh, values in report.get("per_mesh", {}).items():
            key = _pair_key(path, str(mesh))
            if key in rows:
                raise ValueError(f"duplicate PPO pair key for {mode}: {key}")
            extracted = {}
            for name, (field, _direction) in PPO_METRICS.items():
                value = values.get(field)
                if value is None:
                    raise ValueError(
                        f"{path}: mesh {mesh} has no continuous metric {field!r}"
                    )
                extracted[name] = float(value)
            rows[key] = {
                "mode": mode,
                "source": str(path.resolve()),
                "mesh": str(mesh),
                "metrics": extracted,
                "safety": {
                    name: float(values[field])
                    for name, (field, _minimum) in PPO_SAFETY_FLOORS.items()
                },
                "contract": {
                    "checkpoint": report.get("checkpoint"),
                    "stage": report.get("stage"),
                    "steps": values.get("steps"),
                    "num_envs": values.get("num_envs"),
                    "reserve_m": report.get("c3_physical_extension_reserve_m"),
                    # These plant knobs materially change the continuous
                    # metrics.  A latent ablation is paired only when all of
                    # them match, not merely when the checkpoint and mesh do.
                    "lead_caps": report.get("cylinder_target_max_lead_per_axis_m"),
                    "lead_high": report.get("cylinder_target_lead_high_speed_per_axis_m"),
                    "lead_schedule_speed": report.get("cylinder_target_lead_schedule_speed_kmh"),
                    "residual_scale": report.get("online_aam_residual_scale"),
                    "factor_domain": report.get("factor_domain"),
                    "factor_key": report.get("factor_key"),
                    "factor_enabled": report.get("deployment_factor_enabled"),
                    "factor_manifest": report.get("deployment_factor_manifest"),
                    "aam_checkpoint": report.get("online_aam_checkpoint"),
                    "adapter_checkpoint": report.get("online_aam_adapter_checkpoint"),
                    "history_lag": report.get("online_aam_history_lag"),
                    "smoother_window": report.get("online_aam_smoother_window"),
                    "controller_latent_mode": report.get(
                        "online_aam_controller_latent_mode"
                    ),
                    "seed": report.get("evaluation_seed"),
                    "target_leak": report.get("cylinder_target_leak"),
                    "limit_inset": report.get("cylinder_target_limit_inset_m"),
                    "null_gain": report.get("dls_null_space_centering_gain"),
                    "speed_min_kmh": report.get("vehicle_speed_min_kmh"),
                    "speed_max_kmh": report.get("vehicle_speed_max_kmh"),
                },
            }
    return rows


def ppo_gate(estimated_paths: list[Path], zero_paths: list[Path], shuffled_paths: list[Path]) -> dict:
    estimated = _ppo_rows(estimated_paths, "estimated")
    zero = _ppo_rows(zero_paths, "zero")
    shuffled = _ppo_rows(shuffled_paths, "shuffled")
    common = sorted(set(estimated) & set(zero) & set(shuffled))
    if len(common) < 3:
        raise ValueError(f"PPO promotion gate needs at least 3 paired reports, got {len(common)}")

    contract_mismatches = []
    for key in common:
        for mode, table in (("zero", zero), ("shuffled", shuffled)):
            if estimated[key]["contract"] != table[key]["contract"]:
                contract_mismatches.append({"pair": key, "control": mode})
    if contract_mismatches:
        raise ValueError(f"PPO reports are not paired under one contract: {contract_mismatches[:3]}")

    comparisons = {}
    checks = {}
    for control_name, control in (("zero", zero), ("shuffled", shuffled)):
        comparisons[control_name] = {}
        checks[control_name] = {}
        for metric, (_field, direction) in PPO_METRICS.items():
            improvements = []
            for key in common:
                est = estimated[key]["metrics"][metric]
                ctl = control[key]["metrics"][metric]
                improvements.append(ctl - est if direction == "lower" else est - ctl)
            comparisons[control_name][metric] = _stats(improvements)
            checks[control_name][metric] = comparisons[control_name][metric]["pass"]

    safety_checks = {}
    for name, (_field, minimum) in PPO_SAFETY_FLOORS.items():
        observed = {
            mode: min(row["safety"][name] for row in table.values())
            for mode, table in (
                ("estimated", estimated),
                ("zero", zero),
                ("shuffled", shuffled),
            )
        }
        safety_checks[name] = {
            "minimum_required": float(minimum),
            "minimum_observed_by_arm": observed,
            "pass": bool(all(value >= minimum for value in observed.values())),
        }

    return {
        "pass": bool(
            all(value for control in checks.values() for value in control.values())
            and all(row["pass"] for row in safety_checks.values())
        ),
        "n_pairs": len(common),
        "pair_keys": common,
        "checks": checks,
        "safety_checks": safety_checks,
        "comparisons": comparisons,
        "sources": {
            "estimated": [str(path.resolve()) for path in estimated_paths],
            "zero": [str(path.resolve()) for path in zero_paths],
            "shuffled": [str(path.resolve()) for path in shuffled_paths],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd-rollout", required=True, type=Path)
    parser.add_argument("--ppo-estimated", required=True, nargs="+", type=Path)
    parser.add_argument("--ppo-zero", required=True, nargs="+", type=Path)
    parser.add_argument("--ppo-shuffled", required=True, nargs="+", type=Path)
    parser.add_argument("--sd-horizon", type=int, default=20)
    parser.add_argument("--minimum-retention", type=float, default=0.8)
    parser.add_argument("--noninferiority-margin-fraction", type=float, default=0.2)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/stage31_rom/promotion/promotion_verdict.json"),
    )
    args = parser.parse_args()

    errors = []
    sd = None
    ppo = None
    try:
        if not 0.0 < args.minimum_retention <= 1.0:
            raise ValueError("--minimum-retention must be in (0, 1]")
        if not 0.0 <= args.noninferiority_margin_fraction < 1.0:
            raise ValueError("--noninferiority-margin-fraction must be in [0, 1)")
        sd = sd_and_aam_gate(
            args.sd_rollout,
            args.sd_horizon,
            minimum_retention=args.minimum_retention,
            noninferiority_margin_fraction=args.noninferiority_margin_fraction,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"SD/AAM gate: {exc}")
    try:
        ppo = ppo_gate(args.ppo_estimated, args.ppo_zero, args.ppo_shuffled)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"PPO gate: {exc}")

    promoted = not errors and bool(sd and sd["pass"] and ppo and ppo["pass"])
    verdict = {
        "format": "online_adaptive_promotion_v1",
        "promoted": promoted,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "promotion_contract": {
            "sd_true_and_estimated_improve_dynamics": bool(
                sd
                and sd["checks"]["true_beats_zero"]
                and sd["checks"]["true_beats_shuffled"]
                and sd["checks"]["estimated_beats_zero"]
                and sd["checks"]["estimated_beats_shuffled"]
            ),
            "aam_estimated_retains_true_advantage": bool(
                sd
                and sd["checks"]["estimated_not_worse_than_true"]
                and sd["checks"]["minimum_retention"]
            ),
            "ppo_estimated_beats_zero_and_shuffled_continuous_metrics": bool(
                ppo and ppo["pass"]
            ),
        },
        "sd_aam": sd,
        "ppo": ppo,
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    raise SystemExit(0 if promoted else 1)


if __name__ == "__main__":
    main()
