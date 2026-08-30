#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Run the minimum CPU-only gates for the online AAM demonstration.

This check deliberately does not launch Isaac Sim or PPO.  It verifies the
deployment-critical properties that must hold before a small online-adaptation
demonstration is shown: the streaming implementation matches an independent
causal reference, future rows cannot affect an earlier estimate, warm-up and
per-environment reset are isolated, and the bounded adapter falls back to the
frozen ID action on invalid inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PPO_AAM_FD.online_runtime import OnlineAAMIDFDController

DEFAULT_AAM_MODEL = ROOT / "outputs/stage31_rom/aam_cnn/L20_d64"
DEFAULT_ADAPTER_MODEL = ROOT / "outputs/stage31_rom/id_adapter_cnn_l20_s5/zhat_adapter.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_controller(args: argparse.Namespace) -> OnlineAAMIDFDController:
    history_lag = int(
        json.loads((Path(args.aam_model) / "metrics.json").read_text())["lag_ticks"]
    )
    return OnlineAAMIDFDController(
        aam_checkpoint=Path(args.aam_model),
        adapter_checkpoint=Path(args.adapter_model),
        num_envs=args.num_envs,
        device=args.device,
        history_lag=history_lag,
        update_period_ticks=3,
        control_rate_hz=60.0,
        # Keep the check aligned with the promoted runtime configuration.
        smoother_window=5,
        max_inference_latency_ms=args.max_latency_ms,
        residual_scale=0.75,
    )


@torch.no_grad()
def causal_reference(
    controller: OnlineAAMIDFDController, rows: torch.Tensor
) -> torch.Tensor:
    """Independent state-machine reference for the runtime's causal trace."""
    n_envs = rows.shape[1]
    history = torch.zeros(
        (n_envs, controller.history_lag, controller.FEATURE_DIM),
        dtype=torch.float32,
        device=controller.device,
    )
    valid_rows = torch.zeros(n_envs, dtype=torch.long, device=controller.device)
    control_ticks = torch.zeros(n_envs, dtype=torch.long, device=controller.device)
    z_hat = torch.zeros((n_envs, 2), dtype=torch.float32, device=controller.device)
    prediction_history = torch.zeros(
        (n_envs, controller.smoother_window, 2),
        dtype=torch.float32,
        device=controller.device,
    )
    prediction_count = torch.zeros(n_envs, dtype=torch.long, device=controller.device)
    trace = []
    for row in rows:
        history[:, 1:] = history[:, :-1].clone()
        history[:, 0] = row
        valid_rows.add_(1).clamp_max_(controller.history_lag)
        control_ticks.add_(1)
        update = (control_ticks % controller.update_period_ticks == 0) & (
            valid_rows >= controller.history_lag
        )
        if bool(update.any()):
            window = history.reshape(n_envs, -1)
            prediction = controller.aam(
                (window - controller.aam_mean) / controller.aam_std
            )
            if bool(update.any()):
                prediction_history[update, 1:] = prediction_history[update, :-1].clone()
                prediction_history[update, 0] = prediction[update]
                prediction_count[update] = (
                    prediction_count[update] + 1
                ).clamp_max(controller.smoother_window)
                count = prediction_count[update].to(torch.float32).reshape(-1, 1)
                z_hat[update] = prediction_history[update].sum(dim=1) / count
        trace.append(z_hat.clone())
    return torch.stack(trace)


@torch.no_grad()
def streaming_trace(
    controller: OnlineAAMIDFDController, rows: torch.Tensor
) -> tuple[torch.Tensor, list[bool]]:
    trace = []
    ready = []
    for row in rows:
        trace.append(controller.observe(row).clone())
        ready.append(bool(controller.aam_ready.all()))
    return torch.stack(trace), ready


@torch.no_grad()
def sampled_grid_trace(
    controller: OnlineAAMIDFDController,
    rows: torch.Tensor,
    sample_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[bool]]:
    trace = []
    ready = []
    for row, mask in zip(rows, sample_mask):
        trace.append(controller.observe(row, sample_mask=mask).clone())
        ready.append(bool(controller.aam_ready.all()))
    return torch.stack(trace), ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--ticks", type=int, default=72)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-latency-ms", type=float, default=50.0)
    parser.add_argument("--aam-model", default=str(DEFAULT_AAM_MODEL))
    parser.add_argument("--adapter-model", default=str(DEFAULT_ADAPTER_MODEL))
    parser.add_argument(
        "--out",
        default="outputs/stage31_rom/deployment_eval/minimum_online_adaptation_check.json",
    )
    args = parser.parse_args()
    if args.num_envs < 2:
        raise SystemExit("minimum check requires --num-envs >= 2")

    aam_model = Path(args.aam_model).expanduser().resolve()
    adapter_model = Path(args.adapter_model).expanduser().resolve()
    required = [aam_model / "aam.pt", aam_model / "metrics.json", adapter_model]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")

    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(20260810)
    rows = torch.randn(
        (args.ticks, args.num_envs, OnlineAAMIDFDController.FEATURE_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    controller = make_controller(args)
    start = time.perf_counter()
    online, ready = streaming_trace(controller, rows)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    reference = causal_reference(controller, rows)
    trace_error = float(torch.max(torch.abs(online - reference)).cpu())
    trace_pass = bool(torch.allclose(online, reference, rtol=1.0e-6, atol=1.0e-6))

    future_rows = rows.clone()
    cutoff = min(args.ticks - 2, 14)
    future_rows[cutoff + 1 :] = torch.randn(
        future_rows[cutoff + 1 :].shape,
        generator=torch.Generator(device="cpu").manual_seed(77),
        dtype=torch.float32,
    ).to(device) * 1000.0
    future_controller = make_controller(args)
    changed_trace, _ = streaming_trace(future_controller, future_rows)
    prefix_error = float(
        torch.max(torch.abs(online[: cutoff + 1] - changed_trace[: cutoff + 1])).cpu()
    )
    no_future_pass = bool(prefix_error <= 1.0e-6)

    expected_first_ready_index = next(
        index for index in range(args.ticks)
        if index + 1 >= controller.history_lag
        and (index + 1) % controller.update_period_ticks == 0
    )
    warmup_pass = (
        not any(ready[:expected_first_ready_index])
        and all(ready[expected_first_ready_index:])
    )

    # Production appends only the 50 ms model-grid rows while PhysX advances
    # at 60 Hz. Skipped rows must not advance the causal history.
    sampled_controller = make_controller(args)
    grid_mask = torch.zeros(
        args.ticks, args.num_envs, dtype=torch.bool, device=device
    )
    grid_mask[::3] = True
    sampled_online, sampled_ready = sampled_grid_trace(
        sampled_controller, rows, grid_mask
    )
    sampled_rows = int(grid_mask[:, 0].sum().cpu())
    sample_ready_indices = [i for i, value in enumerate(sampled_ready) if value]
    expected_sample_ready = (
        (controller.history_lag - 1) * controller.update_period_ticks
        if sampled_rows >= controller.history_lag
        else None
    )
    skipped_rows_stable = bool(
        torch.equal(sampled_online[2], sampled_online[1])
        and torch.equal(sampled_online[4], sampled_online[3])
    )
    sample_grid_pass = bool(
        sampled_rows >= 5
        and int(sampled_controller.valid_rows[0].cpu()) == min(
            sampled_rows, controller.history_lag
        )
        and int(sampled_controller.control_ticks[0].cpu()) == sampled_rows
        and sample_ready_indices
        and sample_ready_indices[0] == expected_sample_ready
        and skipped_rows_stable
    )

    reset_controller = make_controller(args)
    reset_ticks = min(
        args.ticks,
        (controller.history_lag + 1) * controller.update_period_ticks,
    )
    streaming_trace(reset_controller, rows[:reset_ticks])
    env1_before = {
        "valid_rows": int(reset_controller.valid_rows[1].cpu()),
        "control_ticks": int(reset_controller.control_ticks[1].cpu()),
        "z_hat": reset_controller.z_hat[1].clone(),
    }
    reset_controller.reset(torch.tensor([0], device=reset_controller.device))
    reset_pass = (
        int(reset_controller.valid_rows[0].cpu()) == 0
        and int(reset_controller.control_ticks[0].cpu()) == 0
        and not bool(reset_controller.aam_ready[0])
        and int(reset_controller.valid_rows[1].cpu()) == env1_before["valid_rows"]
        and int(reset_controller.control_ticks[1].cpu()) == env1_before["control_ticks"]
        and bool(torch.equal(reset_controller.z_hat[1], env1_before["z_hat"]))
    )

    base = torch.full((args.num_envs, 4), 0.2, device=controller.device)
    state = torch.zeros((args.num_envs, 4), device=controller.device)
    final, residual = controller.condition_valve(
        u_base=base,
        extension=state,
        velocity=state,
        dP=state,
        force=state,
        target_dx=state,
        target_dv=state,
    )
    bound = controller.residual_bound.abs() + 1.0e-6
    active_residual_max = float(residual.abs().max().cpu())
    adaptation_active_pass = bool(
        controller.aam_update_count > 0
        and bool(torch.isfinite(controller.z_hat).all())
        and float(controller.z_hat.abs().max().cpu()) > 0.0
        and active_residual_max > 0.0
    )
    bounded_pass = bool(
        torch.isfinite(final).all()
        and torch.isfinite(residual).all()
        and (residual.abs() <= bound).all()
    )
    bad_final, bad_residual = controller.condition_valve(
        u_base=base,
        extension=torch.full_like(state, float("nan")),
        velocity=state,
        dP=state,
        force=state,
        target_dx=state,
        target_dv=state,
    )
    fallback_pass = bool(
        torch.isfinite(bad_final).all()
        and torch.equal(bad_residual, torch.zeros_like(bad_residual))
        and torch.allclose(
            bad_final, base.clamp(-controller.command_limits, controller.command_limits)
        )
    )

    checks = {
        "causal_trace_parity": {
            "pass": trace_pass,
            "max_abs_error": trace_error,
            "tolerance": 1.0e-6,
        },
        "no_future_influence": {
            "pass": no_future_pass,
            "cutoff_index": cutoff,
            "prefix_max_abs_error": prefix_error,
        },
        "warmup": {
            "pass": warmup_pass,
            "history_lag_rows": controller.history_lag,
            "update_period_control_ticks": 3,
            "first_ready_index_zero_based": next(
                (i for i, value in enumerate(ready) if value), None
            ),
        },
        "50ms_sample_grid": {
            "pass": sample_grid_pass,
            "sampled_rows": sampled_rows,
            "first_ready_control_index_zero_based": (
                sample_ready_indices[0] if sample_ready_indices else None
            ),
            "expected_first_ready_control_index_zero_based": expected_sample_ready,
            "skipped_rows_do_not_advance_history": skipped_rows_stable,
        },
        "reset_isolation": {"pass": bool(reset_pass), "reset_env_ids": [0]},
        "online_adaptation_active": {
            "pass": adaptation_active_pass,
            "aam_updates": controller.aam_update_count,
            "max_abs_z_hat": float(controller.z_hat.abs().max().cpu()),
            "max_abs_residual_before_fault_test": active_residual_max,
        },
        "bounded_residual": {
            "pass": bounded_pass,
            "residual_scale": 0.75,
            "max_abs_residual": float(residual.abs().max().cpu()),
            "max_abs_bound": float(controller.residual_bound.abs().max().cpu()),
        },
        "invalid_input_fallback": {"pass": fallback_pass},
        "latency": {
            "pass": bool(controller.max_latency_ms_observed <= args.max_latency_ms),
            "max_inference_latency_ms": controller.max_latency_ms_observed,
            "configured_limit_ms": args.max_latency_ms,
            "wall_time_ms": elapsed_ms,
        },
    }
    status = "PASS" if all(bool(value["pass"]) for value in checks.values()) else "FAIL"
    report = {
        "format": "minimum_online_adaptation_check_v1",
        "status": status,
        "protocol": {
            "aam_feature_dim": OnlineAAMIDFDController.FEATURE_DIM,
            "aam_history_lag": controller.history_lag,
            "control_rate_hz": 60.0,
            "aam_update_rate_hz": 20.0,
            "smoother": f"past-only rolling-{controller.smoother_window}",
            "valve_command_in_aam_input": False,
            "device": str(device),
            "num_envs": args.num_envs,
            "ticks": args.ticks,
        },
        "checks": checks,
        "runtime_diagnostics": controller.diagnostics(),
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path) for path in required
        },
    }
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
