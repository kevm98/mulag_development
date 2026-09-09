#!/usr/bin/env python3
"""Stage 31 preflight for the causal AAM -> ID adapter -> FD pipeline.

This is intentionally fast and headless.  It validates artifact contracts,
causal warm-up/update timing, reset isolation, bounded residuals and frozen-ID
fallback before any Isaac/PhysX or PPO process is started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PPO_AAM_FD.online_runtime import OnlineAAMIDFDController

DEFAULT_AAM_MODEL = ROOT / "outputs/stage31_rom/aam_cnn/L20_d64"
DEFAULT_ADAPTER_MODEL = ROOT / "outputs/stage31_rom/id_adapter_cnn_l20_s5/zhat_adapter.pt"
DEFAULT_ID_CHECKPOINT = ROOT / "learned_models/id_fd_model/best_id_cylinder_control.pt"
DEFAULT_FD_CHECKPOINT = ROOT / "learned_models/id_fd_model/best_fd_cylinder_strict.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="outputs/stage31_rom/deployment_eval/preflight.json")
    p.add_argument("--residual-scale", type=float, default=0.75,
                   help="bounded adapter scale to validate (default: selected 0.75)")
    p.add_argument("--aam-model", default=str(DEFAULT_AAM_MODEL))
    p.add_argument("--adapter-model", default=str(DEFAULT_ADAPTER_MODEL))
    p.add_argument("--id-checkpoint", default=str(DEFAULT_ID_CHECKPOINT))
    p.add_argument("--fd-checkpoint", default=str(DEFAULT_FD_CHECKPOINT))
    args = p.parse_args()
    aam_model = Path(args.aam_model).expanduser().resolve()
    adapter_model = Path(args.adapter_model).expanduser().resolve()
    id_checkpoint = Path(args.id_checkpoint).expanduser().resolve()
    fd_checkpoint = Path(args.fd_checkpoint).expanduser().resolve()
    artifacts = {
        "aam": str(aam_model / "aam.pt"),
        "aam_metrics": str(aam_model / "metrics.json"),
        "adapter": str(adapter_model),
        "id": str(id_checkpoint),
        "fd": str(fd_checkpoint),
    }
    missing = [name for name, path in artifacts.items() if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")
    metrics = json.loads((aam_model / "metrics.json").read_text())
    history_lag = int(metrics["lag_ticks"])
    controller = OnlineAAMIDFDController(
        aam_checkpoint=aam_model,
        adapter_checkpoint=adapter_model,
        num_envs=args.num_envs,
        device=args.device,
        history_lag=history_lag,
        update_period_ticks=3,
        control_rate_hz=60.0,
        # The estimator is trained on a 1 s history, but its output smoother
        # is intentionally short (5 predictions = 250 ms) so adaptation does
        # not add a second full-second delay.
        smoother_window=5,
        max_inference_latency_ms=1000.0,
        residual_scale=args.residual_scale,
    )
    row = torch.zeros((args.num_envs, controller.FEATURE_DIM), device=controller.device)
    for _ in range(history_lag + 1):
        controller.observe(row)
    if not bool(controller.aam_ready.any()):
        raise RuntimeError(
            f"AAM did not become ready after causal L{history_lag} warm-up"
        )
    base = torch.full((args.num_envs, 4), 0.2, device=controller.device)
    state = torch.zeros((args.num_envs, 4), device=controller.device)
    final, residual = controller.condition_valve(
        u_base=base, extension=state, velocity=state, dP=state, force=state,
        target_dx=state, target_dv=state,
    )
    if not torch.isfinite(final).all() or float(residual.abs().max()) > float(controller.residual_bound.abs().max()) + 1e-6:
        raise RuntimeError("adapter action is non-finite or exceeds residual bound")
    # NaN input must select the unchanged frozen-ID action.
    bad, bad_residual = controller.condition_valve(
        u_base=base, extension=torch.full_like(state, float("nan")), velocity=state,
        dP=state, force=state, target_dx=state, target_dv=state,
    )
    if not torch.allclose(bad, base.clamp(-controller.command_limits, controller.command_limits)) or bool(bad_residual.abs().any()):
        raise RuntimeError("non-finite adapter input did not fall back to frozen ID")
    controller.reset(torch.tensor([0]))
    if bool(controller.aam_ready[0]) or int(controller.valid_rows[0]) != 0:
        raise RuntimeError("per-environment reset did not clear AAM history")
    report = {
        "status": "PASS",
        "artifacts": artifacts,
        "checks": {
            "causal_history_lag": controller.history_lag,
            "update_period_control_ticks": controller.update_period_ticks,
            "nominal_update_rate_hz": controller.diagnostics()["nominal_update_rate_hz"],
            "warmup_rows": controller.history_lag,
            "bounded_residual": True,
            "residual_scale": args.residual_scale,
            "nonfinite_fallback_to_frozen_id": True,
            "per_environment_reset": True,
        },
        "diagnostics": controller.diagnostics(),
    }
    out = Path(args.out).expanduser().resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
