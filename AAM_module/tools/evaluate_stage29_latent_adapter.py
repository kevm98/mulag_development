#!/usr/bin/env python3
"""Compare frozen ID with the two Stage-29 latent-conditioned adapters."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

TOOL_ROOT = Path(__file__).resolve().parents[2]
AAM_ROOT = TOOL_ROOT / "AAM_module"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import iter_raw_runs, load_runs_info, read_raw_run  # noqa: E402
from aam_data.timebase import rom_tick_indices  # noqa: E402
from mulag_plant.runtime.cylinder_models_runtime import (  # noqa: E402
    CylinderFDRuntime,
    CylinderIDRuntime,
)
from tools.evaluate_stage28_id_transfer import (  # noqa: E402
    aggregate,
    domain_key,
    extract_segments,
    p95_abs,
    per_axis_rmse,
    rmse,
    segment_proxies,
)
from tools.train_stage29_latent_adapter import (  # noqa: E402
    ResidualAdapter,
    load_latent_map,
    load_true_latent,
    runtime_mean,
)


ROM_DT = 0.05


def load_adapter(path: Path, device: torch.device, residual_scale: float = 1.0) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ResidualAdapter(
        int(payload["model_config"]["in_dim"]),
        int(payload["model_config"]["hidden"]),
        int(payload["model_config"]["depth"]),
        int(payload["model_config"]["out_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    payload["residual_bound"] = payload["residual_bound"] * float(residual_scale)
    return {"model": model, **payload}


def one_variant(
    name: str,
    adapter: dict[str, Any] | None,
    latent_kind: str | None,
    *,
    run_ids: list[int],
    domains: list[str],
    splits: list[str],
    metas: list[dict[str, Any]],
    arrays_list: list[dict[str, np.ndarray]],
    id_runtime: CylinderIDRuntime,
    fd_runtime: CylinderFDRuntime,
    dp_mean: np.ndarray,
    force_mean: np.ndarray,
    true_maps: dict[int, dict[int, np.ndarray]],
    zhat_maps: dict[int, dict[int, np.ndarray]],
    jacobian: np.ndarray,
    command_limits: np.ndarray,
    lag: int,
) -> list[dict[str, Any]]:
    ticks_raw = rom_tick_indices(arrays_list[0])
    n_runs, n_ticks = len(run_ids), ticks_raw.size
    actual_x = np.stack([np.asarray(a["cylinder_extension"][ticks_raw], dtype=np.float64) for a in arrays_list])
    actual_v = np.stack([np.asarray(a["cylinder_velocity"][ticks_raw], dtype=np.float64) for a in arrays_list])
    target_x = np.stack([np.asarray(a["desired_cylinder_position"][ticks_raw], dtype=np.float64) for a in arrays_list])
    target_v = np.stack([np.asarray(a["desired_cylinder_velocity"][ticks_raw], dtype=np.float64) for a in arrays_list])
    recorded = np.stack([np.asarray(a["rom_predicted_valve_cmd"][ticks_raw], dtype=np.float64) for a in arrays_list])
    dp = np.broadcast_to(dp_mean, (n_runs, 4)).copy()
    force = np.broadcast_to(force_mean, (n_runs, 4)).copy()
    tf_x = np.zeros_like(actual_x)
    tf_v = np.zeros_like(actual_v)
    tf_u = np.zeros((n_runs, n_ticks, 4))
    tf_pre = np.zeros_like(tf_u)
    closed_x = np.zeros_like(actual_x)
    closed_v = np.zeros_like(actual_v)
    closed_u = np.zeros_like(tf_u)
    closed_pre = np.zeros_like(tf_u)
    closed_x[:, 0] = actual_x[:, 0]
    closed_v[:, 0] = actual_v[:, 0]
    tf_prev = np.zeros((n_runs, 4), dtype=np.float64)
    closed_prev = np.zeros((n_runs, 4), dtype=np.float64)

    def action(x: np.ndarray, v: np.ndarray, targetx: np.ndarray, targetv: np.ndarray,
               prev: np.ndarray, z: np.ndarray, use_adapter: bool) -> tuple[np.ndarray, np.ndarray]:
        id_input = np.concatenate((x, v, dp, force, targetx - x, targetv - v, prev), axis=1)
        base = id_runtime.predict(
            torch.as_tensor(id_input, dtype=torch.float32),
            prev_valve_cmd=torch.as_tensor(prev, dtype=torch.float32),
        ).detach().cpu().numpy().astype(np.float64)
        residual = np.zeros_like(base)
        if adapter is not None and use_adapter:
            feat = np.concatenate((x, v, dp, force, targetx - x, targetv - v, base, z), axis=1)
            dev = next(adapter["model"].parameters()).device
            with torch.no_grad():
                xn = (torch.as_tensor(feat, dtype=torch.float32, device=dev) - adapter["feature_mean"].to(dev)) / adapter["feature_std"].to(dev)
                residual = (adapter["residual_bound"].to(dev) * torch.tanh(adapter["model"](xn))).cpu().numpy().astype(np.float64)
        pre = base + residual
        return np.clip(pre, -command_limits.reshape(1, 4), command_limits.reshape(1, 4)), pre

    for k in range(n_ticks):
        z_tf = np.zeros((n_runs, 2), dtype=np.float64)
        z_cl = np.zeros_like(z_tf)
        adapter_active = latent_kind is not None and k >= max(lag - 1, 0)
        if adapter_active:
            maps = true_maps if latent_kind == "true" else zhat_maps
            declared_lag = adapter.get("latent_lag_ticks") if adapter else None
            if declared_lag is not None and int(declared_lag) != int(lag):
                raise ValueError(
                    f"{name} adapter declares L{int(declared_lag)}, evaluator uses L{lag}"
                )
            smoothing = int(
                adapter.get("smoothing_window_predictions", 1) if adapter else 1
            )
            if smoothing < 1:
                raise ValueError(f"{name} adapter has invalid smoothing window {smoothing}")
            start = max(lag - 1, k - smoothing + 1)
            window_ticks = [int(ticks_raw[index]) for index in range(start, k + 1)]
            # Stage-31 latent maps are keyed by raw 60 Hz control ticks, not
            # local ROM-row indices. Average only past/current predictions to
            # reproduce the online causal smoother exactly.
            z_tf = np.stack([
                np.mean([maps[rid][tick] for tick in window_ticks], axis=0)
                for rid in run_ids
            ])
            z_cl = z_tf
        u, pre = action(actual_x[:, k], actual_v[:, k], target_x[:, k], target_v[:, k], tf_prev, z_tf, adapter_active)
        fd = fd_runtime.step(
            valve_cmd=torch.as_tensor(u, dtype=torch.float32),
            extension=torch.as_tensor(actual_x[:, k], dtype=torch.float32),
            extension_velocity=torch.as_tensor(actual_v[:, k], dtype=torch.float32),
            dP=torch.as_tensor(dp, dtype=torch.float32), force=torch.as_tensor(force, dtype=torch.float32),
        ).raw_output.detach().cpu().numpy().astype(np.float64)
        tf_u[:, k], tf_pre[:, k] = u, pre
        tf_x[:, k] = actual_x[:, k] + fd[:, 0:4]
        tf_v[:, k] = actual_v[:, k] + fd[:, 4:8]
        tf_prev = recorded[:, k]

        u, pre = action(closed_x[:, k], closed_v[:, k], target_x[:, k], target_v[:, k], closed_prev, z_cl, adapter_active)
        fd = fd_runtime.step(
            valve_cmd=torch.as_tensor(u, dtype=torch.float32),
            extension=torch.as_tensor(closed_x[:, k], dtype=torch.float32),
            extension_velocity=torch.as_tensor(closed_v[:, k], dtype=torch.float32),
            dP=torch.as_tensor(dp, dtype=torch.float32), force=torch.as_tensor(force, dtype=torch.float32),
        ).raw_output.detach().cpu().numpy().astype(np.float64)
        closed_u[:, k], closed_pre[:, k] = u, pre
        if k + 1 < n_ticks:
            closed_x[:, k + 1] = closed_x[:, k] + fd[:, 0:4]
            closed_v[:, k + 1] = closed_v[:, k] + fd[:, 4:8]
        closed_prev = u

    records: list[dict[str, Any]] = []
    for i, run_id in enumerate(run_ids):
        tf_err = tf_x[i] - np.stack([np.asarray(a["cylinder_extension"][ticks_raw + 3], dtype=np.float64) for a in arrays_list])[i]
        tf_verr = tf_v[i] - np.stack([np.asarray(a["cylinder_velocity"][ticks_raw + 3], dtype=np.float64) for a in arrays_list])[i]
        closed_track = closed_x[i] - target_x[i]
        transfer = closed_x[i] - actual_x[i]
        sat_pre = np.abs(closed_pre[i]) >= command_limits.reshape(1, 4) - 1.0e-6
        sat_post = np.abs(closed_u[i]) >= command_limits.reshape(1, 4) - 1.0e-6
        horizons = {}
        for hname, h in (("H20_1s", 20), ("H100_5s", 100), ("H400_20s", 400), ("full", n_ticks)):
            stop = min(h, n_ticks)
            horizons[hname] = {
                "cylinder_tracking_rmse_m": rmse(closed_track[:stop]),
                "cylinder_transfer_rmse_vs_measured_m": rmse(transfer[:stop]),
                "valve_preclip_saturation_fraction": float(np.mean(sat_pre[:stop])),
            }
        divergence = np.flatnonzero(np.max(np.abs(closed_x[i]), axis=1) > 1.0)
        records.append({
            "run_id": run_id, "domain": domains[i], "split": splits[i],
            "domain_key": domain_key(domains[i], splits[i]), "variant": name,
            "teacher_forced": {
                "next_extension_rmse_m": rmse(tf_err),
                "next_velocity_rmse_mps": rmse(tf_verr),
                "valve_vs_recorded_rmse": rmse(tf_u[i] - recorded[i]),
            },
            "closed_loop": {
                "cylinder_tracking_rmse_m": rmse(closed_track),
                "cylinder_tracking_rmse_per_axis_m": per_axis_rmse(closed_track),
                "cylinder_tracking_p95_m": p95_abs(closed_track),
                "cylinder_transfer_rmse_vs_measured_m": rmse(transfer),
                "velocity_tracking_rmse_mps": rmse(closed_v[i] - target_v[i]),
                "measured_baseline_tracking_rmse_m": rmse(actual_x[i] - target_x[i]),
                "valve_mean_abs": float(np.mean(np.abs(closed_u[i]))),
                "valve_p95_abs": float(np.percentile(np.abs(closed_u[i]), 95)),
                "valve_max_abs": float(np.max(np.abs(closed_u[i]))),
                "valve_preclip_max_abs": float(np.max(np.abs(closed_pre[i]))),
                "valve_preclip_saturation_fraction": float(np.mean(sat_pre)),
                "valve_postclip_saturation_fraction": float(np.mean(sat_post)),
                "tool_path_transfer_proxy_rmse_m": rmse(transfer @ jacobian.T),
                "tool_path_transfer_proxy_p95_m": p95_abs(transfer @ jacobian.T),
                "max_abs_cylinder_prediction_m": float(np.max(np.abs(closed_x[i]))),
                "first_divergence_time_s_abs_x_gt_1m": float(divergence[0] * ROM_DT) if divergence.size else None,
                "closed_loop_horizons": horizons,
                **segment_proxies(target_x[i], closed_x[i], extract_segments(metas[i], ticks_raw)),
            },
        })
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(AAM_ROOT / "datasets/master_v3_stage22_wide_frozen"))
    p.add_argument("--aam-root", default=None)
    p.add_argument("--models-root", default=str(TOOL_ROOT / "learned_models/id_fd_model/vendor_models"))
    p.add_argument("--id-checkpoint", default=str(TOOL_ROOT / "learned_models/id_fd_model/best_id_cylinder_control.pt"))
    p.add_argument("--fd-checkpoint", default=str(TOOL_ROOT / "learned_models/id_fd_model/best_fd_cylinder_strict.pt"))
    p.add_argument("--true-adapter", required=True)
    p.add_argument("--zhat-adapter", required=True)
    p.add_argument("--zhat", required=True)
    p.add_argument("--lag", type=int, default=5)
    p.add_argument("--jacobian", default=str(TOOL_ROOT / "mulag_plant/models/ee_jacobian_ground_truck3x4.json"))
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--residual-scale", type=float, default=1.0,
        help="scale the checkpoint residual bound for a conservative transfer sweep",
    )
    p.add_argument("--out", default=str(TOOL_ROOT / "outputs/stage29_latent_adapter/evaluation"))
    args = p.parse_args()
    if not 0.0 <= args.residual_scale <= 1.0:
        raise SystemExit("--residual-scale must be in [0, 1]")
    root = Path(args.root).expanduser().resolve()
    aam_root = Path(args.aam_root or root / "aam" / "dataset").expanduser().resolve()
    out = Path(args.out).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    info = load_runs_info(root)
    run_ids, domains, splits, metas, arrays = [], [], [], [], []
    for rid, _ in iter_raw_runs(root):
        rid = int(rid); split = str(info[f"{rid:05d}"]["frozen_split"])
        if split not in ("val", "test"):
            continue
        loaded = read_raw_run(root, rid, verify=True)
        run_ids.append(rid); domains.append(str(loaded["metadata"]["domain"])); splits.append(split)
        metas.append(loaded["metadata"]); arrays.append(loaded["arrays"])
    if not run_ids:
        raise SystemExit("no held-out runs")
    id_rt = CylinderIDRuntime(models_root=args.models_root, checkpoint=args.id_checkpoint,
                              num_envs=len(run_ids), device=device)
    fd_rt = CylinderFDRuntime(models_root=args.models_root, checkpoint=args.fd_checkpoint, device=device)
    dp_mean = runtime_mean(id_rt, [f"dP_{i}" for i in range(1, 5)])
    force_mean = runtime_mean(id_rt, [f"Force_{i}" for i in range(1, 5)])
    command_limits = id_rt.command_limits.detach().cpu().numpy().reshape(-1)
    jacobian = np.asarray(json.loads(Path(args.jacobian).read_text())["jacobian"], dtype=np.float64)
    true_maps = {rid: load_true_latent(aam_root, rid, args.lag) for rid in run_ids}
    zhat_maps = load_latent_map(Path(args.zhat).expanduser().resolve())
    variants = {
        "frozen_id": (None, None),
        "true_latent_adapter": (load_adapter(Path(args.true_adapter), device, args.residual_scale), "true"),
        "aam_latent_adapter": (load_adapter(Path(args.zhat_adapter), device, args.residual_scale), "zhat"),
    }
    records = []
    for name, (adapter, kind) in variants.items():
        print(f"evaluating {name} on {len(run_ids)} held-out runs", flush=True)
        records.extend(one_variant(name, adapter, kind, run_ids=run_ids, domains=domains, splits=splits,
                                   metas=metas, arrays_list=arrays, id_runtime=id_rt, fd_runtime=fd_rt,
                                   dp_mean=dp_mean, force_mean=force_mean, true_maps=true_maps,
                                   zhat_maps=zhat_maps, jacobian=jacobian,
                                   command_limits=command_limits, lag=args.lag))
    metric_paths = [("teacher_forced", "next_extension_rmse_m"), ("teacher_forced", "next_velocity_rmse_mps"),
                    ("teacher_forced", "valve_vs_recorded_rmse"), ("closed_loop", "cylinder_tracking_rmse_m"),
                    ("closed_loop", "cylinder_transfer_rmse_vs_measured_m"), ("closed_loop", "velocity_tracking_rmse_mps"),
                    ("closed_loop", "valve_preclip_saturation_fraction"), ("closed_loop", "tool_path_transfer_proxy_rmse_m"),
                    ("closed_loop", "first_divergence_time_s_abs_x_gt_1m")]
    summary: dict[str, Any] = {"overall": {}, "by_variant_domain_split": {}}
    for variant in variants:
        vr = [r for r in records if r["variant"] == variant]
        summary["overall"][variant] = {"n_runs": len(vr), **{
            "/".join(path): aggregate(vr, (path[0], path[1])) for path in metric_paths}}
        for group_name, group in (("held_out", [r for r in vr if r["split"] == "test"]), ("validation", [r for r in vr if r["split"] == "val"])):
            summary["by_variant_domain_split"].setdefault(variant, {})[group_name] = {
                "/".join(path): aggregate(group, path) for path in metric_paths}
    payload = {"format": "stage29_latent_adapter_evaluation_v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "protocol": {"held_out_splits": ["val", "test"], "adapter_warmup_ticks": args.lag,
                            "frozen_id_checkpoint": str(Path(args.id_checkpoint).resolve()),
                            "frozen_fd_checkpoint": str(Path(args.fd_checkpoint).resolve()),
                            "dP_force": "ID training means", "tool_path": "local 3x4 Jacobian transfer proxy"},
               "summary": summary, "per_run": records}
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Stage 29 evaluation complete -> {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
