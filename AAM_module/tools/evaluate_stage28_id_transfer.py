#!/usr/bin/env python3
"""Evaluate the frozen historical Simscape ID/FD pair on Stage-22 domains.

This is an offline, deployment-faithful transfer evaluation.  The historical
ID receives the observable cylinder state and target motion, while dP and
force use the training means used by the current legacy runtime contract.  The
strict historical FD is then rolled forward autoregressively.  The measured
Stage-22 trajectories are the transfer reference; no Stage-22 data is used to
fit or alter either checkpoint.

The report contains:

* teacher-forced one-step ID->FD errors against the measured next cylinder;
* closed-loop cylinder tracking and transfer error by domain;
* valve/action magnitude, clipping and agreement with the recorded ROM valve;
* bounded overshoot/settling proxies for the available continuous trajectory;
* a cylinder-to-tool transfer-error proxy using the validated local Jacobian.

The Stage-22 collection has no desired end-effector trajectory and no force
sensor, so a direct tool-path tracking error cannot be claimed.  The report
marks that limitation and reports tool-path *transfer* error only.
"""

from __future__ import annotations

import argparse
import json
import sys
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


FACTOR_KEYS = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
DOMAIN_KEYS = ("nominal", "hydraulic", "mechanical", "combined_train", "combined_test")
ROM_DT = 0.05
SETTLE_TOLERANCE_M = 0.005
SETTLE_WINDOW_TICKS = 10


def finite_np(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite input encountered")
    return arr


def tensor_to_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def feature_mean(runtime: Any, names: list[str]) -> np.ndarray:
    feature_names = list(runtime.input_feature_names)
    mean = tensor_to_np(runtime.normalizers["x_mean"])
    indices = [feature_names.index(name) for name in names]
    return mean[np.asarray(indices, dtype=np.int64)]


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def p95_abs(error: np.ndarray) -> float:
    return float(np.percentile(np.abs(error), 95.0))


def per_axis_rmse(error: np.ndarray) -> list[float]:
    return [float(np.sqrt(np.mean(np.square(error[..., i])))) for i in range(error.shape[-1])]


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a1 = np.asarray(a, dtype=np.float64).reshape(-1)
    b1 = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.std(a1) <= 1.0e-12 or np.std(b1) <= 1.0e-12:
        return None
    return float(np.corrcoef(a1, b1)[0, 1])


def domain_key(domain: str, split: str) -> str:
    if domain == "combined":
        return f"combined_{split}"
    return domain


def extract_segments(metadata: dict[str, Any], ticks: np.ndarray) -> list[dict[str, Any]]:
    segments = []
    for raw in metadata.get("segment_ids", []):
        # Stage-22 metadata used start_step/end_step; the frozen Stage-31 ROM
        # collection uses the equivalent start/end names.  Normalize both
        # schemas here so transfer evaluation follows the actual run segments.
        start = raw.get("start_step", raw.get("start"))
        end = raw.get("end_step", raw.get("end"))
        if start is None or end is None:
            raise ValueError(f"segment is missing start/end bounds: {raw}")
        mask = (ticks >= int(start)) & (ticks < int(end))
        idx = np.flatnonzero(mask)
        if idx.size:
            segments.append({**raw, "indices": idx.tolist()})
    return segments


def segment_proxies(
    target: np.ndarray,
    prediction: np.ndarray,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute explicitly bounded proxies, not synthetic step-response claims."""
    overshoot_values: list[float] = []
    settling_values: list[float] = []
    settling_count = 0
    for segment in segments:
        idx = np.asarray(segment["indices"], dtype=np.int64)
        if idx.size == 0:
            continue
        err = prediction[idx] - target[idx]
        peak = np.max(np.abs(err), axis=0)
        start = target[idx[0]]
        end = target[idx[-1]]
        motion = end - start
        for axis in range(target.shape[1]):
            if abs(float(motion[axis])) > 2.0e-3:
                signed_excess = np.sign(motion[axis]) * (prediction[idx, axis] - end[axis])
                overshoot_values.append(float(max(0.0, np.max(signed_excess))))
            else:
                overshoot_values.append(float(peak[axis]))

        # Settling is measured only within a declared segment and only means
        # remaining within 5 mm for 0.5 s. Continuous trajectory segments can
        # legitimately return None.
        max_err = np.max(np.abs(err), axis=1)
        for start_i in range(max(0, len(max_err) - SETTLE_WINDOW_TICKS + 1)):
            tail = max_err[start_i : start_i + SETTLE_WINDOW_TICKS]
            if np.all(tail <= SETTLE_TOLERANCE_M):
                settling_values.append(float(start_i * ROM_DT))
                settling_count += 1
                break
    return {
        "overshoot_proxy_max_m": float(max(overshoot_values)) if overshoot_values else None,
        "overshoot_proxy_mean_m": float(np.mean(overshoot_values)) if overshoot_values else None,
        "settling_time_s_mean": float(np.mean(settling_values)) if settling_values else None,
        "settling_time_s_max": float(max(settling_values)) if settling_values else None,
        "settling_segments_with_result": int(settling_count),
        "settling_definition": "first 10 ROM ticks (0.5 s) continuously within 5 mm inside each declared segment",
        "overshoot_definition": "terminal-target signed excess on moving segments; absolute hold error on near-stationary segments",
    }


def aggregate(records: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    if not values:
        return {"n_runs": 0, "mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_runs": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(AAM_ROOT / "datasets/master_v3_stage22_wide_frozen"))
    parser.add_argument("--models-root", default=str(TOOL_ROOT / "learned_models/id_fd_model/vendor_models"))
    parser.add_argument("--id-checkpoint", default=str(TOOL_ROOT / "learned_models/id_fd_model/best_id_cylinder_control.pt"))
    parser.add_argument("--fd-checkpoint", default=str(TOOL_ROOT / "learned_models/id_fd_model/best_fd_cylinder_strict.pt"))
    parser.add_argument("--jacobian", default=str(TOOL_ROOT / "mulag_plant/models/ee_jacobian_ground_truck3x4.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=str(TOOL_ROOT / "outputs/stage28_id_transfer"))
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    runs_info = load_runs_info(root)
    run_ids: list[int] = []
    domains: list[str] = []
    splits: list[str] = []
    metas: list[dict[str, Any]] = []
    arrays_by_run: list[dict[str, np.ndarray]] = []
    ticks_ref: np.ndarray | None = None
    for run_id, _ in iter_raw_runs(root):
        loaded = read_raw_run(root, run_id, verify=True)
        metadata = loaded["metadata"]
        arrays = loaded["arrays"]
        ticks = rom_tick_indices(arrays)
        if ticks_ref is None:
            ticks_ref = ticks
        elif not np.array_equal(ticks_ref, ticks):
            raise ValueError(f"run {run_id} has a different ROM tick grid")
        run_ids.append(int(run_id))
        domains.append(str(metadata["domain"]))
        splits.append(str(runs_info[f"{run_id:05d}"]["frozen_split"]))
        metas.append(metadata)
        arrays_by_run.append(arrays)

    if not run_ids or ticks_ref is None:
        raise SystemExit("no raw runs found")
    n_runs = len(run_ids)
    ticks = ticks_ref
    n_ticks = int(ticks.size)

    id_runtime = CylinderIDRuntime(
        models_root=args.models_root,
        checkpoint=args.id_checkpoint,
        num_envs=n_runs,
        device=device,
    )
    fd_runtime = CylinderFDRuntime(
        models_root=args.models_root,
        checkpoint=args.fd_checkpoint,
        device=device,
    )
    dp_mean = feature_mean(id_runtime, [f"dP_{i}" for i in range(1, 5)])
    force_mean = feature_mean(id_runtime, [f"Force_{i}" for i in range(1, 5)])
    command_limits = tensor_to_np(id_runtime.command_limits)
    jacobian_data = json.loads(Path(args.jacobian).read_text(encoding="utf-8"))
    jacobian = finite_np(jacobian_data["jacobian"])
    if jacobian.shape != (3, 4):
        raise ValueError(f"expected a 3x4 tool Jacobian, got {jacobian.shape}")

    # Stack all runs so the two frozen networks are evaluated in parallel while
    # the state recursion remains causal in time.
    actual_x = np.stack([finite_np(a["cylinder_extension"][ticks]) for a in arrays_by_run])
    actual_v = np.stack([finite_np(a["cylinder_velocity"][ticks]) for a in arrays_by_run])
    actual_dp = np.stack([finite_np(a["dP"][ticks]) for a in arrays_by_run])
    actual_ee = np.stack([finite_np(a["ee_pos_local"][ticks]) for a in arrays_by_run])
    actual_next_x = np.stack([finite_np(a["cylinder_extension"][ticks + 3]) for a in arrays_by_run])
    actual_next_v = np.stack([finite_np(a["cylinder_velocity"][ticks + 3]) for a in arrays_by_run])
    target_x = np.stack([finite_np(a["desired_cylinder_position"][ticks]) for a in arrays_by_run])
    target_v = np.stack([finite_np(a["desired_cylinder_velocity"][ticks]) for a in arrays_by_run])
    recorded_valve = np.stack([finite_np(a["rom_predicted_valve_cmd"][ticks]) for a in arrays_by_run])
    segments = [extract_segments(m, ticks) for m in metas]

    # The current legacy deployment keeps both hydraulic features at the ID
    # training mean unless a separate hydraulic-state mode is explicitly used.
    dp_input = np.broadcast_to(dp_mean.reshape(1, 4), (n_runs, 4)).copy()
    force_input = np.broadcast_to(force_mean.reshape(1, 4), (n_runs, 4)).copy()

    tf_valve = np.zeros((n_runs, n_ticks, 4), dtype=np.float64)
    tf_valve_preclip = np.zeros_like(tf_valve)
    tf_delta = np.zeros_like(tf_valve)
    tf_fd_delta = np.zeros((n_runs, n_ticks, 12), dtype=np.float64)
    closed_x = np.zeros_like(actual_x)
    closed_v = np.zeros_like(actual_v)
    closed_valve = np.zeros_like(tf_valve)
    closed_valve_preclip = np.zeros_like(tf_valve)
    closed_x[:, 0] = actual_x[:, 0]
    closed_v[:, 0] = actual_v[:, 0]
    closed_dp = dp_input.copy()
    closed_prev = np.zeros((n_runs, 4), dtype=np.float64)
    tf_prev = np.zeros((n_runs, 4), dtype=np.float64)

    for k in range(n_ticks):
        x_actual = actual_x[:, k]
        v_actual = actual_v[:, k]
        target_dx_tf = target_x[:, k] - x_actual
        target_dv_tf = target_v[:, k] - v_actual
        id_input_tf = np.concatenate(
            (x_actual, v_actual, dp_input, force_input, target_dx_tf, target_dv_tf, tf_prev), axis=1
        )
        id_out_tf = id_runtime.predict(torch.from_numpy(id_input_tf.astype(np.float32)), prev_valve_cmd=torch.from_numpy(tf_prev.astype(np.float32)))
        valve_tf = id_out_tf.detach().cpu().numpy().astype(np.float64)
        pre_tf = id_runtime.last_output_preclip.detach().cpu().numpy().astype(np.float64)
        delta_tf = id_runtime.last_output_denormalized.detach().cpu().numpy().astype(np.float64)
        fd_tf = fd_runtime.step(
            valve_cmd=torch.from_numpy(valve_tf.astype(np.float32)),
            extension=torch.from_numpy(x_actual.astype(np.float32)),
            extension_velocity=torch.from_numpy(v_actual.astype(np.float32)),
            dP=torch.from_numpy(dp_input.astype(np.float32)),
            force=torch.from_numpy(force_input.astype(np.float32)),
        ).raw_output.detach().cpu().numpy().astype(np.float64)
        tf_valve[:, k] = valve_tf
        tf_valve_preclip[:, k] = pre_tf
        tf_delta[:, k] = delta_tf
        tf_fd_delta[:, k] = fd_tf
        tf_prev = recorded_valve[:, k]

        x_closed = closed_x[:, k]
        v_closed = closed_v[:, k]
        target_dx_closed = target_x[:, k] - x_closed
        target_dv_closed = target_v[:, k] - v_closed
        id_input_closed = np.concatenate(
            (x_closed, v_closed, closed_dp, force_input, target_dx_closed, target_dv_closed, closed_prev), axis=1
        )
        id_out_closed = id_runtime.predict(
            torch.from_numpy(id_input_closed.astype(np.float32)),
            prev_valve_cmd=torch.from_numpy(closed_prev.astype(np.float32)),
        )
        valve_closed = id_out_closed.detach().cpu().numpy().astype(np.float64)
        pre_closed = id_runtime.last_output_preclip.detach().cpu().numpy().astype(np.float64)
        fd_closed = fd_runtime.step(
            valve_cmd=torch.from_numpy(valve_closed.astype(np.float32)),
            extension=torch.from_numpy(x_closed.astype(np.float32)),
            extension_velocity=torch.from_numpy(v_closed.astype(np.float32)),
            dP=torch.from_numpy(closed_dp.astype(np.float32)),
            force=torch.from_numpy(force_input.astype(np.float32)),
        ).raw_output.detach().cpu().numpy().astype(np.float64)
        closed_valve[:, k] = valve_closed
        closed_valve_preclip[:, k] = pre_closed
        if k + 1 < n_ticks:
            closed_x[:, k + 1] = x_closed + fd_closed[:, 0:4]
            closed_v[:, k + 1] = v_closed + fd_closed[:, 4:8]
            # The deployed legacy mode keeps dP at the training feature mean.
            closed_dp = dp_input.copy()
        closed_prev = valve_closed

    records: list[dict[str, Any]] = []
    for i, run_id in enumerate(run_ids):
        tf_next_x = actual_x[i] + tf_fd_delta[i, :, 0:4]
        tf_next_v = actual_v[i] + tf_fd_delta[i, :, 4:8]
        tf_x_error = tf_next_x - actual_next_x[i]
        tf_v_error = tf_next_v - actual_next_v[i]
        valve_error = tf_valve[i] - recorded_valve[i]
        closed_track_error = closed_x[i] - target_x[i]
        actual_track_error = actual_x[i] - target_x[i]
        closed_transfer_error = closed_x[i] - actual_x[i]
        tool_transfer = (closed_transfer_error @ jacobian.T)
        sat_pre = np.abs(closed_valve_preclip[i]) >= (command_limits.reshape(1, 4) - 1.0e-6)
        sat_post = np.abs(closed_valve[i]) >= (command_limits.reshape(1, 4) - 1.0e-6)
        # Stage-22 ROM logging uses the plant command bound +/-1.0.  This is
        # deliberately separate from the historical ID checkpoint's tighter
        # per-axis limits used for ``sat_pre``/``sat_post``.
        recorded_sat = np.abs(recorded_valve[i]) >= (1.0 - 1.0e-6)
        horizon_metrics: dict[str, dict[str, float]] = {}
        for horizon_name, horizon_ticks in (("H20_1s", 20), ("H100_5s", 100), ("H400_20s", 400), ("full", n_ticks)):
            stop = min(int(horizon_ticks), n_ticks)
            horizon_metrics[horizon_name] = {
                "cylinder_tracking_rmse_m": rmse(closed_track_error[:stop]),
                "cylinder_transfer_rmse_vs_measured_m": rmse(closed_transfer_error[:stop]),
                "velocity_tracking_rmse_mps": rmse((closed_v[i] - target_v[i])[:stop]),
                "valve_preclip_saturation_fraction": float(np.mean(sat_pre[:stop])),
            }
        divergence_idx = np.flatnonzero(np.max(np.abs(closed_x[i]), axis=1) > 1.0)
        record = {
            "run_id": int(run_id),
            "domain": domains[i],
            "split": splits[i],
            "domain_key": domain_key(domains[i], splits[i]),
            "factors": {key: float(metas[i][key]) for key in FACTOR_KEYS},
            "teacher_forced": {
                "next_extension_rmse_m": rmse(tf_x_error),
                "next_extension_rmse_per_axis_m": per_axis_rmse(tf_x_error),
                "next_velocity_rmse_mps": rmse(tf_v_error),
                "next_velocity_rmse_per_axis_mps": per_axis_rmse(tf_v_error),
                "valve_vs_recorded_rmse": rmse(valve_error),
                "valve_vs_recorded_mae": float(np.mean(np.abs(valve_error))),
                "valve_vs_recorded_correlation": correlation(tf_valve[i], recorded_valve[i]),
            },
            "closed_loop": {
                "cylinder_tracking_rmse_m": rmse(closed_track_error),
                "cylinder_tracking_rmse_per_axis_m": per_axis_rmse(closed_track_error),
                "cylinder_tracking_p95_m": p95_abs(closed_track_error),
                "cylinder_transfer_rmse_vs_measured_m": rmse(closed_transfer_error),
                "velocity_tracking_rmse_mps": rmse(closed_v[i] - target_v[i]),
                "measured_baseline_tracking_rmse_m": rmse(actual_track_error),
                "valve_mean_abs": float(np.mean(np.abs(closed_valve[i]))),
                "valve_p95_abs": float(np.percentile(np.abs(closed_valve[i]), 95.0)),
                "valve_max_abs": float(np.max(np.abs(closed_valve[i]))),
                "valve_preclip_max_abs": float(np.max(np.abs(closed_valve_preclip[i]))),
                "valve_preclip_saturation_fraction": float(np.mean(sat_pre)),
                "valve_postclip_saturation_fraction": float(np.mean(sat_post)),
                "recorded_valve_mean_abs": float(np.mean(np.abs(recorded_valve[i]))),
                "recorded_valve_saturation_fraction": float(np.mean(recorded_sat)),
                "tool_path_transfer_proxy_rmse_m": rmse(tool_transfer),
                "tool_path_transfer_proxy_p95_m": p95_abs(tool_transfer),
                "max_abs_cylinder_prediction_m": float(np.max(np.abs(closed_x[i]))),
                "first_divergence_time_s_abs_x_gt_1m": float(divergence_idx[0] * ROM_DT) if divergence_idx.size else None,
                "closed_loop_horizons": horizon_metrics,
                **segment_proxies(target_x[i], closed_x[i], segments[i]),
            },
        }
        records.append(record)

    metric_paths = [
        ("teacher_forced", "next_extension_rmse_m"),
        ("teacher_forced", "next_velocity_rmse_mps"),
        ("teacher_forced", "valve_vs_recorded_rmse"),
        ("closed_loop", "cylinder_tracking_rmse_m"),
        ("closed_loop", "cylinder_transfer_rmse_vs_measured_m"),
        ("closed_loop", "velocity_tracking_rmse_mps"),
        ("closed_loop", "measured_baseline_tracking_rmse_m"),
        ("closed_loop", "valve_mean_abs"),
        ("closed_loop", "recorded_valve_mean_abs"),
        ("closed_loop", "valve_preclip_saturation_fraction"),
        ("closed_loop", "recorded_valve_saturation_fraction"),
        ("closed_loop", "tool_path_transfer_proxy_rmse_m"),
        ("closed_loop", "max_abs_cylinder_prediction_m"),
        ("closed_loop", "first_divergence_time_s_abs_x_gt_1m"),
        ("closed_loop", "overshoot_proxy_max_m"),
        ("closed_loop", "settling_time_s_mean"),
    ]
    for horizon_name in ("H20_1s", "H100_5s", "H400_20s", "full"):
        for metric_name in ("cylinder_tracking_rmse_m", "cylinder_transfer_rmse_vs_measured_m", "velocity_tracking_rmse_mps", "valve_preclip_saturation_fraction"):
            metric_paths.append(("closed_loop", "closed_loop_horizons", horizon_name, metric_name))
    groups: dict[str, list[dict[str, Any]]] = {key: [] for key in DOMAIN_KEYS}
    groups["overall"] = records
    groups["combined"] = []
    groups["held_out"] = []
    for record in records:
        groups.setdefault(record["domain_key"], []).append(record)
        if record["domain"] == "combined":
            groups["combined"].append(record)
            if record["split"] == "test":
                groups["held_out"].append(record)
    summary_groups: dict[str, Any] = {}
    for group, group_records in groups.items():
        if not group_records:
            continue
        summary_groups[group] = {
            "n_runs": len(group_records),
            "metrics": {"/".join(path): aggregate(group_records, path) for path in metric_paths},
        }

    report = {
        "stage": "28",
        "status": "complete",
        "protocol": {
            "description": "Frozen historical Simscape-trained legacy ID plus strict FD evaluated on frozen Stage-22 ROM+PhysX trajectories.",
            "n_runs": n_runs,
            "n_rom_ticks_per_run": n_ticks,
            "rom_dt_s": ROM_DT,
            "teacher_forced": "ID receives measured current cylinder state; FD output is compared with measured next state.",
            "closed_loop": "ID receives the predicted current cylinder state and the recorded desired cylinder reference; strict FD recursively advances the predicted state.",
            "dP_input": "ID training mean, held fixed; matches the current legacy training_mean hydraulic-feature contract.",
            "force_input": "ID training mean, held fixed; force is not present in Stage-22 deployment logs.",
            "no_adaptation": True,
            "no_checkpoint_updates": True,
            "overshoot_settling_note": "Stage-22 contains continuous DLS/PRBS trajectories rather than isolated step labels; bounded proxies are reported with their definitions.",
            "tool_path_note": "No desired end-effector trajectory is recorded. Tool-path values are transfer-error proxies J*(x_pred-x_measured), not direct task tracking error.",
        },
        "artifacts": {
            "id_checkpoint": str(Path(args.id_checkpoint).expanduser().resolve()),
            "fd_checkpoint": str(Path(args.fd_checkpoint).expanduser().resolve()),
            "dataset_root": str(root),
            "jacobian": str(Path(args.jacobian).expanduser().resolve()),
            "device": device,
        },
        "feature_means": {"dP": dp_mean.tolist(), "force": force_mean.tolist()},
        "command_limits": command_limits.tolist(),
        "domains": summary_groups,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "per_run.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "rollouts.npz",
        run_id=np.asarray(run_ids, dtype=np.int64),
        ticks=ticks,
        actual_x=actual_x.astype(np.float32),
        target_x=target_x.astype(np.float32),
        closed_x=closed_x.astype(np.float32),
        closed_v=closed_v.astype(np.float32),
        closed_valve=closed_valve.astype(np.float32),
        closed_valve_preclip=closed_valve_preclip.astype(np.float32),
        teacher_forced_valve=tf_valve.astype(np.float32),
        recorded_valve=recorded_valve.astype(np.float32),
        teacher_forced_fd_delta=tf_fd_delta.astype(np.float32),
    )
    print(json.dumps({"out": str(out), "n_runs": n_runs, "n_ticks": n_ticks, "groups": list(summary_groups)}, indent=2))


if __name__ == "__main__":
    main()
