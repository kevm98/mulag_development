#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Headless Stage 30 evaluation of the pressure-inclusive causal AAM/ID-FD interface.

The AAM receives only the deployment-observable ROM command/state history:
desired cylinder delta and rate, measured cylinder extension and velocity,
measured cylinder pressure, and measured arm joint state. The recorded ROM
valve command is used only as a teacher-forced ID history/diagnostic target;
it is never an AAM input.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "AAM_module") not in sys.path:
    sys.path.insert(0, str(ROOT / "AAM_module"))

from aam_data import iter_raw_runs, load_runs_info, read_raw_run  # noqa: E402
from aam_data.timebase import rom_tick_indices  # noqa: E402
from tools.evaluate_stage28_id_transfer import (  # noqa: E402
    aggregate,
    domain_key,
    extract_segments,
    p95_abs,
    per_axis_rmse,
    rmse,
    segment_proxies,
)
from mulag_plant.runtime.cylinder_models_runtime import (  # noqa: E402
    CylinderFDRuntime,
    CylinderIDRuntime,
)
from PPO_ID_AAM_FD.online_runtime import OnlineAAMIDFDController  # noqa: E402

DEFAULT_AAM_MODEL = ROOT / "outputs/stage31_rom/aam/L5"
DEFAULT_ADAPTER_MODEL = ROOT / "outputs/stage31_rom/id_adapter/zhat_adapter.pt"
DEFAULT_ID_CHECKPOINT = ROOT / "id_fd_model/best_id_cylinder_control.pt"
DEFAULT_FD_CHECKPOINT = ROOT / "id_fd_model/best_fd_cylinder_strict.pt"
DEFAULT_CYLINDER_MODELS_ROOT = ROOT / "id_fd_model/vendor_models"


ROM_DT = 0.05


def run_variant(
    *,
    online: bool,
    run_ids: list[int],
    domains: list[str],
    splits: list[str],
    metas: list[dict[str, Any]],
    arrays: list[dict[str, np.ndarray]],
    id_rt: CylinderIDRuntime,
    fd_rt: CylinderFDRuntime,
    dp_mean: np.ndarray,
    force_mean: np.ndarray,
    command_limits: np.ndarray,
    jacobian: np.ndarray,
    device: torch.device,
    residual_scale: float,
    input_rate_hz: float,
    aam_model: Path,
    adapter_model: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_ticks = rom_tick_indices(arrays[0])
    n_runs, n_ticks = len(run_ids), int(raw_ticks.size)
    actual_x = np.stack([a["cylinder_extension"][raw_ticks] for a in arrays]).astype(np.float64)
    actual_v = np.stack([a["cylinder_velocity"][raw_ticks] for a in arrays]).astype(np.float64)
    next_x = np.stack([a["cylinder_extension"][raw_ticks + 3] for a in arrays]).astype(np.float64)
    next_v = np.stack([a["cylinder_velocity"][raw_ticks + 3] for a in arrays]).astype(np.float64)
    target_x = np.stack([a["desired_cylinder_position"][raw_ticks] for a in arrays]).astype(np.float64)
    target_v = np.stack([a["desired_cylinder_velocity"][raw_ticks] for a in arrays]).astype(np.float64)
    recorded = np.stack([a["rom_predicted_valve_cmd"][raw_ticks] for a in arrays]).astype(np.float64)
    desired_dx = np.stack([a["desired_delta_extension"][raw_ticks] for a in arrays]).astype(np.float64)
    desired_dv = np.stack([a["desired_delta_extension_velocity"][raw_ticks] for a in arrays]).astype(np.float64)
    joint_q = np.stack([a["joint_position"][raw_ticks] for a in arrays]).astype(np.float64)
    joint_v = np.stack([a["joint_velocity"][raw_ticks] for a in arrays]).astype(np.float64)
    # Pressure is measured deployment state and is part of the causal AAM
    # row. Keep the historical ID/FD hydraulic inputs at their legacy means
    # here so this remains an adapter comparison, not an ID retraining change.
    dp_measured = np.stack([a["dP"][raw_ticks] for a in arrays]).astype(np.float64)
    dp = np.broadcast_to(dp_mean, (n_runs, 4)).copy()
    force = np.broadcast_to(force_mean, (n_runs, 4)).copy()

    def make_aam() -> OnlineAAMIDFDController:
        return OnlineAAMIDFDController(
            aam_checkpoint=aam_model, adapter_checkpoint=adapter_model,
            num_envs=n_runs, device=device, history_lag=5,
            update_period_ticks=1, control_rate_hz=input_rate_hz,
            max_inference_latency_ms=1000.0, residual_scale=residual_scale, enabled=online,
        )

    # Teacher-forced and closed-loop trajectories are separate deployment
    # trajectories, so each needs its own causal AAM history. Sharing a
    # history would leak teacher-forced measurements into the closed loop.
    aam_tf = make_aam()
    aam_closed = make_aam()
    tf_x = np.zeros_like(actual_x); tf_v = np.zeros_like(actual_v)
    closed_x = np.zeros_like(actual_x); closed_v = np.zeros_like(actual_v)
    tf_u = np.zeros((n_runs, n_ticks, 4)); pre_u = np.zeros_like(tf_u)
    closed_u = np.zeros_like(tf_u); closed_pre = np.zeros_like(tf_u)
    closed_x[:, 0] = actual_x[:, 0]; closed_v[:, 0] = actual_v[:, 0]
    prev_tf = np.zeros((n_runs, 4)); prev_closed = np.zeros((n_runs, 4))

    for k in range(n_ticks):
        row = np.concatenate((desired_dx[:, k], desired_dv[:, k], actual_x[:, k], actual_v[:, k], dp_measured[:, k], joint_q[:, k], joint_v[:, k]), axis=1)
        aam_tf.observe(torch.as_tensor(row, dtype=torch.float32, device=device))
        id_in = np.concatenate((actual_x[:, k], actual_v[:, k], dp, force, target_x[:, k] - actual_x[:, k], target_v[:, k] - actual_v[:, k], prev_tf), axis=1)
        base = id_rt.predict(torch.as_tensor(id_in, dtype=torch.float32), prev_valve_cmd=torch.as_tensor(prev_tf, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)
        u, res = aam_tf.condition_valve(
            u_base=torch.as_tensor(base, dtype=torch.float32, device=device),
            extension=torch.as_tensor(actual_x[:, k], dtype=torch.float32, device=device),
            velocity=torch.as_tensor(actual_v[:, k], dtype=torch.float32, device=device),
            dP=torch.as_tensor(dp, dtype=torch.float32, device=device), force=torch.as_tensor(force, dtype=torch.float32, device=device),
            target_dx=torch.as_tensor(target_x[:, k] - actual_x[:, k], dtype=torch.float32, device=device),
            target_dv=torch.as_tensor(target_v[:, k] - actual_v[:, k], dtype=torch.float32, device=device),
        )
        u = u.detach().cpu().numpy().astype(np.float64); res = res.detach().cpu().numpy().astype(np.float64)
        fd = fd_rt.step(valve_cmd=torch.as_tensor(u, dtype=torch.float32), extension=torch.as_tensor(actual_x[:, k], dtype=torch.float32), extension_velocity=torch.as_tensor(actual_v[:, k], dtype=torch.float32), dP=torch.as_tensor(dp, dtype=torch.float32), force=torch.as_tensor(force, dtype=torch.float32)).raw_output.detach().cpu().numpy().astype(np.float64)
        tf_x[:, k] = actual_x[:, k] + fd[:, :4]; tf_v[:, k] = actual_v[:, k] + fd[:, 4:8]
        tf_u[:, k] = u; pre_u[:, k] = base + res; prev_tf = recorded[:, k]

        closed_row = np.concatenate((
            desired_dx[:, k], desired_dv[:, k], closed_x[:, k], closed_v[:, k],
            dp_measured[:, k], joint_q[:, k], joint_v[:, k],
        ), axis=1)
        aam_closed.observe(torch.as_tensor(closed_row, dtype=torch.float32, device=device))
        id_in = np.concatenate((closed_x[:, k], closed_v[:, k], dp, force, target_x[:, k] - closed_x[:, k], target_v[:, k] - closed_v[:, k], prev_closed), axis=1)
        base = id_rt.predict(torch.as_tensor(id_in, dtype=torch.float32), prev_valve_cmd=torch.as_tensor(prev_closed, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)
        u, res = aam_closed.condition_valve(
            u_base=torch.as_tensor(base, dtype=torch.float32, device=device),
            extension=torch.as_tensor(closed_x[:, k], dtype=torch.float32, device=device), velocity=torch.as_tensor(closed_v[:, k], dtype=torch.float32, device=device),
            dP=torch.as_tensor(dp, dtype=torch.float32, device=device), force=torch.as_tensor(force, dtype=torch.float32, device=device),
            target_dx=torch.as_tensor(target_x[:, k] - closed_x[:, k], dtype=torch.float32, device=device), target_dv=torch.as_tensor(target_v[:, k] - closed_v[:, k], dtype=torch.float32, device=device),
        )
        u = u.detach().cpu().numpy().astype(np.float64); res = res.detach().cpu().numpy().astype(np.float64)
        fd = fd_rt.step(valve_cmd=torch.as_tensor(u, dtype=torch.float32), extension=torch.as_tensor(closed_x[:, k], dtype=torch.float32), extension_velocity=torch.as_tensor(closed_v[:, k], dtype=torch.float32), dP=torch.as_tensor(dp, dtype=torch.float32), force=torch.as_tensor(force, dtype=torch.float32)).raw_output.detach().cpu().numpy().astype(np.float64)
        closed_u[:, k] = u; closed_pre[:, k] = base + res
        if k + 1 < n_ticks:
            closed_x[:, k + 1] = closed_x[:, k] + fd[:, :4]; closed_v[:, k + 1] = closed_v[:, k] + fd[:, 4:8]
        prev_closed = u

    records = []
    for i, rid in enumerate(run_ids):
        tf_err = tf_x[i] - next_x[i]; closed_err = closed_x[i] - target_x[i]; transfer = closed_x[i] - actual_x[i]
        sat = np.abs(closed_pre[i]) >= command_limits.reshape(1, 4) - 1.0e-6
        div = np.flatnonzero(np.max(np.abs(closed_x[i]), axis=1) > 1.0)
        h = {}
        for name, stop in (("H20_1s", 20), ("H100_5s", 100), ("H400_20s", 400), ("full", n_ticks)):
            stop = min(stop, n_ticks); h[name] = {"cylinder_tracking_rmse_m": rmse(closed_err[:stop]), "cylinder_transfer_rmse_vs_measured_m": rmse(transfer[:stop]), "valve_preclip_saturation_fraction": float(np.mean(sat[:stop]))}
        records.append({"run_id": rid, "domain": domains[i], "split": splits[i], "domain_key": domain_key(domains[i], splits[i]), "variant": "online_aam" if online else "frozen_id", "teacher_forced": {"next_extension_rmse_m": rmse(tf_err), "next_velocity_rmse_mps": rmse(tf_v[i] - next_v[i]), "valve_vs_recorded_rmse": rmse(tf_u[i] - recorded[i])}, "closed_loop": {"cylinder_tracking_rmse_m": rmse(closed_err), "cylinder_tracking_p95_m": p95_abs(closed_err), "cylinder_transfer_rmse_vs_measured_m": rmse(transfer), "velocity_tracking_rmse_mps": rmse(closed_v[i] - target_v[i]), "measured_baseline_tracking_rmse_m": rmse(actual_x[i] - target_x[i]), "valve_mean_abs": float(np.mean(np.abs(closed_u[i]))), "valve_preclip_saturation_fraction": float(np.mean(sat)), "tool_path_transfer_proxy_rmse_m": rmse(transfer @ jacobian.T), "max_abs_cylinder_prediction_m": float(np.max(np.abs(closed_x[i]))), "first_divergence_time_s_abs_x_gt_1m": float(div[0] * ROM_DT) if div.size else None, "closed_loop_horizons": h, **segment_proxies(target_x[i], closed_x[i], extract_segments(metas[i], raw_ticks))}})
    diag = {
        "online_enabled": online,
        "teacher_forced": aam_tf.diagnostics(),
        "closed_loop": aam_closed.diagnostics(),
    }
    return records, diag


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(ROOT / "AAM_module/datasets/verge_rom_classical_stage31_v1_frozen"))
    p.add_argument("--device", default="auto")
    p.add_argument("--residual-scale", type=float, default=0.75)
    p.add_argument("--out", default=str(ROOT / "outputs/stage31_rom/deployment_eval/heldout"))
    p.add_argument("--aam-model", default=str(DEFAULT_AAM_MODEL))
    p.add_argument("--adapter-model", default=str(DEFAULT_ADAPTER_MODEL))
    p.add_argument("--id-checkpoint", default=str(DEFAULT_ID_CHECKPOINT))
    p.add_argument("--fd-checkpoint", default=str(DEFAULT_FD_CHECKPOINT))
    p.add_argument("--models-root", default=str(DEFAULT_CYLINDER_MODELS_ROOT))
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve(); out = Path(args.out).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    aam_model = Path(args.aam_model).expanduser().resolve()
    adapter_model = Path(args.adapter_model).expanduser().resolve()
    id_checkpoint = Path(args.id_checkpoint).expanduser().resolve()
    fd_checkpoint = Path(args.fd_checkpoint).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    info = load_runs_info(root)
    ids=[]; domains=[]; splits=[]; metas=[]; arrays=[]
    for rid, _ in iter_raw_runs(root):
        rid=int(rid); split=str(info[f"{rid:05d}"]["frozen_split"])
        if split not in ("val", "test"): continue
        loaded=read_raw_run(root,rid,verify=True); ids.append(rid); domains.append(str(loaded["metadata"]["domain"])); splits.append(split); metas.append(loaded["metadata"]); arrays.append(loaded["arrays"])
    id_rt=CylinderIDRuntime(models_root=models_root,checkpoint=id_checkpoint,num_envs=len(ids),device=device)
    fd_rt=CylinderFDRuntime(models_root=models_root,checkpoint=fd_checkpoint,device=device)
    names=list(id_rt.input_feature_names); mean=id_rt.normalizers["x_mean"].detach().cpu().numpy().reshape(-1)
    dp_mean=mean[[names.index(f"dP_{i}") for i in range(1,5)]]; force_mean=mean[[names.index(f"Force_{i}") for i in range(1,5)]]
    command_limits=id_rt.command_limits.detach().cpu().numpy().reshape(-1)
    jacobian=np.asarray(json.loads((ROOT/"mulag_plant/models/ee_jacobian_ground_truck3x4.json").read_text())["jacobian"],dtype=np.float64)
    all_records=[]; diagnostics={}
    for online in (False, True):
        print(f"evaluating {'online_aam' if online else 'frozen_id'} on {len(ids)} held-out runs",flush=True)
        rows,diag=run_variant(online=online,run_ids=ids,domains=domains,splits=splits,metas=metas,arrays=arrays,id_rt=id_rt,fd_rt=fd_rt,dp_mean=dp_mean,force_mean=force_mean,command_limits=command_limits,jacobian=jacobian,device=device,residual_scale=args.residual_scale,input_rate_hz=20.0,aam_model=aam_model,adapter_model=adapter_model)
        all_records.extend(rows); diagnostics["online_aam" if online else "frozen_id"]=diag
    paths=[("teacher_forced","next_extension_rmse_m"),("teacher_forced","next_velocity_rmse_mps"),("closed_loop","cylinder_tracking_rmse_m"),("closed_loop","cylinder_transfer_rmse_vs_measured_m"),("closed_loop","valve_preclip_saturation_fraction"),("closed_loop","tool_path_transfer_proxy_rmse_m"),("closed_loop","first_divergence_time_s_abs_x_gt_1m")]
    summary={}
    for variant in ("frozen_id","online_aam"):
        rows=[r for r in all_records if r["variant"]==variant]
        summary[variant]={"overall": {"/".join(x):aggregate(rows,x) for x in paths}, "test": {"/".join(x):aggregate([r for r in rows if r["split"]=="test"],x) for x in paths}, "validation": {"/".join(x):aggregate([r for r in rows if r["split"]=="val"],x) for x in paths}}
    payload={"format":"stage30_online_aam_id_fd_evaluation_v1","created_at":datetime.now(timezone.utc).isoformat(),"protocol":{"held_out_splits":["val","test"],"aam_lag":5,"aam_update_rate_hz":20.0,"deployment_control_rate_hz":60.0,"latent_source":"causal observable history","fallback":"unchanged frozen ID action","fd":"strict historical FD","physx":"not launched by this headless evaluator"},"diagnostics":diagnostics,"summary":summary,"per_run":all_records}
    (out/"summary.json").write_text(json.dumps(payload,indent=2)+"\n"); print(f"Stage 30 evaluation complete -> {out/'summary.json'}",flush=True)


if __name__ == "__main__": main()
