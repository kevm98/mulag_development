#!/usr/bin/env python3
"""Fine-tune the inverse model with causal AAM using recorded ROM valves.

The pretrained ID remains the safety anchor.  A zero-initialized, bounded
residual layer is trained with the deployment data only:

    u_base = ID(x, xdot, dP, reference, previous_valve)
    u_final = clip(u_base + b*tanh(adapter([state, target, u_base, z_hat])))

The objective is valve-command L2 against the classical-controller/ROM
recording.  No ID-FD trajectory is read or collected.  The learned residual is
the fine-tuned ID component used by the online AAM -> ID -> FD stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
AAM_ROOT = ROOT / "AAM_module"
for path in (ROOT, AAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aam_data import load_runs_info, raw_dir, read_raw_run  # noqa: E402
from aam_data.rom_stage31 import rom_ticks  # noqa: E402
from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime  # noqa: E402
from tools.train_stage29_latent_adapter import ResidualAdapter  # noqa: E402


def load_zhat(path: Path) -> dict[int, dict[int, np.ndarray]]:
    payload = json.loads(path.read_text())
    if payload.get("format") != "stage31_aam_zhat_by_window_end_tick_v1":
        raise SystemExit(f"not a Stage 31 causal AAM export: {path}")
    return {
        int(run_id): {int(tick): np.asarray(value, dtype=np.float64) for tick, value in rows.items()}
        for run_id, rows in payload["by_run"].items()
    }


def runtime_mean(runtime: CylinderIDRuntime, names: list[str]) -> np.ndarray:
    names_all = list(runtime.input_feature_names)
    mean = runtime.normalizers["x_mean"].detach().cpu().numpy().reshape(-1)
    return mean[[names_all.index(name) for name in names]].astype(np.float64)


def causal_smooth_zhat(
    zhat: dict[int, np.ndarray], ticks: np.ndarray, window: int
) -> np.ndarray:
    """Match the causal rolling latent filter used by online deployment."""
    if window < 1:
        raise ValueError("AAM smoothing window must be positive")
    values = []
    for tick in ticks:
        value = np.asarray(zhat.get(int(tick), np.full(2, np.nan)), dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"missing causal AAM estimate at ROM tick {int(tick)}")
        values.append(value)
    raw = np.asarray(values, dtype=np.float64)
    smoothed = np.empty_like(raw)
    for index in range(len(raw)):
        smoothed[index] = raw[max(0, index - window + 1): index + 1].mean(axis=0)
    return smoothed


def build_run(
    arrays: dict[str, np.ndarray],
    run_id: int,
    lag: int,
    zhat: dict[int, np.ndarray],
    id_runtime: CylinderIDRuntime,
    dP_mean: np.ndarray,
    force_mean: np.ndarray,
    smoothing_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ticks = rom_ticks(arrays)
    valid = ticks[lag - 1 :]
    z = causal_smooth_zhat(zhat, valid, smoothing_window)
    x = np.asarray(arrays["cylinder_extension"][valid], dtype=np.float64)
    velocity = np.asarray(arrays["cylinder_velocity"][valid], dtype=np.float64)
    desired_x = np.asarray(arrays["desired_cylinder_position"][valid], dtype=np.float64)
    desired_v = np.asarray(arrays["desired_cylinder_velocity"][valid], dtype=np.float64)
    recorded_all = np.asarray(arrays["rom_predicted_valve_cmd"], dtype=np.float64)
    previous_all = np.zeros_like(recorded_all)
    previous_all[3:] = recorded_all[:-3]
    previous = previous_all[valid]
    # Pressure is part of the fresh ROM deployment recording and is available
    # to the ID at deployment. Do not replace it with the historical ID mean;
    # doing so would discard the very hydraulic signal used by AAM.
    dP = np.asarray(arrays["dP"][valid], dtype=np.float64)
    force = np.broadcast_to(force_mean, (len(valid), 4))
    id_input = np.concatenate((x, velocity, dP, force, desired_x - x, desired_v - velocity, previous), axis=1)
    base = []
    with torch.no_grad():
        for start in range(0, len(id_input), 16384):
            batch = torch.as_tensor(id_input[start:start + 16384], dtype=torch.float32, device=id_runtime.device)
            prev = torch.as_tensor(previous[start:start + 16384], dtype=torch.float32, device=id_runtime.device)
            base.append(id_runtime.predict(batch, prev_valve_cmd=prev).cpu().numpy())
    base_np = np.concatenate(base, axis=0).astype(np.float64)
    target = np.asarray(recorded_all[valid], dtype=np.float64)
    features = np.concatenate(
        (x, velocity, dP, force, desired_x - x, desired_v - velocity, base_np, z), axis=1
    )
    if features.shape[1] != 30:
        raise AssertionError(features.shape)
    return features, target, base_np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--zhat", required=True)
    parser.add_argument("--models-root", default=str(ROOT / "learned_models/id_fd_model/vendor_models"))
    parser.add_argument("--id-checkpoint", default=str(ROOT / "learned_models/id_fd_model/best_id_cylinder_control.pt"))
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument("--smoothing-window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--residual-bound", type=float, default=0.15)
    parser.add_argument("--residual-penalty", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    if args.smoothing_window < 1:
        raise SystemExit("--smoothing-window must be positive")
    zhat = load_zhat(Path(args.zhat).expanduser().resolve())
    info = load_runs_info(root)
    id_runtime = CylinderIDRuntime(
        models_root=args.models_root,
        checkpoint=args.id_checkpoint,
        num_envs=1,
        device=device,
    )
    dP_mean = runtime_mean(id_runtime, [f"dP_{i}" for i in range(1, 5)])
    force_mean = runtime_mean(id_runtime, [f"Force_{i}" for i in range(1, 5)])

    sets: dict[str, list[np.ndarray]] = {key: [] for key in ("train_x", "train_y", "train_base", "val_x", "val_y", "val_base")}
    used_runs = {"train": [], "val": []}
    for run_dir in sorted((raw_dir(root) / "runs").iterdir()):
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        run_id = int(run_dir.name)
        split = str(info[f"{run_id:05d}"]["frozen_split"])
        if split not in used_runs:
            continue
        loaded = read_raw_run(root, run_id, verify=True)
        fx, target, base = build_run(
            loaded["arrays"], run_id, args.lag, zhat.get(run_id, {}),
            id_runtime, dP_mean, force_mean, args.smoothing_window,
        )
        prefix = "train" if split == "train" else "val"
        sets[f"{prefix}_x"].append(fx.astype(np.float32))
        sets[f"{prefix}_y"].append(target.astype(np.float32))
        sets[f"{prefix}_base"].append(base.astype(np.float32))
        used_runs[split].append(run_id)
    if not sets["train_x"] or not sets["val_x"]:
        raise SystemExit("Stage 31 ID fine-tuning requires train and validation runs")

    Xtr = torch.as_tensor(np.concatenate(sets["train_x"]), device=device)
    Ytr = torch.as_tensor(np.concatenate(sets["train_y"]), device=device)
    Btr = torch.as_tensor(np.concatenate(sets["train_base"]), device=device)
    Xva = torch.as_tensor(np.concatenate(sets["val_x"]), device=device)
    Yva = torch.as_tensor(np.concatenate(sets["val_y"]), device=device)
    Bva = torch.as_tensor(np.concatenate(sets["val_base"]), device=device)
    feature_mean = Xtr.mean(dim=0, keepdim=True)
    feature_std = Xtr.std(dim=0, keepdim=True).clamp_min(1.0e-5)
    feature_std = torch.maximum(feature_std, feature_mean.abs() * 1.0e-3)
    bound = torch.full((1, 4), args.residual_bound, dtype=torch.float32, device=device)
    limits = torch.as_tensor([0.95, 0.95, 0.8, 0.8], dtype=torch.float32, device=device).reshape(1, 4)
    model = ResidualAdapter(30, args.hidden, args.depth, 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    best = float("inf")
    best_epoch = -1
    stale = 0
    history = []

    def score(X: torch.Tensor, Y: torch.Tensor, base: torch.Tensor) -> tuple[float, float, float]:
        model.eval()
        with torch.no_grad():
            residual = bound * torch.tanh(model((X - feature_mean) / feature_std))
            final = torch.clamp(base + residual, -limits, limits)
            l2 = torch.mean((final - Y) ** 2)
            penalty = torch.mean((residual / bound) ** 2)
            loss = l2 + args.residual_penalty * penalty
        return float(loss), float(l2), float(penalty)

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(Xtr.shape[0], device=device)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            residual = bound * torch.tanh(model((Xtr[idx] - feature_mean) / feature_std))
            final = torch.clamp(Btr[idx] + residual, -limits, limits)
            l2 = torch.mean((final - Ytr[idx]) ** 2)
            penalty = torch.mean((residual / bound) ** 2)
            loss = l2 + args.residual_penalty * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * idx.numel()
        val_loss, val_l2, val_penalty = score(Xva, Yva, Bva)
        history.append({"epoch": epoch, "train_loss": total / Xtr.shape[0], "val_loss": val_loss, "val_valve_l2": val_l2, "val_residual_penalty": val_penalty})
        if val_loss < best - 1.0e-8:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save(
                {
                    "format": "stage31_id_aam_valve_adapter_v1",
                    "model_state": model.state_dict(),
                    "model_config": {"in_dim": 30, "hidden": args.hidden, "depth": args.depth, "out_dim": 4},
                    "feature_mean": feature_mean.detach().cpu(),
                    "feature_std": feature_std.detach().cpu(),
                    "residual_bound": bound.detach().cpu(),
                    "command_limits": limits.detach().cpu(),
                    "lag_ticks": args.lag,
                    "smoothing_window_predictions": args.smoothing_window,
                    "id_checkpoint": str(Path(args.id_checkpoint).expanduser().resolve()),
                    "collection_root": str(root),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                out / "zhat_adapter.pt",
            )
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch {epoch:03d} train={total/Xtr.shape[0]:.6f} val={val_loss:.6f} valve_l2={val_l2:.6f}", flush=True)
        if args.patience > 0 and stale >= args.patience:
            break

    report = {
        "format": "stage31_id_aam_valve_finetune_v1",
        "root": str(root),
        "zhat": str(Path(args.zhat).expanduser().resolve()),
        "train_runs": used_runs["train"],
        "validation_runs": used_runs["val"],
        "held_out_test_not_used": sorted(set(int(k) for k in info) - set(used_runs["train"]) - set(used_runs["val"])),
        "recorded_target": "rom_predicted_valve_cmd at exact ROM tick",
        "objective": "MSE(clipped u_base + bounded AAM residual, recorded valve)",
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "residual_bound": args.residual_bound,
        "history": history,
        "id_checkpoint": str(Path(args.id_checkpoint).expanduser().resolve()),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Stage 31 ID+AAM valve fine-tuning complete -> {out / 'zhat_adapter.pt'}")


if __name__ == "__main__":
    main()
