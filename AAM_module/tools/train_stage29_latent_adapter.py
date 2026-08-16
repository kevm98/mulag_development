#!/usr/bin/env python3
"""Stage 29: train a small residual adapter around frozen historical ID/FD.

The ID and strict FD checkpoints are never updated.  For every exact ROM-grid
transition the frozen ID produces ``u_base`` and the trainable network produces
a bounded residual:

    u_final = clip(u_base + bound * tanh(adapter(features)))

The feature vector is ``[x, xdot, dP, force, target_dx, target_dxdot,
u_base, z]``.  ``z`` is either the privileged Stage-23 latent or the causal,
past-only Stage-27 AAM estimate.  The target is the measured Stage-22 next
state, while the frozen FD supplies the differentiable prediction used by the
loss.  Runs, not individual samples, define the train/validation split.
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
from torch import nn


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


def np_finite(value: Any) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("non-finite dataset value")
    return out


def runtime_mean(runtime: Any, names: list[str]) -> np.ndarray:
    names_all = list(runtime.input_feature_names)
    mean = runtime.normalizers["x_mean"].detach().cpu().numpy().reshape(-1)
    return mean[[names_all.index(name) for name in names]].astype(np.float64)


class ResidualAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, out_dim: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        # A residual adapter must begin as the frozen controller.  Zeroing the
        # last layer prevents a random first step from moving every valve by a
        # large amount before the transfer loss has provided a useful signal.
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_latent_map(
    path: Path, *, expected_lag: int | None = None
) -> dict[int, dict[int, np.ndarray]]:
    payload = json.loads(path.read_text())
    # Stage-31 writes the same causal map with an explicit producer prefix;
    # accept both names so an L20 CNN map cannot be mistaken for an old L5
    # artifact merely because its serialization label changed.
    if payload.get("format") not in {
        "aam_zhat_by_window_end_tick_v1",
        "stage31_aam_zhat_by_window_end_tick_v1",
    }:
        raise ValueError(f"unsupported z_hat file format: {path}")
    declared_lag = payload.get("lag_ticks")
    if (
        expected_lag is not None
        and declared_lag is not None
        and int(declared_lag) != int(expected_lag)
    ):
        raise ValueError(
            f"z_hat lag mismatch: file declares L{int(declared_lag)}, "
            f"but adapter training requested L{int(expected_lag)}"
        )
    result: dict[int, dict[int, np.ndarray]] = {}
    for rid, tick_map in payload["by_run"].items():
        parsed: dict[int, np.ndarray] = {}
        for tick, value in tick_map.items():
            latent = np.asarray(value, dtype=np.float64).reshape(-1)
            if latent.shape != (2,) or not np.isfinite(latent).all():
                raise ValueError(
                    f"invalid two-dimensional latent at run={rid}, tick={tick}"
                )
            parsed[int(tick)] = latent
        result[int(rid)] = parsed
    return result


def load_true_latent(aam_root: Path, run_id: int, lag: int) -> dict[int, np.ndarray]:
    path = aam_root / f"{run_id:05d}" / f"history_L{lag}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {
            int(t): np.asarray(z, dtype=np.float64)
            for t, z in zip(data["window_end_tick"], data["z_target"])
        }


def make_run_samples(
    arrays: dict[str, np.ndarray],
    *,
    id_runtime: CylinderIDRuntime,
    dp_mean: np.ndarray,
    force_mean: np.ndarray,
    latent_by_tick: dict[int, np.ndarray],
    run_id: int,
    lag: int,
    smoothing_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_ticks = rom_tick_indices(arrays)
    n = raw_ticks.size
    if n <= lag or int(raw_ticks[-1]) + 3 >= arrays["cylinder_extension"].shape[0]:
        raise ValueError(f"run {run_id}: insufficient exact ROM transitions")
    # A causal lag is valid after exactly ``lag`` observable rows, i.e. local
    # ROM tick ``lag - 1``. The online runtime exposes the AAM estimate at that
    # same tick, so adapter training must include it rather than starting one
    # tick later.
    k = np.arange(lag - 1, n, dtype=np.int64)
    ticks = raw_ticks[k]
    x = np_finite(arrays["cylinder_extension"][ticks])
    v = np_finite(arrays["cylinder_velocity"][ticks])
    target_x = np_finite(arrays["desired_cylinder_position"][ticks])
    target_v = np_finite(arrays["desired_cylinder_velocity"][ticks])
    next_x = np_finite(arrays["cylinder_extension"][raw_ticks[k] + 3])
    next_v = np_finite(arrays["cylinder_velocity"][raw_ticks[k] + 3])
    recorded = np_finite(arrays["rom_predicted_valve_cmd"][raw_ticks])
    prev_all = np.zeros_like(recorded)
    prev_all[1:] = recorded[:-1]
    prev = prev_all[k]
    dp = np.broadcast_to(dp_mean, (k.size, 4))
    force = np.broadcast_to(force_mean, (k.size, 4))
    id_input = np.concatenate((x, v, dp, force, target_x - x, target_v - v, prev), axis=1)
    with torch.no_grad():
        base_parts = []
        for start in range(0, id_input.shape[0], 16384):
            base_parts.append(id_runtime.predict(
                torch.as_tensor(id_input[start:start + 16384], dtype=torch.float32),
                prev_valve_cmd=torch.as_tensor(prev[start:start + 16384], dtype=torch.float32),
            ).detach().cpu().numpy().astype(np.float64))
    u_base = np.concatenate(base_parts, axis=0)
    # The latent map is indexed by the AAM window-end tick, which is the ROM
    # tick number, not the raw control-step index.
    # The Stage-31 map is keyed by the raw control tick (the same timebase as
    # ``window_end_tick`` in the AAM dataset), not by the local ROM-row index.
    # Using ``k`` here silently paired almost every L20 sample with a missing
    # latent and, for legacy maps, could pair it with the wrong transition.
    z = np.asarray([latent_by_tick.get(int(tick), np.full(2, np.nan)) for tick in ticks], dtype=np.float64)
    if z.shape != (k.size, 2) or not np.isfinite(z).all():
        missing = int(np.sum(~np.isfinite(z).all(axis=1)))
        raise ValueError(f"run {run_id}: latent map missing {missing} samples at L{lag}")
    # Match the deployment controller's causal rolling latent filter.  The
    # first valid AAM row has only one prediction in its history; no future
    # rows are used.  True-latent maps are constant, so this is a no-op for
    # the privileged adapter variant.
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be positive")
    if smoothing_window > 1:
        smoothed = np.empty_like(z)
        for index in range(z.shape[0]):
            smoothed[index] = z[max(0, index - smoothing_window + 1): index + 1].mean(axis=0)
        z = smoothed
    features = np.concatenate((x, v, dp, force, target_x - x, target_v - v, u_base, z), axis=1)
    target = np.concatenate((next_x - x, next_v - v), axis=1)
    if features.shape[1] != 30 or target.shape[1] != 8:
        raise AssertionError((features.shape, target.shape))
    return features, target, u_base


def fd_delta_with_grad(fd: CylinderFDRuntime, features: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    x = features[:, 0:4]
    v = features[:, 4:8]
    dp = features[:, 8:12]
    force = features[:, 12:16]
    row = torch.cat((u, x, v, dp, force), dim=1)
    mean = fd.normalizers["x_mean"].to(device=row.device, dtype=row.dtype).reshape(1, -1)
    std = fd.normalizers["x_std"].to(device=row.device, dtype=row.dtype).reshape(1, -1)
    y_mean = fd.normalizers["y_mean"].to(device=row.device, dtype=row.dtype).reshape(1, -1)
    y_std = fd.normalizers["y_std"].to(device=row.device, dtype=row.dtype).reshape(1, -1)
    y = fd.model((row - mean) / torch.clamp(std, min=1.0e-8)) * y_std + y_mean
    return y[:, 0:8]


def evaluate_loss(
    model: ResidualAdapter,
    features: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    fd: CylinderFDRuntime,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    target_std: torch.Tensor,
    command_limits: torch.Tensor,
    residual_bound: torch.Tensor,
    batch_size: int,
) -> tuple[float, float, float]:
    model.eval()
    total = total_state = total_resid = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            xb = features[sl]
            tb = target[sl]
            ub = base[sl]
            residual = residual_bound * torch.tanh(model((xb - x_mean) / x_std))
            u = torch.clamp(ub + residual, -command_limits, command_limits)
            pred = fd_delta_with_grad(fd, xb, u)
            state_loss = torch.mean(((pred - tb) / target_std) ** 2)
            resid_loss = torch.mean((residual / residual_bound) ** 2)
            n = xb.shape[0]
            total_state += float(state_loss) * n
            total_resid += float(resid_loss) * n
            count += n
    state = total_state / count
    resid = total_resid / count
    return state + 0.01 * resid, state, resid


def train_one(
    name: str,
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_base: np.ndarray,
    val_features: np.ndarray,
    val_target: np.ndarray,
    val_base: np.ndarray,
    *,
    fd: CylinderFDRuntime,
    device: torch.device,
    out: Path,
    hidden: int,
    depth: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    residual_bound: float,
    seed: int,
    latent_lag: int,
    smoothing_window: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(train_target, dtype=torch.float32, device=device)
    train_u = torch.as_tensor(train_base, dtype=torch.float32, device=device)
    val_x = torch.as_tensor(val_features, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(val_target, dtype=torch.float32, device=device)
    val_u = torch.as_tensor(val_base, dtype=torch.float32, device=device)
    x_mean = train_x.mean(dim=0, keepdim=True)
    x_std = train_x.std(dim=0, keepdim=True)
    # dP/force are held at the legacy ID training means.  Those means are
    # large in absolute units, so float32 quantization can create apparent
    # variation in an otherwise constant feature.  A relative floor keeps
    # such channels numerically quiet instead of feeding 1e4-scale inputs to
    # the residual MLP.
    scale_floor = torch.maximum(x_mean.abs() * 1.0e-3,
                                torch.full_like(x_mean, 1.0e-3))
    x_std = torch.maximum(x_std, scale_floor)
    target_std = train_y.std(dim=0, keepdim=True).clamp_min(1.0e-5)
    model = ResidualAdapter(30, hidden, depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    limits = fd.normalizers["x_mean"].new_tensor([0.95, 0.95, 0.8, 0.8]).to(device).reshape(1, 4)
    bound = torch.full((1, 4), residual_bound, dtype=torch.float32, device=device)
    best = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    checkpoint = out / f"{name}_adapter.pt"
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(train_x.shape[0], device=device)
        running = 0.0
        for start in range(0, order.numel(), batch_size):
            idx = order[start:start + batch_size]
            xb, yb, ub = train_x[idx], train_y[idx], train_u[idx]
            residual = bound * torch.tanh(model((xb - x_mean) / x_std))
            u = torch.clamp(ub + residual, -limits, limits)
            pred = fd_delta_with_grad(fd, xb, u)
            state_loss = torch.mean(((pred - yb) / target_std) ** 2)
            resid_loss = torch.mean((residual / bound) ** 2)
            loss = state_loss + 0.01 * resid_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * idx.numel()
        train_loss = running / train_x.shape[0]
        val_loss, val_state, val_resid = evaluate_loss(
            model, val_x, val_y, val_u, fd, x_mean, x_std, target_std,
            limits, bound, batch_size,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_state_loss": val_state, "val_residual_penalty": val_resid})
        print(f"[{name}] epoch={epoch:03d} train={train_loss:.6g} val={val_loss:.6g} "
              f"state={val_state:.6g} resid={val_resid:.6g}", flush=True)
        if val_loss < best - 1.0e-7:
            best = val_loss
            best_epoch = epoch
            stale = 0
            torch.save({
                "format": "stage29_latent_conditioned_residual_adapter_v1",
                "adapter_name": name,
                "model_state": model.state_dict(),
                "model_config": {"in_dim": 30, "hidden": hidden, "depth": depth, "out_dim": 4},
                "feature_mean": x_mean.detach().cpu(),
                "feature_std": x_std.detach().cpu(),
                "target_std": target_std.detach().cpu(),
                "residual_bound": bound.detach().cpu(),
                "command_limits": limits.detach().cpu(),
                "frozen_id_checkpoint": str(getattr(fd, "stage29_id_checkpoint", "")),
                "frozen_fd_checkpoint": str(fd.checkpoint_path),
                "latent_source": name,
                "latent_lag_ticks": int(latent_lag),
                "smoothing_window_predictions": int(smoothing_window),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                print(f"[{name}] early stop at epoch {epoch}; best={best_epoch}", flush=True)
                break
    result = {"name": name, "checkpoint": str(checkpoint), "best_epoch": best_epoch,
              "best_val_loss": best, "history": history}
    (out / f"{name}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(AAM_ROOT / "datasets/master_v3_stage22_wide_frozen"))
    p.add_argument("--aam-root", default=None)
    p.add_argument("--zhat", required=True, help="all-run causal rolling AAM JSON")
    p.add_argument("--lag", type=int, default=20)
    p.add_argument("--smoothing-window", type=int, default=5)
    p.add_argument("--models-root", default=str(TOOL_ROOT / "id_fd_model/vendor_models"))
    p.add_argument("--id-checkpoint", default=str(TOOL_ROOT / "id_fd_model/best_id_cylinder_control.pt"))
    p.add_argument("--fd-checkpoint", default=str(TOOL_ROOT / "id_fd_model/best_fd_cylinder_strict.pt"))
    p.add_argument("--out", default=str(TOOL_ROOT / "outputs/stage29_latent_adapter"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--residual-bound", type=float, default=0.15)
    p.add_argument("--variants", nargs="+", choices=("true", "zhat"),
                   default=("true", "zhat"),
                   help="adapter variants to train; use only zhat for a focused deployment retrain")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=20260809)
    args = p.parse_args()
    if args.smoothing_window < 1:
        raise SystemExit("--smoothing-window must be positive")
    root = Path(args.root).expanduser().resolve()
    aam_root = Path(args.aam_root or root / "aam" / "dataset").expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    info = load_runs_info(root)
    id_runtime = CylinderIDRuntime(models_root=args.models_root, checkpoint=args.id_checkpoint,
                                   num_envs=1, device=device)
    fd = CylinderFDRuntime(models_root=args.models_root, checkpoint=args.fd_checkpoint, device=device)
    id_runtime.model.requires_grad_(False)
    fd.model.requires_grad_(False)
    fd.stage29_id_checkpoint = str(Path(args.id_checkpoint).resolve())
    dp_mean = runtime_mean(id_runtime, [f"dP_{i}" for i in range(1, 5)])
    force_mean = runtime_mean(id_runtime, [f"Force_{i}" for i in range(1, 5)])
    zhat = load_latent_map(
        Path(args.zhat).expanduser().resolve(), expected_lag=args.lag
    )
    splits = {"train": set(), "val": set(), "test": set()}
    # The split manifest is keyed by zero-padded run id in load_runs_info().
    for rid, row in info.items():
        splits[str(row["frozen_split"])].add(int(rid))
    latent_maps: dict[str, dict[int, dict[int, np.ndarray]]] = {"zhat": zhat, "true": {}}
    all_data: dict[str, dict[str, list[np.ndarray]]] = {
        "true": {"train_x": [], "train_y": [], "train_u": [], "val_x": [], "val_y": [], "val_u": []},
        "zhat": {"train_x": [], "train_y": [], "train_u": [], "val_x": [], "val_y": [], "val_u": []},
    }
    raw_runs = []
    for run_id, _ in iter_raw_runs(root):
        run_id = int(run_id)
        split = str(info[f"{run_id:05d}"]["frozen_split"])
        if split not in ("train", "val"):
            continue
        loaded = read_raw_run(root, run_id, verify=True)
        raw_runs.append(run_id)
        true_map = load_true_latent(aam_root, run_id, args.lag)
        for name, latent_map in (("true", true_map), ("zhat", zhat.get(run_id, {}))):
            fx, y, ub = make_run_samples(
                loaded["arrays"], id_runtime=id_runtime, dp_mean=dp_mean,
                force_mean=force_mean, latent_by_tick=latent_map,
                run_id=run_id, lag=args.lag,
                smoothing_window=args.smoothing_window,
            )
            prefix = "train" if split == "train" else "val"
            all_data[name][f"{prefix}_x"].append(fx.astype(np.float32))
            all_data[name][f"{prefix}_y"].append(y.astype(np.float32))
            all_data[name][f"{prefix}_u"].append(ub.astype(np.float32))
    if not raw_runs:
        raise SystemExit("no train/validation runs found")
    results = {}
    for offset, name in enumerate(args.variants):
        d = all_data[name]
        results[name] = train_one(
            name, np.concatenate(d["train_x"]), np.concatenate(d["train_y"]), np.concatenate(d["train_u"]),
            np.concatenate(d["val_x"]), np.concatenate(d["val_y"]), np.concatenate(d["val_u"]),
            fd=fd, device=device, out=out, hidden=args.hidden, depth=args.depth,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
            residual_bound=args.residual_bound, seed=args.seed + offset,
            latent_lag=args.lag,
            smoothing_window=args.smoothing_window,
        )
    manifest = {
        "format": "stage29_latent_adapter_training_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "aam_root": str(aam_root), "lag": args.lag,
        "train_runs": sorted(splits["train"]), "validation_runs": sorted(splits["val"]),
        "held_out_test_unused": sorted(splits["test"]), "n_train_samples": len(np.concatenate(all_data["true"]["train_x"])),
        "n_validation_samples": len(np.concatenate(all_data["true"]["val_x"])),
        "frozen_id_checkpoint": str(Path(args.id_checkpoint).resolve()),
        "frozen_fd_checkpoint": str(Path(args.fd_checkpoint).resolve()),
        "variants": list(args.variants),
        "residual_bound": args.residual_bound, "results": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Stage 29 adapter training complete -> {out}", flush=True)


if __name__ == "__main__":
    main()
