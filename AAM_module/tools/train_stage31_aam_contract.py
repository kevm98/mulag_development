#!/usr/bin/env python3
"""Train the Stage 31 causal AAM under a selectable observation contract.

Identical to ``train_stage31_aam.py`` except for ``--drop-groups``, which
removes named feature groups from every history window before training. It
exists to score the deployable no-pressure contract against the
pressure-inclusive one under a paired design: same runs, same split, same
seed, same objective, same schedule, differing only in the columns supplied.

Dropping (rather than zeroing) reduces the input dimension, so the estimator
is sized for the contract a machine without pressure transducers could
actually deliver.

NOTE: this trains the OFFLINE AAM, which regresses z_AAM against the
privileged target z_SD on the frozen collection. It is evidence that plant
identity is recoverable from causal history. It is NOT the encoder deployed
in the AF stack: that one (phi) has no target and is optimised jointly with
the residual head under RL -- see PPO_AAM_FD/scripts/train_residual_adapter.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_split  # noqa: E402
from aam_data.rom_stage31 import AAM_FEATURE_DIM, AAM_FEATURE_GROUPS  # noqa: E402


def kept_feature_columns(drop_groups: tuple[str, ...]) -> np.ndarray:
    """Column indices of one feature row that survive ``drop_groups``."""
    unknown = [name for name in drop_groups if name not in AAM_FEATURE_GROUPS]
    if unknown:
        raise SystemExit(
            f"unknown feature group(s) {unknown}; "
            f"available: {sorted(AAM_FEATURE_GROUPS)}"
        )
    dropped: set[int] = set()
    for name in drop_groups:
        lo, hi = AAM_FEATURE_GROUPS[name]
        dropped.update(range(lo, hi))
    kept = np.array(
        [i for i in range(AAM_FEATURE_DIM) if i not in dropped], dtype=np.int64
    )
    if kept.size == 0:
        raise SystemExit("refusing to drop every feature group")
    return kept


def apply_contract(window: np.ndarray, lag: int, kept: np.ndarray) -> np.ndarray:
    """Drop feature columns from a flattened (N, lag*AAM_FEATURE_DIM) window.

    The on-disk layout is (lag, feature); reshaping, selecting columns and
    flattening preserves the per-tick ordering the CNN reconstructs.
    """
    if window.shape[1] != lag * AAM_FEATURE_DIM:
        raise SystemExit(
            f"window has {window.shape[1]} columns, expected {lag * AAM_FEATURE_DIM}"
        )
    reshaped = window.reshape(window.shape[0], lag, AAM_FEATURE_DIM)
    return reshaped[:, :, kept].reshape(window.shape[0], lag * kept.size)


class CausalConv1d(nn.Conv1d):
    """Left-padded temporal convolution for an oldest-to-newest sequence."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class CausalTemporalBlock(nn.Module):
    """Residual causal convolution with channel-wise LayerNorm."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, 3, dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x).transpose(1, 2)
        y = self.dropout(self.activation(self.norm(y))).transpose(1, 2)
        return y + self.skip(x)


class AAM(nn.Module):
    """Causal history -> frozen-SD latent estimator.

    ``architecture='cnn'`` consumes the flattened on-disk window but reshapes
    it into ``(batch, time, feature)`` internally.  The stored window format
    is most-recent-first, so the CNN reverses it before applying causal
    convolutions and reads the newest timestep only.  ``architecture='mlp'``
    remains available for loading historical checkpoints.
    """

    def __init__(self, in_dim: int, latent_dim: int, hidden: int = 256,
                 depth: int = 3, dropout: float = 0.05,
                 architecture: str = "mlp", feature_dim: int = AAM_FEATURE_DIM,
                 lag: int | None = None) -> None:
        super().__init__()
        self.architecture = str(architecture).lower()
        self.feature_dim = int(feature_dim)
        self.lag = int(lag or (in_dim // self.feature_dim))
        if self.feature_dim < 1 or self.lag < 1 or self.feature_dim * self.lag != in_dim:
            raise ValueError(
                f"AAM input contract requires in_dim=feature_dim*lag, got "
                f"in_dim={in_dim}, feature_dim={self.feature_dim}, lag={self.lag}"
            )
        if self.architecture == "mlp":
            layers: list[nn.Module] = [
                nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout)
            ]
            for _ in range(max(depth - 1, 0)):
                layers.extend((nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)))
            layers.append(nn.Linear(hidden, latent_dim))
            self.net = nn.Sequential(*layers)
        elif self.architecture == "cnn":
            channels = max(int(hidden), 16)
            blocks: list[nn.Module] = []
            for index in range(max(int(depth), 1)):
                blocks.append(
                    CausalTemporalBlock(
                        self.feature_dim if index == 0 else channels,
                        channels,
                        dilation=2 ** index,
                        dropout=dropout,
                    )
                )
            self.net = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.LayerNorm(channels), nn.SiLU(), nn.Linear(channels, latent_dim)
            )
        else:
            raise ValueError(f"unsupported AAM architecture: {architecture!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "mlp":
            return self.net(x)
        if x.ndim == 2:
            x = x.reshape(x.shape[0], self.lag, self.feature_dim)
        if x.ndim != 3 or x.shape[1:] != (self.lag, self.feature_dim):
            raise ValueError(
                f"CNN AAM expects (batch, {self.lag}, {self.feature_dim}) or "
                f"flattened ({self.lag * self.feature_dim},), got {tuple(x.shape)}"
            )
        # causal_stack stores current -> past; convolutions run oldest -> current.
        sequence = x.flip(dims=(1,)).transpose(1, 2)
        features = self.net(sequence).transpose(1, 2)
        return self.head(features[:, -1, :])


def load_windows(root: Path, run_id: int, lag: int) -> dict | None:
    path = root / f"{run_id:05d}" / f"history_L{lag}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {
            "window": data["window_input"].astype(np.float64),
            "z": data["z_target"].astype(np.float64),
            "ticks": data["window_end_tick"].astype(np.int64),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="frozen ROM collection")
    parser.add_argument("--aam-root", required=True, help="Stage 31 AAM dataset")
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument(
        "--architecture",
        choices=("mlp", "cnn"),
        default="mlp",
        help=(
            "history encoder; cnn applies causal temporal convolutions over "
            "the lag x 18 observable window"
        ),
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temporal-consistency-weight", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--drop-groups",
        default="",
        help=(
            "comma-separated feature groups to remove from every window, e.g. "
            "'cylinder_pressure'. Empty keeps the full 18-D contract."
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.root).expanduser().resolve()
    aam_root = Path(args.aam_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )

    split_runs = {split: load_split(root, split) for split in ("train", "val", "test")}
    loaded: dict[str, dict[int, dict]] = {}
    for split, ids in split_runs.items():
        loaded[split] = {}
        for run_id in ids:
            data = load_windows(aam_root, int(run_id), args.lag)
            if data is not None and data["window"].shape[0]:
                loaded[split][int(run_id)] = data
    if not loaded["train"] or not loaded["val"]:
        raise SystemExit("Stage 31 AAM requires non-empty train and validation windows")
    drop_groups = tuple(
        name.strip() for name in args.drop_groups.split(",") if name.strip()
    )
    kept = kept_feature_columns(drop_groups)
    contract_dim = int(kept.size)
    # Apply the contract once, before any statistic is computed, so that the
    # normaliser is fitted on exactly the columns the model will receive.
    for split_data in loaded.values():
        for value in split_data.values():
            value["window"] = apply_contract(value["window"], args.lag, kept)

    x_train = np.concatenate([value["window"] for value in loaded["train"].values()])
    y_train = np.concatenate([value["z"] for value in loaded["train"].values()])
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0).clip(min=1.0e-6)
    latent_dim = int(y_train.shape[1])
    expected = args.lag * contract_dim
    if x_train.shape[1] != expected:
        raise SystemExit(f"AAM window has {x_train.shape[1]} features, expected {expected}")

    def tensor_split(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.concatenate([value["window"] for value in loaded[name].values()])
        y = np.concatenate([value["z"] for value in loaded[name].values()])
        return (
            torch.as_tensor(((x - x_mean) / x_std).astype(np.float32), device=device),
            torch.as_tensor(y.astype(np.float32), device=device),
        )

    Xtr, Ytr = tensor_split("train")
    Xva, Yva = tensor_split("val")
    Xte, Yte = tensor_split("test") if loaded["test"] else (
        torch.zeros((0, expected), device=device), torch.zeros((0, latent_dim), device=device)
    )
    model = AAM(
        expected,
        latent_dim,
        args.hidden,
        args.depth,
        args.dropout,
        architecture=args.architecture,
        feature_dim=contract_dim,
        lag=args.lag,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 1.0e-2)
    loss_fn = nn.MSELoss()

    def evaluate(X: torch.Tensor, Y: torch.Tensor) -> dict[str, float]:
        model.eval()
        if X.shape[0] == 0:
            return {"loss": float("nan"), "rmse": float("nan")}
        with torch.no_grad():
            prediction = model(X)
            loss = torch.mean((prediction - Y) ** 2)
        return {"loss": float(loss.cpu()), "rmse": float(torch.sqrt(loss).cpu())}

    train_slices: dict[int, tuple[int, int]] = {}
    cursor = 0
    for run_id, value in loaded["train"].items():
        count = value["window"].shape[0]
        train_slices[run_id] = (cursor, cursor + count)
        cursor += count
    consistency_runs = [run_id for run_id, (lo, hi) in train_slices.items() if hi - lo > 1]

    def consistency_batch(batch: int) -> torch.Tensor:
        if not consistency_runs or args.temporal_consistency_weight <= 0:
            return torch.zeros((), device=device)
        # The supervised batch may be large for the 1-D CNN.  Keep the paired
        # temporal-consistency probe bounded so it cannot duplicate a full
        # activation graph and exhaust a deployment-sized GPU.
        batch = min(int(batch), 1024)
        ids = np.random.choice(consistency_runs, size=batch, replace=True)
        starts = [np.random.randint(train_slices[int(run_id)][0], train_slices[int(run_id)][1] - 1) for run_id in ids]
        index = torch.as_tensor(starts, dtype=torch.long, device=device)
        return torch.mean((model(Xtr[index]) - model(Xtr[index + 1])) ** 2)

    best = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(Xtr.shape[0], device=device)
        total = 0.0
        count = 0
        for start in range(0, Xtr.shape[0], args.batch_size):
            indices = permutation[start:start + args.batch_size]
            supervised = loss_fn(model(Xtr[indices]), Ytr[indices])
            consistency = consistency_batch(int(indices.numel()))
            loss = supervised + args.temporal_consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * indices.numel()
            count += int(indices.numel())
        scheduler.step()
        val = evaluate(Xva, Yva)
        history.append({"epoch": epoch, "train_loss": total / max(count, 1), "val_loss": val["loss"], "val_rmse": val["rmse"]})
        if val["loss"] < best:
            best = val["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), out / "aam.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:03d}/{args.epochs} train={total/max(count,1):.6f} val={val['loss']:.6f} rmse={val['rmse']:.6f} elapsed={time.time()-start_time:.0f}s", flush=True)
        if args.patience > 0 and stale >= args.patience:
            break

    model.load_state_dict(torch.load(out / "aam.pt", map_location=device, weights_only=False))
    val = evaluate(Xva, Yva)
    test = evaluate(Xte, Yte)
    report = {
        "format": "stage31_rom_aam_contract_v1",
        "root": str(root),
        "aam_root": str(aam_root),
        "feature_dim": contract_dim,
        "source_feature_dim": AAM_FEATURE_DIM,
        "dropped_groups": list(drop_groups),
        "kept_columns": [int(i) for i in kept],
        "feature_contract": (
            "joint_q + joint_qdot + dP + previous_valve" if not drop_groups
            else "joint_q + joint_qdot + previous_valve (dP removed)"
        ),
        "lag_ticks": args.lag,
        "lag_seconds": args.lag * 0.05,
        "in_dim": expected,
        "architecture": args.architecture,
        "latent_dim": latent_dim,
        "hidden": args.hidden,
        "depth": args.depth,
        "dropout": args.dropout,
        "temporal_consistency_weight": args.temporal_consistency_weight,
        "train_runs": len(loaded["train"]),
        "val_runs": len(loaded["val"]),
        "test_runs": len(loaded["test"]),
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "input_normalization": {
            "computed_on": "train_windows_only",
            "per_position_mean": [float(value) for value in x_mean],
            "per_position_std": [float(value) for value in x_std],
        },
        "val": val,
        "test": test,
        "history": history,
        "seed": args.seed,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Stage 31 AAM complete: val_rmse={val['rmse']:.6f} test_rmse={test['rmse']:.6f} -> {out}")


if __name__ == "__main__":
    main()
