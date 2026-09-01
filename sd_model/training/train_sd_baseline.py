#!/usr/bin/env python
"""Stage 19: train the no-latent SD baseline ``SD_0(s_t, u_t) -> Delta s_{t+1}``.

The baseline answers the Stage 19 question: can the current observable
hydraulic state and command explain the changing deployment dynamics WITHOUT
an explicit domain representation? It therefore consumes ONLY observable
inputs -- the 12-D hydraulic state ``s_t^H``, the 8-D ROM command ``u_t``,
and (optional) the 10-D joint context ``q_t, qdot_t`` -- never the privileged
factor vector ``z_i``.

Input modes (all exclude ``z``):

    su       SD_0([s_t^H; u_t])                    20-D instantaneous (mandatory)
    su_delta SD_0([s_t^H; u_t; s_t^H - s_{t-1}^H]) 32-D instantaneous + previous
                 state change (cheap substitute for short history; drops the
                 first tick of each run)
    squ      SD_0([s_t^H; q_t, qdot_t; u_t])       30-D instantaneous
    hist3    SD_0([history L=3 of s^H;u; s_t^H; u_t])   60-D causal history
    hist6    SD_0([history L=6 of s^H;u; s_t^H; u_t])  120-D causal history

Target: the 12-D measured ``Delta s_{t+1}^H = s_{t+1}^H - s_t^H`` over one
aligned 50 ms ROM transition (Stage 17/18 exact rows, no interpolation).

The trainer reads the SD dataset built by ``tools/build_sd_dataset.py`` from a
frozen master dataset (Stage 16): ``<root>/sd/dataset/<run_id>/`` +
``<root>/splits/{train,val,test}.json``. Samples are grouped by run so the
train/val/test split is run-level (no configuration leak); the val split may
be empty for single-key development subsets. Inputs and targets are
z-scored with training-set statistics only. Checkpoints and metrics are
written under ``--out``; a progress log is written to ``--log``.

Usage:
    python train_sd_baseline.py --root datasets/master_v1_frozen \
        --input su --epochs 80 --out outputs/sd_baseline
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
from torch import nn


AAM_ROOT = Path(__file__).resolve().parents[2] / "AAM_module"  # SD tools live in sd_model/; aam_data and datasets stay under AAM_module
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_split, splits_dir  # noqa: E402


HYD_STATE_DIM = 12
MECH_CONTEXT_DIM = 10
CMD_DIM = 8
TARGET_DIM = 12
CHANNEL_GROUPS = {
    "extension": (0, 4),
    "velocity": (4, 8),
    "dP": (8, 12),
}


def input_dim_for(mode: str, state_keep: list[int] | None = None) -> int:
    state_dim = len(state_keep) if state_keep is not None else HYD_STATE_DIM
    if mode == "su":
        return state_dim + CMD_DIM
    if mode == "su_delta":
        return state_dim + CMD_DIM + state_dim
    if mode == "squ":
        return state_dim + MECH_CONTEXT_DIM + CMD_DIM
    if mode == "hist3":
        return 3 * (HYD_STATE_DIM + CMD_DIM)
    if mode == "hist6":
        return 6 * (HYD_STATE_DIM + CMD_DIM)
    raise ValueError(f"unknown input mode {mode!r}")


def load_run_samples(
    run_dir: Path, mode: str, state_keep: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (X, y) for one run's SD samples; None if the run has no data."""
    samples_path = run_dir / "samples.npz"
    if not samples_path.is_file():
        return None
    with np.load(samples_path, allow_pickle=False) as z:
        s_h = z["s_h"]
        u = z["u"]
        target = z["target"]
        q = z["q"]
    if state_keep is not None:
        s_h = s_h[:, state_keep]
    if mode == "su_delta":
        # Previous-state-change feature: s_t^H - s_{t-1}^H (drops tick 0).
        prev = s_h[1:] - s_h[:-1]
        return (
            np.concatenate([s_h[1:], u[1:], prev], axis=-1).astype(np.float32),
            target[1:].astype(np.float32),
        )
    if mode == "su":
        return np.concatenate([s_h, u], axis=-1).astype(np.float32), target.astype(np.float32)
    if mode == "squ":
        return (
            np.concatenate([s_h, q, u], axis=-1).astype(np.float32),
            target.astype(np.float32),
        )
    lag = 3 if mode == "hist3" else 6
    hist_path = run_dir / f"history_L{lag}.npz"
    if not hist_path.is_file():
        return None
    with np.load(hist_path, allow_pickle=False) as z:
        # history_input is already compacted to the valid rows (n_valid x L*20);
        # valid_mask marks which of the run's sample rows those are.
        hist = z["history_input"]
        valid = z["valid_mask"]
    if not bool(valid.any()):
        return None
    idx = np.flatnonzero(valid)
    # The first lag-1 ticks of the run are dropped (never padded); the target
    # rows are the sample rows at the same indices as the valid history rows.
    return hist.astype(np.float32), target[idx].astype(np.float32)


def zscore_fit(X: np.ndarray, y: np.ndarray):
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0).clip(min=1e-6)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0).clip(min=1e-6)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def zscore_apply(X: np.ndarray, stats) -> np.ndarray:
    return (X - stats["x_mean"]) / stats["x_std"]


def unnormalize(y_norm: np.ndarray, stats) -> np.ndarray:
    return y_norm * stats["y_std"] + stats["y_mean"]


class SD0MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = TARGET_DIM, hidden: int = 256,
                 depth: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def group_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (lo, hi) in CHANNEL_GROUPS.items():
        out[name] = float(np.sqrt(np.mean((y_true[:, lo:hi] - y_pred[:, lo:hi]) ** 2)))
    out["all"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(AAM_ROOT / "datasets/master_v1_frozen"))
    p.add_argument(
        "--input",
        choices=("su", "su_delta", "squ", "hist3", "hist6"),
        default="su",
    )
    p.add_argument(
        "--obs-drop",
        default="",
        help="comma-separated 0-based state channels to HIDE from the input "
        "(partial-observation no-latent baseline; matches train_sd_z.py "
        "--obs-drop). Only valid for su/su_delta/squ; history modes are "
        "built from precomputed full-channel tensors.",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(AAM_ROOT / "outputs/sd_baseline"))
    p.add_argument("--log", default=None)
    p.add_argument("--eval-every", type=int, default=1)
    args = p.parse_args()

    obs_drop = sorted({int(x) for x in args.obs_drop.split(",") if x.strip()})
    if obs_drop and args.input in ("hist3", "hist6"):
        raise SystemExit("--obs-drop is not supported for history modes")
    if any(i < 0 or i >= HYD_STATE_DIM for i in obs_drop):
        raise SystemExit(f"--obs-drop indices must be in 0..{HYD_STATE_DIM-1}: {obs_drop}")
    state_keep = [i for i in range(HYD_STATE_DIM) if i not in set(obs_drop)]
    state_keep = state_keep if obs_drop else None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"device: {device} | cuda available: {torch.cuda.is_available()}", flush=True)

    root = Path(args.root).expanduser().resolve()
    sd_root = root / "sd" / "dataset"
    if not (root / "splits" / "train.json").is_file():
        raise SystemExit(f"not a split frozen dataset: {root} (missing splits/train.json)")
    if not sd_root.is_dir():
        raise SystemExit(f"SD dataset not built: {sd_root} (run tools/build_sd_dataset.py)")

    split_ids = {
        name: load_split(root, name)
        for name in ("train", "val", "test")
    }
    all_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        name: {} for name in split_ids
    }
    for name, run_ids in split_ids.items():
        for run_id in run_ids:
            data = load_run_samples(sd_root / f"{int(run_id):05d}", args.input, state_keep)
            if data is not None:
                all_data[name][int(run_id)] = data

    def concat(group: str) -> tuple[np.ndarray, np.ndarray]:
        xs = [all_data[group][rid][0] for rid in sorted(all_data[group])]
        ys = [all_data[group][rid][1] for rid in sorted(all_data[group])]
        if not xs:
            return np.zeros((0, input_dim_for(args.input, state_keep)), np.float32), np.zeros((0, TARGET_DIM), np.float32)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = concat("train")
    X_val, y_val = concat("val")
    X_test, y_test = concat("test")
    print(
        f"input={args.input} dim={input_dim_for(args.input, state_keep)} "
        f"samples train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}",
        flush=True,
    )
    if X_train.shape[0] < 64:
        raise SystemExit("train set too small; collect more runs first")

    stats = zscore_fit(X_train, y_train)
    Xn = {name: zscore_apply(X, stats) for name, X in
          (("train", X_train), ("val", X_val), ("test", X_test))}
    yn = {name: (y - stats["y_mean"]) / stats["y_std"] for name, y in
          (("train", y_train), ("val", y_val), ("test", y_test))}

    model = SD0MLP(
        input_dim_for(args.input, state_keep), hidden=args.hidden, depth=args.depth, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    loss_fn = nn.MSELoss()

    def numpy_batches(X: np.ndarray, y: np.ndarray, shuffle: bool):
        n = X.shape[0]
        idx = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, args.batch_size):
            b = idx[start : start + args.batch_size]
            yield torch.as_tensor(X[b], device=device), torch.as_tensor(y[b], device=device)

    @torch.no_grad()
    def evaluate(X: np.ndarray, y_raw: np.ndarray) -> dict:
        model.eval()
        preds = []
        for xb, _ in numpy_batches(X, y_raw, shuffle=False):
            preds.append(model(xb).detach().cpu().numpy())
        if not preds:
            return {"loss": float("nan"), "rmse": {"all": float("nan")}}
        y_pred_norm = np.concatenate(preds, axis=0)
        y_pred = unnormalize(y_pred_norm, stats)
        rmse = group_rmse(y_raw, y_pred)
        return {
            "loss": float(np.mean((y_pred_norm - (y_raw - stats["y_mean"]) / stats["y_std"]) ** 2)),
            "rmse": rmse,
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else out / "training.log"
    log_fh = log_path.open("a")

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    history: list[dict] = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    train_t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, yb in numpy_batches(Xn["train"], yn["train"], shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * xb.shape[0]
            n_batches += xb.shape[0]
        scheduler.step()
        train_loss = train_loss_sum / max(n_batches, 1)

        row: dict = {"epoch": epoch, "train_loss": train_loss}
        if epoch % args.eval_every == 0:
            val_metrics = evaluate(Xn["val"], y_val) if X_val.shape[0] else None
            if val_metrics is not None:
                row["val_loss"] = val_metrics["loss"]
                row["val_rmse"] = val_metrics["rmse"]
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                    torch.save(best_state, out / "best.pt")
                else:
                    epochs_no_improve += 1
            test_metrics = evaluate(Xn["test"], y_test) if X_test.shape[0] else None
            if test_metrics is not None:
                row["test_loss"] = test_metrics["loss"]
                row["test_rmse"] = test_metrics["rmse"]
        history.append(row)
        if epoch % max(args.eval_every, 5) == 0 or epoch == 1 or epoch == args.epochs:
            detail = " | ".join(
                f"val_{k}={v:.3g}" for k, v in row.get("val_rmse", {}).items()
            )
            log(
                f"epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.6g} "
                f"{detail} elapsed={time.time()-train_t0:.0f}s"
            )
        if args.patience and epochs_no_improve >= args.patience:
            log(f"early stopping at epoch {epoch} (no val improvement for {args.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"best epoch {best_epoch} val_loss={best_val_loss:.6g}")
    else:
        torch.save(model.state_dict(), out / "best.pt")

    report = {
        "input_mode": args.input,
        "input_dim": input_dim_for(args.input),
        "device": str(device),
        "seed": args.seed,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if X_val.shape[0] else None,
        "samples": {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]), "test": int(X_test.shape[0])},
        "stats": {
            "y_std": [round(float(v), 6) for v in stats["y_std"]],
            "y_mean": [round(float(v), 6) for v in stats["y_mean"]],
            "x_std": [round(float(v), 6) for v in stats["x_std"]],
            "x_mean": [round(float(v), 6) for v in stats["x_mean"]],
        },
        "test_rmse": evaluate(Xn["test"], y_test)["rmse"] if X_test.shape[0] else None,
        "val_rmse": evaluate(Xn["val"], y_val)["rmse"] if X_val.shape[0] else None,
        "train_rmse": evaluate(Xn["train"], y_train)["rmse"],
        "history": history,
        "frozen_root": str(root),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {out / 'metrics.json'}")
    log_fh.close()


if __name__ == "__main__":
    main()
