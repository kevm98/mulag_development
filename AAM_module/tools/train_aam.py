#!/usr/bin/env python
"""Train the AAM latent estimator for one history length (AAM Stage 25/26).

``z_hat = AAM(H_t)`` maps a causal window of observable history to the
privileged latent. The supervised target is the stopped-gradient
``z = mu(e)`` from the frozen Stage 23 encoder, stored per window in the
Stage 24 AAM dataset.

The AAM is a small MLP over the flattened window (``lag * 30`` inputs ->
``latent_dim`` outputs). Training uses TRAIN runs only; early stopping and
latent-RMSE reporting use the VALIDATION runs. The reserved test split is
never used for early stopping or selection (Stage 25 rule).

After training, the best checkpoint predicts ``z_hat`` per window for every
held-out run and the per-run latent is the mean over the run's valid windows
(the latent is stationary per run). The map is written as ``zhat.json``
``{run_id: [z0, z1]}`` for the Stage 25 downstream ``SD_z(s, u, z_hat)``
scoring.

Usage:
    python train_aam.py --root datasets/master_v3_stage22_wide_frozen \\
        --aam-root <root>/aam/dataset --lag 5 --out outputs/aam_stage25/L5

NOTE: this trains the OFFLINE AAM, which regresses z_AAM against the
privileged target z_SD on the frozen collection. It is evidence that plant
identity is recoverable from causal history. It is NOT the encoder deployed
in the AF stack: that one (phi) has no target and is optimised jointly with
the residual head under RL, in a mechanism since withdrawn.
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


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import AAM_FEATURE_DIM, load_split  # noqa: E402


class AAM(nn.Module):
    """MLP z_hat = AAM(window); window is lag * AAM_FEATURE_DIM flattened."""

    def __init__(self, in_dim: int, latent_dim: int, hidden: int = 256,
                 depth: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_windows(aam_root: Path, run_id: int, lag: int) -> dict | None:
    path = aam_root / f"{int(run_id):05d}" / f"history_L{lag}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        tick_key = "window_end_tick" if "window_end_tick" in z.files else "window_start_tick"
        return {
            "window": z["window_input"].astype(np.float64),
            "z": z["z_target"].astype(np.float64),
            "window_end_tick": z[tick_key].astype(np.int64),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=str(AAM_ROOT / "datasets/master_v3_stage22_wide_frozen"),
        help="frozen master dataset root (for splits)",
    )
    p.add_argument(
        "--aam-root",
        default=None,
        help="AAM dataset dir (default: <root>/aam/dataset)",
    )
    p.add_argument("--lag", type=int, required=True,
                   help="causal history length in 50 ms ROM ticks")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument(
        "--temporal-consistency-weight",
        type=float,
        default=0.0,
        help="weight for MSE(AAM(H_t), AAM(H_{t+1})) on adjacent training "
        "windows from the same run; 0 preserves the prior objective",
    )
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--log", default=None)
    args = p.parse_args()
    if args.temporal_consistency_weight < 0:
        raise SystemExit("--temporal-consistency-weight must be non-negative")

    root = Path(args.root).expanduser().resolve()
    aam_root = Path(args.aam_root or str(root / "aam" / "dataset")).expanduser().resolve()
    out = Path(args.out or str(AAM_ROOT / f"outputs/aam_stage25/L{args.lag}")).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else out / "training.log"
    log_fh = log_path.open("a")

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    train_ids = load_split(root, "train")
    val_ids = load_split(root, "val")
    test_ids = load_split(root, "test")

    def load_split_windows(ids: list[int]) -> dict[int, dict]:
        runs = {}
        for rid in ids:
            w = load_windows(aam_root, int(rid), args.lag)
            if w is not None:
                runs[int(rid)] = w
        return runs

    tr = load_split_windows(train_ids)
    va = load_split_windows(val_ids)
    te = load_split_windows(test_ids)
    if not tr:
        raise SystemExit(f"no training windows for lag {args.lag} in {aam_root}")

    Xtr = np.concatenate([r["window"] for r in tr.values()], axis=0)
    Ytr = np.concatenate([r["z"] for r in tr.values()], axis=0)
    x_mean = Xtr.mean(0)
    x_std = Xtr.std(0).clip(min=1e-6)
    latent_dim = Ytr.shape[1]
    in_dim = Xtr.shape[1]
    assert in_dim == args.lag * AAM_FEATURE_DIM, (in_dim, args.lag, AAM_FEATURE_DIM)

    def prep(runs: dict[int, dict]) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        X = np.concatenate([r["window"] for r in runs.values()], axis=0)
        Y = np.concatenate([r["z"] for r in runs.values()], axis=0)
        Xn = ((X - x_mean) / x_std).astype(np.float32)
        run_ids = np.concatenate(
            [[rid] * r["window"].shape[0] for rid, r in runs.items()]
        )
        return (
            torch.as_tensor(Xn, device=device),
            torch.as_tensor(Y, dtype=torch.float32, device=device),
            {rid: torch.as_tensor(r["window"], dtype=torch.float32, device=device)
             for rid, r in runs.items()},
        )

    Xtr_t, Ytr_t, _ = prep(tr)
    Xva_t, Yva_t, va_windows = prep(va)
    _, _, te_windows = prep(te)

    # Xtr_t is concatenated in the same run order as ``tr``. Keep only slice
    # metadata so adjacent pairs can be sampled without materializing a second
    # copy of the full (large) training-window tensor.
    train_slices: dict[int, tuple[int, int]] = {}
    cursor = 0
    for rid, run in tr.items():
        count = int(run["window"].shape[0])
        train_slices[int(rid)] = (cursor, cursor + count)
        cursor += count
    consistency_run_ids = [rid for rid, (lo, hi) in train_slices.items() if hi - lo >= 2]
    if args.temporal_consistency_weight > 0 and not consistency_run_ids:
        raise SystemExit("no training run has adjacent AAM windows")

    model = AAM(in_dim, latent_dim, hidden=args.hidden, depth=args.depth,
                dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    loss_fn = nn.MSELoss()

    def sample_adjacent_training_windows(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample normalized H_t/H_{t+1} pairs within training runs only."""
        rids = np.random.choice(consistency_run_ids, size=batch_size, replace=True)
        starts = np.asarray(
            [np.random.randint(train_slices[int(rid)][0], train_slices[int(rid)][1] - 1)
             for rid in rids],
            dtype=np.int64,
        )
        idx0 = torch.as_tensor(starts, dtype=torch.long, device=device)
        return Xtr_t[idx0], Xtr_t[idx0 + 1]

    @torch.no_grad()
    def evaluate(X, Y) -> dict:
        model.eval()
        preds = []
        for start in range(0, X.shape[0], args.batch_size):
            preds.append(model(X[start : start + args.batch_size]).detach().cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        y = Y.detach().cpu().numpy()
        rmse_per = np.sqrt(((pred - y) ** 2).mean(0))
        return {
            "loss": float(((pred - y) ** 2).mean()),
            "rmse_per_dim": [float(v) for v in rmse_per],
            "rmse_all": float(np.sqrt(((pred - y) ** 2).mean())),
        }

    best_val, best_epoch, no_improve = float("inf"), -1, 0
    history, t0 = [], time.time()
    n = Xtr_t.shape[0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot, tot_supervised, tot_consistency, nb = 0.0, 0.0, 0.0, 0
        for start in range(0, n, args.batch_size):
            b = perm[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            supervised = loss_fn(model(Xtr_t[b]), Ytr_t[b])
            consistency = torch.zeros((), device=device)
            if args.temporal_consistency_weight > 0:
                x0, x1 = sample_adjacent_training_windows(int(b.shape[0]))
                consistency = loss_fn(model(x0), model(x1))
            loss = supervised + args.temporal_consistency_weight * consistency
            loss.backward()
            optimizer.step()
            tot += float(loss) * b.shape[0]
            tot_supervised += float(supervised) * b.shape[0]
            tot_consistency += float(consistency) * b.shape[0]
            nb += b.shape[0]
        scheduler.step()
        v = evaluate(Xva_t, Yva_t)
        history.append({
            "epoch": epoch,
            "train_loss": tot / max(nb, 1),
            "train_supervised_loss": tot_supervised / max(nb, 1),
            "train_consistency_loss": tot_consistency / max(nb, 1),
            "val_loss": v["loss"],
        })
        if v["loss"] < best_val:
            best_val, best_epoch, no_improve = v["loss"], epoch, 0
            torch.save(model.state_dict(), out / "aam.pt")
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == 1:
            log(f"epoch {epoch:03d}/{args.epochs} train={tot/max(nb,1):.6f} "
                f"sup={tot_supervised/max(nb,1):.6f} "
                f"cons={tot_consistency/max(nb,1):.6f} "
                f"val={v['loss']:.6f} val_rmse={v['rmse_all']:.5f} "
                f"elapsed={time.time()-t0:.0f}s")
        if args.patience and no_improve >= args.patience:
            log(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(out / "aam.pt", map_location=device))
    val = evaluate(Xva_t, Yva_t)
    test = evaluate(
        torch.as_tensor(((np.concatenate([r["window"] for r in te.values()], 0) - x_mean) / x_std).astype(np.float32), device=device) if te else torch.zeros(0, in_dim, device=device),
        torch.as_tensor(np.concatenate([r["z"] for r in te.values()], 0), dtype=torch.float32, device=device) if te else torch.zeros(0, latent_dim, device=device),
    )

    # Per-run z_hat = mean over the run's valid windows (stationary latent).
    @torch.no_grad()
    def per_run_zhat(windows: dict[int, torch.Tensor]) -> dict[int, dict]:
        model.eval()
        x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
        x_std_t = torch.as_tensor(x_std, dtype=torch.float32, device=device)
        out_map = {}
        for rid, W in sorted(windows.items()):
            Wn = (W - x_mean_t) / x_std_t
            pred = []
            for start in range(0, Wn.shape[0], 8192):
                pred.append(model(Wn[start : start + 8192]).detach().cpu().numpy())
            pred = np.concatenate(pred, axis=0)
            out_map[int(rid)] = {
                "mean": [float(v) for v in pred.mean(0)],
                "std": [float(v) for v in pred.std(0)],
                "n_windows": int(pred.shape[0]),
            }
        return out_map

    held_zhat = {**per_run_zhat(va_windows), **per_run_zhat(te_windows)}
    (out / "zhat.json").write_text(
        json.dumps(
            {rid: v["mean"] for rid, v in sorted(held_zhat.items())}, indent=2
        )
        + "\n"
    )
    (out / "zhat_stats.json").write_text(
        json.dumps(held_zhat, indent=2) + "\n"
    )

    report = {
        "stage": "25",
        "lag_ticks": args.lag,
        "lag_seconds": args.lag * 0.05,
        "in_dim": in_dim,
        "latent_dim": latent_dim,
        "architecture": f"MLP {in_dim}->{args.hidden}x{args.depth}->{latent_dim}",
        "hidden": args.hidden,
        "depth": args.depth,
        "dropout": args.dropout,
        "temporal_consistency_weight": args.temporal_consistency_weight,
        "train_runs": len(tr),
        "val_runs": len(va),
        "test_runs": len(te),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "input_normalization": {
            "computed_on": "train_windows_only",
            "per_position_mean": [float(v) for v in x_mean],
            "per_position_std": [float(v) for v in x_std],
        },
        "val": val,
        "test": test,
        "history": history,
        "seed": args.seed,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    log(
        f"lag={args.lag} best_epoch={best_epoch} "
        f"val_rmse={val['rmse_all']:.5f} test_rmse={test['rmse_all']:.5f}"
    )
    log_fh.close()


if __name__ == "__main__":
    main()
