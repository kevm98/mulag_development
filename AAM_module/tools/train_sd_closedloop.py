#!/usr/bin/env python
"""Closed-loop-aware training for the Stage 19 SD baseline (deployment fix).

Motivation (measured on the one-step baseline): one-step RMSE is small but
multi-step rollouts drift -- the best one-step model diverges past the
persistence baseline at 5 s. This is the classic *exposure bias* / compounding
error problem: the model is trained on recorded states (teacher forcing) but
deployed with its own predictions fed back, so the input distribution shifts
and errors accumulate.

Literature-grounded mitigations implemented here:

- **Scheduled sampling** (Bengio et al., NeurIPS 2015, "Scheduled Sampling
  for Sequence Prediction with Recurrent Neural Networks"): anneal the
  teacher-forcing ratio from 1.0 toward a small floor, so the model
  progressively trains on its own predictions.
- **DAgger principle** (Ross, Gordon, Bagnell, AISTATS 2011, "A Reduction of
  Imitation Learning and Structured Prediction to No-Regret Online Learning"):
  the training distribution should include states the learner itself visits;
  the unrolled loss does exactly that for short horizons.
- **Input noise injection**: perturb the state input by noise scaled to the
  per-channel state std, the standard regularizer for learned dynamics models
  used in closed loop (the model must be robust to its own input error).

Training objective: a short unrolled loss over ``--unroll`` 50 ms ticks with
annealed teacher forcing; validation and early stopping use the *free-running*
unrolled loss at ``--val-unroll`` ticks, i.e. the deployment criterion itself,
not the one-step loss.

Usage:
    python train_sd_closedloop.py --root datasets/master_v1_frozen --mode hist3 \
        --unroll 10 --noise-std 0.05 --epochs 200 --out outputs/sd_cl/checkpoints
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


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_split  # noqa: E402
from tools.train_sd_baseline import (  # noqa: E402
    CHANNEL_GROUPS,
    SD0MLP,
    input_dim_for,
)


HYD_STATE_DIM = 12
CMD_DIM = 8
MECH_CONTEXT_DIM = 10


def load_run_seqs(root: Path, run_id: int) -> dict | None:
    """Load one run's aligned transition sequences (float32)."""
    path = root / "sd" / "dataset" / f"{int(run_id):05d}" / "samples.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {
            "s": z["s_h"].astype(np.float32),
            "u": z["u"].astype(np.float32),
            "q": z["q"].astype(np.float32),
            "target": z["target"].astype(np.float32),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(AAM_ROOT / "datasets/master_v1_frozen"))
    p.add_argument("--mode", choices=("su", "squ", "hist3", "hist6"), default="hist3")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--unroll", type=int, default=15, help="unrolled loss horizon (ticks)")
    p.add_argument("--val-unroll", type=int, default=15, help="free-running val horizon (ticks)")
    p.add_argument("--tf-start", type=float, default=1.0)
    p.add_argument("--tf-floor", type=float, default=0.2)
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="state-input noise as a fraction of per-channel state std")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(AAM_ROOT / "outputs/sd_cl"))
    p.add_argument("--log", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"device: {device}", flush=True)

    root = Path(args.root)
    train_ids = load_split(root, "train")
    val_ids = load_split(root, "val")
    train_runs = {rid: load_run_seqs(root, rid) for rid in train_ids}
    train_runs = {k: v for k, v in train_runs.items() if v is not None}
    val_runs = {rid: load_run_seqs(root, rid) for rid in val_ids}
    val_runs = {k: v for k, v in val_runs.items() if v is not None}
    if not train_runs:
        raise SystemExit("no train runs")
    print(f"train runs={len(train_runs)} val runs={len(val_runs)} mode={args.mode}", flush=True)

    # Z-score statistics from the training set only (same convention as the
    # one-step baseline trainer). Inputs and targets are normalized so the
    # unrolled loss is balanced across channels (Pa-scale dP vs mm-scale ext).
    def stack(field: str) -> np.ndarray:
        return np.concatenate([r[field] for r in train_runs.values()], axis=0)

    S = stack("s")
    U = stack("u")
    T = stack("target")
    s_mean, s_std = S.mean(axis=0), S.std(axis=0).clip(min=1e-6)
    u_mean, u_std = U.mean(axis=0), U.std(axis=0).clip(min=1e-6)
    y_mean, y_std = T.mean(axis=0), T.std(axis=0).clip(min=1e-6)
    q_mean = q_std = None
    if args.mode == "squ":
        Q = stack("q")
        q_mean, q_std = Q.mean(axis=0), Q.std(axis=0).clip(min=1e-6)
    s_mean_t = torch.as_tensor(s_mean, dtype=torch.float32, device=device)
    s_std_t = torch.as_tensor(s_std, dtype=torch.float32, device=device)
    u_mean_t = torch.as_tensor(u_mean, dtype=torch.float32, device=device)
    u_std_t = torch.as_tensor(u_std, dtype=torch.float32, device=device)
    q_mean_t = torch.as_tensor(q_mean, dtype=torch.float32, device=device) if q_mean is not None else None
    q_std_t = torch.as_tensor(q_std, dtype=torch.float32, device=device) if q_std is not None else None
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    noise = torch.as_tensor(args.noise_std, dtype=torch.float32, device=device)

    model = SD0MLP(
        input_dim_for(args.mode), hidden=args.hidden, depth=args.depth, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    loss_fn = nn.MSELoss()

    run_ids = list(train_runs)
    # Cache tensors per run for fast gathering.
    t_train = {rid: {k: torch.as_tensor(v, device=device) for k, v in r.items()} for rid, r in train_runs.items()}
    t_val = {rid: {k: torch.as_tensor(v, device=device) for k, v in r.items()} for rid, r in val_runs.items()}
    lag = 3 if args.mode == "hist3" else 6

    def build_input(s_cur: torch.Tensor, u: torch.Tensor, q: torch.Tensor,
                    hist: torch.Tensor | None, hist_u: torch.Tensor | None) -> torch.Tensor:
        s_n = (s_cur - s_mean_t) / s_std_t
        u_n = (u - u_mean_t) / u_std_t
        if args.mode == "su":
            return torch.cat([s_n, u_n], dim=-1)
        if args.mode == "squ":
            q_n = (q - q_mean_t) / q_std_t
            return torch.cat([s_n, q_n, u_n], dim=-1)
        blocks = torch.cat(
            [(hist - s_mean_t) / s_std_t, (hist_u - u_mean_t) / u_std_t], dim=-1
        )  # (B, lag, 20)
        return blocks.reshape(-1, lag * (HYD_STATE_DIM + CMD_DIM))

    def unroll_loss(runs: dict[int, dict], ids: list[int], starts: torch.Tensor,
                    teacher_ratio: float, noise_on: bool, max_horizon: int) -> torch.Tensor:
        """Vectorized unrolled loss over max_horizon ticks for a batch."""
        B = starts.shape[0]
        s_cur = torch.stack([runs[rid]["s"][st] for rid, st in zip(ids, starts.tolist())])
        u0 = torch.stack([runs[rid]["u"][st] for rid, st in zip(ids, starts.tolist())])
        if args.mode == "squ":
            q0 = torch.stack([runs[rid]["q"][st] for rid, st in zip(ids, starts.tolist())])
        hist = hist_u = None
        if args.mode in ("hist3", "hist6"):
            hist = torch.zeros(B, lag, HYD_STATE_DIM, device=device)
            hist_u = torch.zeros(B, lag, CMD_DIM, device=device)
            for i, (rid, st) in enumerate(zip(ids, starts.tolist())):
                r = runs[rid]
                for k in range(lag):
                    t0 = max(st - k, 0)
                    hist[i, k] = r["s"][t0]
                    hist_u[i, k] = r["u"][t0]
        total = torch.zeros((), device=device)
        for j in range(max_horizon):
            u = torch.stack([runs[rid]["u"][st + j] for rid, st in zip(ids, starts.tolist())])
            q = torch.stack([runs[rid]["q"][st + j] for rid, st in zip(ids, starts.tolist())]) if args.mode == "squ" else None
            target = torch.stack([runs[rid]["target"][st + j] for rid, st in zip(ids, starts.tolist())])
            s_in = s_cur + (noise * s_std_t * torch.randn_like(s_cur) if noise_on else 0.0)
            X = build_input(s_in, u, q, hist, hist_u)
            pred_norm = model(X)
            target_norm = (target - y_mean_t) / y_std_t
            total = total + loss_fn(pred_norm, target_norm)
            delta_raw = pred_norm * y_std_t + y_mean_t
            s_next = s_cur + delta_raw
            s_true = torch.stack([runs[rid]["s"][st + j + 1] for rid, st in zip(ids, starts.tolist())])
            tf = torch.rand(B, device=device) < teacher_ratio
            s_cur = torch.where(tf[:, None], s_true, s_next)
            if args.mode in ("hist3", "hist6"):
                hist = torch.cat([hist[:, 1:], s_cur[:, None, :]], dim=1)
                hist_u = torch.cat([hist_u[:, 1:], u[:, None, :]], dim=1)
        return total / max_horizon

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        starts_all, ids_all = [], []
        for rid, r in val_runs.items():
            n = r["s"].shape[0] - args.val_unroll
            ids_all += [rid] * n
            starts_all += list(range(n))
        if not ids_all:
            return float("nan")
        rng = np.random.default_rng(0)
        pick = rng.choice(len(ids_all), size=min(2048, len(ids_all)), replace=False)
        ids = [ids_all[i] for i in pick]
        starts = torch.tensor([starts_all[i] for i in pick], dtype=torch.long)
        loss = unroll_loss(t_val, ids, starts, teacher_ratio=0.0, noise_on=False,
                           max_horizon=args.val_unroll)
        model.train()
        return float(loss.item())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else out / "training.log"
    log_fh = log_path.open("a")

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    history = []
    t0 = time.time()
    n_train = sum(r["s"].shape[0] for r in train_runs.values())

    for epoch in range(1, args.epochs + 1):
        # Anneal teacher forcing linearly from start toward the floor.
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        tf_ratio = args.tf_start + (args.tf_floor - args.tf_start) * progress
        model.train()
        steps = max(int(n_train / args.batch_size), 8)
        train_loss = 0.0
        for _ in range(steps):
            ids = list(np.random.choice(run_ids, size=args.batch_size, replace=True))
            starts = []
            for rid in ids:
                n = train_runs[rid]["s"].shape[0]
                starts.append(int(np.random.randint(0, max(n - args.unroll - 1, 1))))
            starts_t = torch.tensor(starts, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = unroll_loss(t_train, ids, starts_t, teacher_ratio=tf_ratio,
                               noise_on=args.noise_std > 0.0, max_horizon=args.unroll)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item())
        scheduler.step()
        train_loss /= max(steps, 1)
        v = val_loss()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": v, "tf": tf_ratio})
        if v < best_val:
            best_val = v
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == 1:
            log(
                f"epoch {epoch:03d}/{args.epochs} train={train_loss:.5f} "
                f"val(unroll{args.val_unroll})={v:.5f} tf={tf_ratio:.2f} "
                f"elapsed={time.time()-t0:.0f}s"
            )
        if args.patience and no_improve >= args.patience:
            log(f"early stopping at epoch {epoch}")
            break

    if not (out / "best.pt").is_file():
        torch.save(model.state_dict(), out / "best.pt")
    report = {
        "mode": args.mode,
        "unroll": args.unroll,
        "val_unroll": args.val_unroll,
        "noise_std": args.noise_std,
        "tf_start": args.tf_start,
        "tf_floor": args.tf_floor,
        "best_epoch": best_epoch,
        "best_val_unrolled_loss": best_val,
        "train_runs": len(train_runs),
        "state_std": [float(v) for v in s_std],
        "history": history,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"best epoch {best_epoch} val(unroll{args.val_unroll})={best_val:.5f} -> {out / 'best.pt'}")
    log_fh.close()


if __name__ == "__main__":
    main()
