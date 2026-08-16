#!/usr/bin/env python3
"""End-to-end AAM + SD: learn z_hat from observable history, no privileged encoder.

Why this exists
---------------
The privileged route is ``FactorEncoder: e(6 true factors) -> z(2)``, trained
only on dynamics prediction. Two consequences were measured on the frozen
Stage 31 model:

* the 6->2 squeeze discards whatever does not help predict cylinder dynamics,
  so the latent decodes ``seal`` at R2 0.956 and ``mass_scale`` at -0.008; and
* it can only ever represent quantities that HAVE a label ``e``. The
  Simscape <-> IsaacLab-ROM discrepancy has no such label, so the privileged
  route structurally cannot encode the very gap the AAM is deployed to cover.

This trainer removes the privileged encoder entirely::

    history h_t --> AAM --> z_hat -----.
                                        v
    causal [s;u] window --> GRU --> h_t (belief)
                                        v
                 s_t, u_t, q_t, z_hat, h_t --> SD --> Delta s_{t+1}

``z_hat`` is whatever helps SD predict the next state from observable history.
No factor labels are used at any point, so nothing has to be labelled for the
latent to represent it.

The belief ``h_t`` carries instantaneous plant state and ``z_hat`` the slowly
varying configuration; they are never fused, matching the Stage 22 protocol.

Alignment
---------
SD samples carry ``tick_control_step``; AAM windows carry ``window_end_tick``.
Both index the same 60 Hz control grid (AAM windows sit on every third step,
the 50 ms ROM tick). Rows are joined on that tick and the join is asserted, not
assumed -- a silent misalignment would train the latent against the wrong
transition and still converge.

Output layout matches the frozen-AAM contract (``aam.pt`` + ``metrics.json``
with ``in_dim``/``lag_ticks``/``feature_dim``/``latent_dim`` and
``input_normalization``) so the result loads directly in
``PPO_ID_AAM_FD/online_runtime.py``. There is deliberately no ``encoder.pt``:
this path has no privileged encoder, and downstream label regeneration via
``build_stage31_aam_dataset.py`` is neither needed nor valid here.
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

AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_split  # noqa: E402
from tools.train_stage31_aam import AAM  # noqa: E402
from tools.train_sd_z import (  # noqa: E402
    BeliefEncoder, SDz, causal_windows_np, group_rmse,
    HYD_STATE_DIM, CMD_DIM, MECH_CONTEXT_DIM, TARGET_DIM,
)

AAM_FEATURE_DIM = 18


def load_aligned(root: Path, run_id: int, lag: int, belief_lag: int) -> dict | None:
    """Join SD samples and AAM windows on the control tick."""
    sd_path = root / "sd" / "dataset" / f"{run_id:05d}" / "samples.npz"
    aam_path = root / "aam" / "dataset_multilag" / f"{run_id:05d}" / f"history_L{lag}.npz"
    if not sd_path.is_file() or not aam_path.is_file():
        return None
    with np.load(sd_path, allow_pickle=False) as z:
        sd_tick = z["tick_control_step"].astype(np.int64)
        s = z["s_h"].astype(np.float64)
        u = z["u"].astype(np.float64)
        q = z["q"].astype(np.float64)
        target = z["target"].astype(np.float64)
    with np.load(aam_path, allow_pickle=False) as z:
        aam_tick = z["window_end_tick"].astype(np.int64)
        window = z["window_input"].astype(np.float64)

    # belief windows are indexed by SD row; row r covers ticks [r-belief_lag+1, r]
    blocks = causal_windows_np(s, u, belief_lag)          # (n-belief_lag+1, L, blk)
    block_row0 = belief_lag - 1

    order = np.argsort(sd_tick)
    sd_tick_sorted = sd_tick[order]
    pos = np.searchsorted(sd_tick_sorted, aam_tick)
    ok = (pos < len(sd_tick_sorted)) & (sd_tick_sorted[np.clip(pos, 0, len(sd_tick_sorted) - 1)] == aam_tick)
    sd_rows = order[np.clip(pos, 0, len(sd_tick_sorted) - 1)]
    ok &= sd_rows >= block_row0                            # belief window must exist
    if not ok.any():
        return None
    sd_rows = sd_rows[ok]
    win = window[ok]
    if not np.array_equal(sd_tick[sd_rows], aam_tick[ok]):
        raise RuntimeError(f"run {run_id}: tick join is not exact")
    return {
        "window": win,
        "s": s[sd_rows], "u": u[sd_rows], "q": q[sd_rows],
        "target": target[sd_rows],
        "blocks": blocks[sd_rows - block_row0],
        "tick": aam_tick[ok],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--lag", type=int, default=20, help="AAM history lag (ROM ticks)")
    p.add_argument("--belief-lag", type=int, default=6)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--belief-dim", type=int, default=16)
    p.add_argument("--aam-hidden", type=int, default=64)
    p.add_argument("--aam-depth", type=int, default=4)
    p.add_argument("--aam-dropout", type=float, default=0.05)
    p.add_argument("--sd-hidden", type=int, default=256)
    p.add_argument("--sd-depth", type=int, default=3)
    p.add_argument("--sd-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temporal-consistency-weight", type=float, default=0.1,
                   help="penalise |z_t - z_{t-1}| so the latent stays a slowly "
                        "varying configuration rather than tracking fast state")
    p.add_argument("--no-belief", action="store_true",
                   help="drop the GRU belief h_t. With h_t present the latent "
                        "can COLLAPSE: h_t already supplies what SD needs, so "
                        "nothing pressures z to carry plant identity (measured "
                        "between/within = 0.068 vs 4.74 for the privileged "
                        "latent). Removing h_t makes z the only channel.")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-rows-per-run", type=int, default=0,
                   help="subsample rows per run (0 = all)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device)

    splits = {s: load_split(a.root, s) for s in ("train", "val", "test")}
    data: dict[str, list[dict]] = {}
    for split, ids in splits.items():
        rows = []
        for rid in ids:
            r = load_aligned(a.root, int(rid), a.lag, a.belief_lag)
            if r is None or not r["window"].shape[0]:
                continue
            if a.max_rows_per_run:
                sel = slice(0, None, max(r["window"].shape[0] // a.max_rows_per_run, 1))
                r = {k: v[sel] for k, v in r.items()}
            rows.append(r)
        data[split] = rows
        print(f"{split}: {len(rows)} runs, {sum(x['window'].shape[0] for x in rows)} rows")
    if not data["train"] or not data["val"]:
        raise SystemExit("need non-empty train and val splits")

    def cat(split, key):
        return np.concatenate([r[key] for r in data[split]])

    win_tr = cat("train", "window")
    w_mean, w_std = win_tr.mean(0), win_tr.std(0).clip(min=1e-6)
    s_tr, u_tr, q_tr = cat("train", "s"), cat("train", "u"), cat("train", "q")
    blk_tr, y_tr = cat("train", "blocks"), cat("train", "target")
    s_m, s_s = s_tr.mean(0), s_tr.std(0).clip(min=1e-6)
    u_m, u_s = u_tr.mean(0), u_tr.std(0).clip(min=1e-6)
    q_m, q_s = q_tr.mean(0), q_tr.std(0).clip(min=1e-6)
    y_m, y_s = y_tr.mean(0), y_tr.std(0).clip(min=1e-6)
    b_m = blk_tr.reshape(-1, blk_tr.shape[-1]).mean(0)
    b_s = blk_tr.reshape(-1, blk_tr.shape[-1]).std(0).clip(min=1e-6)

    def tensors(split):
        t = lambda x: torch.as_tensor(x.astype(np.float32), device=device)
        return {
            "w": t((cat(split, "window") - w_mean) / w_std),
            "s": t((cat(split, "s") - s_m) / s_s),
            "u": t((cat(split, "u") - u_m) / u_s),
            "q": t((cat(split, "q") - q_m) / q_s),
            "b": t((cat(split, "blocks") - b_m) / b_s),
            "y": t((cat(split, "target") - y_m) / y_s),
        }

    T = {s: tensors(s) for s in ("train", "val", "test") if data[s]}
    in_dim = T["train"]["w"].shape[1]
    block_dim = T["train"]["b"].shape[-1]

    aam = AAM(in_dim, a.latent_dim, hidden=a.aam_hidden, depth=a.aam_depth,
              dropout=a.aam_dropout, architecture="cnn",
              feature_dim=AAM_FEATURE_DIM, lag=a.lag).to(device)
    belief = BeliefEncoder(block_dim, a.belief_dim).to(device)
    use_belief = not a.no_belief
    sd = SDz(a.latent_dim, state_dim=HYD_STATE_DIM, hidden=a.sd_hidden,
             depth=a.sd_depth, dropout=a.sd_dropout, use_q=True,
             use_belief=use_belief, belief_dim=a.belief_dim).to(device)

    params = list(aam.parameters()) + list(sd.parameters())
    if use_belief:
        params += list(belief.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)

    def forward(d, idx):
        z = aam(d["w"][idx])
        parts = [d["s"][idx], d["u"][idx], d["q"][idx], z]
        if use_belief:
            parts.append(belief.init_from_window(d["b"][idx]))
        return sd(torch.cat(parts, dim=-1)), z

    def evaluate(split):
        for m in (aam, belief, sd):
            m.eval()
        d = T[split]
        n = d["w"].shape[0]
        outs = []
        with torch.no_grad():
            for i in range(0, n, a.batch_size):
                idx = torch.arange(i, min(i + a.batch_size, n), device=device)
                outs.append(forward(d, idx)[0])
        pred = torch.cat(outs)
        return float(nn.functional.mse_loss(pred, d["y"])), pred

    best, best_state, bad = float("inf"), None, 0
    n_train = T["train"]["w"].shape[0]
    for epoch in range(a.epochs):
        for m in (aam, belief, sd):
            m.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, a.batch_size):
            idx = perm[i:i + a.batch_size]
            opt.zero_grad(set_to_none=True)
            pred, z = forward(T["train"], idx)
            loss = nn.functional.mse_loss(pred, T["train"]["y"][idx])
            if a.temporal_consistency_weight > 0 and z.shape[0] > 1:
                loss = loss + a.temporal_consistency_weight * (z[1:] - z[:-1]).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        vl, _ = evaluate("val")
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {
                "aam": {k: v.detach().clone() for k, v in aam.state_dict().items()},
                "belief": {k: v.detach().clone() for k, v in belief.state_dict().items()},
                "sd": {k: v.detach().clone() for k, v in sd.state_dict().items()},
                "epoch": epoch,
            }
        else:
            bad += 1
            if bad >= a.patience:
                break
        if epoch % 5 == 0 or bad == 0:
            print(f"epoch {epoch:3d}  val_mse={vl:.6f}  best={best:.6f}", flush=True)

    if best_state:
        aam.load_state_dict(best_state["aam"])
        belief.load_state_dict(best_state["belief"])
        sd.load_state_dict(best_state["sd"])

    report = {"best_val_mse": best, "best_epoch": best_state["epoch"] if best_state else None}
    for split in ("val", "test"):
        if split not in T:
            continue
        mse, pred = evaluate(split)
        yt = T[split]["y"].cpu().numpy() * y_s + y_m
        yp = pred.cpu().numpy() * y_s + y_m
        report[f"{split}_mse_standardised"] = mse
        report[f"{split}_rmse"] = group_rmse(yt, yp)

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(aam.state_dict(), a.out / "aam.pt")
    torch.save(belief.state_dict(), a.out / "belief.pt")
    torch.save(sd.state_dict(), a.out / "sd.pt")
    metrics = {
        "format": "stage31_sd_aam_e2e_v1",
        "latent_source": "aam_end_to_end",
        "privileged_factors_used": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # --- frozen-AAM contract, so online_runtime.py can load aam.pt ---
        "architecture": "cnn",
        "feature_dim": AAM_FEATURE_DIM,
        "lag_ticks": a.lag,
        "in_dim": in_dim,
        "latent_dim": a.latent_dim,
        "hidden": a.aam_hidden,
        "depth": a.aam_depth,
        "dropout": a.aam_dropout,
        "input_normalization": {
            "per_position_mean": w_mean.tolist(),
            "per_position_std": w_std.tolist(),
            "computed_on": "train_windows_only",
        },
        # --- SD side ---
        "use_belief": use_belief,
        "belief_dim": a.belief_dim if use_belief else 0,
        "belief_lag": a.belief_lag,
        "sd_hidden": a.sd_hidden,
        "sd_depth": a.sd_depth,
        "temporal_consistency_weight": a.temporal_consistency_weight,
        "seed": a.seed,
        "runs": {s: len(v) for s, v in data.items()},
        "rows": {s: int(T[s]["w"].shape[0]) for s in T},
        **report,
    }
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
