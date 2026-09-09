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
``PPO_AAM_FD/online_runtime.py``. There is deliberately no ``encoder.pt``:
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

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AAM_ROOT = Path(__file__).resolve().parents[3] / "AAM_module"  # SD tools live in learned_models/sd_model/; aam_data and datasets stay under AAM_module
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_split  # noqa: E402
from aam_data.sample_mask import usable_rows  # noqa: E402
from tools.train_stage31_aam import AAM  # noqa: E402
from learned_models.sd_model.training.train_sd_z import (  # noqa: E402
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
        usable = usable_rows(z)
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
        "usable": usable[sd_rows].astype(np.float32),
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
    p.add_argument("--unroll", type=int, default=1,
                   help="rollout horizon in 50 ms ticks. >1 adds the accumulated "
                        "normalised state-error term, matching train_sd_z.py so "
                        "the e2e latent can be compared with SD_0/SD_z on the "
                        "same objective. 1 reproduces the one-step trainer.")
    p.add_argument("--rollout-weight", type=float, default=1.0,
                   help="target weight of the rollout term once ramped in")
    p.add_argument("--l1-weight", type=float, default=1.0,
                   help="weight of the teacher-forced one-step term")
    p.add_argument("--tf-floor", type=float, default=0.2,
                   help="scheduled-sampling teacher-forcing floor")
    p.add_argument("--rollout-ramp-epochs", type=int, default=0,
                   help="ramp the rollout weight linearly from 0 over N epochs")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-rows-per-run", type=int, default=0,
                   help="subsample rows per run (0 = all)")
    p.add_argument("--eval-only", type=Path, default=None,
                   help="load aam/belief/sd from this dir, skip training, and "
                        "score the online / run-mean / shuffled latent ablation")
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
            "m": t(cat(split, "usable")),
        }

    T = {s: tensors(s) for s in ("train", "val", "test") if data[s]}
    in_dim = T["train"]["w"].shape[1]
    block_dim = T["train"]["b"].shape[-1]

    # ---- per-run padded views, needed only for the unrolled objective ------
    # The flat tensors above lose run boundaries, so a rollout built from them
    # would silently step across a run boundary. These keep the run structure
    # and record, per row, whether the next `unroll` rows are the physically
    # consecutive 50 ms ticks of the SAME run.
    TICK_STRIDE = 3  # SD rows sit on every third 60 Hz control step

    def padded(split):
        runs = data[split]
        n = [int(r["window"].shape[0]) for r in runs]
        R, N = len(runs), max(n)
        f32 = np.float32
        W = np.zeros((R, N, in_dim), f32)
        Sf = np.zeros((R, N + 1, HYD_STATE_DIM), f32)   # states, plus the last next-state
        U = np.zeros((R, N, T["train"]["u"].shape[1]), f32)
        Q = np.zeros((R, N, T["train"]["q"].shape[1]), f32)
        Y = np.zeros((R, N, TARGET_DIM), f32)
        Bk = np.zeros((R, N, a.belief_lag, block_dim), f32)  # (run, row, lag, dim)
        M = np.zeros((R, N), f32)          # per-row training usability
        ok = np.zeros((R, N), bool)
        for i, r in enumerate(runs):
            m = n[i]
            W[i, :m] = (r["window"] - w_mean) / w_std
            Sf[i, :m] = r["s"]
            Sf[i, m] = r["s"][-1] + r["target"][-1]      # s_{r+1} = s_r + target_r
            U[i, :m], Q[i, :m], Y[i, :m], Bk[i, :m] = r["u"], r["q"], r["target"], r["blocks"]
            M[i, :m] = r["usable"]
            contiguous = np.diff(r["tick"]) == TICK_STRIDE
            ok[i, :m] = True
            ok[i, :m - 1] &= contiguous                  # row r usable as a step
            ok[i, m - 1] = False                         # no next state recorded
        dev = lambda x: torch.as_tensor(x, device=device)
        return {"W": dev(W), "S": dev(Sf), "U": dev(U), "Q": dev(Q),
                "Y": dev(Y), "B": dev(Bk), "M": dev(M), "ok": dev(ok), "n": n}

    PAD = {s: padded(s) for s in T} if a.unroll > 1 else {}

    def rollout_starts(split, horizon):
        """(run, row) pairs whose whole horizon is contiguous within one run."""
        ok = PAD[split]["ok"]
        run_ok = ok.clone()
        for j in range(1, horizon):
            run_ok[:, :-j] &= ok[:, j:]
        run_ok[:, -horizon:] = False
        idx = run_ok.nonzero(as_tuple=False)
        return idx[:, 0], idx[:, 1]

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

    def masked_mse(pred, tgt, m):
        """MSE over usable rows only. Saturated targets contribute no gradient."""
        w = m[:, None]
        denom = w.sum() * pred.shape[1]
        if float(denom) == 0.0:
            return pred.sum() * 0.0
        return ((pred - tgt) ** 2 * w).sum() / denom

    def forward(d, idx):
        z = aam(d["w"][idx])
        parts = [d["s"][idx], d["u"][idx], d["q"][idx], z]
        if use_belief:
            parts.append(belief.init_from_window(d["b"][idx]))
        return sd(torch.cat(parts, dim=-1)), z

    s_mean_t = torch.as_tensor(s_m.astype(np.float32), device=device)
    s_std_t = torch.as_tensor(s_s.astype(np.float32), device=device)
    u_mean_t = torch.as_tensor(u_m.astype(np.float32), device=device)
    u_std_t = torch.as_tensor(u_s.astype(np.float32), device=device)
    q_mean_t = torch.as_tensor(q_m.astype(np.float32), device=device)
    q_std_t = torch.as_tensor(q_s.astype(np.float32), device=device)
    y_mean_t = torch.as_tensor(y_m.astype(np.float32), device=device)
    y_std_t = torch.as_tensor(y_s.astype(np.float32), device=device)

    # --- latent substitution, used only by the --eval-only sweep -------------
    # Training always runs with "online". The eval modes answer claim 1: if a
    # per-run CONSTANT latent scores the same as the online one, the model is
    # being conditioned on plant identity and nothing is adapting within the
    # episode. "shuffled" is the floor -- same latent distribution, wrong plant.
    latent_mode = "online"
    run_latent: dict[str, torch.Tensor] = {}

    def latent_for(split, rows, starts, W):
        if latent_mode == "online":
            return aam(W[rows, starts])
        return run_latent[split][rows]

    @torch.no_grad()
    def build_run_latent(split, mode, seed=0, chunk=4096):
        """(R, latent_dim) per-run mean z_hat over that run's valid ticks."""
        d = PAD[split]
        W, ok = d["W"], d["ok"]
        out = torch.zeros((W.shape[0], a.latent_dim), device=device)
        for r in range(W.shape[0]):
            idx = ok[r].nonzero(as_tuple=False).squeeze(-1)
            if not idx.numel():
                continue
            zs = [aam(W[r, idx[i:i + chunk]]) for i in range(0, idx.numel(), chunk)]
            out[r] = torch.cat(zs).mean(0)
        if mode == "shuffled":
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(W.shape[0], generator=g).to(device)
            if W.shape[0] > 1 and bool((perm == torch.arange(
                    W.shape[0], device=device)).any()):
                perm = torch.roll(perm, 1)
            out = out[perm]
        return out

    def unrolled_loss(split, rows, starts, horizon, tf_ratio):
        """One-step delta loss and accumulated normalised state error.

        Semantics are copied from ``train_sd_z.unrolled_loss`` so the two are
        scored on the same objective: the model integrates its own predicted
        deltas and is charged for where it actually lands. The latent is read
        from the AAM window at the rollout START and held for the horizon,
        which is what deployment does -- z updates on the 20 Hz grid and is a
        slowly varying plant descriptor, not a per-step correction.
        """
        d = PAD[split]
        S, U, Q, Y, W, Bk = d["S"], d["U"], d["Q"], d["Y"], d["W"], d["B"]
        z = latent_for(split, rows, starts, W)
        h = belief.init_from_window(Bk[rows, starts]) if use_belief else None
        s_cur = S[rows, starts]
        l1 = torch.zeros((), device=device)
        st = torch.zeros((), device=device)
        for j in range(horizon):
            ticks = starts + j
            s_keep_n = (s_cur - s_mean_t) / s_std_t
            u_n = (U[rows, ticks] - u_mean_t) / u_std_t
            q_n = (Q[rows, ticks] - q_mean_t) / q_std_t
            if use_belief:
                h = belief.step(torch.cat([s_keep_n, u_n], dim=-1), h)
            parts = [s_keep_n, u_n, q_n, z]
            if use_belief:
                parts.append(h)
            pred_norm = sd(torch.cat(parts, dim=-1))
            # A saturated step still PROPAGATES (deployment cannot skip it) but
            # contributes no gradient, because its target says nothing about
            # which plant this is.
            mj = d["M"][rows, ticks]
            l1 = l1 + masked_mse(pred_norm, (Y[rows, ticks] - y_mean_t) / y_std_t, mj)
            s_next = s_cur + (pred_norm * y_std_t + y_mean_t)
            s_true_next = S[rows, ticks + 1]
            st = st + masked_mse(s_next / s_std_t, s_true_next / s_std_t, mj)
            tf = torch.rand(rows.shape[0], device=device) < tf_ratio
            s_cur = torch.where(tf[:, None], s_true_next, s_next)
        return l1 / horizon, st / horizon, z

    def evaluate_rollout(split, horizon, chunk=2048):
        """Free-running (tf_ratio=0) accumulated normalised state error.

        This is the criterion train_sd_z.py selects on, so an e2e checkpoint
        chosen by it is comparable with SD_0 and SD_z rather than merely
        adjacent to them.
        """
        for m in (aam, belief, sd):
            m.eval()
        rows, starts = rollout_starts(split, horizon)
        if not rows.shape[0]:
            return float("nan")
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, rows.shape[0], chunk):
                r, s0 = rows[i:i + chunk], starts[i:i + chunk]
                _, st, _ = unrolled_loss(split, r, s0, horizon, tf_ratio=0.0)
                tot += float(st) * r.shape[0]
                n += r.shape[0]
        return tot / max(n, 1)

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
        return float(masked_mse(pred, d["y"], d["m"])), pred

    if a.eval_only:
        src = Path(a.eval_only)
        aam.load_state_dict(torch.load(src / "aam.pt", map_location=device))
        belief.load_state_dict(torch.load(src / "belief.pt", map_location=device))
        sd.load_state_dict(torch.load(src / "sd.pt", map_location=device))
        H = max(a.unroll, 1)
        report = {"checkpoint": str(src), "horizon": H,
                  "estimand": "free-running accumulated normalised state error, "
                              "tf_ratio=0", "modes": {}}
        for mode in ("online", "run_mean", "shuffled"):
            latent_mode = mode
            if mode != "online":
                for sp in ("val", "test"):
                    run_latent[sp] = build_run_latent(sp, mode, seed=a.seed)
            report["modes"][mode] = {
                sp: evaluate_rollout(sp, H) for sp in ("val", "test")}
            print(f"  {mode:>9}  val {report['modes'][mode]['val']:.5f}   "
                  f"test {report['modes'][mode]['test']:.5f}")
        base = report["modes"]["online"]["val"]
        report["run_mean_vs_online_pct"] = 100.0 * (
            report["modes"]["run_mean"]["val"] / base - 1.0)
        report["shuffled_vs_online_pct"] = 100.0 * (
            report["modes"]["shuffled"]["val"] / base - 1.0)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "latent_ablation.json").write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out / 'latent_ablation.json'}")
        return

    best, best_state, bad = float("inf"), None, 0
    n_train = T["train"]["w"].shape[0]
    if a.unroll > 1:
        roll_rows, roll_starts = rollout_starts("train", a.unroll)
        n_roll = roll_rows.shape[0]
        if not n_roll:
            raise SystemExit(f"no contiguous rollout window of {a.unroll} ticks")
        print(f"rollout: {n_roll} usable starts at horizon {a.unroll}")
    for epoch in range(a.epochs):
        for m in (aam, belief, sd):
            m.train()
        if a.unroll > 1:
            # scheduled sampling: fully teacher-forced at first, decaying to
            # the floor, so the model is optimised for recursive use.
            frac = epoch / max(a.epochs - 1, 1)
            tf_ratio = max(a.tf_floor, 1.0 - frac)
            lam_r = a.rollout_weight
            if a.rollout_ramp_epochs > 0:
                lam_r *= min(1.0, epoch / a.rollout_ramp_epochs)
            roll_perm = torch.randperm(n_roll, device=device)
        perm = torch.randperm(n_train, device=device)
        for bi, i in enumerate(range(0, n_train, a.batch_size)):
            idx = perm[i:i + a.batch_size]
            opt.zero_grad(set_to_none=True)
            pred, z = forward(T["train"], idx)
            loss = a.l1_weight * masked_mse(pred, T["train"]["y"][idx], T["train"]["m"][idx])
            if a.unroll > 1:
                j = (bi * a.batch_size) % n_roll
                sel = roll_perm[j:j + a.batch_size]
                if sel.numel():
                    l1_r, st_r, z_r = unrolled_loss(
                        "train", roll_rows[sel], roll_starts[sel], a.unroll, tf_ratio)
                    loss = loss + lam_r * st_r
                    z = torch.cat((z, z_r), dim=0)
            if a.temporal_consistency_weight > 0 and z.shape[0] > 1:
                loss = loss + a.temporal_consistency_weight * (z[1:] - z[:-1]).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        vl, _ = evaluate("val")
        if a.unroll > 1:
            vl = evaluate_rollout("val", a.unroll)
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

    report = {"best_val_mse": best, "best_epoch": best_state["epoch"] if best_state else None,
              "selection_criterion": ("val_rollout_state_mse" if a.unroll > 1
                                      else "val_one_step_mse")}
    if a.unroll > 1:
        for split in ("val", "test"):
            if split in PAD:
                report[f"{split}_rollout_state_mse"] = evaluate_rollout(split, a.unroll)
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
        # predict_stage31_aam.py copies this into its output header; the e2e
        # AAM consumes the identical 18-feature rows as the stage31 one.
        "feature_contract": "joint_q + joint_qdot + dP + previous_valve",
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
        "sample_mask": {
            "applied": True,
            "flags": ["command_saturation_flag", "cylinder_limit_flag",
                      "invalid_physics_flag"],
            "note": "masked in the LOSS only; rows stay in place so causal "
                    "history windows remain contiguous",
        },
        "unroll": a.unroll,
        "rollout_weight": a.rollout_weight if a.unroll > 1 else 0.0,
        "l1_weight": a.l1_weight,
        "tf_floor": a.tf_floor if a.unroll > 1 else None,
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
