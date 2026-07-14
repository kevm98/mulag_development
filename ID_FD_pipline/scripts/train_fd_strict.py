#!/usr/bin/env python3
"""Stage B/C: retrain FD on the strict open-loop excitation set, with MAT replay.

Why retrain rather than nudge: the deployed FD, evaluated on the strict plant data at its own
0.05 s timebase, scores R^2 = -3.9..-8.7 on DeltaExtensionVelocity and -1.0..-3.0 on DeltadP
(worse than predicting the mean), is beaten by the trivial xd*dt baseline on DeltaExtension,
and its d(DeltadP)/d(valve) has the WRONG SIGN on cylinder 1. It is not a model of this plant.

The valve IS identifiable in the strict set (valve history alone predicts dP at R^2 0.80-0.85;
VIF of each valve column against the other 19 inputs is 7-10, so the partials are well posed).
It was NOT identifiable in the MAT, which is why the old FD's valve gain came out 3-18x hot.

Loss = Huber(strict)                 <- the plant we must actually model. Huber, not MSE: the
                                        hydraulics sit on the relief valve 9-22% of the time
                                        and one 0.05 s step can swing dP by the full 1.1e8 Pa.
     + w_mat * Huber(MAT replay)      <- state coverage; measured to help (dExt R2 .48-.95 ->
                                        .86-.92) and to REDUCE valve-gain error, so keep it.
     + lam_roll * MSE(H-step rollout) <- one-step loss cannot see drift; this is what actually
                                        constrains the valve's integrated authority.

NO constant-Jacobian anchor. It was tried and rejected on evidence: forcing dFD/dvalve to the
global OLS partial pins the gain ratio to 1.00 but collapses one-step R2 to ~0, because the
valve gain is NOT constant. Measured d(DeltaExtVel_i)/du_i by |dP| quartile:

    cyl1 [-0.762, -0.053, -0.036, -0.031]      cyl2 [-0.446, +0.165, -0.009, -0.024]
    cyl3 [-3.468, +0.128, -0.022, -0.046]      cyl4 [-0.868, +0.218, -0.058, -0.058]

i.e. a 15-75x swing with pressure, and a sign change. That is orifice physics (Q ~ u*sqrt(Ps-dP)):
near the relief valve the valve loses authority. A single constant J is wrong by construction,
and the old "FD's valve gain is 9-19x too hot" finding was an artifact of comparing a global
OLS partial (dominated by the high-dP, low-gain 80% of samples) with the net's local slope.
The gain is therefore validated PER PRESSURE BIN against the data, not against one number.

Rollout: x/xd/dP integrate from FD's own predictions; valve and Force are exogenous and are
teacher-forced from the data. Chains are built by decimating each run by 5 at every phase
offset, so a chain never crosses a run boundary.

Validation is the strict temporal held-out block (never trained on, never used to pick alpha).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
FDD = ROOT / "ID_FD_pipline/models/fd"
CSV = ROOT / "ID_FD_pipline/training_data/Excitation_FD_Strict.csv"
FD_DIR = Path("/home/rrlab/rrlab_mulag_newobs_pruned_20260701/kevin_integration/"
              "new_observations_pipeline/Models_cylinder /fd_cylinder_space ")

p = argparse.ArgumentParser()
p.add_argument("--src", default=str(FDD / "best_fd_cylinder_control.pt"))
p.add_argument("--out", default=str(FDD / "best_fd_cylinder_strict.pt"))
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--w_mat", type=float, default=0.25)
p.add_argument("--lam_jac", type=float, default=0.0)   # rejected on evidence; see docstring
p.add_argument("--lam_roll", type=float, default=1.0)
p.add_argument("--lam_kin", type=float, default=10.0)   # trapezoid consistency; see docstring
p.add_argument("--roll_h", type=int, default=10)      # 0.5 s
p.add_argument("--roll_batch", type=int, default=128)
p.add_argument("--jac_batch", type=int, default=512)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
sys.path.insert(0, str(FD_DIR))
spec = importlib.util.spec_from_file_location("fd_common", FD_DIR / "fd_cylinder_common.py")
common = importlib.util.module_from_spec(spec)
sys.modules["fd_common"] = common
spec.loader.exec_module(common)

U = [f"ValveCmd_{i}" for i in range(1, 5)]
XC = [f"Extension_{i}" for i in range(1, 5)]
XDC = [f"ExtensionVelocity_{i}" for i in range(1, 5)]
DPC = [f"dP_{i}" for i in range(1, 5)]
FC = [f"Force_{i}" for i in range(1, 5)]
S_U, S_X, S_XD, S_DP, S_F = (slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16),
                             slice(16, 20))
O_DX, O_DXD, O_DDP = slice(0, 4), slice(4, 8), slice(8, 12)
DEC, NBLOCK, VAL_EVERY = 5, 20, 5      # must match build_strict_fd_dataset.py

# ---------------------------------------------------------------- strict transitions
d = np.load(FDD / "strict_fd_transitions_0p05.npz", allow_pickle=True)
Xn, Yn = torch.tensor(d["X_train"], device=dev), torch.tensor(d["Y_train"], device=dev)
Xnv, Ynv = torch.tensor(d["X_val"], device=dev), torch.tensor(d["Y_val"], device=dev)
J_data = torch.tensor(d["J_valve_train"], dtype=torch.float32, device=dev)   # (12,4)
print(f"[strict] train {tuple(Xn.shape)} val {tuple(Xnv.shape)}", flush=True)

# ---------------------------------------------------------------- rollout chains (strict)
df = pd.read_csv(CSV)
chains_tr, chains_va = [], []
for rid, s in df.groupby("run_id", sort=True):
    s = s.reset_index(drop=True)
    n = len(s)
    edges = np.linspace(0, n - DEC, NBLOCK + 1).astype(int)   # same blocks as the transitions
    for bi in range(NBLOCK):
        lo, hi = edges[bi], edges[bi + 1]
        seg = s.iloc[lo:hi]
        if not seg["valid"].all():
            continue
        for ph in range(DEC):                     # every phase offset -> a 0.05 s chain
            arr = np.c_[seg[U].to_numpy()[ph::DEC], seg[XC].to_numpy()[ph::DEC],
                        seg[XDC].to_numpy()[ph::DEC], seg[DPC].to_numpy()[ph::DEC],
                        seg[FC].to_numpy()[ph::DEC]].astype(np.float32)
            if len(arr) <= a.roll_h + 1:
                continue
            t = torch.tensor(arr, device=dev)
            (chains_va if bi % VAL_EVERY == VAL_EVERY - 1 else chains_tr).append(t)
print(f"[roll] train chains {len(chains_tr)} (len ~{len(chains_tr[0])})  "
      f"val chains {len(chains_va)} (len ~{len(chains_va[0])})", flush=True)

# ---------------------------------------------------------------- MAT replay
with h5py.File(FDD / "FD_cylinder_data.mat", "r") as f:
    Xm = torch.tensor(np.array(f["X_train_fd"]).T.astype(np.float32), device=dev)
    Ym = torch.tensor(np.array(f["Y_train_fd"]).T.astype(np.float32), device=dev)
print(f"[mat]    replay {tuple(Xm.shape)}", flush=True)

# ---------------------------------------------------------------- model + normalisation
model, ck, _ = common.load_model_from_checkpoint(Path(a.src), device=dev)

T = lambda v: torch.tensor(np.asarray(v).ravel(), dtype=torch.float32, device=dev)
# the checkpoint's OWN stats -- the "before" baseline must be scored with these, or we would be
# feeding the pretrained weights inputs they were never normalised for and slandering them.
xm0, xs0 = T(ck["x_mean"]), T(ck["x_std"]).clamp(min=1e-8)
ym0, ys0 = T(ck["y_mean"]), T(ck["y_std"]).clamp(min=1e-8)

# renormalise over the COMBINED set for training: the strict deltas are 4-8x smaller than the
# MAT's, and the old MAT-only stats would down-weight exactly the regime we must now fit.
Xc, Yc = torch.cat([Xn, Xm], 0), torch.cat([Yn, Ym], 0)
xm_, xs_ = Xc.mean(0), Xc.std(0).clamp(min=1e-8)
ym_, ys_ = Yc.mean(0), Yc.std(0).clamp(min=1e-8)
del Xc, Yc

NORM = [xm0, xs0, ym0, ys0]          # swapped to the new stats after the "before" eval


def fd_raw(x):
    m, s, om, os_ = NORM
    return model((x - m) / s) * os_ + om


def valve_jac(rows, eps=0.05):
    cols = []
    for c in range(4):
        pp, mm = rows.clone(), rows.clone()
        pp[:, c] += eps
        mm[:, c] -= eps
        cols.append(((fd_raw(pp) - fd_raw(mm)) / (2 * eps)).mean(0))
    return torch.stack(cols, dim=1)                       # (12, 4)


# ---- data-side gain per |dP| quartile: the reference the FD must reproduce ----
Xn_np, Yn_np = Xn.cpu().numpy().astype(np.float64), Yn.cpu().numpy().astype(np.float64)
QBINS, gain_data, bin_masks = 4, np.zeros((4, 4)), []
for i in range(4):
    dpi = np.abs(Xn_np[:, 12 + i])
    qs = np.quantile(dpi, np.linspace(0, 1, QBINS + 1))
    ms = []
    for k in range(QBINS):
        m = (dpi >= qs[k]) & (dpi <= qs[k + 1])
        A = np.c_[Xn_np[m], np.ones(m.sum())]
        w, *_ = np.linalg.lstsq(A, Yn_np[m, 4 + i], rcond=None)
        gain_data[i, k] = w[i]                            # d(DeltaExtVel_i)/du_i in this bin
        ms.append(torch.tensor(np.where(m)[0], device=dev))
    bin_masks.append(ms)


def gain_by_bin():
    """FD's own d(DeltaExtVel_i)/du_i, evaluated on the data points of each |dP_i| quartile."""
    g = np.zeros((4, QBINS))
    with torch.no_grad():
        for i in range(4):
            for k in range(QBINS):
                idx = bin_masks[i][k]
                idx = idx[torch.randint(0, len(idx), (min(2000, len(idx)),), device=dev)]
                rows = Xn[idx]
                pp, mm = rows.clone(), rows.clone()
                pp[:, i] += 0.05
                mm[:, i] -= 0.05
                g[i, k] = float(((fd_raw(pp)[:, 4 + i] - fd_raw(mm)[:, 4 + i]) / 0.10).mean())
    return g


def rollout_loss(chains, H, nb):
    """Integrate FD forward H steps within a chain; valve+Force teacher-forced."""
    tot = 0.0
    for _ in range(nb):
        ch = chains[np.random.randint(len(chains))]
        t0 = np.random.randint(0, len(ch) - H - 1)
        cur = ch[t0].clone().unsqueeze(0)
        loss = 0.0
        for k in range(H):
            dpred = fd_raw(cur)
            nxt = cur.clone()
            nxt[:, S_X] = cur[:, S_X] + dpred[:, O_DX]
            nxt[:, S_XD] = cur[:, S_XD] + dpred[:, O_DXD]
            nxt[:, S_DP] = cur[:, S_DP] + dpred[:, O_DDP]
            truth = ch[t0 + k + 1].unsqueeze(0)
            nxt[:, S_U] = truth[:, S_U]                   # exogenous
            nxt[:, S_F] = truth[:, S_F]                   # exogenous
            # scale-free: compare in normalised state units
            for sl in (S_X, S_XD, S_DP):
                loss = loss + (((nxt[:, sl] - truth[:, sl]) / xs_[sl]) ** 2).mean()
            cur = nxt
        tot = tot + loss / H
    return tot / nb


def evaluate(tag):
    model.eval()
    with torch.no_grad():
        P = fd_raw(Xnv)
        r2 = [float(1 - ((Ynv[:, j] - P[:, j]) ** 2).sum()
                    / ((Ynv[:, j] - Ynv[:, j].mean()) ** 2).sum()) for j in range(12)]
        nrmse = [float((((Ynv[:, j] - P[:, j]) ** 2).mean().sqrt()) / Yn[:, j].std())
                 for j in range(12)]   # error relative to the TRAIN spread: stable under
                                       # the quiet val blocks that make R2 explode
        J = valve_jac(Xn[torch.randint(0, len(Xn), (4000,), device=dev)])
        # closed-loop rollout on the held-out chains
        rmse = []
        for ch in chains_va:
            H = min(a.roll_h, len(ch) - 2)
            cur = ch[:len(ch) - H - 1].clone()
            for k in range(H):
                dp = fd_raw(cur)
                cur[:, S_X] += dp[:, O_DX]
                cur[:, S_XD] += dp[:, O_DXD]
                cur[:, S_DP] += dp[:, O_DDP]
                nx = ch[k + 1:k + 1 + len(cur)]
                cur[:, S_U] = nx[:, S_U]
                cur[:, S_F] = nx[:, S_F]
            tgt = ch[H:H + len(cur)][:, S_X]
            rmse.append(float(((cur[:, S_X] - tgt) ** 2).mean().sqrt()))
    model.train()
    g = lambda s: np.round(r2[s], 3).tolist()
    q = lambda s: np.round(nrmse[s], 3).tolist()
    ratio = (torch.diag(J[O_DX]) / torch.diag(J_data[O_DX])).cpu().numpy()
    for nm, sl in (("dExt", O_DX), ("dVel", O_DXD), ("ddP ", O_DDP)):
        print(f"[{tag}] {nm}  R2={g(sl)}   NRMSE/train-sigma={q(sl)}", flush=True)
    gb = gain_by_bin()
    print(f"[{tag}] valve gain d(dVel_i)/du_i by |dP| quartile  (FD vs DATA):", flush=True)
    for i in range(4):
        fd_s = " ".join(f"{v:+8.4f}" for v in gb[i])
        da_s = " ".join(f"{v:+8.4f}" for v in gain_data[i])
        print(f"[{tag}]   cyl{i+1} FD  ={fd_s}", flush=True)
        print(f"[{tag}]        DATA={da_s}", flush=True)
    print(f"[{tag}] {a.roll_h}-step ({a.roll_h*0.05:.2f}s) rollout ext RMSE = "
          f"{np.mean(rmse):.5f} m", flush=True)
    return dict(r2=r2, nrmse=nrmse, valve_ratio_global=ratio.tolist(),
                gain_by_bin_fd=gb.tolist(), gain_by_bin_data=gain_data.tolist(),
                rollout_rmse=float(np.mean(rmse)), jac=J.detach().cpu().numpy().tolist())


print("\n--- BEFORE (deployed FD, scored with its OWN normalisation) ---", flush=True)
before = evaluate("before")

NORM[:] = [xm_, xs_, ym_, ys_]        # from here on, train/score with the combined stats

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
nb = max(1, len(Xn) // a.batch)
jscale = torch.diag(J_data[O_DX]).abs().clamp(min=1e-9)

print(f"\n--- TRAIN ({a.epochs} epochs x {nb} batches) ---", flush=True)
best = None
for ep in range(a.epochs):
    perm = torch.randperm(len(Xn), device=dev)
    acc = np.zeros(4)
    for b in range(nb):
        i = perm[b * a.batch:(b + 1) * a.batch]
        opt.zero_grad()
        # Huber, not MSE: the hydraulics sit on the relief valve 20-41% of the time and a
        # single 0.05 s step can swing dP by the full 1.1e8 Pa range. Under MSE those rare
        # events own the gradient and the model fits the spikes instead of the plant.
        hub = lambda p, t: torch.nn.functional.smooth_l1_loss(p, t, beta=1.0)
        l_new = hub(model((Xn[i] - xm_) / xs_), (Yn[i] - ym_) / ys_)
        j = torch.randint(0, len(Xm), (a.batch,), device=dev)
        l_mat = hub(model((Xm[j] - xm_) / xs_), (Ym[j] - ym_) / ys_)
        # kinematic consistency: DeltaExtension is not a free output. dx = dt*xd + (dt/2)*dxd
        # holds in the data at R^2 0.994-0.9997 (exact trapezoid rule). Forcing the two heads
        # to agree hands DeltaExtension the accuracy of DeltaExtensionVelocity instead of
        # making the net rediscover an identity -- without it, FD's one-step dx is 4x worse
        # than the trivial x + v*dt.
        pred = fd_raw(Xn[i])
        dx_kin = 0.05 * Xn[i][:, S_XD] + 0.025 * pred[:, O_DXD]
        l_kin = (((pred[:, O_DX] - dx_kin) / ys_[O_DX]) ** 2).mean()

        l_jac = torch.zeros((), device=dev)
        if a.lam_jac > 0:
            Jb = valve_jac(Xn[perm[torch.randint(0, len(Xn), (a.jac_batch,), device=dev)]])
            l_jac = (((Jb[O_DX] - J_data[O_DX]) / jscale.reshape(1, 4)) ** 2).mean()
        l_roll = rollout_loss(chains_tr, a.roll_h, 4)
        loss = (l_new + a.w_mat * l_mat + a.lam_jac * l_jac + a.lam_roll * l_roll
                + a.lam_kin * l_kin)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc += [float(l_new), float(l_mat), float(l_kin), float(l_roll)]
    sched.step()
    acc /= nb
    if (ep + 1) % 10 == 0 or ep == 0:
        print(f"[ep {ep+1:3d}] new={acc[0]:.5f} mat={acc[1]:.5f} kin={acc[2]:.5f} "
              f"roll={acc[3]:.5f}", flush=True)

print("\n--- AFTER ---", flush=True)
after = evaluate("after")

ck_out = dict(ck)
ck_out["model_state_dict"] = model.state_dict()
for k, v in (("x_mean", xm_), ("x_std", xs_), ("y_mean", ym_), ("y_std", ys_)):
    ck_out[k] = v.detach().cpu().numpy()
ck_out["strict_finetune"] = dict(
    source=a.src, dataset="Excitation_FD_Strict (4 runs x 6400 @0.01s -> 0.05s transitions)",
    epochs=a.epochs, lr=a.lr, w_mat=a.w_mat, lam_jac=a.lam_jac, lam_roll=a.lam_roll,
    roll_h=a.roll_h, J_data=J_data.cpu().numpy().tolist(),
    before=before, after=after,
    note="Renormalised over strict+MAT. The deployed FD scored negative R2 on dVel/ddP on "
         "real plant data and had a sign-flipped d(ddP)/d(valve) on cylinder 1.")
torch.save(ck_out, a.out)
json.dump(dict(before=before, after=after),
          open(FDD / "fd_strict_training_report.json", "w"), indent=2)
print(f"\n[OUT] {a.out}", flush=True)
