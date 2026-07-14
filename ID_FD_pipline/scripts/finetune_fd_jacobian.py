#!/usr/bin/env python3
"""Stage 1: fine-tune FD so its VALVE SENSITIVITY matches the source data.

Why this is needed (docs/idfd_wiring_audit.md §3):

The deployed FD scores R^2 = 0.993-0.997 on its own training data, yet its valve gain
d(DeltaExtension)/d(ValveCmd) is 9-19x LARGER than the data's:

    diag(data) = [-9.4e-05, -5.2e-05, -4.49e-04, -3.06e-04]
    diag(FD)   = [-1.69e-03, -9.77e-04, -4.88e-03, -2.83e-03]
    ratio      = [18.0, 18.8, 10.9, 9.3]

It can do both because ExtensionVelocity dominates the variance of DeltaExtension
(dx ~ xdot*dt), so one-step MSE barely constrains the valve term -- and the valve term is
the ONE thing the controller depends on. Plain MSE fine-tuning cannot fix this; the loss
must constrain the JACOBIAN.

Loss = MSE(one-step prediction)                     [keep the fidelity it already has]
     + lambda_jac * || dFD/dvalve - J_target ||^2   [fix the gain]

J_target is the source-data partial derivative (OLS over all 20 inputs, VIF ~ 1.0 so the
partials are identifiable). No fabricated valve labels: the data is the Simscape set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FD_MAT = ROOT / "ID_FD_pipline/models/fd/FD_cylinder_data.mat"
FD_DIR = Path("/home/rrlab/rrlab_mulag_newobs_pruned_20260701/kevin_integration/"
              "new_observations_pipeline/Models_cylinder /fd_cylinder_space ")

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="ID_FD_pipline/models/fd/best_fd_cylinder_control.pt")
parser.add_argument("--out", default="ID_FD_pipline/models/fd/best_fd_cylinder_jac.pt")
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lambda_jac", type=float, default=30.0)
parser.add_argument("--jac_batch", type=int, default=512)
parser.add_argument("--valve_eps", type=float, default=0.05)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
sys.path.insert(0, str(FD_DIR))
spec = importlib.util.spec_from_file_location("fd_common", FD_DIR / "fd_cylinder_common.py")
common = importlib.util.module_from_spec(spec)
sys.modules["fd_common"] = common
spec.loader.exec_module(common)

# ---------------------------------------------------------------- data
with h5py.File(FD_MAT, "r") as f:
    X = np.array(f["X_train_fd"]).T.astype(np.float32)
    Y = np.array(f["Y_train_fd"]).T.astype(np.float32)
    Xv = np.array(f["X_val_fd"]).T.astype(np.float32)
    Yv = np.array(f["Y_val_fd"]).T.astype(np.float32)
print(f"[FD] train {X.shape}  val {Xv.shape}", flush=True)

# source-data valve partials (the target Jacobian), in RAW units
A = np.c_[X.astype(np.float64), np.ones(len(X))]
coef, *_ = np.linalg.lstsq(A, Y[:, 0:4].astype(np.float64), rcond=None)
J_target_raw = torch.tensor(coef[0:4, :].T, dtype=torch.float32, device=dev)  # (4 out, 4 valve)
print(f"[FD] target diag (from data) = "
      f"{np.round(np.diag(J_target_raw.cpu().numpy()), 6).tolist()}", flush=True)

model, ck, _ = common.load_model_from_checkpoint(ROOT / args.src, device=dev)
xm = torch.tensor(np.asarray(ck["x_mean"]).ravel(), dtype=torch.float32, device=dev)
xs = torch.tensor(np.asarray(ck["x_std"]).ravel(), dtype=torch.float32, device=dev).clamp(min=1e-8)
ym = torch.tensor(np.asarray(ck["y_mean"]).ravel(), dtype=torch.float32, device=dev)
ys = torch.tensor(np.asarray(ck["y_std"]).ravel(), dtype=torch.float32, device=dev)

Xt = torch.from_numpy(X).to(dev); Yt = torch.from_numpy(Y).to(dev)
Xvt = torch.from_numpy(Xv).to(dev); Yvt = torch.from_numpy(Yv).to(dev)


def fd_raw(row):
    """raw 20-D input -> raw 12-D output."""
    return model((row - xm) / xs) * ys + ym


def valve_jacobian(rows):
    """Central-difference dFD_dx[0:4] / d(valve), averaged over `rows`. Differentiable."""
    cols = []
    for c in range(4):
        p = rows.clone(); m = rows.clone()
        p[:, c] = p[:, c] + args.valve_eps
        m[:, c] = m[:, c] - args.valve_eps
        cols.append(((fd_raw(p)[:, 0:4] - fd_raw(m)[:, 0:4]) / (2 * args.valve_eps)).mean(0))
    return torch.stack(cols, dim=1)          # (4 out, 4 valve)


def report(tag):
    model.eval()
    with torch.no_grad():
        pv = fd_raw(Xvt)[:, 0:4]
        r2 = [float(1 - ((Yvt[:, j] - pv[:, j]) ** 2).sum()
                    / ((Yvt[:, j] - Yvt[:, j].mean()) ** 2).sum()) for j in range(4)]
        idx = torch.randint(0, len(Xt), (4000,), device=dev)
        J = valve_jacobian(Xt[idx])
    ratio = (torch.diag(J) / torch.diag(J_target_raw)).cpu().numpy()
    print(f"[{tag}] val R^2 = {np.round(r2, 4).tolist()}", flush=True)
    print(f"[{tag}] diag(J) = {np.round(torch.diag(J).cpu().numpy(), 6).tolist()}", flush=True)
    print(f"[{tag}] diag ratio vs data = {np.round(ratio, 2).tolist()}   (want ~1.0)", flush=True)
    model.train()
    return r2, J.detach().cpu().numpy(), ratio.tolist()


print("\n--- BEFORE ---", flush=True)
r2_0, J0, ratio0 = report("before")

opt = torch.optim.Adam(model.parameters(), lr=args.lr)
n = len(Xt)
for ep in range(args.epochs):
    perm = torch.randperm(n, device=dev)
    tot_m = tot_j = 0.0
    nb = 0
    for i in range(0, n, args.batch):
        b = perm[i:i + args.batch]
        opt.zero_grad()
        # normalized-space MSE, exactly the original objective
        pred_n = model((Xt[b] - xm) / xs)
        targ_n = (Yt[b] - ym) / ys.clamp(min=1e-8)
        loss_mse = torch.nn.functional.mse_loss(pred_n, targ_n)
        # Jacobian anchor: scale-free (relative to the data's own diagonal magnitudes)
        jb = perm[torch.randint(0, n, (args.jac_batch,), device=dev)]
        J = valve_jacobian(Xt[jb])
        scale = torch.diag(J_target_raw).abs().clamp(min=1e-9)
        loss_jac = (((J - J_target_raw) / scale.reshape(1, 4)) ** 2).mean()
        loss = loss_mse + args.lambda_jac * loss_jac
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_m += float(loss_mse); tot_j += float(loss_jac); nb += 1
    print(f"[FD] epoch {ep+1}/{args.epochs}  mse={tot_m/nb:.6f}  jac={tot_j/nb:.6f}", flush=True)

print("\n--- AFTER ---", flush=True)
r2_1, J1, ratio1 = report("after")

ck_out = dict(ck)
ck_out["model_state_dict"] = model.state_dict()
ck_out["jacobian_finetune"] = dict(
    source=str(args.src), lambda_jac=args.lambda_jac, epochs=args.epochs, lr=args.lr,
    valve_eps=args.valve_eps,
    target_jacobian=J_target_raw.cpu().numpy().tolist(),
    jacobian_before=J0.tolist(), jacobian_after=J1.tolist(),
    diag_ratio_before=ratio0, diag_ratio_after=ratio1,
    val_r2_before=r2_0, val_r2_after=r2_1,
    note="Anchors dFD/dvalve to the source-data OLS partials. One-step MSE alone leaves "
         "the valve sensitivity unconstrained (R^2 0.993 with an 18x gain error).",
)
torch.save(ck_out, ROOT / args.out)
print(f"\n[FD] wrote {ROOT / args.out}", flush=True)
