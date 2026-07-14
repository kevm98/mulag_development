#!/usr/bin/env python3
"""Offline half of the wiring audit: source-data and learned-FD 4x4 response matrices.

No Isaac needed.  Establishes the ground truth that the runtime must reproduce:

  source-data matrix : d(DeltaExtension_next) / d(ValveCmd), by OLS on the training set
                       (partialled out against the other 16 FD inputs)
  learned-FD matrix  : the same derivative, finite-differenced through the FD network
                       at the training-data mean state

If these two disagree, the FD checkpoint did not learn the data.
If they agree but the runtime composite disagrees with them, the runtime is miswired.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
FD_MAT = ROOT / "ID_FD_pipline/models/fd/FD_cylinder_data.mat"
FD_CKPT = ROOT / "ID_FD_pipline/models/fd/best_fd_cylinder_control.pt"
ID_CKPT = ROOT / "ID_FD_pipline/models/id/best_id_cylinder_control.pt"
OUT = ROOT / "outputs/lower_level_tracker"
OUT.mkdir(parents=True, exist_ok=True)

VALVE_EPS = 0.05  # identical perturbation used for every matrix

# --------------------------------------------------------------------------- #
# 1. source-data matrix
# --------------------------------------------------------------------------- #
with h5py.File(FD_MAT, "r") as f:
    X = np.array(f["X_train_fd"]).T.astype(np.float64)   # (N, 20)
    Y = np.array(f["Y_train_fd"]).T.astype(np.float64)   # (N, 12)
    st = np.array(f["TrainSampleTime"]).ravel()

print(f"[DATA] {X.shape[0]} samples, sample_time = {np.median(st):.4f} s "
      f"(min {st.min():.4f}, max {st.max():.4f})", flush=True)

# OLS with intercept: DeltaExtension_next(4) ~ all 20 inputs.  The 4x4 valve block is
# the partial derivative wrt valve, holding the state constant -- exactly what a
# finite-difference of the FD net computes.
A = np.c_[X, np.ones(len(X))]
coef, *_ = np.linalg.lstsq(A, Y[:, 0:4], rcond=None)     # (21, 4)
J_data = coef[0:4, :].T                                  # (4 out, 4 valve)

pred = A @ coef
r2 = 1.0 - ((Y[:, 0:4] - pred) ** 2).sum(0) / ((Y[:, 0:4] - Y[:, 0:4].mean(0)) ** 2).sum(0)
print(f"[DATA] linear-fit R^2 per output: {np.round(r2, 3).tolist()}", flush=True)

# --------------------------------------------------------------------------- #
# 2. learned-FD matrix (finite difference through the network)
# --------------------------------------------------------------------------- #
import sys
fd_dir = Path("/home/rrlab/rrlab_mulag_newobs_pruned_20260701/kevin_integration/new_observations_pipeline/Models_cylinder /fd_cylinder_space ")
sys.path.insert(0, str(fd_dir))
import importlib.util
spec = importlib.util.spec_from_file_location("fd_common", fd_dir / "fd_cylinder_common.py")
common = importlib.util.module_from_spec(spec)
sys.modules["fd_common"] = common  # dataclass() needs the module resolvable by name
spec.loader.exec_module(common)

model, ckpt, norm = common.load_model_from_checkpoint(FD_CKPT, device="cpu")
model.eval()
x_mean = torch.as_tensor(np.asarray(ckpt["x_mean"]).ravel(), dtype=torch.float32)
x_std = torch.as_tensor(np.asarray(ckpt["x_std"]).ravel(), dtype=torch.float32)
y_mean = torch.as_tensor(np.asarray(ckpt["y_mean"]).ravel(), dtype=torch.float32)
y_std = torch.as_tensor(np.asarray(ckpt["y_std"]).ravel(), dtype=torch.float32)


def fd_delta(row: torch.Tensor) -> torch.Tensor:
    """row: (B, 20) raw -> (B, 4) DeltaExtension_next raw."""
    xn = (row - x_mean) / x_std.clamp(min=1e-8)
    with torch.no_grad():
        yn = model(xn)
    return (yn * y_std + y_mean)[:, 0:4]


base = torch.as_tensor(X.mean(0), dtype=torch.float32).reshape(1, 20)
J_fd = np.zeros((4, 4))
for c in range(4):
    plus, minus = base.clone(), base.clone()
    plus[0, c] += VALVE_EPS
    minus[0, c] -= VALVE_EPS
    J_fd[:, c] = ((fd_delta(plus) - fd_delta(minus)) / (2 * VALVE_EPS)).numpy().ravel()

# --------------------------------------------------------------------------- #
def describe(name, J):
    sv = np.linalg.svd(J, compute_uv=False)
    diag = np.diag(J)
    off = J - np.diag(diag)
    return dict(
        name=name,
        matrix=J.tolist(),
        diag=diag.tolist(),
        singular_values=sv.tolist(),
        condition=float(sv[0] / max(sv[-1], 1e-15)),
        rank=int((sv > 1e-6 * sv[0]).sum()),
        diag_dominance=float(np.abs(diag).min() / max(np.abs(off).max(), 1e-15)),
        max_offdiag=float(np.abs(off).max()),
    )


def show(name, J):
    print(f"\n===== {name}  d(DeltaExtension)/d(ValveCmd) =====", flush=True)
    print("           v1         v2         v3         v4", flush=True)
    for i in range(4):
        print(f"  dx{i+1}  " + "".join(f"{J[i, j]:+11.6f}" for j in range(4)), flush=True)
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"  diag={np.round(np.diag(J), 6).tolist()}", flush=True)
    print(f"  singular values={np.round(sv, 6).tolist()}  cond={sv[0]/max(sv[-1],1e-15):.1f}", flush=True)


show("SOURCE DATA (OLS)", J_data)
show("LEARNED FD (finite diff)", J_fd)

# agreement between the two
num = np.abs(J_data - J_fd).max()
rel = num / max(np.abs(J_data).max(), 1e-15)
cos = float((J_data.ravel() @ J_fd.ravel()) /
            (np.linalg.norm(J_data) * np.linalg.norm(J_fd) + 1e-15))
sign_agree = float((np.sign(J_data) == np.sign(J_fd)).mean())
print(f"\n[COMPARE] max|data - fd| = {num:.6f}  (rel {rel:.2%})", flush=True)
print(f"[COMPARE] cosine similarity = {cos:+.4f}   sign agreement = {sign_agree:.0%}", flush=True)
print(f"[COMPARE] diag(data) = {np.round(np.diag(J_data), 6).tolist()}", flush=True)
print(f"[COMPARE] diag(fd)   = {np.round(np.diag(J_fd), 6).tolist()}", flush=True)

# --------------------------------------------------------------------------- #
# 3. ID feature-convention audit against the data
# --------------------------------------------------------------------------- #
id_ck = torch.load(ID_CKPT, map_location="cpu", weights_only=False)
ID_MAT = Path("/home/rrlab/rrlab_mulag_newobs_pruned_20260701/kevin_integration/"
              "new_observations_pipeline/Models_cylinder /id_cylinder_space /ID_cylinder_data.mat")
feat = {}
if ID_MAT.exists():
    with h5py.File(ID_MAT, "r") as f:
        Xi = np.array(f["X_train_id"]).T.astype(np.float64)
    TDX, TDXV, EV = Xi[:, 16:20], Xi[:, 20:24], Xi[:, 4:8]
    dt_model = 0.05
    dt_runtime = 1.0 / 60.0
    feat = dict(
        model_sample_time_s=dt_model,
        runtime_learned_model_dt_s=dt_runtime,
        corr_TDX_TDXV_in_data=[float(np.corrcoef(TDX[:, i], TDXV[:, i])[0, 1]) for i in range(4)],
        std_TDX=TDX.std(0).tolist(),
        std_TDXV=TDXV.std(0).tolist(),
        runtime_formula_rel_error=[
            float(np.abs((TDX[:, i] / dt_runtime - EV[:, i]) - TDXV[:, i]).mean() / TDXV[:, i].std())
            for i in range(4)
        ],
        runtime_formula_rel_error_at_model_dt=[
            float(np.abs((TDX[:, i] / dt_model - EV[:, i]) - TDXV[:, i]).mean() / TDXV[:, i].std())
            for i in range(4)
        ],
        note="data defines TDXV = xdot(k+1)-xdot(k) (a velocity INCREMENT). "
             "The runtime computes target_delta_x/learned_model_dt - xdot_current "
             "(a velocity ERROR). Different quantity, and built at 1/60 s not 0.05 s.",
    )
    print("\n===== ID FEATURE CONVENTION =====", flush=True)
    print(f"  data sample time      : {dt_model} s   runtime learned_model_dt: {dt_runtime:.5f} s", flush=True)
    print(f"  corr(TDX,TDXV) in data: {np.round(feat['corr_TDX_TDXV_in_data'], 4).tolist()}"
          "   <- ~0: they are INDEPENDENT quantities", flush=True)
    print(f"  runtime formula rel err vs data TDXV @1/60s : "
          f"{np.round(feat['runtime_formula_rel_error'], 3).tolist()}", flush=True)
    print(f"  runtime formula rel err vs data TDXV @0.05s : "
          f"{np.round(feat['runtime_formula_rel_error_at_model_dt'], 3).tolist()}", flush=True)

(OUT / "wiring_audit_offline.json").write_text(json.dumps(dict(
    valve_eps=VALVE_EPS,
    source_data=describe("source_data", J_data),
    learned_fd=describe("learned_fd", J_fd),
    data_fit_r2=r2.tolist(),
    agreement=dict(max_abs_diff=float(num), relative=float(rel),
                   cosine=cos, sign_agreement=sign_agree),
    id_feature_convention=feat,
), indent=2))
print(f"\n[OUT] {OUT / 'wiring_audit_offline.json'}", flush=True)
