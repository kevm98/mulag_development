#!/usr/bin/env python3
"""Ablation suite on the original FD MAT dataset (FD_cylinder_data.mat).

Question: how much of FD's fit comes from kinematic persistence (dx ~ xd*dt) vs the
valve command? Runs the ablations on the PRESERVED run split (936 train / 104 val runs),
one-step AND multi-step rollout. Never differences or rolls across a run boundary.

Feature groups in the 20-D input:  u=0:4  x=4:8  xd=8:12  dP=12:16  F=16:20
Output groups in the 12-D target:  dx=0:4  dxd=4:8  ddP=8:12

Ablations (ridge, alpha picked on an internal split of TRAIN runs, reported on VAL runs):
  full            all 20 features
  valve_zeroed    trained with valve, valve column zeroed at eval
  valve_removed   retrained without the 4 valve features
  valve_only      only the 4 valve features
  xd_only / dP_only / F_only / x_only
  hist_state      state lags {0,1,2,4,8}, no valve
  hist_valve      valve lags {0,1,2,4,8}, no state
  hist_both       both lag sets

Rollout (closed-loop, within-run): integrate predicted dx/dxd/ddP forward H steps with
u and F teacher-forced from data; report extension RMSE at H in {1,5,10,20} (0.05-1.0 s)
for full / valve_zeroed / valve_removed. If full ~= valve_zeroed at every H, the valve
carries no identifiable authority in this dataset either.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "ID_FD_pipline/models/fd/FD_cylinder_data.mat"
OUT = ROOT / "ID_FD_pipline/models/fd/mat_fd_ablation_results.json"

RUN_LEN = 440
LAGS = [0, 1, 2, 4, 8]
ALPHAS = [1e-8, 1e-6, 1e-4, 1e-2, 1e0]
HORIZONS = [1, 5, 10, 20]
G_IN = dict(u=slice(0, 4), x=slice(4, 8), xd=slice(8, 12), dP=slice(12, 16), F=slice(16, 20))
G_OUT = dict(dx=slice(0, 4), dxd=slice(4, 8), ddP=slice(8, 12))
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- load, keep run split
with h5py.File(MAT, "r") as f:
    Xtr = np.array(f["X_train_fd"]).T.astype(np.float64)
    Ytr = np.array(f["Y_train_fd"]).T.astype(np.float64)
    Xva = np.array(f["X_val_fd"]).T.astype(np.float64)
    Yva = np.array(f["Y_val_fd"]).T.astype(np.float64)
    rid_tr = np.array(f["TrainRunID"]).ravel().astype(int)
    tix_tr = np.array(f["TrainTimeIndex"]).ravel().astype(int)
    rid_va = np.array(f["ValRunID"]).ravel().astype(int)
    tix_va = np.array(f["ValTimeIndex"]).ravel().astype(int)
    dt = float(np.array(f["TrainSampleTime"]).ravel()[0])


def order(X, Y, rid, tix):
    o = np.lexsort((tix, rid))
    X, Y, rid, tix = X[o], Y[o], rid[o], tix[o]
    runs = np.unique(rid)
    assert len(X) == len(runs) * RUN_LEN, (len(X), len(runs))
    # verify contiguous time index within every run
    t = tix.reshape(len(runs), RUN_LEN)
    assert np.all(t == t[:, :1] + np.arange(RUN_LEN)), "time index not contiguous within runs"
    return X.reshape(len(runs), RUN_LEN, -1), Y.reshape(len(runs), RUN_LEN, -1), runs


Xtr3, Ytr3, runs_tr = order(Xtr, Ytr, rid_tr, tix_tr)
Xva3, Yva3, runs_va = order(Xva, Yva, rid_va, tix_va)
print(f"[MAT] dt={dt}s  train runs={len(runs_tr)}  val runs={len(runs_va)}  "
      f"run overlap={len(np.intersect1d(runs_tr, runs_va))}", flush=True)

# internal alpha-selection split: hold out the last 10% of TRAIN runs (never touch val)
n_hold = max(1, len(runs_tr) // 10)
fit3, hold3 = (Xtr3[:-n_hold], Ytr3[:-n_hold]), (Xtr3[-n_hold:], Ytr3[-n_hold:])


def flat(x3):
    return x3.reshape(-1, x3.shape[-1])


class Ridge:
    def fit(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        A = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        AtA, AtY = A.T @ A, A.T @ Y
        self.ws = {a: np.linalg.solve(AtA + a * len(A) * np.eye(A.shape[1]), AtY)
                   for a in ALPHAS}
        return self

    def predict(self, X, a):
        return np.c_[(X - self.mu) / self.sd, np.ones(len(X))] @ self.ws[a]

    def select(self, Xh, Yh):
        self.alpha = min(ALPHAS, key=lambda a: float(((Yh - self.predict(Xh, a)) ** 2).sum()))
        self.w = self.ws[self.alpha]
        return self


def r2_groups(Y, P):
    out = {}
    for g, sl in G_OUT.items():
        ss = 1 - ((Y[:, sl] - P[:, sl]) ** 2).sum(0) / ((Y[:, sl] - Y[:, sl].mean(0)) ** 2).sum(0)
        out[g] = float(np.mean(ss))
    return out


def run_ablation(cols=None, zero_at_eval=None, lag_groups=None):
    """cols: input column indices to keep. zero_at_eval: cols zeroed only at predict time.
    lag_groups: list of slices to build lagged features from (uses LAGS, within-run)."""
    if lag_groups is not None:
        def design(X3):
            m = max(LAGS)
            blocks = [X3[:, m - L:RUN_LEN - L, sl] for sl in lag_groups for L in LAGS]
            return np.concatenate(blocks, axis=2).reshape(-1, sum(
                (sl.stop - sl.start) for sl in lag_groups) * len(LAGS))

        def target(Y3):
            return Y3[:, max(LAGS):, :].reshape(-1, Y3.shape[-1])
        Xf, Yf = design(fit3[0]), target(fit3[1])
        Xh, Yh = design(hold3[0]), target(hold3[1])
        Xv, Yv = design(Xva3), target(Yva3)
    else:
        cols = np.arange(20) if cols is None else np.asarray(cols)
        Xf, Yf = flat(fit3[0])[:, cols], flat(fit3[1])
        Xh, Yh = flat(hold3[0])[:, cols], flat(hold3[1])
        Xv, Yv = flat(Xva3)[:, cols], flat(Yva3)
        if zero_at_eval is not None:
            Xv = Xv.copy()
            Xv[:, np.isin(cols, zero_at_eval)] = 0.0
    m = Ridge().fit(Xf, Yf).select(Xh, Yh)
    return r2_groups(Yv, m.predict(Xv, m.alpha)), m


IDX = {g: np.arange(sl.start, sl.stop) for g, sl in G_IN.items()}
state_cols = np.concatenate([IDX["x"], IDX["xd"], IDX["dP"], IDX["F"]])

ABL = {
    "full": dict(cols=None),
    "valve_zeroed": dict(cols=None, zero_at_eval=IDX["u"]),
    "valve_removed": dict(cols=state_cols),
    "valve_only": dict(cols=IDX["u"]),
    "x_only": dict(cols=IDX["x"]),
    "xd_only": dict(cols=IDX["xd"]),
    "dP_only": dict(cols=IDX["dP"]),
    "F_only": dict(cols=IDX["F"]),
    "hist_state": dict(lag_groups=[G_IN["x"], G_IN["xd"], G_IN["dP"], G_IN["F"]]),
    "hist_valve": dict(lag_groups=[G_IN["u"]]),
    "hist_both": dict(lag_groups=[G_IN["u"], G_IN["x"], G_IN["xd"], G_IN["dP"], G_IN["F"]]),
}

results, models = {}, {}
print("\n================ one-step val R^2 (mean over 4 cylinders) ================")
print(f"{'ablation':16s} {'dx':>9s} {'dxd':>9s} {'ddP':>9s}   alpha")
for name, kw in ABL.items():
    r2, m = run_ablation(**kw)
    results[name] = dict(r2=r2, alpha=m.alpha)
    models[name] = m
    print(f"{name:16s} {r2['dx']:+9.4f} {r2['dxd']:+9.4f} {r2['ddP']:+9.4f}   {m.alpha:g}",
          flush=True)

# ---------------------------------------------------------------- multi-step rollout
def rollout(model, cols, H, zero_valve=False):
    """Closed-loop within-run: x/xd/dP evolve by prediction; u,F teacher-forced."""
    S = Xva3.copy()                       # (R, T, 20) working state
    n_steps = RUN_LEN - H
    errs = []
    cur = S[:, :n_steps, :].copy()        # rollout start states, vectorized over (R, t0)
    for k in range(H):
        feats = cur.reshape(-1, 20)[:, cols]
        if zero_valve:
            feats = feats.copy()
            feats[:, np.isin(cols, IDX["u"])] = 0.0
        d = model.predict(feats, model.alpha).reshape(cur.shape[0], n_steps, 12)
        cur[..., G_IN["x"]] += d[..., G_OUT["dx"]]
        cur[..., G_IN["xd"]] += d[..., G_OUT["dxd"]]
        cur[..., G_IN["dP"]] += d[..., G_OUT["ddP"]]
        if k < H - 1:                     # teacher-force exogenous u, F from data at t0+k+1
            nxt = S[:, k + 1:k + 1 + n_steps, :]
            cur[..., G_IN["u"]] = nxt[..., G_IN["u"]]
            cur[..., G_IN["F"]] = nxt[..., G_IN["F"]]
    truth = S[:, H:H + n_steps, G_IN["x"]]
    return float(np.sqrt(((cur[..., G_IN["x"]] - truth) ** 2).mean()))


print("\n================ closed-loop extension RMSE [m] at horizon H ================")
print(f"{'H (s)':>8s} {'full':>12s} {'valve_zeroed':>12s} {'valve_removed':>13s}")
roll = {}
for H in HORIZONS:
    r_full = rollout(models["full"], np.arange(20), H)
    r_vz = rollout(models["full"], np.arange(20), H, zero_valve=True)
    r_vr = rollout(models["valve_removed"], state_cols, H)
    roll[H] = dict(full=r_full, valve_zeroed=r_vz, valve_removed=r_vr)
    print(f"{H * dt:8.2f} {r_full:12.6f} {r_vz:12.6f} {r_vr:13.6f}", flush=True)

OUT.write_text(json.dumps(dict(
    dt_s=dt, lags=LAGS, alphas=ALPHAS, horizons=HORIZONS,
    split="official MAT run split preserved; alpha chosen on last 10% of train runs",
    one_step=results if all(isinstance(v, dict) for v in results.values()) else results,
    rollout_extension_rmse_m={str(h): roll[h] for h in HORIZONS},
), indent=2, default=float))
print(f"\n[OUT] {OUT}", flush=True)
