#!/usr/bin/env python3
"""Valve identifiability on the STRICT open-loop excitation dataset (4 runs x 6400, dt=0.01s).

The valve sequence is identical across runs by design (one excitation, four plant draws), so
identifiability is tested WITHIN each run on a temporal split: first 70% train, next 15% val
(picks the ridge alpha), last 15% test (reported). Nothing is ever differenced, lagged or
windowed across a run boundary.

For horizons h (samples): compare matched ridge models
    y_{t+h} = f(state history)          vs      y_{t+h} = f(state history, valve history)
and report Delta_R2 = R2(state+valve) - R2(state only) on the held-out test block.

POSITIVE CONTROLS are mandatory: an unregularised version of this test once produced a false
null because its own controls failed too. Controls here:
    ddP <- xd   (physically causal)     xd <- dP   (physically causal)
If a control does not come out positive, the detector is broken and the valve verdict is void.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = (HERE.parent / "Fine_tune_dataset-20260714T085628Z-1-001" / "Fine_tune_dataset"
       / "Excitation_FD_Strict.csv")
OUT = HERE / "strict_horizon_identifiability.json"

U = [f"ValveCmd_{i}" for i in range(1, 5)]
X = [f"Extension_{i}" for i in range(1, 5)]
XD = [f"ExtensionVelocity_{i}" for i in range(1, 5)]
DP = [f"dP_{i}" for i in range(1, 5)]
FF = [f"Force_{i}" for i in range(1, 5)]
STATE = X + XD + DP + FF

LAGS = [0, 1, 2, 3, 5, 8, 12, 20]                  # 0 .. 0.20 s
HORIZONS = [1, 2, 5, 10, 20, 50, 100]              # 0.01 .. 1.00 s
ALPHAS = [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]
DT = 0.01

df = pd.read_csv(CSV)
print(f"[strict] rows={len(df)} runs={df.run_id.nunique()} valid={df.valid.mean():.4f}",
      flush=True)


def ridge_fit_select(Xtr, ytr, Xva, yva):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    B = np.c_[(Xva - mu) / sd, np.ones(len(Xva))]
    AtA, Aty = A.T @ A, A.T @ ytr
    best, bw = None, None
    for a in ALPHAS:
        w = np.linalg.solve(AtA + a * len(A) * np.eye(A.shape[1]), Aty)
        e = float(((yva - B @ w) ** 2).sum())
        if best is None or e < best:
            best, bw, ba = e, w, a
    return (mu, sd, bw, ba)


def ridge_r2(m, Xte, yte, ymean):
    mu, sd, w, _ = m
    p = np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ w
    sst = ((yte - ymean) ** 2).sum()
    return float(1.0 - ((yte - p) ** 2).sum() / max(sst, 1e-30))


def blocks(run_df, h, feat_cols):
    """Lagged design within ONE run. Returns dict of feature blocks + targets, temporal split."""
    n = len(run_df)
    m = max(LAGS)
    idx = np.arange(m, n - h)
    ok = run_df["valid"].to_numpy().astype(bool)
    keep = ok[idx] & ok[idx + h]
    idx = idx[keep]

    def lag(cols):
        return np.concatenate([run_df[cols].to_numpy()[idx - L] for L in LAGS], axis=1)

    F_ = {k: lag(v) for k, v in feat_cols.items()}
    tg = {}
    for i in range(1, 5):
        for nm, col in (("dP", f"dP_{i}"), ("xd", f"ExtensionVelocity_{i}"),
                        ("x", f"Extension_{i}")):
            a = run_df[col].to_numpy()
            tg[f"{nm}{i}_abs"] = a[idx + h]
            tg[f"{nm}{i}_delta"] = a[idx + h] - a[idx]
    n_ = len(idx)
    tr, va = slice(0, int(0.70 * n_)), slice(int(0.70 * n_), int(0.85 * n_))
    te = slice(int(0.85 * n_), n_)
    return F_, tg, tr, va, te


def compare(run_df, h, base, extra, targets):
    """Delta_R2 from adding `extra` features to `base`, per target, within one run."""
    F_, tg, tr, va, te = blocks(run_df, h, {**base, **extra})
    B = np.concatenate([F_[k] for k in base], axis=1)
    E = np.concatenate([F_[k] for k in list(base) + list(extra)], axis=1)
    out = {}
    for t in targets:
        y = tg[t]
        ym = y[tr].mean()
        m0 = ridge_fit_select(B[tr], y[tr], B[va], y[va])
        m1 = ridge_fit_select(E[tr], y[tr], E[va], y[va])
        r0 = ridge_r2(m0, B[te], y[te], ym)
        r1 = ridge_r2(m1, E[te], y[te], ym)
        out[t] = dict(r2_base=r0, r2_ext=r1, delta_r2=r1 - r0)
    return out


TGTS = [f"{k}{i}_{s}" for k in ("dP", "xd", "x") for i in range(1, 5) for s in ("delta", "abs")]
runs = {int(r): s.reset_index(drop=True) for r, s in df.groupby("run_id")}

# ------------------------------------------------------------------ positive controls (h=1)
print("\n=== POSITIVE CONTROLS (h=1, must be positive or the test is void) ===", flush=True)
ctrl = {}
for name, base, extra, tg in [
    ("ddP <- xd", {"x": X, "dP": DP, "F": FF}, {"xd": XD}, [f"dP{i}_delta" for i in range(1, 5)]),
    ("xd  <- dP", {"x": X, "xd": XD, "F": FF}, {"dP": DP}, [f"xd{i}_delta" for i in range(1, 5)]),
]:
    per = [compare(runs[r], 1, base, extra, tg) for r in runs]
    d = {t: float(np.mean([p[t]["delta_r2"] for p in per])) for t in tg}
    ctrl[name] = d
    print(f"  {name}:  " + "  ".join(f"{t.split('_')[0]}={v:+.4f}" for t, v in d.items()),
          flush=True)
ok = all(v > 0 for d in ctrl.values() for v in d.values())
print(f"  controls {'PASS' if ok else 'FAIL -- verdict void'}", flush=True)

# ------------------------------------------------------------------ valve, per horizon
print("\n=== Delta_R2 from adding VALVE history to STATE history (per-run, test block) ===",
      flush=True)
res = {}
for h in HORIZONS:
    per = {r: compare(runs[r], h, {"s": STATE}, {"u": U}, TGTS) for r in runs}
    res[h] = {t: dict(
        delta_r2=float(np.mean([per[r][t]["delta_r2"] for r in runs])),
        delta_r2_min=float(np.min([per[r][t]["delta_r2"] for r in runs])),
        r2_state=float(np.mean([per[r][t]["r2_base"] for r in runs])),
        r2_state_valve=float(np.mean([per[r][t]["r2_ext"] for r in runs])),
    ) for t in TGTS}
    print(f"[C] h={h:3d} ({h*DT:.2f}s) done", flush=True)

for kind in ("dP", "xd", "x"):
    for suf in ("delta", "abs"):
        print(f"\n--- target {kind}_{suf}:  mean Delta_R2 over 4 runs ---")
        print("   h(s) |" + "".join(f"   cyl{i}  " for i in range(1, 5)))
        for h in HORIZONS:
            v = [res[h][f"{kind}{i}_{suf}"]["delta_r2"] for i in range(1, 5)]
            print(f"  {h*DT:5.2f} |" + "".join(f" {x:+8.4f}" for x in v))

OUT.write_text(json.dumps(dict(
    dataset=CSV.name, runs=sorted(runs), lags=LAGS, horizons=HORIZONS, dt=DT,
    split="within-run temporal 70/15/15; alpha on val block; R2 on test block",
    positive_controls=ctrl, controls_pass=bool(ok),
    results={str(h): res[h] for h in HORIZONS},
), indent=2))
print(f"\n[OUT] {OUT}", flush=True)
