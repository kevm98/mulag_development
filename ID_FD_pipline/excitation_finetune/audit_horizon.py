#!/usr/bin/env python3
"""Section C: causal / horizon identifiability.

The previous audit's mistake was to test the valve on a ONE-STEP dx target. At dt = 0.01 s,
dx ~ xdot*dt, so current velocity explains dx almost perfectly and hides the valve. That
tests the TARGET, not the valve.

Here: for horizons h = 1..100 samples (0.01-1.00 s), compare matched models

    y_{t+h} = f(state history)              vs      y_{t+h} = f(state history, valve history)

and report Delta_R2(h) = R2_state+valve - R2_state_only, on RUN-WISE splits (never random
rows, never windows crossing a run boundary).

Targets, per cylinder: dP_{t+h}, Delta_dP, xd_{t+h}, Delta_xd, x_{t+h}, Delta_x.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from audit_lib import DP, DT, FF, HERE, U, X, XD, add_derivatives, build_masks, load, run_type

LAGS = [0, 1, 2, 3, 5, 8, 12, 20]        # 0 .. 0.20 s of history
HORIZONS = [1, 2, 5, 10, 20, 50, 100]    # 0.01 .. 1.00 s
STATE = X + XD + DP + FF                 # 16 signals
RIDGE = 1e-6

df = add_derivatives(load())
masks = build_masks(df)
df["_ok"] = masks["VALID_ACTIVE"] | masks["ZERO_INPUT_BASELINE"]
df["_type"] = df.groupby("run_id")["run_id"].transform(
    lambda s: run_type(df.loc[s.index]))


def build(run_df: pd.DataFrame, h: int):
    """Lagged design matrices for ONE run. Returns (S, V, targets, ok) or None."""
    n = len(run_df)
    maxlag = max(LAGS)
    if n <= maxlag + h:
        return None
    idx = np.arange(maxlag, n - h)          # t indices with full history AND t+h in-run
    S = np.concatenate([run_df[STATE].to_numpy()[idx - L] for L in LAGS], axis=1)
    V = np.concatenate([run_df[U].to_numpy()[idx - L] for L in LAGS], axis=1)
    tg = {}
    for i in range(1, 5):
        for nm, arr in (("dP", run_df[f"dP{i}"].to_numpy()),
                        ("xd", run_df[f"xd{i}"].to_numpy()),
                        ("x", run_df[f"x{i}"].to_numpy())):
            tg[f"{nm}{i}_abs"] = arr[idx + h]
            tg[f"{nm}{i}_delta"] = arr[idx + h] - arr[idx]
    ok = run_df["_ok"].to_numpy()[idx]
    return S, V, tg, ok, idx


def ridge_r2(Xtr, ytr, Xte, yte):
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]
    Xte = np.c_[Xte, np.ones(len(Xte))]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    G = A.T @ A + RIDGE * len(A) * np.eye(A.shape[1])
    w = np.linalg.solve(G, A.T @ ytr)
    p = B @ w
    sst = ((yte - ytr.mean()) ** 2).sum()
    return float(1.0 - ((yte - p) ** 2).sum() / max(sst, 1e-30))


# 4 parameter-condition groups: runs 1-6, 7-12, 13-18, 19-24. Hold out one group at a time.
GROUPS = {1: range(1, 7), 2: range(7, 13), 3: range(13, 19), 4: range(19, 25)}
run_group = {r: g for g, rs in GROUPS.items() for r in rs}

# use only runs with active valves for the identifiability test
active_runs = [int(r) for r, s in df.groupby("run_id") if run_type(s) != "CH5"]
print(f"[C] valve-active runs: {active_runs}", flush=True)

results = {}
for h in HORIZONS:
    per_run = {}
    for rid in active_runs:
        b = build(df[df.run_id == rid].reset_index(drop=True), h)
        if b is None:
            continue
        S, V, tg, ok, _ = b
        per_run[rid] = (S[ok], V[ok], {k: v[ok] for k, v in tg.items()})
    tgt_names = list(next(iter(per_run.values()))[2].keys())
    acc = {t: {"state": [], "state_valve": []} for t in tgt_names}
    for gh in GROUPS:                      # leave-one-group-out over runs
        tr = [r for r in per_run if run_group[r] != gh]
        te = [r for r in per_run if run_group[r] == gh]
        if not tr or not te:
            continue
        Str = np.vstack([per_run[r][0] for r in tr])
        Vtr = np.vstack([per_run[r][1] for r in tr])
        Ste = np.vstack([per_run[r][0] for r in te])
        Vte = np.vstack([per_run[r][1] for r in te])
        for t in tgt_names:
            ytr = np.concatenate([per_run[r][2][t] for r in tr])
            yte = np.concatenate([per_run[r][2][t] for r in te])
            acc[t]["state"].append(ridge_r2(Str, ytr, Ste, yte))
            acc[t]["state_valve"].append(
                ridge_r2(np.c_[Str, Vtr], ytr, np.c_[Ste, Vte], yte))
    results[h] = {t: dict(
        r2_state=float(np.mean(acc[t]["state"])),
        r2_state_valve=float(np.mean(acc[t]["state_valve"])),
        delta_r2=float(np.mean(acc[t]["state_valve"]) - np.mean(acc[t]["state"])),
    ) for t in tgt_names}
    print(f"[C] horizon h={h:3d} ({h*DT:.2f}s) done", flush=True)

# ---------------------------------------------------------------- report
print("\n================ Delta_R2 = R2(state+valve) - R2(state only) ================")
print("run-wise leave-one-group-out; positive = valve history carries information\n")
for kind in ("dP", "xd", "x"):
    for suf in ("delta", "abs"):
        print(f"--- target {kind}_{suf} ---")
        print("   h(s) |" + "".join(f"   cyl{i}  " for i in range(1, 5)))
        for h in HORIZONS:
            vals = [results[h][f"{kind}{i}_{suf}"]["delta_r2"] for i in range(1, 5)]
            print(f"  {h*DT:5.2f} |" + "".join(f" {v:+8.4f}" for v in vals))
        print()

(HERE / "open_loop_horizon_identifiability.json").write_text(json.dumps(dict(
    lags=LAGS, horizons=HORIZONS, dt=DT, state_signals=STATE,
    split="leave-one-parameter-group-out over runs (groups 1-6, 7-12, 13-18, 19-24)",
    note="Delta_R2 > 0 means valve history improves prediction beyond state history.",
    results={str(h): results[h] for h in HORIZONS},
), indent=2))
print(f"[OUT] open_loop_horizon_identifiability.json", flush=True)
