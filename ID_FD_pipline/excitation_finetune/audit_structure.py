#!/usr/bin/env python3
"""Section A + B: structural audit, signal-quality audit, sample masks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from audit_lib import (CSV, COLUMNS, DDP, DP, DT, FF, HERE, Q, QD, U, X, XD,
                       Thresholds, add_derivatives, build_masks, load, run_type)

df = load()
print(f"[A] rows={len(df)}  cols={df.shape[1]}  runs={df['run_id'].nunique()}", flush=True)

# ---------------------------------------------------------------- A. structure
issues = []
per_run = []
for rid, sub in df.groupby("run_id", sort=True):
    t = sub["t"].to_numpy()
    dt = np.diff(t)
    rt = run_type(sub)
    act = [c for c in U if float(sub[c].abs().max()) > 1e-9]
    sat = {c: float((sub[c].abs() >= 0.999).mean()) for c in U + ["u5"]}
    row = dict(
        run_id=int(rid), n=len(sub), type=rt,
        mode=int(sub["mode"].iloc[0]), exc_ch=int(sub["exc_ch"].iloc[0]),
        draw_id=int(sub["draw_id"].iloc[0]),
        t_min=float(t.min()), t_max=float(t.max()),
        dt_min=float(dt.min()) if len(dt) else np.nan,
        dt_max=float(dt.max()) if len(dt) else np.nan,
        monotonic=bool(np.all(dt > 0)) if len(dt) else True,
        dup_t=int(len(t) - len(np.unique(t))),
        active_valves=",".join(act) if act else "none",
        u5_active=bool(float(sub["u5"].abs().max()) > 1e-9),
        valid_frac=float(sub["valid"].mean()),
        **{f"sat_{k}": v for k, v in sat.items()},
        **{f"u{i}_absmax": float(sub[f"u{i}"].abs().max()) for i in range(1, 5)},
    )
    per_run.append(row)
    if not row["monotonic"]:
        issues.append(f"run {rid}: time not strictly increasing")
    if row["dup_t"]:
        issues.append(f"run {rid}: {row['dup_t']} duplicate timestamps")
    if len(dt) and (abs(row["dt_min"] - DT) > 1e-9 or abs(row["dt_max"] - DT) > 1e-9):
        issues.append(f"run {rid}: dt not constant ({row['dt_min']}..{row['dt_max']})")

runs = pd.DataFrame(per_run)
print(f"[A] dt constant at {DT}s across all runs: "
      f"{bool(np.allclose(runs.dt_min.dropna(), DT) and np.allclose(runs.dt_max.dropna(), DT))}",
      flush=True)
print(f"[A] samples/run unique: {sorted(runs.n.unique().tolist())}", flush=True)
print(f"[A] NaN/Inf in file: {int(df.isna().sum().sum())} / "
      f"{int(np.isinf(df.select_dtypes('number')).sum().sum())}", flush=True)
print(f"[A] run types:\n{runs.groupby('type').run_id.apply(list).to_string()}", flush=True)
print(f"[A] mode values: {sorted(df['mode'].unique().tolist())}   "
      f"exc_ch values: {sorted(df['exc_ch'].unique().tolist())}", flush=True)
print(f"[A] valid column: unique={sorted(df['valid'].unique().tolist())}", flush=True)

# cross-check: does exc_ch agree with the channel that is actually excited?
mism = []
for _, r in runs.iterrows():
    ch = r["exc_ch"]
    if r["type"].startswith("SISO_"):
        want = int(r["type"].split("u")[1])
        if ch != want:
            mism.append((int(r.run_id), int(ch), want))
print(f"[A] SISO exc_ch vs actually-excited channel mismatches: {mism if mism else 'none'}",
      flush=True)

# ddP identity check
dfd = add_derivatives(df)
g = dfd.groupby("run_id", sort=False)
err = []
for i in range(1, 5):
    nxt = g[f"dP{i}"].shift(-1)
    e = (nxt - dfd[f"dP{i}"]) - dfd[f"ddP{i}"]
    err.append(float(np.nanmax(np.abs(e[~dfd["last_of_run"]]))))
print(f"[A] max |ddP_i - (dP_i(k+1)-dP_i(k))| within runs = {err}", flush=True)

if issues:
    print("[A] ISSUES:", *issues, sep="\n   ")
else:
    print("[A] no structural issues", flush=True)

# ---------------------------------------------------------------- B. signal quality
SIGNALS = U + ["u5"] + X + XD + DP + FF + Q + QD
qs = [0.001, 0.01, 0.5, 0.99, 0.999]
rows = []
for rid, sub in dfd.groupby("run_id", sort=True):
    for c in SIGNALS:
        v = sub[c].to_numpy()
        d = np.diff(v)
        rows.append(dict(
            run_id=int(rid), signal=c,
            min=float(v.min()), max=float(v.max()), median=float(np.median(v)),
            std=float(v.std()),
            **{f"q{int(q*1000)}": float(np.quantile(v, q)) for q in qs},
            max_abs_jump=float(np.abs(d).max()) if len(d) else 0.0,
            stuck_frac=float((np.abs(d) < 1e-12).mean()) if len(d) else 1.0,
        ))
sig = pd.DataFrame(rows)
sig.to_csv(HERE / "open_loop_signal_stats.csv", index=False)

masks = build_masks(dfd)
counts = {r: int(masks[r].sum()) for r in masks.columns}
print("\n[B] mask counts (samples):", flush=True)
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"    {k:32s} {v:7d}  ({v/len(dfd)*100:5.2f}%)", flush=True)

out = pd.concat([dfd[["run_id", "t"]], masks.astype(np.int8)], axis=1)
out.to_csv(HERE / "open_loop_sample_masks.csv", index=False)

# per-run mask summary joined into the run table
for r in masks.columns:
    runs[f"n_{r}"] = masks.groupby(dfd["run_id"]).sum()[r].values
runs.to_csv(HERE / "open_loop_run_summary.csv", index=False)

schema = dict(
    file=CSV.name, rows=int(len(df)), columns=int(df.shape[1]),
    runs=int(df["run_id"].nunique()), samples_per_run=sorted(runs.n.unique().tolist()),
    dt_s=DT, time_range_s=[float(df["t"].min()), float(df["t"].max())],
    column_mapping={f"DATA_{i+1}": c for i, c in enumerate(COLUMNS)},
    derived=["dx_i(k)=x_i(k+1)-x_i(k)", "dxd_i(k)=xd_i(k+1)-xd_i(k)",
             "computed WITHIN run only; last sample of each run is NaN"],
    ddP_identity_max_err=err,
    run_types={str(int(r.run_id)): r.type for _, r in runs.iterrows()},
    mask_reason_codes=list(masks.columns),
    mask_counts=counts,
    structural_issues=issues,
)
(HERE / "open_loop_csv_schema.json").write_text(json.dumps(schema, indent=2))
print(f"\n[OUT] open_loop_csv_schema.json, open_loop_run_summary.csv, "
      f"open_loop_sample_masks.csv, open_loop_signal_stats.csv", flush=True)
