#!/usr/bin/env python3
"""Build FD fine-tuning data at FD's native 0.05 s timebase from the strict open-loop set.

Source: Excitation_FD_Strict.csv -- 4 runs x 6400 samples @ 0.01 s, MIMO PRBS on all four
valves (command rank 4), four plant draws (Nominal / Random Hydraulics / Random Mechanics /
Fully Randomized). Verified: 4/4 distinct state trajectories, all 12 target columns match
within-run k+1-k to machine precision, valve->dP causal (R^2 0.80-0.85 from valve history
alone).

FD semantics (from its checkpoint): input  [ValveCmd, Extension, ExtensionVelocity, dP, Force] x4
                                   output [DeltaExtension, DeltaExtensionVelocity, DeltadP] x4
                                   sample_time_s = 0.05

The excitation is logged at 0.01 s, so a 0.05 s transition spans 5 raw samples over which the
valve is NOT constant. At runtime the valve IS held for the whole control step, so the input
that matches deployment is the held command. We export both conventions and let the trainer
pick:  u_hold = u(k)   and   u_mean = mean(u(k..k+4)).

Split design. Each run is ONE continuous 64 s sweep through the workspace, so a last-20%
temporal split is an extrapolation to an unseen pose/energy regime (measured: val target std
1.3-2.3x train, extension mean shifted up to 0.19 m) -- it neither trains nor scores the model
fairly. Instead each run is cut into NBLOCK contiguous blocks and every VAL_EVERY-th block is
held out, with a GUARD of DEC samples dropped on each side of a boundary so no 0.05 s window
straddles the train/val cut. Blocks never cross a run boundary; samples are kept only where
valid==1 at both k and k+DEC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ID_FD_pipline/training_data/Excitation_FD_Strict.csv"
OUT = ROOT / "ID_FD_pipline/models/fd/strict_fd_transitions_0p05.npz"

DEC = 5                     # 0.01 s * 5 = 0.05 s
NBLOCK = 20                 # contiguous blocks per run
VAL_EVERY = 5               # every 5th block is validation -> 20% val, same regime as train
GUARD = DEC                 # samples dropped either side of a block boundary

U = [f"ValveCmd_{i}" for i in range(1, 5)]
X = [f"Extension_{i}" for i in range(1, 5)]
XD = [f"ExtensionVelocity_{i}" for i in range(1, 5)]
DP = [f"dP_{i}" for i in range(1, 5)]
FF = [f"Force_{i}" for i in range(1, 5)]

df = pd.read_csv(SRC)
if len(df.columns) != 49:
    raise SystemExit(f"[FATAL] expected 49 columns, got {len(df.columns)}")

Xtr, Ytr, Xva, Yva, Rtr, Rva, Utr_m, Uva_m = [], [], [], [], [], [], [], []
for rid, s in df.groupby("run_id", sort=True):
    s = s.reset_index(drop=True)
    n = len(s)
    ok = s["valid"].to_numpy().astype(bool)
    idx = np.arange(0, n - DEC)
    idx = idx[ok[idx] & ok[idx + DEC]]

    u_hold = s[U].to_numpy()[idx]
    u_mean = np.stack([s[U].to_numpy()[idx + j] for j in range(DEC)]).mean(0)
    xin = np.c_[u_hold, s[X].to_numpy()[idx], s[XD].to_numpy()[idx],
                s[DP].to_numpy()[idx], s[FF].to_numpy()[idx]]
    yout = np.c_[s[X].to_numpy()[idx + DEC] - s[X].to_numpy()[idx],
                 s[XD].to_numpy()[idx + DEC] - s[XD].to_numpy()[idx],
                 s[DP].to_numpy()[idx + DEC] - s[DP].to_numpy()[idx]]

    # blocked-interleaved split, guarded so no 0.05 s window straddles the cut
    edges = np.linspace(0, n - DEC, NBLOCK + 1).astype(int)
    blk = np.searchsorted(edges, idx, side="right") - 1
    is_val = (blk % VAL_EVERY) == (VAL_EVERY - 1)
    near = np.zeros(len(idx), bool)
    for e in edges[1:-1]:
        near |= np.abs(idx - e) < GUARD
    tr_m, va_m = (~is_val) & (~near), is_val & (~near)

    Xtr.append(xin[tr_m]); Ytr.append(yout[tr_m]); Utr_m.append(u_mean[tr_m])
    Xva.append(xin[va_m]); Yva.append(yout[va_m]); Uva_m.append(u_mean[va_m])
    Rtr.append(np.full(tr_m.sum(), rid)); Rva.append(np.full(va_m.sum(), rid))
    print(f"[run {rid}] {len(idx)} transitions -> train {tr_m.sum()} / val {va_m.sum()} "
          f"(guard-dropped {int(near.sum())})", flush=True)

pack = lambda L: np.vstack(L).astype(np.float32)
cat = lambda L: np.concatenate(L).astype(np.int32)
Xtr, Ytr, Xva, Yva = pack(Xtr), pack(Ytr), pack(Xva), pack(Yva)
Utr_m, Uva_m = pack(Utr_m), pack(Uva_m)

# valve Jacobian target: OLS partials on the TRAIN block only (never on val)
A = np.c_[Xtr.astype(np.float64), np.ones(len(Xtr))]
coef, *_ = np.linalg.lstsq(A, Ytr.astype(np.float64), rcond=None)
J_valve = coef[0:4, :].T                            # (12 out, 4 valve)
print(f"\n[J] diag d(DeltaExtension)/d(valve) from TRAIN = "
      f"{np.round(np.diag(J_valve[0:4]), 6).tolist()}")

np.savez_compressed(
    OUT, X_train=Xtr, Y_train=Ytr, X_val=Xva, Y_val=Yva,
    run_train=cat(Rtr), run_val=cat(Rva), u_mean_train=Utr_m, u_mean_val=Uva_m,
    J_valve_train=J_valve.astype(np.float64), dt=0.05, decimation=DEC,
    input_names=np.array(U + X + XD + DP + FF),
    output_names=np.array([f"DeltaExtension_{i}" for i in range(1, 5)]
                          + [f"DeltaExtensionVelocity_{i}" for i in range(1, 5)]
                          + [f"DeltadP_{i}" for i in range(1, 5)]),
    note=np.array("temporal 80/20 within run; valid-gated at k and k+5; u=held command"))
print(f"\n[OUT] {OUT}  train {Xtr.shape} val {Xva.shape}")
