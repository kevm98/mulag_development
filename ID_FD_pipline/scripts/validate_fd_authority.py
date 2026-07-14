#!/usr/bin/env python3
"""Does the FD actually have the plant's VALVE AUTHORITY? (closed-loop, not one-step)

Why the one-step Jacobian is the wrong instrument. dP is a MEDIATOR of the valve's effect:

        u  ->  dP  ->  force  ->  acceleration  ->  motion
         \______________ direct ______________/

FD takes dP as an *input*, so d(FD)/du at fixed dP measures only the small DIRECT arm and
blocks the dominant path through pressure. Every "valve gain" number computed that way -- the
old 9-19x, the per-quartile table, the OLS partial -- conditions on a mediator and is not the
control-relevant gain. The quantity that matters is the TOTAL effect, which only appears once
dP is free to respond, i.e. over a multi-step rollout in which FD integrates its own DeltadP.

So: roll FD forward H steps from real states, integrating x/xd/dP from its own predictions with
valve and Force teacher-forced from data, and compare against the plant.

  fd            trained FD
  fd_valve0     same FD, valve forced to 0 during the rollout  <- if this is as good as `fd`,
                                                                  the FD has NO valve authority
  hold          xd frozen at its initial value (constant-velocity dead reckoning)
  zero          no motion at all

Authority is real only if  err(fd) << err(fd_valve0).  That comparison cannot be faked by a
model that merely extrapolates kinematics, which is exactly how the old FD scored R^2 0.99.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
FDD = ROOT / "ID_FD_pipline/models/fd"
CSV = (ROOT / "ID_FD_pipline/Fine_tune_dataset-20260714T085628Z-1-001/Fine_tune_dataset"
       / "Excitation_FD_Strict.csv")
FD_DIR = Path("/home/rrlab/rrlab_mulag_newobs_pruned_20260701/kevin_integration/"
              "new_observations_pipeline/Models_cylinder /fd_cylinder_space ")

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=[str(FDD / "best_fd_cylinder_control.pt"),
                                               str(FDD / "best_fd_cylinder_strict.pt")])
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
sys.path.insert(0, str(FD_DIR))
spec = importlib.util.spec_from_file_location("fd_common", FD_DIR / "fd_cylinder_common.py")
common = importlib.util.module_from_spec(spec)
sys.modules["fd_common"] = common
spec.loader.exec_module(common)

DEC, NBLOCK, VAL_EVERY = 5, 20, 5
S_U, S_X, S_XD, S_DP, S_F = slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16), slice(16, 20)
O_DX, O_DXD, O_DDP = slice(0, 4), slice(4, 8), slice(8, 12)
HS = [1, 2, 5, 10, 20]
U = [f"ValveCmd_{i}" for i in range(1, 5)]
XC = [f"Extension_{i}" for i in range(1, 5)]
XDC = [f"ExtensionVelocity_{i}" for i in range(1, 5)]
DPC = [f"dP_{i}" for i in range(1, 5)]
FC = [f"Force_{i}" for i in range(1, 5)]

# held-out (validation) chains only -- same blocks the trainer never saw
df = pd.read_csv(CSV)
chains = []
for rid, s in df.groupby("run_id", sort=True):
    s = s.reset_index(drop=True)
    edges = np.linspace(0, len(s) - DEC, NBLOCK + 1).astype(int)
    for bi in range(NBLOCK):
        if bi % VAL_EVERY != VAL_EVERY - 1:
            continue
        seg = s.iloc[edges[bi]:edges[bi + 1]]
        if not seg["valid"].all():
            continue
        for ph in range(DEC):
            arr = np.c_[seg[U].to_numpy()[ph::DEC], seg[XC].to_numpy()[ph::DEC],
                        seg[XDC].to_numpy()[ph::DEC], seg[DPC].to_numpy()[ph::DEC],
                        seg[FC].to_numpy()[ph::DEC]].astype(np.float32)
            if len(arr) > max(HS) + 1:
                chains.append(torch.tensor(arr, device=dev))
print(f"[val] {len(chains)} held-out chains of length ~{len(chains[0])} @0.05s", flush=True)


def load(path):
    m, ck, _ = common.load_model_from_checkpoint(Path(path), device=dev)
    m.eval()
    T = lambda v: torch.tensor(np.asarray(v).ravel(), dtype=torch.float32, device=dev)
    xm, xs = T(ck["x_mean"]), T(ck["x_std"]).clamp(min=1e-8)
    ym, ys = T(ck["y_mean"]), T(ck["y_std"]).clamp(min=1e-8)
    return lambda x: m((x - xm) / xs) * ys + ym


@torch.no_grad()
def rollout(fd, H, valve0=False):
    """Extension RMSE at horizon H, plus the plant's own motion over H for reference."""
    e, mot = [], []
    for ch in chains:
        n = len(ch) - H
        cur = ch[:n].clone()
        if valve0:
            cur[:, S_U] = 0.0
        for k in range(H):
            d = fd(cur)
            cur[:, S_X] += d[:, O_DX]
            cur[:, S_XD] += d[:, O_DXD]
            cur[:, S_DP] += d[:, O_DDP]
            nx = ch[k + 1:k + 1 + n]
            cur[:, S_U] = 0.0 if valve0 else nx[:, S_U]
            cur[:, S_F] = nx[:, S_F]
        tgt = ch[H:H + n][:, S_X]
        e.append(((cur[:, S_X] - tgt) ** 2).mean(0).cpu().numpy())
        mot.append(((ch[:n][:, S_X] - tgt) ** 2).mean(0).cpu().numpy())
    return np.sqrt(np.mean(e, 0)), np.sqrt(np.mean(mot, 0))


@torch.no_grad()
def baselines(H):
    e_hold, e_zero = [], []
    for ch in chains:
        n = len(ch) - H
        tgt = ch[H:H + n][:, S_X]
        x0, v0 = ch[:n][:, S_X], ch[:n][:, S_XD]
        e_hold.append((((x0 + v0 * 0.05 * H) - tgt) ** 2).mean(0).cpu().numpy())
        e_zero.append(((x0 - tgt) ** 2).mean(0).cpu().numpy())
    return np.sqrt(np.mean(e_hold, 0)), np.sqrt(np.mean(e_zero, 0))


rows = {}
for path in a.models:
    fd = load(path)
    name = Path(path).stem
    rows[name] = {}
    print(f"\n================ {name} ================", flush=True)
    print(f"{'H':>4s} {'t[s]':>6s} | {'fd':>9s} {'fd_valve0':>10s} {'hold':>9s} {'zero':>9s} "
          f"| {'authority':>9s}", flush=True)
    for H in HS:
        r_fd, motion = rollout(fd, H)
        r_v0, _ = rollout(fd, H, valve0=True)
        r_hold, r_zero = baselines(H)
        f, v0, ho, ze = (float(r_fd.mean()), float(r_v0.mean()),
                         float(r_hold.mean()), float(r_zero.mean()))
        auth = v0 / max(f, 1e-12)
        rows[name][H] = dict(fd=f, fd_valve0=v0, hold=ho, zero=ze, authority=auth,
                             motion=float(motion.mean()),
                             fd_per_cyl=r_fd.tolist(), valve0_per_cyl=r_v0.tolist())
        print(f"{H:4d} {H*0.05:6.2f} | {f:9.5f} {v0:10.5f} {ho:9.5f} {ze:9.5f} "
              f"| {auth:8.2f}x", flush=True)
    print("  fd        = FD rollout, true valve      fd_valve0 = same FD, valve forced to 0")
    print("  hold      = constant-velocity baseline  zero      = no motion")
    print("  authority = err(valve0)/err(fd): >1 means the FD genuinely uses the valve")

json.dump(rows, open(FDD / "fd_authority_report.json", "w"), indent=2)
print(f"\n[OUT] {FDD/'fd_authority_report.json'}", flush=True)
