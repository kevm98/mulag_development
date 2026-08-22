#!/usr/bin/env python3
"""Step 1 analysis: pick the factor-recon weight using seed repeats.

A single seed cannot select a weight -- held-out mass R2 moved
0.68 / 0.35 / 0.44 / 0.61 across weights on 16 test runs, which is sampling
noise. This aggregates 3 seeds per weight and reports median +/- spread.

Both decodes are reported on the SAME held-out test runs:

  LINEAR    affine z -> e.  A lower bound on legibility.
  MLP       small net z -> e, early-stopped on the val split.  PPO and ID
            consume z through neural networks, so THIS is the operative
            criterion.  Judging by the linear number alone already produced
            one wrong conclusion in this project.

seal is the POSITIVE CONTROL: it must score well in the w=0.0 column
(the deployed latent decodes it at ~0.92).  If it does not, the decode is
broken and no other column means anything.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "AAM_module/datasets/rom_deployment_stage31_v1_frozen"
FACTORS = ["flow", "compliance", "seal", "pressure", "mass_scale", "com_z"]
CONTROL = "seal"
CONTROL_MIN = 0.5


class FactorEncoder(nn.Module):
    """Mirrors train_sd_z.FactorEncoder: e_tilde (6-D) -> z (d_z)."""

    def __init__(self, latent_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim)
        )

    def forward(self, e):
        return self.net(e)


def r2(y, p, ref):
    return [float(1 - ((y[:, j] - p[:, j]) ** 2).sum()
                  / ((y[:, j] - ref[j]) ** 2).sum()) for j in range(6)]


def lin_decode(Ztr, Ytr, Zte):
    A = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    W, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    return np.hstack([Zte, np.ones((len(Zte), 1))]) @ W


def mlp_decode(Ztr, Ytr, Zva, Yva, Zte, seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Ztr.shape[1], 32), nn.SiLU(), nn.Linear(32, 6))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-3)
    X, Y = (torch.as_tensor(a, dtype=torch.float32) for a in (Ztr, Ytr))
    Xv, Yv = (torch.as_tensor(a, dtype=torch.float32) for a in (Zva, Yva))
    best, state, bad = float("inf"), None, 0
    for _ in range(4000):
        opt.zero_grad()
        nn.functional.mse_loss(m(X), Y).backward()
        opt.step()
        with torch.no_grad():
            vl = float(nn.functional.mse_loss(m(Xv), Yv))
        if vl < best - 1e-7:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 300:
                break
    if state:
        m.load_state_dict(state)
    with torch.no_grad():
        return m(torch.as_tensor(Zte, dtype=torch.float32)).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=Path, default=REPO / "outputs/factor_recon_seeds")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "results/factor_recon_seeds.json")
    a = p.parse_args()

    ri = json.loads((ROOT / "splits/runs_info.json").read_text())
    tr, va, te = (json.loads((ROOT / f"splits/{s}.json").read_text())
                  for s in ("train", "val", "test"))
    E = {int(k): np.array(v["key"], float) for k, v in ri.items()}
    mu = np.stack([E[i] for i in tr]).mean(0)
    sd = np.stack([E[i] for i in tr]).std(0).clip(min=1e-9)
    Y = {s: (np.stack([E[i] for i in ids]) - mu) / sd
         for s, ids in (("train", tr), ("val", va), ("test", te))}
    ref = Y["train"].mean(0)

    per_w = defaultdict(lambda: defaultdict(list))
    vloss = defaultdict(list)
    for d in sorted(x for x in a.sweep.glob("w*") if x.is_dir()):
        enc_p, met_p = d / "encoder.pt", d / "metrics.json"
        if not enc_p.is_file() or not met_p.is_file():
            print(f"skip incomplete {d.name}")
            continue
        name = d.name                       # w<W>_s<SEED>
        w = name.split("_")[0][1:]
        met = json.loads(met_p.read_text())
        vloss[w].append(float(met["best_val_loss"]))

        st = torch.load(enc_p, map_location="cpu", weights_only=False)
        if hasattr(st, "state_dict"):
            st = st.state_dict()
        ld = [v for k, v in st.items() if k.endswith("weight")][-1].shape[0]
        enc = FactorEncoder(ld)
        enc.load_state_dict(st)
        enc.eval()

        def Z(ids):
            x = torch.as_tensor(((np.stack([E[i] for i in ids]) - mu) / sd).astype(np.float32))
            with torch.no_grad():
                return enc(x).numpy()

        Ztr, Zva, Zte = Z(tr), Z(va), Z(te)
        per_w[w]["linear"].append(r2(Y["test"], lin_decode(Ztr, Y["train"], Zte), ref))
        per_w[w]["mlp"].append(
            r2(Y["test"], mlp_decode(Ztr, Y["train"], Zva, Y["val"], Zte), ref))

    if not per_w:
        raise SystemExit(f"no completed runs under {a.sweep}")

    weights = sorted(per_w, key=float)
    out = {}
    for kind in ("linear", "mlp"):
        print(f"\n=== {kind.upper()} decode — median over seeds (min..max) ===")
        print(f"{'w':6} {'n':>2} " + "".join(f"{f:>19}" for f in FACTORS))
        for w in weights:
            A = np.array(per_w[w][kind])            # (seeds, 6)
            med, lo, hi = np.median(A, 0), A.min(0), A.max(0)
            cells = "".join(f"{med[j]:>8.2f}({lo[j]:.2f}-{hi[j]:.2f})" for j in range(6))
            print(f"{w:6} {len(A):>2} {cells}")
            out.setdefault(w, {})[kind] = {
                "median": med.tolist(), "min": lo.tolist(), "max": hi.tolist(),
                "n_seeds": int(len(A)),
                "per_factor": {f: float(med[j]) for j, f in enumerate(FACTORS)},
            }

    print(f"\n=== dynamics cost ===\n{'w':6} {'n':>2} {'best_val_loss median':>22}")
    for w in weights:
        v = np.array(vloss[w])
        print(f"{w:6} {len(v):>2} {np.median(v):>22.6f}")
        out[w]["best_val_loss_median"] = float(np.median(v))

    ci = FACTORS.index(CONTROL)
    mi = FACTORS.index("mass_scale")
    ctrl = out.get("0.0", {}).get("mlp", {}).get("median", [np.nan] * 6)[ci]
    print(f"\n=== positive control: {CONTROL} at w=0.0 (MLP) = {ctrl:.3f} ===")
    if not (ctrl > CONTROL_MIN):
        print("  CONTROL FAILED — decode is broken; ignore every other number.")
    else:
        print("  control passed.")
        print(f"\n{'w':6} {'mass (MLP)':>12} {'seal (MLP)':>12} {'val_loss':>11}")
        for w in weights:
            m = out[w]["mlp"]["median"]
            print(f"{w:6} {m[mi]:>12.3f} {m[ci]:>12.3f} {out[w]['best_val_loss_median']:>11.5f}")
        best = max((w for w in weights if w != "0.0"),
                   key=lambda w: out[w]["mlp"]["median"][mi])
        print(f"\n  highest median mass R2 among w>0: w={best} "
              f"({out[best]['mlp']['median'][mi]:.3f}), "
              f"seal there = {out[best]['mlp']['median'][ci]:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"format": "factor_recon_seeds_v1", "factors": FACTORS,
         "control": {"factor": CONTROL, "min": CONTROL_MIN, "observed_w0_mlp": ctrl},
         "weights": out}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
