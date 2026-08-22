#!/usr/bin/env python3
"""Experiment 1 (step 1): how wide must the AAM latent be to identify the plant?

Question
--------
The deployed AAM emits ``z_hat`` of width 2, because it regresses onto the
frozen SD latent and SD is 2-D by construction (extension + velocity channels).
Held-out decoding of the true plant factors from that latent gives
``seal`` R2=0.956 but ``mass_scale`` R2=-0.008 -- mass is randomized during
training yet invisible in the latent.

This probe separates two competing explanations:

  (a) WIDTH      -- 2 numbers cannot carry 6 factors, or
  (b) TARGET     -- 2 numbers could carry more, but regressing onto the SD
                    latent spends them on the wrong thing.

Design
------
Same causal-CNN trunk and same 18x20 history contract as the deployed AAM.
Only the bottleneck width changes::

    history h_t -> CNN trunk -> z (d_z)  -> linear head -> theta_hat (6)

``z`` is the bottleneck, i.e. exactly the vector that would be handed to PPO
and ID.  The head is LINEAR on purpose: the deployed consumers see ``z``, so
the factors must be linearly legible in ``z`` itself, not recoverable only by
a deep decoder.

Running ``--widths 2 4 6 8`` gives:
  * d_z=2 trained on theta  -> the (b) control.  If this beats the deployed
    2-D latent, the current target is the problem, not the width.
  * d_z>=6                  -> the information ceiling: how much of theta is
    recoverable from history at all.

theta is constant within a run, so windows are heavily redundant; ``--stride``
subsamples them.  Run diversity (82 train runs) is what carries this probe,
not window count.

CPU only.  No simulator.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "AAM_module"))

from tools.train_stage31_aam import AAM, load_windows  # noqa: E402
from aam_data import load_split  # noqa: E402

FACTORS = ["flow", "compliance", "seal", "pressure", "mass_scale", "com_z"]
DEFAULT_ROOT = REPO / "AAM_module/datasets/rom_deployment_stage31_v1_frozen"


class WidthProbe(nn.Module):
    """Deployed AAM trunk, bottlenecked to ``latent_dim``, linear head to theta."""

    def __init__(self, in_dim: int, latent_dim: int, n_factors: int,
                 hidden: int, depth: int, dropout: float, lag: int) -> None:
        super().__init__()
        self.trunk = AAM(
            in_dim=in_dim, latent_dim=latent_dim, hidden=hidden, depth=depth,
            dropout=dropout, architecture="cnn", lag=lag,
        )
        self.head = nn.Linear(latent_dim, n_factors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def build_split(aam_root: Path, runs_info: dict, run_ids, lag: int, stride: int):
    """Return (windows, theta) with theta broadcast from the run's true key."""
    xs, ys = [], []
    for run_id in run_ids:
        data = load_windows(aam_root, int(run_id), lag)
        if data is None or not data["window"].shape[0]:
            continue
        info = runs_info.get(f"{int(run_id):05d}")
        if info is None:
            continue
        w = data["window"][::stride]
        xs.append(w)
        ys.append(np.repeat(np.asarray(info["key"], dtype=np.float64)[None, :], w.shape[0], axis=0))
    if not xs:
        raise SystemExit("no windows loaded; check --root/--aam-root/--lag")
    return np.concatenate(xs), np.concatenate(ys)


def r2_per_factor(y_true: np.ndarray, y_pred: np.ndarray, train_mean: np.ndarray) -> dict:
    """R2 against the TRAIN mean as the null model (honest held-out R2)."""
    out = {}
    for j, name in enumerate(FACTORS):
        ss_res = float(((y_true[:, j] - y_pred[:, j]) ** 2).sum())
        ss_tot = float(((y_true[:, j] - train_mean[j]) ** 2).sum())
        out[name] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--aam-root", type=Path, default=None)
    p.add_argument("--lag", type=int, default=20)
    p.add_argument("--widths", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--stride", type=int, default=5, help="window subsample stride")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "results/latent_width_ceiling.json")
    a = p.parse_args()

    aam_root = a.aam_root or (a.root / "aam/dataset_multilag")
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    runs_info = json.loads((a.root / "splits/runs_info.json").read_text())
    splits = {s: load_split(a.root, s) for s in ("train", "val", "test")}
    print(f"runs  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    raw = {s: build_split(aam_root, runs_info, ids, a.lag, a.stride) for s, ids in splits.items()}
    Xtr_np, Ytr_np = raw["train"]
    print(f"windows (stride={a.stride})  " +
          "  ".join(f"{s}={raw[s][0].shape[0]}" for s in raw))

    # normalisation computed on TRAIN ONLY (mirrors the deployed AAM contract)
    x_mean, x_std = Xtr_np.mean(0), Xtr_np.std(0).clip(min=1e-6)
    y_mean, y_std = Ytr_np.mean(0), Ytr_np.std(0).clip(min=1e-9)
    print("\ntrue-factor spread in TRAIN (sd): " +
          "  ".join(f"{n}={s:.4f}" for n, s in zip(FACTORS, Ytr_np.std(0))))

    def to_t(split):
        x, y = raw[split]
        return (torch.as_tensor(((x - x_mean) / x_std).astype(np.float32), device=device),
                torch.as_tensor(((y - y_mean) / y_std).astype(np.float32), device=device))

    Xtr, Ytr = to_t("train")
    Xva, Yva = to_t("val")
    Xte, Yte = to_t("test")
    in_dim = Xtr.shape[1]

    results = {}
    for width in a.widths:
        t0 = time.time()
        torch.manual_seed(a.seed)
        model = WidthProbe(in_dim, width, len(FACTORS), a.hidden, a.depth,
                           a.dropout, a.lag).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        best, best_state, bad = float("inf"), None, 0
        n = Xtr.shape[0]
        for epoch in range(a.epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, a.batch_size):
                idx = perm[i:i + a.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(nn.functional.mse_loss(model(Xva), Yva))
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(Xte).cpu().numpy() * y_std + y_mean
        y_true = raw["test"][1]
        r2 = r2_per_factor(y_true, pred, y_mean)
        results[str(width)] = {
            "latent_dim": width,
            "val_mse_standardised": best,
            "epochs_run": epoch + 1,
            "seconds": round(time.time() - t0, 1),
            "r2_per_factor": r2,
            "r2_mean": float(np.nanmean(list(r2.values()))),
        }
        print(f"\nd_z={width:<2} val_mse={best:.4f} epochs={epoch+1} "
              f"({time.time()-t0:.0f}s)")
        for k, v in r2.items():
            print(f"    {k:12} R2={v:+.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "aam_latent_width_ceiling_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "how wide must z be for the plant factors to be linearly legible in it?",
        "deployed_baseline": {
            "latent_dim": 2,
            "target": "frozen SD latent (extension, velocity)",
            "r2_per_factor": {
                "flow": 0.114, "compliance": 0.448, "seal": 0.956,
                "pressure": 0.081, "mass_scale": -0.008, "com_z": 0.056,
            },
            "note": "measured 2026-08-16 by linear decode of per-run mean z_hat",
        },
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "widths": results,
    }
    a.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {a.out}")

    print("\n" + "=" * 72)
    print("SUMMARY   R2 per factor vs latent width (held-out test runs)")
    print("=" * 72)
    print(f"{'factor':12} " + "".join(f"{'d_z='+str(w):>10}" for w in a.widths) + f"{'deployed':>11}")
    base = payload["deployed_baseline"]["r2_per_factor"]
    for f in FACTORS:
        row = "".join(f"{results[str(w)]['r2_per_factor'][f]:>10.3f}" for w in a.widths)
        print(f"{f:12} {row}{base[f]:>11.3f}")
    print(f"{'mean':12} " +
          "".join(f"{results[str(w)]['r2_mean']:>10.3f}" for w in a.widths) +
          f"{np.mean(list(base.values())):>11.3f}")


if __name__ == "__main__":
    main()
