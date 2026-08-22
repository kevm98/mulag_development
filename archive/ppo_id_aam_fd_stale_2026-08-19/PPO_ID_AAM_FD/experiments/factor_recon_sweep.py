#!/usr/bin/env python3
"""Held-out factor decode of the SD latent across the factor-recon sweep.

For each trained model in ``--sweep-dir/w<W>`` (factor-recon-weight sweep
{0.0, 0.1, 0.5, 1.0, 2.0}):

- load its privileged ``FactorEncoder`` (6 -> d_z=2, saved as ``encoder.pt``);
- for every run in the TRAIN and TEST splits compute the per-run latent
  ``z = encoder(e_tilde)`` from that run's true factor vector ``e``, where
  ``e_tilde = (e - mu_e)/sigma_e`` uses the TRAIN-only normalization recorded
  in the run's ``metrics.json``;
- fit a held-out AFFINE decode ``z -> e`` on the train-split runs and score
  per-factor R^2 on the test-split runs (the factor configs of test are
  disjoint from train, per splits/split_summary.json);
- record the model's own validation cost (``best_val_loss``, ``best_epoch``).

Positive control (mandatory): ``seal`` must decode at held-out R^2 > 0.5 in
the ``w=0.0`` column. If it does not, the decode procedure is broken and the
whole table is void.

CPU only. No training, no simulator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AAM_ROOT = REPO / "AAM_module"
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_split  # noqa: E402


FACTOR_KEYS = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
WEIGHTS = ("0.0", "0.1", "0.5", "1.0", "2.0")
POSITIVE_CONTROL = "seal"
POSITIVE_CONTROL_THRESHOLD = 0.5
DEFAULT_ROOT = REPO / "AAM_module/datasets/rom_deployment_stage31_v1_frozen"
DEFAULT_SWEEP = REPO / "outputs/factor_recon_sweep"
DEFAULT_OUT = Path(__file__).resolve().parent / "results/factor_recon_sweep.json"


class FactorEncoder(nn.Module):
    """Must mirror AAM_module/tools/train_sd_z.py FactorEncoder (6 -> d_z)."""

    def __init__(self, latent_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim)
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


def held_out_r2_per_factor(Z_train, E_train, Z_test, E_test):
    """Affine decode z -> e trained on train runs, R^2 scored on test runs.

    Returns (r2_per_factor dict, corr_per_factor dict, decoded_test array).
    """
    n_tr, n_te = Z_train.shape[0], Z_test.shape[0]
    X_tr = np.hstack([Z_train, np.ones((n_tr, 1))])
    X_te = np.hstack([Z_test, np.ones((n_te, 1))])
    r2, corr = {}, {}
    decoded = np.empty_like(E_test)
    for j, name in enumerate(FACTOR_KEYS):
        y_tr = E_train[:, j]
        y_te = E_test[:, j]
        w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        y_hat = X_te @ w
        decoded[:, j] = y_hat
        ss_res = float(np.sum((y_te - y_hat) ** 2))
        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
        r2[name] = float(1.0 - ss_res / ss_tot)
        corr[name] = float(np.corrcoef(y_te, y_hat)[0, 1])
    return r2, corr, decoded


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()

    runs_info = json.loads((a.root / "splits/runs_info.json").read_text())
    key_of = {int(run_id): np.asarray(info["key"], dtype=np.float64)
              for run_id, info in runs_info.items()}
    train_ids = load_split(a.root, "train")
    test_ids = load_split(a.root, "test")

    E_train_all = np.stack([key_of[r] for r in train_ids])   # (N_train, 6)
    E_test_all = np.stack([key_of[r] for r in test_ids])     # (N_test, 6)

    table = {}
    for w in WEIGHTS:
        run_dir = a.sweep_dir / f"w{w}"
        metrics = json.loads((run_dir / "metrics.json").read_text())
        e_mean = np.asarray(metrics["e_normalization"]["mean"], dtype=np.float64)
        e_std = np.asarray(metrics["e_normalization"]["std"], dtype=np.float64)

        enc = FactorEncoder(metrics["latent_dim"])
        enc.load_state_dict(torch.load(run_dir / "encoder.pt", map_location="cpu"))
        enc.eval()

        def latents(ids):
            E = np.stack([key_of[r] for r in ids])
            E_n = (E - e_mean) / e_std
            with torch.no_grad():
                Z = enc(torch.as_tensor(E_n, dtype=torch.float32)).numpy()
            return Z, E

        Z_tr, E_tr = latents(train_ids)
        Z_te, E_te = latents(test_ids)

        r2, corr, decoded = held_out_r2_per_factor(Z_tr, E_tr, Z_te, E_te)

        table[w] = {
            "factor_recon_weight": float(metrics["factor_recon_weight"]),
            "best_epoch": int(metrics["best_epoch"]),
            "best_val_loss": float(metrics["best_val_loss"]),
            "test_r2_all_dynamics": float(metrics["test"]["r2_all"]),
            "test_rmse_all_dynamics": float(metrics["test"]["rmse"]["all"]),
            "val_loss_1step": float(metrics["val"]["loss"]),
            "r2_held_out_test": {k: r2[k] for k in FACTOR_KEYS},
            "corr_held_out_test": {k: corr[k] for k in FACTOR_KEYS},
            "n_train_runs": int(len(train_ids)),
            "n_test_runs": int(len(test_ids)),
            "decoded_e_test": decoded.tolist(),
            "true_e_test": E_test_all.tolist(),
        }

    # ---- report table ----
    header = f"{'w':>4} {'best_epoch':>10} {'best_val_loss':>13}" + "".join(
        f" {k:>9}" for k in FACTOR_KEYS
    )
    print(header)
    print("-" * len(header))
    for w in WEIGHTS:
        t = table[w]
        r2 = t["r2_held_out_test"]
        print(f"{w:>4} {t['best_epoch']:>10} {t['best_val_loss']:>13.5f}" + "".join(
            f" {r2[k]:>9.3f}" for k in FACTOR_KEYS
        ))

    # ---- positive control ----
    seal_r2_00 = table["0.0"]["r2_held_out_test"][POSITIVE_CONTROL]
    control_pass = seal_r2_00 > POSITIVE_CONTROL_THRESHOLD
    print("\n--- positive control ---")
    print(f"  seal held-out R2 at w=0.0: {seal_r2_00:.4f} "
          f"(threshold > {POSITIVE_CONTROL_THRESHOLD}) -> "
          f"{'PASS' if control_pass else 'FAIL (decode void)'}")

    # ---- trade-off summary ----
    base = table["0.0"]["best_val_loss"]
    print("\n--- dynamics cost relative to w=0.0 ---")
    for w in WEIGHTS:
        t = table[w]
        print(f"  w={w:>4}: best_val_loss={t['best_val_loss']:.5f} "
              f"delta={t['best_val_loss'] - base:+.5f} "
              f"best_epoch={t['best_epoch']}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "format": "factor_recon_sweep_v1",
        "method": ("held-out affine decode z(2-D) -> e(6) trained on train-split "
                   "runs, scored as per-factor R2 on test-split runs; z = "
                   "FactorEncoder(e_tilde) with train-only normalization"),
        "positive_control": {
            "factor": POSITIVE_CONTROL,
            "threshold": POSITIVE_CONTROL_THRESHOLD,
            "seal_r2_at_w0": seal_r2_00,
            "pass": control_pass,
        },
        "factor_keys": list(FACTOR_KEYS),
        "weights": list(WEIGHTS),
        "per_weight": table,
        "dynamics_cost_relative_to_w0": {
            w: table[w]["best_val_loss"] - base for w in WEIGHTS
        },
    }, indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
