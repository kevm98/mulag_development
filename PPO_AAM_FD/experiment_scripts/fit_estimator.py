"""Fit the online payload estimator from the Stage A observation dumps.

Ridge regression from the 66 TASK observation columns to the normalised
payload. The context columns are not in the dump at all, so the estimator
cannot be trained on the answer -- that leakage is the failure the RMA design
document names explicitly.

Reported in payload multiples, not normalised units: the context normalisation
is mass_scale_norm = (m - 1) / 0.5, so an error of e in normalised units is
0.5 * e in multiples of the nominal cutter mass.

Fit and test are DIFFERENT runs. The mixed-payload run is the training set and
the fixed-payload runs are held out, so the reported accuracy is not a
training fit. A ridge that only works on the run it was fitted to would sail
through an in-sample number and fail the moment it is deployed.

Usage:  fit_estimator.py --dir outputs/estimator --prefix ctxonly \
                        --out outputs/estimator/ctxonly_fit.npz

--prefix selects which run's dumps to use, so a campaign that replaces the
policy does not silently fit on the previous campaign's data.
"""
import glob
import os
import sys

import numpy as np


def load(path):
    d = np.load(path)
    return d["x"].astype(np.float64), d["y"].astype(np.float64)


def fit_ridge(x, y, lam):
    mu, sigma = x.mean(0), x.std(0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    z = (x - mu) / sigma
    n_features = z.shape[1]
    a = z.T @ z + lam * np.eye(n_features)
    w = np.linalg.solve(a, z.T @ (y - y.mean()))
    b = y.mean()
    return w, b, mu, sigma


def score(w, b, mu, sigma, x, y):
    z = (x - mu) / np.where(sigma < 1e-6, 1.0, sigma)
    pred = z @ w + b
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    denom = float(np.var(y))
    r2 = float(1.0 - np.mean((pred - y) ** 2) / denom) if denom > 1e-12 else float("nan")
    return pred, rmse, r2


if __name__ == "__main__":
    root = (sys.argv[sys.argv.index("--dir") + 1]
            if "--dir" in sys.argv else "outputs/estimator")
    out = (sys.argv[sys.argv.index("--out") + 1]
           if "--out" in sys.argv else os.path.join(root, "payload_fit.npz"))

    prefix = (sys.argv[sys.argv.index("--prefix") + 1]
              if "--prefix" in sys.argv else "adaptive")
    train_path = os.path.join(root, f"{prefix}_mixed.npz")
    if not os.path.exists(train_path):
        # Auto-discover rather than die on a prefix mismatch. A campaign that
        # renames its dumps should not lose the estimator at the last step of a
        # multi-hour run; the newest mixed-payload dump is the right training
        # set by construction, since that is the run that just produced it.
        candidates = sorted(glob.glob(os.path.join(root, "*_mixed.npz")),
                            key=os.path.getmtime)
        if not candidates:
            raise SystemExit(
                f"no mixed-payload dump under {root} (looked for "
                f"{prefix}_mixed.npz and *_mixed.npz)"
            )
        train_path = candidates[-1]
        prefix = os.path.basename(train_path)[: -len("_mixed.npz")]
        print(f"prefix auto-detected as {prefix!r} from {train_path}")
    xtr, ytr = load(train_path)
    print(f"train: {train_path}  x={xtr.shape}  payload range "
          f"{1 + 0.5 * ytr.min():.2f}x - {1 + 0.5 * ytr.max():.2f}x")

    tests = sorted(p for p in glob.glob(os.path.join(root, f"{prefix}_p*.npz")))
    if not tests:
        print("WARNING: no held-out fixed-payload dumps; accuracy would be in-sample")

    # Pick lambda on the HELD-OUT sets, not on the training set, and report the
    # ladder so the choice is visible rather than asserted.
    print(f"\n{'lambda':>10} {'train R2':>9} {'held-out RMSE (x)':>19}")
    best = None
    for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        w, b, mu, sigma = fit_ridge(xtr, ytr, lam)
        _, _, r2tr = score(w, b, mu, sigma, xtr, ytr)
        errs = []
        for path in tests:
            xte, yte = load(path)
            _, rmse, _ = score(w, b, mu, sigma, xte, yte)
            errs.append(rmse)
        held = float(np.mean(errs)) * 0.5 if errs else float("nan")
        print(f"{lam:>10.0f} {r2tr:>9.3f} {held:>19.3f}")
        if best is None or (errs and held < best[0]):
            best = (held, lam)

    lam = best[1]
    w, b, mu, sigma = fit_ridge(xtr, ytr, lam)
    print(f"\nselected lambda = {lam:.0f}")
    for path in tests:
        xte, yte = load(path)
        pred, rmse, r2 = score(w, b, mu, sigma, xte, yte)
        true_x = 1 + 0.5 * float(np.mean(yte))
        pred_x = 1 + 0.5 * float(np.mean(pred))
        print(f"  {os.path.basename(path):>24}  true {true_x:.2f}x  "
              f"mean estimate {pred_x:.2f}x  RMSE {0.5 * rmse:.3f}x  "
              f"bias {pred_x - true_x:+.3f}x")

    np.savez(out, w=w.astype(np.float32), b=np.float32(b),
             mu=mu.astype(np.float32), sigma=sigma.astype(np.float32))
    print(f"\nwrote {out}")
