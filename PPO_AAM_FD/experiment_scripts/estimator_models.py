"""Compare feature sets for the online payload estimator, offline.

The deployed ridge reaches R2 0.726 and estimates 1.41x / 3.11x / 4.10x against
a true 1 / 3 / 5: accurate in the middle, shrunk at the extremes. Shrinkage that
survives a lambda sweep is not a regularisation choice, it is the features
failing to determine the payload. So this compares feature sets rather than
retuning lambda.

Costs no GPU: it reads the dumps the scoring runs already wrote.

Candidates, in increasing order of what they assume:

  linear     the 66 task columns as they are
  +squares   plus their squares, for any curvature in effort vs motion
  +effort x  plus products of the cylinder-effort block (62:66) with the
             velocity block (16:20) and the model-error block (58:62). Payload
             enters the dynamics through force against acceleration, which is a
             PRODUCT of measured quantities and cannot be expressed by any
             linear combination of them.
  mlp        a small tanh network, as the nonlinear reference

Held out the same way as the deployed fit: trained on the mixed-payload run,
tested on the fixed-payload runs, reported in payload multiples.

Usage:  estimator_models.py [--dir outputs/estimator] [--prefix ctxonly]
"""
import glob
import os
import sys

import numpy as np

EFFORT = slice(62, 66)
VELOCITY = slice(16, 20)
MODEL_ERR = slice(58, 62)


def load(path):
    d = np.load(path)
    return d["x"].astype(np.float64), d["y"].astype(np.float64)


def feats(x, kind):
    if kind == "linear":
        return x
    if kind == "+squares":
        return np.hstack([x, x ** 2])
    if kind == "+effort x":
        e, v, m = x[:, EFFORT], x[:, VELOCITY], x[:, MODEL_ERR]
        prods = np.hstack([
            (e[:, :, None] * v[:, None, :]).reshape(len(x), -1),
            (e[:, :, None] * m[:, None, :]).reshape(len(x), -1),
        ])
        return np.hstack([x, x ** 2, prods])
    raise ValueError(kind)


def ridge(z, y, lam):
    mu, sd = z.mean(0), z.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    zn = (z - mu) / sd
    w = np.linalg.solve(zn.T @ zn + lam * np.eye(zn.shape[1]),
                        zn.T @ (y - y.mean()))
    return w, y.mean(), mu, sd


def predict(fit, z):
    w, b, mu, sd = fit
    return ((z - mu) / sd) @ w + b


def mlp(ztr, ytr, hidden=64, epochs=400, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = ztr.mean(0), ztr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    zn = (ztr - mu) / sd
    ym, ys = ytr.mean(), ytr.std() or 1.0
    t = (ytr - ym) / ys
    w1 = rng.normal(0, 1 / np.sqrt(zn.shape[1]), (zn.shape[1], hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden)
    b2 = 0.0
    for _ in range(epochs):
        h = np.tanh(zn @ w1 + b1)
        p = h @ w2 + b2
        g = (p - t) / len(t)
        gw2 = h.T @ g
        gb2 = g.sum()
        gh = np.outer(g, w2) * (1 - h ** 2)
        gw1 = zn.T @ gh
        gb1 = gh.sum(0)
        w1 -= lr * gw1; b1 -= lr * gb1; w2 -= lr * gw2; b2 -= lr * gb2
    return lambda z: (np.tanh(((z - mu) / sd) @ w1 + b1) @ w2 + b2) * ys + ym


if __name__ == "__main__":
    root = (sys.argv[sys.argv.index("--dir") + 1]
            if "--dir" in sys.argv else "outputs/estimator")
    prefix = (sys.argv[sys.argv.index("--prefix") + 1]
              if "--prefix" in sys.argv else "ctxonly")
    xtr, ytr = load(os.path.join(root, f"{prefix}_mixed.npz"))
    tests = sorted(glob.glob(os.path.join(root, f"{prefix}_p*.npz")))
    print(f"train x={xtr.shape}, held out {[os.path.basename(t) for t in tests]}\n")
    print(f"{'model':>12} {'dims':>6} {'held RMSE (x)':>14}   per-payload estimate")
    for kind in ("linear", "+squares", "+effort x"):
        ztr = feats(xtr, kind)
        best = None
        for lam in (1.0, 10.0, 100.0, 1000.0):
            fit = ridge(ztr, ytr, lam)
            errs, shows = [], []
            for t in tests:
                xte, yte = load(t)
                p = predict(fit, feats(xte, kind))
                errs.append(np.sqrt(np.mean((p - yte) ** 2)))
                shows.append(1 + 0.5 * p.mean())
            r = float(np.mean(errs)) * 0.5
            if best is None or r < best[0]:
                best = (r, lam, shows)
        r, lam, shows = best
        print(f"{kind:>12} {ztr.shape[1]:>6} {r:>14.3f}   "
              f"{'  '.join(f'{v:.2f}x' for v in shows)}  (lam {lam:.0f})")
    ztr = feats(xtr, "+effort x")
    f = mlp(ztr, ytr)
    errs, shows = [], []
    for t in tests:
        xte, yte = load(t)
        p = f(feats(xte, "+effort x"))
        errs.append(np.sqrt(np.mean((p - yte) ** 2)))
        shows.append(1 + 0.5 * p.mean())
    print(f"{'mlp':>12} {ztr.shape[1]:>6} {float(np.mean(errs)) * 0.5:>14.3f}   "
          f"{'  '.join(f'{v:.2f}x' for v in shows)}")
    print("\ntrue payloads: " + "  ".join(
        os.path.basename(t).split('_p')[1][:-4] + 'x' for t in tests))
