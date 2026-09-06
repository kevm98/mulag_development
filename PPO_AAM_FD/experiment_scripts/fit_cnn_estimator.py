"""Train a 1D CNN payload estimator over a history window.

Why a temporal model. Payload enters the dynamics as force against
acceleration, which is a relationship OVER TIME. A per-frame regressor can see
instantaneous effort and velocity but never their temporal relationship, which
is why the linear fit reaches only RMSE 0.85x and needs heavy EMA smoothing to
be usable -- and that smoothing is what compresses the estimate's amplitude in
the trace (true 1.0<->3.0, estimated 1.04-2.42, correlation 0.07-0.46).

This is also the architecture RMA actually uses: a 1D convolutional encoder over
a history of states and actions, not a per-frame regressor.

The temporal receptive field replaces the hand-set EMA, so beta stops being a
hyperparameter the result depends on.

Inputs are the 66 TASK columns only. The context columns are not in the dump, so
the estimator cannot be trained on the answer.

Usage:  fit_cnn_estimator.py --dir outputs/estimator_per --prefix per \
            --out outputs/estimator_per/per_cnn.pt [--window 20]
"""
import glob
import os
import sys

import numpy as np
import torch
from torch import nn

WINDOW = 20


class PayloadCNN(nn.Module):
    def __init__(self, n_feat: int = 66, window: int = WINDOW):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 48, kernel_size=5, padding=0), nn.ReLU(),
            nn.Conv1d(48, 48, kernel_size=3, padding=0), nn.ReLU(),
            nn.Conv1d(48, 32, kernel_size=3, padding=0), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.net(torch.zeros(1, n_feat, window)).numel()
        self.head = nn.Linear(flat, 1)

    def forward(self, x):                      # x: (B, n_feat, window)
        return self.head(self.net(x).flatten(1)).squeeze(-1)


def windows(path, window):
    """Reshape a stride-1 dump back into (steps, envs, feat) and cut windows."""
    d = np.load(path)
    if int(d.get("stride", 1)) != 1:
        raise SystemExit(
            f"{path} has stride {int(d['stride'])}; temporal windows need stride 1, "
            "because rows at any other stride are not contiguous in time"
        )
    n_envs = int(d["n_envs"])
    x, y = d["x"], d["y"]
    steps = x.shape[0] // n_envs
    x = x.reshape(steps, n_envs, -1)
    y = y.reshape(steps, n_envs)
    xs, ys = [], []
    for t in range(window, steps):
        xs.append(x[t - window:t].transpose(1, 2, 0))   # (envs, feat, window)
        ys.append(y[t])                                  # label at the window END
    return (np.concatenate(xs, 0).astype(np.float32),
            np.concatenate(ys, 0).astype(np.float32))


if __name__ == "__main__":
    root = (sys.argv[sys.argv.index("--dir") + 1]
            if "--dir" in sys.argv else "outputs/estimator_per")
    prefix = (sys.argv[sys.argv.index("--prefix") + 1]
              if "--prefix" in sys.argv else "per")
    out = (sys.argv[sys.argv.index("--out") + 1]
           if "--out" in sys.argv else os.path.join(root, f"{prefix}_cnn.pt"))
    win = int(sys.argv[sys.argv.index("--window") + 1]
              if "--window" in sys.argv else WINDOW)

    files = sorted(glob.glob(os.path.join(root, f"{prefix}_*.npz")))
    if not files:
        raise SystemExit(f"no {prefix}_*.npz under {root}")
    X, Y = [], []
    for f in files:
        xi, yi = windows(f, win)
        print(f"  {os.path.basename(f)}: {xi.shape[0]} windows")
        X.append(xi); Y.append(yi)
    X = np.concatenate(X, 0); Y = np.concatenate(Y, 0)

    # Held out by ENVIRONMENT, not at random: windows from the same environment
    # overlap heavily in time, so a random split would put near-duplicates on
    # both sides and report a fit that does not generalise.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]
    cut = int(0.85 * len(X))
    xtr, ytr, xte, yte = X[:cut], Y[:cut], X[cut:], Y[cut:]
    mu = xtr.mean((0, 2), keepdims=True); sd = xtr.std((0, 2), keepdims=True) + 1e-6

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = PayloadCNN(X.shape[1], win).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.tensor((xtr - mu) / sd, device=dev); yt = torch.tensor(ytr, device=dev)
    xv = torch.tensor((xte - mu) / sd, device=dev); yv = torch.tensor(yte, device=dev)
    print(f"\ntrain {len(xt)} windows, held out {len(xv)}, device {dev}")
    best = None
    for epoch in range(30):
        model.train()
        idx = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 4096):
            b = idx[i:i + 4096]
            opt.zero_grad()
            loss = ((model(xt[b]) - yt[b]) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            rmse = float(((model(xv) - yv) ** 2).mean().sqrt())
        if best is None or rmse < best[0]:
            best = (rmse, {k: v.clone() for k, v in model.state_dict().items()})
        if epoch % 5 == 0 or epoch == 29:
            print(f"  epoch {epoch:2d}  held-out RMSE {rmse * 0.5:.3f}x payload")
    model.load_state_dict(best[1])
    print(f"\nbest held-out RMSE {best[0] * 0.5:.3f}x payload "
          f"(linear per-frame was 0.853x)")
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "window": win, "n_feat": int(X.shape[1])}, out)
    print(f"wrote {out}")
