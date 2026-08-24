#!/usr/bin/env python3
"""Run-disjoint NumPy Ridge/MLP diagnostics for direct AAM_v2 factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEYS = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")


def r2(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    denom = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ((y - p) ** 2).sum(0) / np.maximum(denom, 1e-12)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xm, xs = x.mean(0), x.std(0).clip(1e-6)
    xn = (x - xm) / xs
    ym = y.mean(0)
    w = np.linalg.solve(xn.T @ xn + alpha * np.eye(xn.shape[1]), xn.T @ (y - ym))
    return w, xm, xs, ym


def ridge_predict(model, x: np.ndarray) -> np.ndarray:
    w, xm, xs, ym = model
    return (x - xm) / xs @ w + ym


def mlp_fit(x: np.ndarray, y: np.ndarray, seed: int = 20260825, epochs: int = 70) -> tuple[dict, np.ndarray, np.ndarray]:
    """Small ReLU MLP with Adam; intentionally dependency-free and CPU-only."""
    rng = np.random.default_rng(seed)
    xm, xs = x.mean(0), x.std(0).clip(1e-6)
    xn = ((x - xm) / xs).astype(np.float32)
    ym = y.mean(0).astype(np.float32)
    yt = y.astype(np.float32) - ym
    h = 32
    params = {"w1": rng.normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], h)).astype(np.float32),
              "b1": np.zeros(h, np.float32),
              "w2": rng.normal(0, 1 / np.sqrt(h), (h, 6)).astype(np.float32),
              "b2": np.zeros(6, np.float32)}
    m = {k: np.zeros_like(v) for k, v in params.items()}; v = {k: np.zeros_like(z) for k, z in params.items()}
    order = np.arange(len(xn)); step = 0
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 128):
            ix = order[start:start + 128]; a = xn[ix] @ params["w1"] + params["b1"]; z = np.maximum(a, 0); pred = z @ params["w2"] + params["b2"]
            d = 2.0 * (pred - yt[ix]) / len(ix)
            grads = {"w2": z.T @ d, "b2": d.sum(0), "w1": xn[ix].T @ ((d @ params["w2"].T) * (a > 0)), "b1": ((d @ params["w2"].T) * (a > 0)).sum(0)}
            step += 1
            for k in params:
                m[k] = .9 * m[k] + .1 * grads[k]; v[k] = .999 * v[k] + .001 * grads[k] ** 2
                params[k] -= 2e-3 * (m[k] / (1 - .9 ** step)) / (np.sqrt(v[k] / (1 - .999 ** step)) + 1e-8)
    return params, xm, xs, ym


def mlp_predict(model, x: np.ndarray) -> np.ndarray:
    params, xm, xs, ym = model
    return np.maximum((x - xm) / xs @ params["w1"] + params["b1"], 0) @ params["w2"] + params["b2"] + ym


def rows(root: Path, dataset: Path, run_id: int, lag: int, feature_set: str, sample_n: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(dataset / f"{run_id:05d}" / "history_L20.npz", allow_pickle=False) as d:
        h0_rows = np.asarray(d["window_input"], np.float32).reshape(-1, 20, 18)
        target = np.asarray(d["e_target_normalized"], np.float32)[0]
    if feature_set == "H0" and lag == 20:
        data = h0_rows
    else:
        # ``h0_rows[:, 0]`` is the current 18-D row at each original L20
        # window end; starts are omitted uniformly for every split.
        base = h0_rows[:, 0, :]
        if feature_set == "H1":
            with np.load(root / "raw" / "runs" / f"{run_id:05d}" / "trajectory.npz", allow_pickle=False) as raw, np.load(root / "aam" / "rom_rows" / f"{run_id:05d}" / "rom_rows.npz", allow_pickle=False) as rr:
                ticks = np.asarray(rr["tick_control_step"], np.int64)[19:]
                ext = np.asarray(raw["cylinder_extension"], np.float32)[ticks]
                vel = np.asarray(raw["cylinder_velocity"], np.float32)[ticks]
            base = np.concatenate((base, ext, vel), axis=1)
        # Build newest-first causal windows without crossing this run.
        data = np.stack([np.concatenate([base[end-k] for k in range(lag)]) for end in range(lag-1, len(base))])
    ix = np.linspace(0, len(data)-1, min(sample_n, len(data)), dtype=int)
    return data[ix].reshape(len(ix), -1), np.repeat(target[None, :], len(ix), axis=0)


def load_split(root, dataset, split, lag, feature_set, n):
    ids = json.loads((root / "splits" / f"{split}.json").read_text())
    x, y, groups = [], [], []
    for rid in ids:
        a, b = rows(root, dataset, int(rid), lag, feature_set, n); x.append(a); y.append(b); groups.extend([int(rid)] * len(a))
    return np.concatenate(x), np.concatenate(y), np.asarray(groups)


def scored(predict, x, y, groups):
    ids = np.unique(groups)
    actual = np.stack([y[groups == rid][0] for rid in ids])
    predicted = np.stack([predict(x[groups == rid]).mean(0) for rid in ids])
    return r2(actual, predicted).tolist()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True); p.add_argument("--dataset", required=True); p.add_argument("--feature-set", choices=("H0", "H1"), required=True); p.add_argument("--lag", type=int, required=True); p.add_argument("--model", choices=("ridge", "mlp"), required=True); p.add_argument("--out", required=True); p.add_argument("--test", action="store_true", help="evaluate reserved test only after selection")
    a = p.parse_args(); root=Path(a.root); ds=Path(a.dataset)
    tr=load_split(root,ds,"train",a.lag,a.feature_set,24); va=load_split(root,ds,"val",a.lag,a.feature_set,80)
    model = ridge_fit(tr[0],tr[1]) if a.model == "ridge" else mlp_fit(tr[0],tr[1])
    predict = lambda x: ridge_predict(model,x) if a.model == "ridge" else mlp_predict(model,x)
    report={"model":a.model,"feature_set":a.feature_set,"lag":a.lag,"factor_keys":KEYS,"val_r2_run_mean":scored(predict,*va),"run_counts":{"train":82,"val":20,"test":16}}
    if a.test:
        te=load_split(root,ds,"test",a.lag,a.feature_set,80); report["test_r2_run_mean"]=scored(predict,*te)
    Path(a.out).write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__": main()
