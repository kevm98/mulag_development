#!/usr/bin/env python3
"""Does a plant latent pay off for a parameter that is INVISIBLE instantaneously?

Motivation
----------
Conditioning the ID residual on the AAM latent measurably HURT when the model
saw pressures and forces (val loss: none 3.92092 < zhat 3.92253 < true
3.92397), and HELPED once those were masked (true 4.09981 < zhat 4.10966 <
none 4.13550). The latent pays off exactly when the observation is
insufficient -- RMA's condition.

Every hydraulic factor tried so far (seal, compliance, flow, pressure, valve
effectiveness) acts on the instantaneous state, so a single slice largely
reveals it. A TRANSPORT DELAY does not: it is a property of the command ->
response mapping over time. No instantaneous measurement contains it, so it
can only be identified from history. That makes it the sharpest available test
of the architecture.

Construction
------------
Real recorded dynamics, synthetic delay. For run r draw d_r in {0,2,4,6,8} and
present the command stream shifted forward by d_r:

    u~(t) = u(t + d_r)     =>     u~(t - d_r) = u(t)

The response recorded at t was caused by u(t), which now sits at lag d_r of the
presented history. To predict it the model must read the command d_r steps back
-- and since d_r varies per run and appears nowhere in a single slice, it can
only be recovered from history.

This is a well-posed prediction task on real dynamics: nothing about the plant
is fabricated, only the index at which the causal command is presented.

Arms (identical predictor, identical inputs apart from the 2-D latent)
    none    latent zeroed
    true    a learned 2-D embedding of the TRUE d -- the ceiling
    zhat    2-D output of a causal encoder over recent history -- the AAM
            analogue, which must infer d the way the AAM must infer the plant

`true` beating `none` establishes that delay is a parameter worth knowing.
`zhat` approaching `true` establishes that it is recoverable from history.
Both are needed: the first alone says nothing about estimation, and the second
is meaningless if the first is flat.

RESULT 2026-08-17: INCONCLUSIVE, AND THE HEADLINE NUMBER IS AN ARTIFACT.
--------------------------------------------------------------------
    none  test 0.29998
    true  test 0.29831   (-0.6%)    oracle delay
    zhat  test 0.24221   (-19.3%)   learned encoder

`zhat` beat `true` by 19%, which is impossible if the latent's job is to convey
the delay: an estimate cannot beat the quantity it estimates. The arms are not
information-matched. The encoder reads an 8x80 history of commands AND
responses; nothing constrains its 2 dims to encode delay, and the easiest thing
to encode is recent response magnitude, which predicts the next response
directly. So `zhat`'s gain is generic history information, not identification.
DO NOT cite the -19.3% as an adaptation result.

What is valid here: the oracle delay is worth only 0.6%. Even perfectly known,
the delay barely helps next-state prediction on this data, because with 10-step
command holds u(t-d) ~ u(t+-5) and precise lag selection changes little.

Delay classification from history reached 0.326 against 0.200 chance -- the
delay is only weakly identifiable, the same persistent-excitation limit.
(The "instantaneous" probe's 0.294 is also mislabelled: its input includes the
44-lag command window, which is history. A genuine single-slice probe would use
`state` alone.)

A VALID VERSION NEEDS BOTH:
  1. Information-matched arms -- train the encoder SUPERVISED to predict d,
     freeze it, then use its output as the latent. That is what the AAM
     actually does, and it stops the encoder smuggling in unrelated features.
  2. Excitation fast enough to identify a delay: --probe-hold-steps 1-3, not
     10. That needs a NEW collection; no re-slicing of the existing data can
     fix excitation that was never there.

CPU or GPU, no simulator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "AAM_module"))

from aam_data import iter_raw_runs, load_runs_info, read_raw_run  # noqa: E402

# The PRBS probe holds each command value for 10 control steps, so a delay
# SHORTER than one hold leaves u(t-d) equal to u(t) and is unidentifiable by
# construction -- you cannot identify a delay unless the input changes faster
# than the delay. Measured on this data: corr(u(t), dx(t+lag)) is 0.600, 0.601,
# 0.603, 0.606, 0.609 for lag 0..4, i.e. flat. The delays below are whole
# multiples of the hold, which is the coarsest grid this excitation supports.
DELAYS = (0, 10, 20, 30, 40)
CMD_WINDOW = 44      # presented command lags; must span the largest delay
HIST = 80            # encoder history, ~1.3 s at 60 Hz
LATENT = 2


def build_run(arrays: dict, delay: int) -> tuple[np.ndarray, ...]:
    """Return (state, cmd_window, hist, target) for one run at one delay."""
    x = np.asarray(arrays["cylinder_extension"], dtype=np.float32)
    v = np.asarray(arrays["cylinder_velocity"], dtype=np.float32)
    # applied_valve_cmd is all-NaN in ROM collections: the ROM PRODUCES the
    # valve rather than consuming one, so only the predicted field is written.
    u = np.asarray(arrays["rom_predicted_valve_cmd"], dtype=np.float32)
    if not np.isfinite(u).all():
        raise SystemExit("valve command contains non-finite values")
    n = len(u)
    # need t+delay <= n-1 and t-CMD_WINDOW+1+delay >= 0, plus history and t+1
    lo = max(HIST, CMD_WINDOW - 1 - delay, 0)
    hi = min(n - 1 - delay, len(x) - 2)
    if hi <= lo:
        return tuple(np.empty((0, 1), np.float32) for _ in range(4))
    t = np.arange(lo, hi, dtype=np.int64)

    state = np.concatenate([x[t], v[t]], axis=1)                       # 8
    # presented command stream: u~(s) = u(s + delay)
    lags = np.arange(CMD_WINDOW)
    cmd = u[(t[:, None] + delay - lags[None, :])].reshape(len(t), -1)  # 4*W
    # encoder history: presented command and the measured response beside it,
    # which is the only pairing from which a delay can be read
    hl = np.arange(HIST)
    hu = u[(t[:, None] + delay - hl[None, :])]
    hdx = x[(t[:, None] - hl[None, :]) + 1] - x[(t[:, None] - hl[None, :])]
    hist = np.concatenate([hu, hdx], axis=2).reshape(len(t), -1)       # 8*HIST
    target = np.concatenate([x[t + 1] - x[t], v[t + 1] - v[t]], axis=1)  # 8
    return state, cmd, hist, target


class Encoder(nn.Module):
    """Causal history -> 2-D latent. The AAM analogue."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, LATENT),
        )

    def forward(self, h):
        return self.net(h)


class Predictor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + LATENT, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 8),
        )

    def forward(self, feat, z):
        return self.net(torch.cat([feat, z], dim=-1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path,
                   default=ROOT / "AAM_module/datasets/rom_phase2_v2_frozen")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=ROOT / "PPO_ID_AAM_FD/experiments/results/delay_identification.json")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    info = load_runs_info(a.root)

    data = {s: {k: [] for k in ("state", "cmd", "hist", "y", "d")}
            for s in ("train", "val", "test")}
    assigned, per_split = {}, {}
    for rid, _ in iter_raw_runs(a.root):
        rid = int(rid)
        split = str(info[f"{rid:05d}"]["frozen_split"])
        if split not in data:
            continue
        # round-robin WITHIN each split, so every split sees every delay
        per_split[split] = per_split.get(split, 0) + 1
        d = DELAYS[per_split[split] % len(DELAYS)]
        assigned[rid] = d
        arrays = read_raw_run(a.root, rid, verify=True)["arrays"]
        s, c, h, y = build_run(arrays, d)
        if len(s) == 0:
            continue
        # thin: neighbouring 60 Hz samples are near-duplicates
        keep = np.arange(0, len(s), 3)
        data[split]["state"].append(s[keep])
        data[split]["cmd"].append(c[keep])
        data[split]["hist"].append(h[keep])
        data[split]["y"].append(y[keep])
        data[split]["d"].append(np.full(len(keep), DELAYS.index(d), np.int64))

    P = {}
    for s in data:
        if not data[s]["state"]:
            raise SystemExit(f"split {s} is empty")
        P[s] = {k: np.concatenate(v) for k, v in data[s].items()}
        print(f"{s}: {len(P[s]['state'])} samples, "
              f"{len(set(P[s]['d'].tolist()))} distinct delays")

    # normalize on train
    def norm(key):
        mu = P["train"][key].mean(0)
        sd = P["train"][key].std(0).clip(1e-6)
        for s in P:
            P[s][key] = (P[s][key] - mu) / sd
    for k in ("state", "cmd", "hist"):
        norm(k)
    ymu, ysd = P["train"]["y"].mean(0), P["train"]["y"].std(0).clip(1e-9)
    for s in P:
        P[s]["y"] = (P[s]["y"] - ymu) / ysd

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = {s: {k: torch.as_tensor(v).to(dev) for k, v in P[s].items()} for s in P}
    feat_dim = P["train"]["state"].shape[1] + P["train"]["cmd"].shape[1]

    # ---- is the delay visible in a single slice? the contrast that matters --
    # For every hydraulic factor tried so far, the instantaneous slice recovers
    # the plant at R^2 ~ 0.75. If that collapses here, the delay is genuinely a
    # history-only parameter and the latent has something only it can supply.
    def probe(inputs: list[str], label: str) -> float:
        X = torch.cat([T["train"][k] for k in inputs], dim=-1)
        Xt = torch.cat([T["test"][k] for k in inputs], dim=-1)
        m = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(),
                          nn.Linear(128, len(DELAYS))).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        yb = T["train"]["d"]
        for _ in range(2500):
            i = torch.randint(0, len(X), (4096,), device=dev)
            opt.zero_grad()
            nn.functional.cross_entropy(m(X[i]), yb[i]).backward()
            opt.step()
        with torch.no_grad():
            acc = (m(Xt).argmax(-1) == T["test"]["d"]).float().mean().item()
        print(f"  delay classification from {label:28} {acc:.3f} "
              f"(chance {1/len(DELAYS):.3f})")
        return acc

    print("\nIS THE DELAY VISIBLE WITHOUT HISTORY?")
    acc_slice = probe(["state", "cmd"], "one slice + cmd window")
    acc_hist = probe(["hist"], "history")

    # ---- the three arms -----------------------------------------------------
    results = {}
    for arm in ("none", "true", "zhat"):
        torch.manual_seed(a.seed)
        pred = Predictor(feat_dim).to(dev)
        mods = [pred]
        emb = enc = None
        if arm == "true":
            emb = nn.Embedding(len(DELAYS), LATENT).to(dev); mods.append(emb)
        if arm == "zhat":
            enc = Encoder(P["train"]["hist"].shape[1]).to(dev); mods.append(enc)
        opt = torch.optim.Adam([q for m in mods for q in m.parameters()], lr=1e-3)

        def latent(split, idx=None):
            n = len(T[split]["state"]) if idx is None else len(idx)
            if arm == "none":
                return torch.zeros(n, LATENT, device=dev)
            if arm == "true":
                return emb(T[split]["d"] if idx is None else T[split]["d"][idx])
            return enc(T[split]["hist"] if idx is None else T[split]["hist"][idx])

        best, bad, state = float("inf"), 0, None
        for ep in range(a.epochs):
            for m in mods:
                m.train()
            perm = torch.randperm(len(T["train"]["state"]), device=dev)
            for k in range(0, len(perm), a.batch):
                i = perm[k:k + a.batch]
                f = torch.cat([T["train"]["state"][i], T["train"]["cmd"][i]], -1)
                opt.zero_grad()
                loss = nn.functional.mse_loss(pred(f, latent("train", i)),
                                              T["train"]["y"][i])
                loss.backward(); opt.step()
            for m in mods:
                m.eval()
            with torch.no_grad():
                f = torch.cat([T["val"]["state"], T["val"]["cmd"]], -1)
                vl = float(nn.functional.mse_loss(pred(f, latent("val")),
                                                  T["val"]["y"]))
            if vl < best - 1e-6:
                best, bad = vl, 0
                state = [{k: q.clone() for k, q in m.state_dict().items()} for m in mods]
            else:
                bad += 1
                if bad >= 10:
                    break
        if state is None:
            raise SystemExit(
                f"arm {arm!r} never produced a finite validation loss; "
                "check the inputs for non-finite values"
            )
        for m, st in zip(mods, state):
            m.load_state_dict(st)
        with torch.no_grad():
            f = torch.cat([T["test"]["state"], T["test"]["cmd"]], -1)
            te = float(nn.functional.mse_loss(pred(f, latent("test")), T["test"]["y"]))
        results[arm] = {"val_mse": best, "test_mse": te}
        print(f"  {arm:5} val {best:.5f}   test {te:.5f}")

    n, t, z = (results[k]["test_mse"] for k in ("none", "true", "zhat"))
    print("\nRESULT")
    print(f"  true vs none : {100*(t-n)/n:+.1f}%   (is the delay worth knowing)")
    print(f"  zhat vs none : {100*(z-n)/n:+.1f}%   (is it recoverable from history)")
    if t < n:
        print(f"  zhat recovers {100*(n-z)/max(n-t,1e-12):.0f}% of the privileged gain")
    else:
        print("  the privileged latent did NOT help; nothing below it is meaningful")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "format": "delay_identification_v1",
        "delays": list(DELAYS), "cmd_window": CMD_WINDOW, "hist": HIST,
        "delay_classification_accuracy": {
            "instantaneous_slice_plus_cmd_window": acc_slice,
            "history": acc_hist, "chance": 1 / len(DELAYS),
        },
        "arms": results,
        "run_delays": {str(k): int(v) for k, v in assigned.items()},
    }, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
