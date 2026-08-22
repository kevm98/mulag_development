#!/usr/bin/env python3
"""Step 2 gate: does the AAM's ESTIMATED latent track the plant factors?

The factor-reconstruction sweep fixed the TARGET -- the SD latent now retains
mass. This asks the question that actually gates the thesis experiment:

    the AAM sees only [q(5), qdot(5), dP(4), previous_valve(4)].
    Can it INFER the mass component of that new latent from history?

The SD-side result does not answer this. A latent can encode mass perfectly and
still be unestimable from observable history.

Method
------
For each AAM model directory, load its z_hat trace (zhat_all.json, produced by
tools/predict_stage31_aam.py), reduce to a per-run mean, and decode the six true
factors on HELD-OUT test runs. Two decodes, same runs:

  LINEAR  affine z_hat -> e     (lower bound)
  MLP     small net, val-early-stopped   <-- operative: PPO and ID are networks

seal is the POSITIVE CONTROL. The deployed AAM decodes seal at ~0.956. If a
model's seal score collapses, its decode is suspect and its mass number means
nothing.

Compare a `--baseline` (the deployed, mass-blind AAM) against a `--candidate`
(retrained on factor-recon labels) to see whether mass moved.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "AAM_module/datasets/rom_deployment_stage31_v1_frozen"
FACTORS = ["flow", "compliance", "seal", "pressure", "mass_scale", "com_z"]
CONTROL, CONTROL_MIN = "seal", 0.5


def load_zhat(model_dir: Path) -> dict[int, np.ndarray]:
    """run_id -> per-run mean z_hat, from zhat_all.json."""
    p = model_dir / "zhat_all.json"
    if not p.is_file():
        raise SystemExit(
            f"missing {p}\nGenerate it first:\n"
            f"  python3 AAM_module/tools/predict_stage31_aam.py \\\n"
            f"    --root {ROOT} --aam-root <aam dataset> \\\n"
            f"    --model-dir {model_dir} --out {p}")
    by_run = json.loads(p.read_text())["by_run"]
    out = {}
    for rid, ticks in by_run.items():
        arr = np.array([ticks[t] for t in sorted(ticks, key=int)], dtype=float)
        if arr.ndim == 2 and arr.shape[0]:
            out[int(rid)] = arr.mean(0)
    return out


def r2(y, p, ref):
    return [float(1 - ((y[:, j] - p[:, j]) ** 2).sum()
                  / ((y[:, j] - ref[j]) ** 2).sum()) for j in range(6)]


def lin(Xtr, Ytr, Xte):
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    W, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    return np.hstack([Xte, np.ones((len(Xte), 1))]) @ W


def mlp(Xtr, Ytr, Xva, Yva, Xte, seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 32), nn.SiLU(), nn.Linear(32, 6))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-3)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    X, Y, Xv, Yv = T(Xtr), T(Ytr), T(Xva), T(Yva)
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
        return m(T(Xte)).numpy()


def evaluate(model_dir: Path, E, splits, mu, sd) -> dict:
    z = load_zhat(model_dir)
    tr, va, te = ([i for i in splits[s] if i in z] for s in ("train", "val", "test"))
    Z = {n: np.stack([z[i] for i in ids]) for n, ids in (("tr", tr), ("va", va), ("te", te))}
    Y = {n: (np.stack([E[i] for i in ids]) - mu) / sd
         for n, ids in (("tr", tr), ("va", va), ("te", te))}
    ref = Y["tr"].mean(0)
    return {
        "n_runs": {"train": len(tr), "val": len(va), "test": len(te)},
        "latent_dim": int(Z["tr"].shape[1]),
        "linear": r2(Y["te"], lin(Z["tr"], Y["tr"], Z["te"]), ref),
        "mlp": r2(Y["te"], mlp(Z["tr"], Y["tr"], Z["va"], Y["va"], Z["te"]), ref),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path,
                   default=REPO / "outputs/stage31_rom/aam_cnn/L20_d64",
                   help="deployed (mass-blind) AAM, for reference")
    p.add_argument("--candidate", type=Path, required=True,
                   help="AAM retrained on factor-recon labels")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "results/zhat_identification.json")
    a = p.parse_args()

    ri = json.loads((ROOT / "splits/runs_info.json").read_text())
    splits = {s: json.loads((ROOT / f"splits/{s}.json").read_text())
              for s in ("train", "val", "test")}
    E = {int(k): np.array(v["key"], float) for k, v in ri.items()}
    tr = np.stack([E[i] for i in splits["train"]])
    mu, sd = tr.mean(0), tr.std(0).clip(min=1e-9)

    res = {}
    for name, d in (("baseline", a.baseline), ("candidate", a.candidate)):
        if d is None or not d.is_dir():
            print(f"skip {name}: {d}")
            continue
        res[name] = evaluate(d, E, splits, mu, sd)
        res[name]["model_dir"] = str(d)

    ci, mi = FACTORS.index(CONTROL), FACTORS.index("mass_scale")
    for kind in ("linear", "mlp"):
        print(f"\n=== {kind.upper()} decode of z_hat -> true factors (held-out test) ===")
        print(f"{'model':11} " + "".join(f"{f:>12}" for f in FACTORS))
        for name, r in res.items():
            print(f"{name:11} " + "".join(f"{v:>12.3f}" for v in r[kind]))

    print("\n=== gate ===")
    for name, r in res.items():
        ctrl, mass = r["mlp"][ci], r["mlp"][mi]
        ok = "pass" if ctrl > CONTROL_MIN else "FAIL"
        print(f"  {name:11} seal(control)={ctrl:+.3f} [{ok}]   mass={mass:+.3f}")

    if "candidate" in res and "baseline" in res:
        b, c = res["baseline"]["mlp"][mi], res["candidate"]["mlp"][mi]
        cc = res["candidate"]["mlp"][ci]
        print(f"\n  mass_scale: baseline {b:+.3f} -> candidate {c:+.3f}  (delta {c-b:+.3f})")
        if cc <= CONTROL_MIN:
            print("  VERDICT: control failed on the candidate; its mass number is not")
            print("           interpretable. Investigate the retrain before concluding.")
        elif c > 0.5:
            print("  VERDICT: z_hat DOES track mass from history.")
            print("           -> Experiment 1 (EE mass) is viable; proceed to the 2x2.")
        elif c > 0.2:
            print("  VERDICT: partial. z_hat carries some mass signal but weakly.")
            print("           -> EE mass is marginal; hydraulic delay is the safer lead.")
        else:
            print("  VERDICT: z_hat does NOT track mass despite the latent containing it.")
            print("           -> mass is not observable from [q, qdot, dP, valve].")
            print("           -> lead with hydraulic delay, which IS visible in these signals.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"format": "zhat_identification_v1",
                                 "factors": FACTORS, "results": res}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
