#!/usr/bin/env python3
"""Latent response to a constructed plant switch.

No recorded run contains a plant change: each run holds one fixed factor
vector. A switch is CONSTRUCTED by splicing two held-out runs at a matched ROM
tick, which is admissible because every run executes the identical
eleven-segment excitation schedule, so at a matched tick the commanded motion
is the same and only the plant differs. The result is a constructed sequence,
not a recorded one.

Writes the per-tick latent track so the figure can be regenerated without
re-running inference.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

TOOL_ROOT = Path(__file__).resolve().parents[2]
AAM_ROOT = TOOL_ROOT / "AAM_module"
for p in (TOOL_ROOT, AAM_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aam_data.rom_stage31 import causal_stack          # noqa: E402
from tools.train_stage31_aam import AAM                # noqa: E402

ROM_DT = 0.05


def load_aam(model_dir: Path, device):
    met = json.loads((model_dir / "metrics.json").read_text())
    lag, fdim = int(met["lag_ticks"]), int(met["feature_dim"])
    m = AAM(lag * fdim, int(met["latent_dim"]), int(met["hidden"]),
            int(met["depth"]), float(met["dropout"]),
            architecture=met.get("architecture", "mlp"),
            feature_dim=fdim, lag=lag).to(device)
    m.load_state_dict(torch.load(model_dir / "aam.pt", map_location=device,
                                 weights_only=False))
    m.eval()
    nrm = met["input_normalization"]
    return (m, lag,
            np.asarray(nrm["per_position_mean"], dtype=np.float64),
            np.asarray(nrm["per_position_std"], dtype=np.float64).clip(min=1e-6))


def rows_of(root: Path, rid: int) -> np.ndarray:
    with np.load(root / "aam" / "rom_rows" / f"{rid:05d}" / "rom_rows.npz",
                 allow_pickle=False) as d:
        return np.asarray(d["features"], dtype=np.float64)


def ztarget_of(root: Path, rid: int, lag: int) -> np.ndarray:
    with np.load(root / "aam" / "dataset" / f"{rid:05d}" / f"history_L{lag}.npz",
                 allow_pickle=False) as d:
        return np.asarray(d["z_target"], dtype=np.float64).mean(axis=0)


def infer(model, rows, lag, mean, std, device):
    win, _ = causal_stack(rows, lag)
    with torch.no_grad():
        z = model(torch.as_tensor((win - mean) / std, dtype=torch.float32,
                                  device=device))
    return z.cpu().numpy().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(AAM_ROOT / "datasets/rom_deployment_stage31_v1_frozen"))
    ap.add_argument("--model-dir", default=str(TOOL_ROOT / "outputs/stage31_rom/aam_cnn/L20_d64"))
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--keep-tracks", type=int, default=3,
                    help="how many per-tick latent tracks to store for plotting")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cpu")
    root = Path(a.root)
    model, lag, mean, std = load_aam(Path(a.model_dir), dev)

    info = json.loads((root / "splits" / "runs_info.json").read_text())
    held = sorted(int(r) for r, m in info.items() if m.get("frozen_split") == "test")
    z_of = {r: ztarget_of(root, r, lag) for r in held}

    cands = sorted(((float(np.linalg.norm(z_of[ra] - z_of[rb])), ra, rb)
                    for i, ra in enumerate(held) for rb in held[i + 1:]),
                   reverse=True)[: a.pairs]
    print(f"AAM lag={lag}  held-out={len(held)}  pairs={len(cands)}")

    results, tracks = [], []
    for k, (sep, ra, rb) in enumerate(cands):
        A, B = rows_of(root, ra), rows_of(root, rb)
        n = min(len(A), len(B)); tc = n // 2
        zh = infer(model, np.vstack([A[:tc], B[tc:n]]), lag, mean, std, dev)
        off, zA, zB = lag - 1, z_of[ra], z_of[rb]
        dA = np.linalg.norm(zh - zA, axis=1)
        dB = np.linalg.norm(zh - zB, axis=1)
        cut = tc - off
        pre, post = slice(max(0, cut - 200), cut), slice(cut, min(len(zh), cut + 400))
        closer = np.flatnonzero(dB[cut:] < dA[cut:])
        t_cross = float(closer[0] * ROM_DT) if closer.size else float("nan")
        z_pre = zh[pre].mean(axis=0); z_set = zh[max(0, len(zh) - 200):].mean(axis=0)
        span = float(np.linalg.norm(z_set - z_pre))
        t90 = float("nan")
        if span > 1e-9:
            hit = np.flatnonzero(np.linalg.norm(zh[cut:] - z_pre, axis=1) / span >= 0.9)
            if hit.size:
                t90 = float(hit[0] * ROM_DT)
        results.append({"run_A": ra, "run_B": rb, "latent_separation": sep,
                        "dA_pre": float(dA[pre].mean()), "dB_pre": float(dB[pre].mean()),
                        "dA_post": float(dA[post].mean()), "dB_post": float(dB[post].mean()),
                        "t_cross_s": t_cross, "t90_s": t90, "span": span})
        if k < a.keep_tracks:
            lo, hi = max(0, cut - 300), min(len(zh), cut + 500)
            tracks.append({"run_A": ra, "run_B": rb, "cut_index": int(cut - lo),
                           "t": [(i - (cut - lo)) * ROM_DT for i in range(hi - lo)],
                           "dA": dA[lo:hi].tolist(), "dB": dB[lo:hi].tolist()})
        print(f"  A={ra} B={rb} sep={sep:.3f} cross={t_cross:.2f}s t90={t90:.2f}s")

    ok = sum(1 for r in results if r["dB_post"] < r["dA_post"])
    mv = sum(1 for r in results if r["dA_post"] > r["dA_pre"])
    cr = [r["t_cross_s"] for r in results if r["t_cross_s"] == r["t_cross_s"]]
    t9 = [r["t90_s"] for r in results if r["t90_s"] == r["t90_s"]]
    summary = {"pairs_total": len(results), "ends_closer_to_B": ok,
               "moves_away_from_A": mv,
               "median_cross_s": float(np.median(cr)) if cr else None,
               "median_t90_s": float(np.median(t9)) if t9 else None,
               "window_s": lag * ROM_DT}
    print(f"\ncloser to B {ok}/{len(results)}, away from A {mv}/{len(results)}, "
          f"cross {summary['median_cross_s']:.2f}s, t90 {summary['median_t90_s']:.2f}s")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lag_ticks": lag, "rom_dt": ROM_DT,
                               "model_dir": str(a.model_dir),
                               "summary": summary, "pairs": results,
                               "tracks": tracks}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
