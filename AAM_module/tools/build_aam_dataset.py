#!/usr/bin/env python
"""Build the AAM dataset from the frozen master runs (AAM Stage 24).

For every raw run in the frozen 118-run dataset, on the exact 50 ms ROM tick
grid, this tool builds the deployment-observable feature stream:

    applied action 8  +  cylinder extension 4  +  cylinder velocity 4
        +  measured cylinder pressure 4  +  joint position 5
        +  joint velocity 5                              = 30-D per tick

and stores causal history windows for the Stage 25 candidate lags
(5/10/20/40 ticks = 0.25/0.5/1.0/2.0 s). Excluded by construction: physical
factors ``e``, true latent ``z`` (target only), ROM hidden state, future
observations, and the desired task reference. Measured ``dP`` is included as
deployment-observable reduced-order hydraulic state.

Each window receives the STOPPED-GRADIENT target ``z = mu(e)`` computed by
the frozen Stage 23 encoder (``encoder.pt`` + the train-only factor
normalization from the frozen ``metrics.json``), evaluated in ``eval`` mode
with ``no_grad`` so no gradient path exists.

Layout (mirrors the SD derived dataset):

    <root>/aam/dataset/
      index.json
      <run_id:05d>/
        features.npz          # (T, 30) feature rows + tick indices + flags
        history_L<lag>.npz    # window_input (N, lag*30), z_target (N, 2),
                              # valid_mask, window_end_tick
        sample_meta.json      # run_id, domain, split, z, history bounds

Usage:
    python build_aam_dataset.py --root datasets/master_v3_stage22_wide_frozen \\
        --model-dir ../sd_model/stage23_frozen/dz2_nodp_holdq
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]  # for sd_model.* (SD tools moved there)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import (  # noqa: E402
    AAM_FEATURE_DIM,
    FEATURE_GROUPS,
    HISTORY_LAG_CANDIDATES,
    ROM_DT,
    aam_window_stack,
    build_aam_features,
    iter_raw_runs,
    load_runs_info,
    raw_dir,
    read_raw_run,
    sha256_file,
)
from aam_data.frozen import CHECKSUM_FILE  # noqa: E402
from aam_data.sd import FACTOR_KEYS  # noqa: E402
from sd_model.training.train_sd_z import FactorEncoder  # noqa: E402


def load_frozen_encoder(model_dir: Path) -> tuple[torch.nn.Module, dict]:
    """Load the frozen Stage 23 encoder and its train-only normalization."""
    metrics = json.loads((model_dir / "metrics.json").read_text())
    if metrics.get("latent_source") != "encoder":
        raise SystemExit(f"model is not an encoder model: {model_dir}")
    latent_dim = int(metrics["latent_dim"])
    encoder = FactorEncoder(latent_dim)
    encoder.load_state_dict(
        torch.load(model_dir / "encoder.pt", map_location="cpu")
    )
    encoder.eval()
    return encoder, metrics


def encode_z(
    encoder: torch.nn.Module,
    e: np.ndarray,
    e_mean: np.ndarray,
    e_std: np.ndarray,
) -> np.ndarray:
    """Stopped-gradient z = mu(e_tilde) for one run (train-only stats)."""
    e_n = torch.as_tensor((e - e_mean) / e_std, dtype=torch.float32)
    with torch.no_grad():
        z = encoder(e_n[None, :]).detach().numpy()[0]
    return z.astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=str(AAM_ROOT / "datasets/master_v3_stage22_wide_frozen"),
        help="frozen master dataset root",
    )
    p.add_argument(
        "--model-dir",
        default=str(AAM_ROOT.parent / "sd_model/stage23_frozen/dz2_nodp_holdq"),
        help="frozen Stage 23 model dir (encoder.pt + metrics.json)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="derived AAM dataset dir (default: <root>/aam/dataset)",
    )
    p.add_argument("--lags", type=int, nargs="+", default=list(HISTORY_LAG_CANDIDATES))
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    raw = raw_dir(root)
    if not (raw / "manifest.csv").is_file():
        raise SystemExit(f"not a frozen dataset: {root}")
    out = Path(args.out or str(root / "aam" / "dataset")).expanduser().resolve()
    if out == raw or raw in out.parents:
        raise SystemExit(f"refusing to write AAM data into raw/: {out}")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output exists and is not empty: {out} (use --force)")
    if out.exists():
        shutil.rmtree(out)
    lags = tuple(sorted(set(args.lags)))

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not (model_dir / "encoder.pt").is_file():
        raise SystemExit(f"frozen encoder not found: {model_dir / 'encoder.pt'}")
    encoder, metrics = load_frozen_encoder(model_dir)
    e_mean = np.asarray(metrics["e_normalization"]["mean"], dtype=np.float64)
    e_std = np.asarray(metrics["e_normalization"]["std"], dtype=np.float64)
    latent_dim = int(metrics["latent_dim"])

    runs_info = load_runs_info(root)
    raw_checksums_digest = sha256_file(raw / CHECKSUM_FILE)
    model_checksums_digest = sha256_file(model_dir / CHECKSUM_FILE)

    per_run: list[dict] = []
    n_total_ticks = 0
    for run_id, _run_path in sorted((int(r), p) for r, p in iter_raw_runs(root)):
        if args.limit is not None and run_id > args.limit - 1:
            break
        data = read_raw_run(root, run_id, verify=True)
        meta, arrays = data["metadata"], data["arrays"]
        info = runs_info.get(f"{run_id:05d}", {})
        built = build_aam_features(arrays)
        features = built["features"]
        n = built["n_ticks"]
        n_total_ticks += n

        e = np.asarray([float(meta[k]) for k in FACTOR_KEYS], dtype=np.float64)
        z = encode_z(encoder, e, e_mean, e_std)
        assert z.shape == (latent_dim,)
        if not np.isfinite(z).all():
            raise SystemExit(f"run {run_id:05d}: non-finite z {z}")

        run_out = out / f"{run_id:05d}"
        run_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_out / "features.npz",
            features=features,
            tick_control_step=built["tick_control_step"],
            cylinder_limit_flag=built["flags"]["cylinder_limit_flag"],
            command_saturation_flag=built["flags"]["command_saturation_flag"],
            valve_saturation_flag=built["flags"]["valve_saturation_flag"],
            invalid_physics_flag=built["flags"]["invalid_physics_flag"],
        )
        history_bounds: dict[str, dict] = {}
        for lag in lags:
            stack, valid = aam_window_stack(features, lag)
            n_valid = int(valid.sum())
            # ``valid`` is indexed in the compressed ROM-tick stream, while
            # downstream SD rollout maps are keyed by the original control
            # step (0, 3, 6, ...). Keeping the compressed index here silently
            # shifted every causal prediction after the first window.
            valid_indices = np.flatnonzero(valid).astype(np.int64)
            window_end_ticks = built["tick_control_step"][valid_indices]
            np.savez_compressed(
                run_out / f"history_L{lag}.npz",
                window_input=stack,
                z_target=np.repeat(z[None, :], n_valid, axis=0),
                valid_mask=valid,
                window_end_tick=window_end_ticks,
            )
            history_bounds[str(lag)] = {
                "n_valid": n_valid,
                "n_dropped": n - n_valid,
                "first_window_end_tick": int(window_end_ticks[0]) if n_valid else None,
                "last_window_end_tick": int(window_end_ticks[-1]) if n_valid else None,
                "window_ticks_per_row": lag,
            }
        (run_out / "sample_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "domain": meta["domain"],
                    "frozen_split": info.get("frozen_split"),
                    "n_ticks": n,
                    "feature_dim": AAM_FEATURE_DIM,
                    "feature_groups": FEATURE_GROUPS,
                    "z": [float(v) for v in z],
                    "z_note": "stopped-gradient mu(e) from frozen stage23 encoder",
                    "e_raw": {k: float(meta[k]) for k in FACTOR_KEYS},
                    "history_lags": history_bounds,
                    "grid_uniform": built["grid"]["uniform"],
                    "no_interpolation": True,
                    "excluded": [
                        "physical factors",
                        "true z (target only)",
                        "ROM hidden state",
                        "future observations",
                        "desired task reference",
                    ],
                    "built_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        per_run.append(
            {
                "run_id": run_id,
                "domain": meta["domain"],
                "frozen_split": info.get("frozen_split"),
                "n_ticks": n,
                "history_bounds": history_bounds,
            }
        )
        print(
            f"run {run_id:05d} [{meta['domain']}/{info.get('frozen_split')}]: "
            f"{n} ticks, z={np.round(z, 4).tolist()}",
            flush=True,
        )

    (out / "index.json").write_text(
        json.dumps(
            {
                "frozen_root": str(root),
                "raw_checksums_digest": raw_checksums_digest,
                "model_dir": str(model_dir),
                "model_checksums_digest": model_checksums_digest,
                "encoder": {
                    "latent_source": metrics["latent_source"],
                    "latent_dim": latent_dim,
                    "e_normalization": metrics["e_normalization"],
                    "do_not_finetune": True,
                },
                "timebase": {"rom_dt": ROM_DT},
                "feature": {
                    "dim": AAM_FEATURE_DIM,
                    "groups": FEATURE_GROUPS,
                    "per_tick": (
                        "applied action 8 + cylinder extension 4 + "
                        "cylinder velocity 4 + joint position 5 + "
                        "joint velocity 5"
                    ),
                },
                "history_lags": list(lags),
                "n_runs": len(per_run),
                "n_ticks_total": n_total_ticks,
                "command": " ".join(sys.argv),
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\n{len(per_run)} runs, {n_total_ticks} ticks -> {out}")
    print("history bounds per lag (valid ticks / run):")
    for lag in lags:
        nv = [r["history_bounds"][str(lag)]["n_valid"] for r in per_run]
        print(f"  L{lag}: min {min(nv)} max {max(nv)} mean {np.mean(nv):.1f}")


if __name__ == "__main__":
    main()
