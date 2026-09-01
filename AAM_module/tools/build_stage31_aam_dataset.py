#!/usr/bin/env python3
"""Attach frozen SD latents to the causal Stage 31 ROM-observable rows."""

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

from aam_data import FACTOR_KEYS, load_runs_info  # noqa: E402
from aam_data.rom_stage31 import (  # noqa: E402
    AAM_FEATURE_DIM,
    AAM_FEATURE_GROUPS,
    causal_stack,
)
from sd_model.training.train_sd_z import FactorEncoder  # noqa: E402


def load_factor_track(
    root: Path, run_id: int, ticks: np.ndarray, metadata: dict
) -> tuple[np.ndarray, str]:
    """Per-tick true factors ``e(t)`` for one run, shape ``(len(ticks), 6)``.

    The AAM answers "what plant am I on RIGHT NOW", so the label has to follow
    the plant through the episode. A run whose factors never change yields a
    constant track and therefore exactly the previous per-run behaviour --
    this is a strict generalisation, not a change of meaning for existing data.

    Three sources, in priority order:

    1. ``raw/runs/<id>/factor_track.npz`` with ``tick_control_step`` (n,) and
       ``factors`` (n, 6) in FACTOR_KEYS order. Fully general; use for
       continuously varying plants. Values are held piecewise-constant between
       the supplied ticks (last value carries forward).
    2. ``metadata["factor_schedule"]``: a list of
       ``{"from_tick": int, "factors": {name: value, ...}}`` segments, applied
       in tick order. Compact and readable; use for step changes.
    3. ``metadata[<factor>]`` scalars -- the constant plant, i.e. legacy runs.
    """
    n = len(ticks)
    track_path = root / "raw" / "runs" / f"{run_id:05d}" / "factor_track.npz"
    if track_path.is_file():
        with np.load(track_path, allow_pickle=False) as data:
            src_ticks = np.asarray(data["tick_control_step"], dtype=np.int64)
            src_factors = np.asarray(data["factors"], dtype=np.float32)
        if src_factors.ndim != 2 or src_factors.shape[1] != len(FACTOR_KEYS):
            raise SystemExit(
                f"run {run_id:05d}: factor_track factors must be (n, {len(FACTOR_KEYS)}), "
                f"got {src_factors.shape}"
            )
        if src_ticks.shape[0] != src_factors.shape[0]:
            raise SystemExit(f"run {run_id:05d}: factor_track tick/factor length mismatch")
        order = np.argsort(src_ticks)
        src_ticks, src_factors = src_ticks[order], src_factors[order]
        # hold the most recent value at or before each tick
        idx = np.searchsorted(src_ticks, ticks, side="right") - 1
        if (idx < 0).any():
            raise SystemExit(
                f"run {run_id:05d}: factor_track starts at tick {int(src_ticks[0])}, "
                f"after the first row tick {int(ticks[0])}"
            )
        return src_factors[idx], "factor_track.npz"

    base = np.asarray([float(metadata[key]) for key in FACTOR_KEYS], dtype=np.float32)
    schedule = metadata.get("factor_schedule")
    if schedule:
        track = np.repeat(base[None, :], n, axis=0)
        for segment in sorted(schedule, key=lambda s: int(s["from_tick"])):
            from_tick = int(segment["from_tick"])
            values = segment.get("factors", {})
            unknown = set(values) - set(FACTOR_KEYS)
            if unknown:
                raise SystemExit(
                    f"run {run_id:05d}: factor_schedule has unknown factors {sorted(unknown)}"
                )
            mask = ticks >= from_tick
            for name, value in values.items():
                track[mask, FACTOR_KEYS.index(name)] = float(value)
        return track, "metadata.factor_schedule"

    return np.repeat(base[None, :], n, axis=0), "metadata.constant"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="frozen ROM collection")
    parser.add_argument("--sd-model", required=True, help="frozen SD directory")
    parser.add_argument("--rows", default=None, help="rom_rows directory")
    parser.add_argument("--out", default=None, help="AAM dataset directory")
    parser.add_argument("--lags", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    sd_model = Path(args.sd_model).expanduser().resolve()
    rows_root = Path(args.rows or root / "aam" / "rom_rows").expanduser().resolve()
    out = Path(args.out or root / "aam" / "dataset").expanduser().resolve()
    if not (sd_model / "encoder.pt").is_file() or not (sd_model / "metrics.json").is_file():
        raise SystemExit(f"frozen SD encoder is incomplete: {sd_model}")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output exists and is not empty: {out} (use --force)")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    lags = tuple(sorted(set(int(lag) for lag in args.lags)))
    if not lags or any(lag < 1 for lag in lags):
        raise SystemExit("--lags must contain positive integers")

    metrics = json.loads((sd_model / "metrics.json").read_text())
    if metrics.get("latent_source") != "encoder":
        raise SystemExit("Stage 31 AAM targets require an encoder-based frozen SD")
    latent_dim = int(metrics["latent_dim"])
    e_norm = metrics["e_normalization"]
    e_mean = np.asarray(e_norm["mean"], dtype=np.float32)
    e_std = np.asarray(e_norm["std"], dtype=np.float32).clip(min=1.0e-6)
    encoder = FactorEncoder(latent_dim)
    encoder.load_state_dict(torch.load(sd_model / "encoder.pt", map_location="cpu", weights_only=False))
    encoder.eval()
    info = load_runs_info(root)

    per_run = []
    for run_dir in sorted(path for path in rows_root.iterdir() if path.is_dir() and path.name.isdigit()):
        run_id = int(run_dir.name)
        rows_path = run_dir / "rom_rows.npz"
        if not rows_path.is_file():
            raise SystemExit(f"missing ROM rows: {rows_path}")
        with np.load(rows_path, allow_pickle=False) as rows:
            features = np.asarray(rows["features"], dtype=np.float64)
            ticks = np.asarray(rows["tick_control_step"], dtype=np.int64)
            desired_x = np.asarray(rows["desired_cylinder_position"], dtype=np.float64)
            desired_v = np.asarray(rows["desired_cylinder_velocity"], dtype=np.float64)
            previous_valve = np.asarray(rows["previous_valve"], dtype=np.float64)
            current_valve = np.asarray(rows["current_valve"], dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != AAM_FEATURE_DIM:
            raise SystemExit(f"run {run_id:05d}: expected features (*, {AAM_FEATURE_DIM}), got {features.shape}")
        metadata = json.loads((root / "raw" / "runs" / f"{run_id:05d}" / "metadata.json").read_text())
        # Per-tick plant, not a per-run constant. z_target(t) = f(e(t)) so the
        # latent is a "what plant am I on now" estimate that can follow an
        # in-episode change. A constant-plant run yields a constant track and
        # therefore the identical labels to the previous per-run behaviour.
        e_track, factor_source = load_factor_track(root, run_id, ticks, metadata)
        with torch.no_grad():
            z_track = encoder(
                torch.as_tensor((e_track - e_mean) / e_std, dtype=torch.float32)
            ).numpy()
        factors_vary = bool(np.ptp(e_track, axis=0).max() > 0.0)
        run_out = out / f"{run_id:05d}"
        run_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_out / "features.npz",
            features=features,
            tick_control_step=ticks,
            desired_cylinder_position=desired_x,
            desired_cylinder_velocity=desired_v,
            previous_valve=previous_valve,
            current_valve=current_valve,
        )
        bounds = {}
        for lag in lags:
            window, valid = causal_stack(features, lag)
            valid_idx = np.flatnonzero(valid)
            end_ticks = ticks[valid_idx]
            # END-OF-WINDOW labelling: valid_idx are window end indices, so the
            # target is the plant as it stands at the newest row of the window.
            # A window straddling a change therefore carries the NEW plant,
            # which teaches the fastest tracking the history allows and leaves
            # the response lag as an emergent, measurable property rather than
            # something baked into the label.
            np.savez_compressed(
                run_out / f"history_L{lag}.npz",
                window_input=window,
                z_target=z_track[valid_idx],
                e_target=e_track[valid_idx],
                valid_mask=valid,
                window_end_tick=end_ticks,
            )
            bounds[str(lag)] = {
                "n_valid": int(len(valid_idx)),
                "first_window_end_tick": int(end_ticks[0]) if len(end_ticks) else None,
                "last_window_end_tick": int(end_ticks[-1]) if len(end_ticks) else None,
                "z_target_varies": bool(
                    len(valid_idx) > 1
                    and np.ptp(z_track[valid_idx], axis=0).max() > 0.0
                ),
            }
        (run_out / "sample_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "domain": metadata["domain"],
                    "frozen_split": info[f"{run_id:05d}"]["frozen_split"],
                    "feature_dim": AAM_FEATURE_DIM,
                    "feature_groups": AAM_FEATURE_GROUPS,
                    "latent_dim": latent_dim,
                    "z_target": "stopped-gradient frozen SD factor encoder, per-window (end-of-window plant)",
                    "z_target_semantics": "current_plant_at_window_end",
                    "factor_source": factor_source,
                    "factors_vary_within_run": factors_vary,
                    "history_bounds": bounds,
                    "excluded_from_input": ["factors", "z", "desired reference", "future rows", "ROM hidden state"],
                },
                indent=2,
            )
            + "\n"
        )
        per_run.append({"run_id": run_id, "n_ticks": int(len(ticks)), "frozen_split": info[f"{run_id:05d}"]["frozen_split"]})
        if factors_vary:
            print(
                f"run {run_id:05d}: TIME-VARYING plant ({factor_source}); "
                f"z_target range {np.round(np.ptp(z_track, axis=0), 4).tolist()}",
                flush=True,
            )
        else:
            print(
                f"run {run_id:05d}: constant plant; latent target "
                f"{np.round(z_track[0], 4).tolist()}",
                flush=True,
            )

    (out / "index.json").write_text(
        json.dumps(
            {
                "format": "stage31_causal_aam_dataset_v1",
                "root": str(root),
                "rows_root": str(rows_root),
                "sd_model": str(sd_model),
                "feature_dim": AAM_FEATURE_DIM,
                "feature_groups": AAM_FEATURE_GROUPS,
                "causal_row": "joint_q + joint_qdot + dP + previous_valve",
                "latent_dim": latent_dim,
                "lags": list(lags),
                "runs": per_run,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"built Stage 31 AAM dataset for {len(per_run)} runs -> {out}")


if __name__ == "__main__":
    main()

