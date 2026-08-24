"""Pre-declared evaluation protocol: episode set, thresholds, classification.

Everything in this file is fixed BEFORE any stress sweep runs and is identical
at every stress level.  Nothing here may be re-tuned after seeing curves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from robustness_env import DATASET_ROOT, HISTORY_LAG, Episode

# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------
HORIZON = 48                 # 48 ticks x 50 ms = 2.4 s
N_EPISODES = 64              # matched episodes, identical at every stress level
EPISODE_SEED = 20260825      # frozen
SUPPORT_Z = 3.0              # frozen PPO training-support gate on extensions

# A tracking task the plant can satisfy without moving is not a tracking task.
# The first episode set drew from the static bulk of the dataset: its median
# cylinder excursion over the whole 2.4 s horizon was 0.03 mm, so a do-nothing
# null controller beat the frozen PPO and every stress ratio measured surrogate
# drift instead of control.  Both screens below use ONLY recorded data and the
# frozen normalizer, so they remain independent of stress level and of any
# measured closed-loop performance.
MIN_MOTION_M = 0.005         # >= 5 mm recorded peak excursion inside the window
MAX_DRIFT_FRACTION = 0.25    # open-loop replay error at the horizon must stay
                             # below this fraction of the recorded excursion

# ---------------------------------------------------------------------------
# Pre-declared degradation / breakdown criteria (section 11 and 12)
# ---------------------------------------------------------------------------
DEGRADATION_RATIO = 1.5      # E(rho)/E(nominal) >= 1.5 -> degradation onset
RETURN_DROP = 0.20           # >= 20% worse normalized return -> secondary onset
BREAKDOWN_RATIO = 3.0        # E(rho)/E(nominal) >= 3.0 -> functional breakdown
BREAKDOWN_SATURATION = 0.90  # or >= 90% of steps with a saturated action
ROBUST_RATIO = 1.1           # below this the plant change is not meaningful


def frozen_episode_set(actor_obs_mean: torch.Tensor, actor_obs_std: torch.Tensor,
                       *, min_motion_m: float = MIN_MOTION_M,
                       n_episodes: int = N_EPISODES) -> list[Episode]:
    """Sample matched episodes that lie in the PPO's support AND actually move.

    Selection uses ONLY the frozen policy's observation-normalizer statistics
    and the frozen dataset.  It is independent of any stress level and of any
    measured performance, so the same episodes serve every sweep point.
    """
    mu = actor_obs_mean.flatten()[12:16].numpy()
    sd = actor_obs_std.flatten()[12:16].numpy()
    sd_root = DATASET_ROOT / "sd" / "dataset"
    row_root = DATASET_ROOT / "aam" / "rom_rows"

    candidates: list[Episode] = []
    for rid in range(118):
        sd_path = sd_root / f"{rid:05d}" / "samples.npz"
        row_path = row_root / f"{rid:05d}" / "rom_rows.npz"
        if not sd_path.is_file() or not row_path.is_file():
            continue
        with np.load(sd_path) as s_npz, np.load(row_path) as r_npz:
            n = min(len(s_npz["s_h"]), len(r_npz["desired_cylinder_position"]))
            x = s_npz["s_h"][:n, :4].astype(np.float32)
        inside = (np.abs((x - mu) / sd).max(1) <= SUPPORT_Z)
        first = HISTORY_LAG - 1
        for start in range(first, n - HORIZON - 1):
            if not inside[start:start + HORIZON + 1].all():
                continue
            window = x[start:start + HORIZON + 1]
            if float(np.abs(window - window[0]).max()) < min_motion_m:
                continue
            candidates.append(Episode(run_id=rid, start=start))
    if len(candidates) < n_episodes:
        raise RuntimeError(f"only {len(candidates)} in-support moving windows; need {n_episodes}")

    # Spread the draw across runs so no single trajectory dominates.
    rng = np.random.default_rng(EPISODE_SEED)
    by_run: dict[int, list[Episode]] = {}
    for ep in candidates:
        by_run.setdefault(ep.run_id, []).append(ep)
    runs = sorted(by_run)
    rng.shuffle(runs)
    chosen: list[Episode] = []
    cursor = 0
    while len(chosen) < n_episodes:
        pool = by_run[runs[cursor % len(runs)]]
        chosen.append(pool[int(rng.integers(0, len(pool)))])
        cursor += 1
    return chosen[:n_episodes]


@dataclass(frozen=True)
class ValidityEnvelope:
    """Surrogate-validity limits measured on the frozen SD training data."""

    state_z_max: float
    action_z_max: float
    extension_min: tuple
    extension_max: tuple

    def to_dict(self) -> dict:
        return {
            "sd_state_z_max": self.state_z_max,
            "sd_action_z_max": self.action_z_max,
            "extension_min": list(self.extension_min),
            "extension_max": list(self.extension_max),
        }


def measure_validity_envelope(metrics: dict, *, quantile: float = 0.9999) -> ValidityEnvelope:
    """Derive the surrogate's own validity envelope from its training support.

    The SD model is credible where its inputs resemble what it was fitted on.
    We take a high quantile of the standard score that the FROZEN RECORDED data
    itself attains; anything beyond that is extrapolation by the surrogate, not
    a controller result.
    """
    s_mean = np.asarray(metrics["s_stats"]["mean"], dtype=np.float64)
    s_std = np.asarray(metrics["s_stats"]["std"], dtype=np.float64)
    u_mean = np.asarray(metrics["u_stats"]["mean"], dtype=np.float64)
    u_std = np.asarray(metrics["u_stats"]["std"], dtype=np.float64)
    sd_root = DATASET_ROOT / "sd" / "dataset"

    state_z, action_z, ext = [], [], []
    for rid in range(118):
        path = sd_root / f"{rid:05d}" / "samples.npz"
        if not path.is_file():
            continue
        with np.load(path) as npz:
            s = npz["s_h"].astype(np.float64)
            u = npz["u"].astype(np.float64)
        state_z.append(np.abs((s - s_mean) / s_std).max(1))
        action_z.append(np.abs((u - u_mean) / u_std).max(1))
        ext.append(s[:, :4])
    state_z = np.concatenate(state_z)
    action_z = np.concatenate(action_z)
    ext = np.concatenate(ext, axis=0)
    return ValidityEnvelope(
        state_z_max=float(np.quantile(state_z, quantile)),
        action_z_max=float(np.quantile(action_z, quantile)),
        extension_min=tuple(float(v) for v in ext.min(0)),
        extension_max=tuple(float(v) for v in ext.max(0)),
    )


def classify(*, error_ratio: float, saturation: float, surrogate_valid: bool,
             is_nominal: bool) -> str:
    """Assign one of the five pre-declared regions to a sweep point."""
    if not surrogate_valid:
        return "SURROGATE_INVALID"
    if is_nominal:
        return "NOMINAL"
    if error_ratio >= BREAKDOWN_RATIO or saturation >= BREAKDOWN_SATURATION:
        return "BREAKDOWN"
    if error_ratio >= DEGRADATION_RATIO:
        return "DEGRADED"
    return "ROBUST"
