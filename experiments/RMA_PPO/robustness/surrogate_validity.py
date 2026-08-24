"""Is the frozen SD surrogate credible on a given episode at all?

A learned one-step delta model is integrated for the whole horizon here, so its
error accumulates.  If that accumulated drift is comparable to the motion the
task itself demands, every closed-loop number measured on the surrogate is
drift, not control -- and no stress ratio computed against it means anything.

The test is open loop and uses no policy: replay the RECORDED valve commands
through the surrogate from a recorded initial state and compare the predicted
extension trajectory with the recorded one.  Nothing about a controller, a
stress level or a reward enters, so the screen cannot bias a robustness curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from robustness_env import HISTORY_LAG, RobustnessEnv, Episode


@dataclass(frozen=True)
class DriftReport:
    """Per-episode open-loop replay drift against recorded motion."""

    drift_m: np.ndarray          # |predicted - recorded| extension at the horizon
    motion_m: np.ndarray         # recorded peak excursion inside the window
    fraction: np.ndarray         # drift / motion, the surrogate's noise-to-signal

    def passing(self, max_fraction: float) -> np.ndarray:
        return self.fraction <= max_fraction

    def summary(self, max_fraction: float) -> dict:
        keep = self.passing(max_fraction)
        return {
            "episodes": int(len(self.fraction)),
            "median_drift_mm": float(np.median(self.drift_m) * 1000.0),
            "median_motion_mm": float(np.median(self.motion_m) * 1000.0),
            "median_noise_to_signal": float(np.median(self.fraction)),
            "p90_noise_to_signal": float(np.percentile(self.fraction, 90)),
            "max_fraction": max_fraction,
            "passing": int(keep.sum()),
            "passing_fraction": float(keep.mean()),
        }


@torch.no_grad()
def replay_drift(env: RobustnessEnv, episodes: list[Episode], horizon: int) -> DriftReport:
    """Replay recorded valve commands through the surrogate, episode by episode."""
    drift, motion = [], []
    for ep in episodes:
        run = env.runs[ep.run_id]
        st = ep.start
        s_true = torch.from_numpy(run["s"][st:st + horizon + 1])
        q_true = torch.from_numpy(run["q"][st:st + horizon + 1])
        u_true = torch.from_numpy(run["u"][st:st + horizon + 1])
        window = np.concatenate([
            (run["s"][st - HISTORY_LAG + 1:st + 1] - env._s_mean.numpy()) / env._s_std.numpy(),
            (run["u"][st - HISTORY_LAG + 1:st + 1] - env._u_mean.numpy()) / env._u_std.numpy(),
        ], axis=-1)
        belief = env.belief.init_from_window(torch.from_numpy(window[None]).float())[0]

        s = s_true[0].clone()
        for t in range(horizon):
            s_n = (s - env._s_mean) / env._s_std
            u_n = (u_true[t] - env._u_mean) / env._u_std
            q_n = (q_true[t] - env._q_mean) / env._q_std
            belief = env.belief.step(torch.cat((s_n, u_n))[None], belief[None])[0]
            delta = env.sd(torch.cat((s_n, u_n, q_n, belief))[None])[0] * env._y_std + env._y_mean
            s = s + delta

        drift.append(float((s[:4] - s_true[-1, :4]).abs().max()))
        excursion = (s_true[:, :4] - s_true[0, :4]).abs().max()
        motion.append(float(excursion))

    drift = np.asarray(drift)
    motion = np.asarray(motion)
    return DriftReport(drift_m=drift, motion_m=motion,
                       fraction=drift / np.maximum(motion, 1.0e-9))


def screen(env: RobustnessEnv, episodes: list[Episode], *, horizon: int,
           max_fraction: float) -> tuple[list[Episode], DriftReport]:
    """Keep only the episodes on which the surrogate is credible."""
    report = replay_drift(env, episodes, horizon)
    keep = report.passing(max_fraction)
    return [ep for ep, ok in zip(episodes, keep) if ok], report
