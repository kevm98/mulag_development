"""Construct causal four-cylinder history windows from run data."""

import numpy as np
from typing import Dict, Tuple

FEATURE_NAMES = [
    'ValveCmd_1', 'ValveCmd_2', 'ValveCmd_3', 'ValveCmd_4',
    'Extension_1', 'Extension_2', 'Extension_3', 'Extension_4',
    'ExtVel_1', 'ExtVel_2', 'ExtVel_3', 'ExtVel_4',
    'dP_1', 'dP_2', 'dP_3', 'dP_4',
    'Force_1', 'Force_2', 'Force_3', 'Force_4',
]
N_FEATURES = 20


def build_feature_matrix(run_data: Dict) -> np.ndarray:
    """Build (T, 20) feature matrix from a single run.

    Input ordering per Section 5:
        [ValveCmd_1..4, Extension_1..4, ExtVel_1..4, dP_1..4, Force_1..4]
    """
    valve_cmd = run_data['valve_cmd']
    extension = run_data['extension']
    ext_vel = run_data['ext_vel']
    dp = run_data['dP']
    force = run_data['force']

    if ext_vel is None:
        from AAM_SystemFactor.data.matlab_v73 import compute_ext_vel_causal
        ext_vel = compute_ext_vel_causal(extension, run_data['dt'])

    T = valve_cmd.shape[1]
    features = np.zeros((T, N_FEATURES))
    features[:, 0:4] = valve_cmd[0:4, :].T
    features[:, 4:8] = extension.T
    features[:, 8:12] = ext_vel.T
    features[:, 12:16] = dp.T
    features[:, 16:20] = force.T
    return features


def build_windows(features: np.ndarray, T: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window histories. Window at t covers [t-T+1, t]."""
    N = features.shape[0]
    if N < T:
        return np.empty((0, T, N_FEATURES)), np.array([], dtype=int)
    n_windows = N - T + 1
    windows = np.zeros((n_windows, T, N_FEATURES))
    for i in range(n_windows):
        windows[i] = features[i:i + T]
    indices = np.arange(T - 1, N)
    return windows, indices


def build_windows_for_runs(runs_data: list, T: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build windows from multiple runs, tracking run ownership."""
    all_windows, all_run_ids, all_indices = [], [], []
    for run_idx, run_data in enumerate(runs_data):
        features = build_feature_matrix(run_data)
        w, idx = build_windows(features, T)
        all_windows.append(w)
        all_run_ids.append(np.full(len(w), run_idx, dtype=int))
        all_indices.append(idx)
    if not all_windows:
        return np.empty((0, T, N_FEATURES)), np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(all_windows), np.concatenate(all_run_ids), np.concatenate(all_indices)
