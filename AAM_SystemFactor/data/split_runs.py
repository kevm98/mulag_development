"""Deterministic run-level splitting with zero window leakage."""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def split_runs_deterministic(
    run_ids: List[str],
    run_types: List[str],
    source_types: List[str],
    n_samples: List[int],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split runs deterministically into train/val/test.

    Never splits windows from one run across multiple splits.

    Returns:
        train_mask, val_mask, test_mask: boolean arrays
        assignments: str array ['train', 'val', 'test']
    """
    rng = np.random.RandomState(seed)
    N = len(run_ids)

    # Process open-loop separately: 4 runs -> 2/1/1
    ol_idx = [i for i, s in enumerate(source_types) if s == 'open_loop']
    cl_idx = [i for i, s in enumerate(source_types) if s == 'closed_loop']

    assignments = np.array(['train'] * N, dtype=object)

    # Open-loop: 2/1/1
    if len(ol_idx) >= 4:
        ol_perm = rng.permutation(ol_idx)
        assignments[ol_perm[0:2]] = 'train'
        if len(ol_perm) >= 3:
            assignments[ol_perm[2]] = 'val'
        if len(ol_perm) >= 4:
            assignments[ol_perm[3]] = 'test'
    elif len(ol_idx) >= 3:
        ol_perm = rng.permutation(ol_idx)
        assignments[ol_perm[0]] = 'train'
        assignments[ol_perm[1]] = 'val'
        assignments[ol_perm[2]] = 'test'

    # Closed-loop: stratified 80/10/10
    if cl_idx:
        cl_perm = rng.permutation(cl_idx)
        n_cl = len(cl_idx)
        n_train = max(1, int(n_cl * train_frac))
        n_val = max(1, int(n_cl * val_frac))
        assignments[cl_perm[:n_train]] = 'train'
        assignments[cl_perm[n_train:n_train + n_val]] = 'val'
        assignments[cl_perm[n_train + n_val:]] = 'test'

    train_mask = assignments == 'train'
    val_mask = assignments == 'val'
    test_mask = assignments == 'test'

    return train_mask, val_mask, test_mask, assignments


def save_split_manifest(
    path: str,
    run_ids: List[str],
    source_types: List[str],
    run_types: List[str],
    assignments: np.ndarray,
    n_samples: List[int],
    n_windows: List[int],
) -> None:
    """Save run split manifest as CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'source_file', 'canonical_source', 'source_type',
            'run_id', 'run_type', 'split',
            'number_of_samples', 'number_of_windows',
        ])
        for i in range(len(run_ids)):
            writer.writerow([
                run_ids[i],
                'per_run_mat' if source_types[i] == 'open_loop' else 'curriculum_mat',
                source_types[i],
                run_ids[i],
                run_types[i],
                assignments[i],
                n_samples[i],
                n_windows[i],
            ])


def verify_no_leakage(
    train_ids: set,
    val_ids: set,
    test_ids: set,
) -> bool:
    """Assert zero overlap of run IDs across splits."""
    assert len(train_ids & val_ids) == 0, "Train/val leakage!"
    assert len(train_ids & test_ids) == 0, "Train/test leakage!"
    assert len(val_ids & test_ids) == 0, "Val/test leakage!"
    return True
