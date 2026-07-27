"""Build canonical dataset manifest from open-loop and closed-loop sources."""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from AAM_SystemFactor.data.matlab_v73 import (
    load_open_loop_run, load_closed_loop_run, compute_ext_vel_causal,
)
from AAM_SystemFactor.data.factor_targets import compute_targets, TARGET_NAMES
from AAM_SystemFactor.data.cylinder_windows import build_windows

# Dataset roots
THESIS_BASE = os.path.expanduser(
    "~/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis/"
    "Domain_Randomization_Data_Collection-20260720T142849Z-1-001/"
    "Domain_Randomization_Data_Collection"
)

OL_RUNS_DIR = os.path.join(THESIS_BASE, "Randomization_open_loop/Excitation_Finetune_25/runs")
CL_MAT = os.path.join(THESIS_BASE, "Randomization_closed_loop/Hydraulic_Mechanic_Dataset_Curriculum_1000.mat")

OL_RUN_FILES = sorted([
    f for f in os.listdir(OL_RUNS_DIR) if f.endswith('.mat')
])

N_CL_RUNS = 1000


def build_manifest(T: int = 10) -> Dict:
    """Build the complete dataset manifest.

    Returns:
        manifest dict with keys:
            runs: list of per-run metadata
            targets: (N_runs, 7) array
            splits: assignment array
            T: history length
            n_total_windows: total windows across all runs
    """
    runs = []
    targets = []
    source_types = []
    run_types = []

    # Open-loop
    for fname in OL_RUN_FILES:
        path = os.path.join(OL_RUNS_DIR, fname)
        run_data = load_open_loop_run(path)

        # Ensure ext_vel is present
        if run_data['ext_vel'] is None:
            run_data['ext_vel'] = compute_ext_vel_causal(
                run_data['extension'], run_data['dt']
            )

        t = compute_targets(run_data['params'])
        n_windows = max(0, run_data['n_transitions'] - T + 1)

        runs.append({
            'run_id': fname.replace('.mat', ''),
            'source_file': path,
            'source_type': 'open_loop',
            'run_type': run_data['run_type'],
            'n_transitions': run_data['n_transitions'],
            'dt': run_data['dt'],
            'n_windows': n_windows,
        })
        targets.append(t)
        source_types.append('open_loop')
        run_types.append(run_data['run_type'])

    # Closed-loop
    for i in range(N_CL_RUNS):
        run_data = load_closed_loop_run(CL_MAT, i)
        if run_data['ext_vel'] is None:
            run_data['ext_vel'] = compute_ext_vel_causal(
                run_data['extension'], run_data['dt']
            )

        t = compute_targets(run_data['params'])
        n_windows = max(0, run_data['n_transitions'] - T + 1)

        runs.append({
            'run_id': f'cl_{i:04d}',
            'source_file': CL_MAT,
            'source_type': 'closed_loop',
            'run_type': run_data['run_type'],
            'n_transitions': run_data['n_transitions'],
            'dt': run_data['dt'],
            'n_windows': n_windows,
            'run_index': i,
        })
        targets.append(t)
        source_types.append('closed_loop')
        run_types.append(run_data['run_type'])

    return {
        'runs': runs,
        'targets': np.array(targets),
        'T': T,
        'source_types': source_types,
        'run_types': run_types,
        'n_total_runs': len(runs),
        'n_total_windows': sum(r['n_windows'] for r in runs),
    }


if __name__ == '__main__':
    manifest = build_manifest(T=10)
    print(f"Total runs: {manifest['n_total_runs']}")
    print(f"Total windows: {manifest['n_total_windows']}")
    print(f"Targets shape: {manifest['targets'].shape}")
    print(f"Target names: {TARGET_NAMES}")
    # Quick stats per source
    for st in ['open_loop', 'closed_loop']:
        runs_st = [r for r in manifest['runs'] if r['source_type'] == st]
        print(f"\n{st}: {len(runs_st)} runs, {sum(r['n_windows'] for r in runs_st)} windows")
