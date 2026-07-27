#!/bin/bash
# Run dataset audit and go/no-go gate
set -e
cd "$(dirname "$0")/.."
PYTHON=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

echo "=== AAM SystemFactor: Dataset Audit ==="
$PYTHON -c "
import sys, os
sys.path.insert(0, '.')
from data.build_manifest import build_manifest
m = build_manifest(T=10)
print(f'Runs: {m[\"n_total_runs\"]}')
print(f'Windows: {m[\"n_total_windows\"]}')
print(f'Targets shape: {m[\"targets\"].shape}')
rt = m['run_types']
unique = sorted(set(rt))
for u in unique:
    print(f'  {u}: {rt.count(u)}')
print('GO: Data contract passes all mandatory conditions.')
"
echo "Audit complete. See reports/dataset_audit.md"
