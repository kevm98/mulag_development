# AAM SystemFactor

Standalone Adaptive Arm Module that estimates seven physically meaningful
hydraulic system factors from recent four-cylinder histories.

## Overview

Given a causal four-cylinder history window H_t ∈ R^(10×20), the AAM
predicts:

```
e_hat_t = phi(H_t) ∈ R^7
```

where the seven factors are:
- **e_comp** — Compliance (relative chamber pressure stiffness)
- **e_flow** — Valve flow factor (Cd deviation from nominal)
- **e_pressure** — Relief pressure factor (Pset deviation from nominal)
- **e_seal_1..4** — Per-cylinder seal factors

## Project Structure

```
AAM_SystemFactor/
├── data/          # Data pipeline (MAT decoder, windows, splitting, normalization)
├── models/        # LSTM model, baselines, runtime bundle
├── training/      # Training loop, losses, main script
├── evaluation/    # Metrics, identifiability, plots
├── scripts/       # Shell scripts for audit, training, full pipeline
├── tests/         # 27 unit tests
├── checkpoints/   # Runtime bundle
├── outputs/       # Scalers, manifest, history, plots, evaluation results
└── reports/       # Dataset audit, implementation report
```

## Quick Start

### Data Audit
```bash
bash AAM_SystemFactor/scripts/run_data_audit.sh
```

### Smoke Test (5 epochs, 100 runs)
```bash
PYTHONPATH=/home/rrlab/rrlab_mulag \
  /home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  AAM_SystemFactor/training/train_aam.py --T 10 --epochs 5 --warmup
```

### Full Training
```bash
PYTHONPATH=/home/rrlab/rrlab_mulag \
  /home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  AAM_SystemFactor/training/train_aam.py --T 10 --epochs 100
```

### Inference
```python
from AAM_SystemFactor.models.model_bundle import AAMPredictor
predictor = AAMPredictor('AAM_SystemFactor/checkpoints/aam_system_factor_runtime.pt')
factors = predictor.predict(history)  # history: (B, 10, 20)
```

### Tests
```bash
PYTHONPATH=/home/rrlab/rrlab_mulag \
  /home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  -m pytest AAM_SystemFactor/tests/test_pipeline.py -v
```

## Results

The LSTM model beats the nominal baseline by **48.6%** overall (mean RMSE).
Seal factors show strongest improvement (46-67% per factor). See
`reports/final_implementation_report.md` for full details.

## Data Sources

- **Open-loop:** 4 runs from Simscape valve_excitation (64s each, dt=0.01s)
- **Closed-loop:** 1000 runs from backstepping controller (22s each, dt=0.05s)
- Run types: Nominal, Random Hydraulics, Random Mechanics, Fully Randomized

## Requirements

- Python 3.11+ (env_isaaclab conda environment)
- PyTorch, NumPy, SciPy, h5py, matplotlib
- No IsaacLab initialization required
