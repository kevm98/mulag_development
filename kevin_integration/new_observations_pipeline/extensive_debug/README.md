# Extensive Debug — Mulag Hydraulic Arm Direction/Mapping Investigation

This folder contains diagnostic scripts to empirically investigate the mapping
between Isaac raw cylinder/joint coordinates, the Simscape/model convention used
by learned ID/FD/ROM models, TCP/end-effector movement, PPO action convention,
and actual actuator command effects in simulation.

## Folder Structure

```
extensive_debug/
├── README.md
├── run_all_debug_tests.py          # Orchestrator: run all tests sequentially
├── debug_configs/                  # Debug-specific config overrides (if needed)
├── scripts/                        # Individual test scripts
│   ├── debug_utils.py              # Shared utilities (env construction, save helpers)
│   ├── test1_articulation_inspection.py
│   ├── test3_single_joint_perturbation.py
│   ├── test4_single_cylinder_perturbation.py
│   ├── test5_cylinder_sign_order_matrix.py
│   ├── test6_numerical_cylinder_jacobian.py
│   ├── test7_q4_q5_coupling.py
│   ├── test8_safe_velocity_tracking.py
│   └── test9_ab_comparison.py
├── logs/                           # Per-test log files
├── results/                        # JSON, CSV result files
├── plots/                          # Generated plots (if matplotlib available)
└── reports/                        # Markdown reports
    ├── current_pipeline_summary.md
    ├── cylinder_sign_order_matrix.md
    ├── numerical_jacobian_validation.md
    ├── q4_q5_coupling.md
    ├── ab_comparison.md
    └── FINAL_EXTENSIVE_DEBUG_REPORT.md
```

## Prerequisites

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/rrlab_mulag
```

## Quick Start: Run All Tests

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/run_all_debug_tests.py \
  --device cuda:0
```

This runs all tests sequentially and saves all outputs. Tests that fail are
logged with their error messages and the orchestrator continues.

## Individual Test Commands

### Test 1: Articulation Inspection

Prints all joint names, body names, limits, actuator configs, TCP pose.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/scripts/test1_articulation_inspection.py \
  --device cuda:0
```

With GUI viewer:
```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/scripts/test1_articulation_inspection.py \
  --device cuda:0 --viewer
```

### Test 3: Single-Joint Perturbation

Perturbs each arm joint by +0.005 rad and measures TCP/cylinder response.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/scripts/test3_single_joint_perturbation.py \
  --device cuda:0
```

### Test 4: Single-Cylinder Perturbation

Perturbs each cylinder by +0.002 m and measures which joints moved and TCP response.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/scripts/test4_single_cylinder_perturbation.py \
  --device cuda:0
```

### Test 5: Cylinder Sign/Order Matrix

Estimates the mapping between Isaac and Simscape conventions using perturbation
results or inline measurements.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/scripts/test5_cylinder_sign_order_matrix.py \
  --device cuda:0
```

### Test 6: Numerical Cylinder Jacobian

Builds J_cyl = d(TCP)/d(s_cylinder) via central finite differences and validates
with damped pseudoinverse.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/scripts/test6_numerical_cylinder_jacobian.py \
  --device cuda:0
```

### Test 7: q4/q5 Coupling Check

Investigates whether Messerkopf_joint is mechanically coupled to Messerkopf_Schwenk_joint.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/scripts/test7_q4_q5_coupling.py \
  --device cuda:0
```

### Test 8: Safe Velocity Tracking

Conservative velocity/cylinder tracking controller with PD + feedforward.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/scripts/test8_safe_velocity_tracking.py \
  --device cuda:0
```

### Test 9: A/B Comparison

Compares current convention vs corrected sign, rate limiting, and numerical Jacobian approaches.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/extensive_debug/scripts/test9_ab_comparison.py \
  --device cuda:0
```

## Test Dependencies

```
Test 1 (articulation)    — no dependencies
Test 3 (joint perturb)   — no dependencies
Test 4 (cyl perturb)     — no dependencies
Test 5 (sign/order)      — uses Test 3+4 results if available, or runs inline
Test 6 (Jacobian)        — no dependencies
Test 7 (q4/q5)           — no dependencies
Test 8 (velocity track)  — no dependencies
Test 9 (A/B comparison)  — no dependencies
```

## Expected Runtime

Each test takes ~30–120 seconds on a single GPU (including env construction).
Total sequential runtime: ~10–20 minutes.

## Troubleshooting

If IsaacLab fails to launch:
- Check GPU availability: `nvidia-smi`
- Ensure conda env is activated: `conda activate env_isaaclab`
- Verify IsaacLab import: `python -c "import isaaclab"`

If a specific test fails, the error is saved in both the JSON result file
and the log file. Re-run with `--viewer` to see the simulation visually.
