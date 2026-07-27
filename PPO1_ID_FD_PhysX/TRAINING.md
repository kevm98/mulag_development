# ID-FD training and fine-tuning provenance

This document describes the exact lineage of the two checkpoints used by the current MULAG
arm controller:

```text
models/id/best_id_cylinder_control.pt
models/fd/best_fd_cylinder_strict.pt
```

The runtime pair was not trained in one step. It came from four stages:

1. supervised inverse-dynamics (ID) and forward-dynamics (FD) pretraining;
2. rank-preserving FD fine-tuning;
3. paired ID/FD control fine-tuning;
4. strict open-loop FD retraining with rollout and kinematic consistency.

The final deployed controller then wraps the learned pair in a reference-dominant hybrid.
That runtime repair is part of the accepted control contract and must not be confused with an
additional neural-network training stage.

## 1. Model semantics

### Inverse dynamics

The ID network maps the current hydraulic/cylinder state and a desired cylinder increment to
a change in valve command.

```text
input, 28 values:
  Extension[4]
  ExtensionVelocity[4]
  dP[4]
  Force[4]
  TargetDeltaExtension[4]
  TargetDeltaExtensionVelocity[4]
  PreviousValveCmd[4]

output, 4 values:
  DeltaValveCmd[4]
```

The deployed valve command is `PreviousValveCmd + DeltaValveCmd`, clipped by the checkpoint
limits `[0.95, 0.95, 0.8, 0.8]`.

Architecture:

- `IDCylinderMLP`
- 28 inputs, 4 outputs
- hidden width 512
- four residual blocks
- SiLU activations
- layer normalization
- dropout 0.02

### Forward dynamics

The FD network predicts the cylinder-state increment over its native 0.05 s sample time.

```text
input, 20 values:
  ValveCmd[4]
  Extension[4]
  ExtensionVelocity[4]
  dP[4]
  Force[4]

output, 12 values:
  DeltaExtension[4]
  DeltaExtensionVelocity[4]
  DeltaDP[4]
```

Architecture:

- `FDCylinderMLP`
- 20 inputs, 12 outputs
- hidden width 512
- four residual blocks
- SiLU activations
- layer normalization
- dropout 0.02

Every checkpoint contains its own input/output means and standard deviations. Runtime must
normalize with the statistics stored in the checkpoint being evaluated; mixing statistics
between checkpoints invalidates comparisons.

## 2. Stage A: original supervised training

### Original ID

Source data: `models/id/ID_cylinder_data.mat`.

Training configuration embedded in the checkpoint:

- seed 42
- maximum 300 epochs
- selected epoch 120
- batch size 8192; validation batch size 16384
- Adam-style learning rate 1e-3
- weight decay 1e-4
- Smooth-L1 loss with beta 0.5
- gradient norm limit 10
- early-stopping patience 45
- previous valve command enabled
- delta-valve target mode
- saturation samples weighted 1.25

The selected model had validation loss 0.033616. Raw validation R2 by valve axis was
0.99962, 0.99727, 0.99808, and 0.99737. Those high supervised scores showed that the model
could reproduce logged inverse labels, but did not by themselves prove that the composite
`ID -> FD` map had useful closed-loop authority.

### Original FD

Source data: `models/fd/FD_cylinder_data.mat`.

Training configuration embedded in the checkpoint lineage:

- seed 42
- maximum 300 epochs
- selected epoch 291
- batch size 8192; validation batch size 16384
- learning rate 1e-4
- weight decay 1e-4
- normalized MSE on all three output groups
- rollout-loss weight 0.25
- rollout horizons 2, 5, 10, and 20 steps
- rollout batch size 512
- checkpoint selected by validation rollout loss

The original FD scored very well on its own MATLAB distribution, including grouped R2 near
0.999 for extension increments. Later strict excitation proved that these scores were partly
kinematic extrapolation: removing the valve input barely changed long-horizon error. High
one-step R2 was therefore not accepted as proof of valve authority.

## 3. Stage B: rank-preserving FD fine-tuning

Script: `scripts/finetune_fd_rank.py`

Output: `models/fd/best_fd_cylinder_rank4.pt`

The original FD's valve-to-extension differential map was poorly conditioned for inversion.
This stage computed the valve partial effect after regressing the other 16 state inputs out of
both valve and target. A finite-difference Jacobian loss was added while retaining supervised
validation loss.

Key settings:

- normalized-space finite-difference epsilon 0.01
- Jacobian weight 1.0
- learning rate 1e-4
- no weight decay
- original FD normalization contract retained

The selected checkpoint had a rank-4 valve Jacobian with condition number about 2.29 and did
not sacrifice the original validation objective. This checkpoint is retained as provenance,
but is not the active FD.

## 4. Stage C: paired ID/FD control fine-tuning

Script: `scripts/finetune_idfd_control.py`

Outputs:

```text
models/id/best_id_cylinder_control.pt
models/fd/best_fd_cylinder_control.pt
```

This stage jointly fine-tuned the ID and rank-4 FD for target intervention. Target blocks were
shuffled across valid states to prevent the networks from succeeding by merely reconstructing
the logged state trajectory. The control loss used the paired differential map:

```text
u_target = ID(state, target, previous_u)
u_zero   = ID(state, zero_target, previous_u)

predicted_delta = FD(u_target, state) - FD(u_zero, state)
control_loss    = normalized_MSE(predicted_delta_extension, target_delta_extension)
```

Supervised ID and FD losses remained in the objective to limit regression from their original
data contracts.

The promoted checkpoint metadata records:

| Metric | Before paired tuning | After paired tuning |
|---|---:|---:|
| Composite normalized control MSE | 1.0011 | 0.1086 |
| Axis slopes | 0.003, 0.002, 0.001, 0.004 | 0.818, 0.928, 0.849, 0.912 |
| Axis correlations | 0.135, 0.090, 0.037, 0.249 | 0.920, 0.967, 0.933, 0.958 |

The resulting ID is still the active ID. The paired-control FD became the source checkpoint
for strict retraining.

## 5. Stage D: strict open-loop FD dataset

Raw source: `training_data/Excitation_FD_Strict.csv`

The strict export contains four 64 s runs sampled at 0.01 s, 6400 samples per run, and 49
columns. All four valves receive a rank-4 MIMO PRBS command. The same excitation is applied to
four distinct plant draws:

1. nominal;
2. randomized hydraulics;
3. randomized mechanics;
4. fully randomized hydraulics and mechanics.

The columns contain run/time/valid flags, current valve and state features, and the 12 next
state deltas. Dataset validation established:

- four of four distinct state trajectories;
- all target deltas agree with within-run `k+1-k` differences;
- time step is 0.01 s to numerical precision;
- valve history predicts pressure response with R2 about 0.80-0.85;
- valve conditioning VIF is about 7-10 rather than degenerate;
- relief saturation occurs roughly 9-22% of the time.

The older excitation export was rejected because half of its runs duplicated state
trajectories and the valve was not causally identifiable. That export and its reports are not
part of the maintained workflow.

### Converting to the FD time base

Script: `scripts/build_strict_fd_dataset.py`

Output: `models/fd/strict_fd_transitions_0p05.npz`

Five 0.01 s samples are combined into one 0.05 s FD transition. Inputs use the valve held at
the beginning of the window; the dataset also records the mean valve over the five samples for
analysis.

Each continuous run is divided into 20 temporal blocks. Every fifth block is validation, and
a five-sample guard is removed at each boundary. This blocked-interleaved split gives train
and validation samples from comparable pose and energy regimes without allowing a transition
window to cross a split boundary. A simple last-20% split was rejected because it changed
target spread by 1.3-2.3 times and tested extrapolation to a different regime.

## 6. Stage E: strict FD retraining

Script: `scripts/train_fd_strict.py`

Source checkpoint: `models/fd/best_fd_cylinder_control.pt`

Output checkpoint: `models/fd/best_fd_cylinder_strict.pt`

The promoted run used:

- 150 epochs
- learning rate 3e-4
- strict-data Huber loss
- original-MAT replay weight 0.25
- 10-step rollout horizon (0.5 s)
- rollout weight 2.0
- trapezoidal kinematic-consistency weight 10.0
- constant-Jacobian weight 0.0
- combined strict/MAT normalization statistics

The objective was:

```text
L = Huber(strict one-step prediction)
  + 0.25 * Huber(original MAT replay)
  + 2.0  * rollout_loss(10 steps)
  + 10.0 * trapezoid_consistency
```

The consistency term enforces the measured identity

```text
DeltaExtension = dt * ExtensionVelocity
               + 0.5 * dt * DeltaExtensionVelocity
```

which holds in the strict data at R2 0.994-0.9997. Huber loss is used because rare relief
events otherwise dominate MSE. MAT replay remains because ablation showed that it improves
state coverage and extension R2 while reducing valve-gain error.

A constant valve-Jacobian anchor is deliberately absent. Measured valve authority changes by
15-75 times across pressure quartiles and sometimes changes sign. Pinning one global Jacobian
gave an attractive mean gain but collapsed one-step prediction, so it was rejected.

### Strict validation result

| Metric | Paired-control FD before strict retrain | Strict FD |
|---|---:|---:|
| Valve-zero error ratio at 0.5 s | about 0.94 (no authority) | about 1.81 |
| Extension rollout RMSE at 0.5 s | 0.0752 m | 0.0124-0.0125 m |
| Extension rollout RMSE at 1.0 s | 0.1579 m | 0.0273 m |
| DeltaExtension R2 | -5.7 to 0.60 | 0.76 to 0.98 |

DeltaExtensionVelocity remains harder, with modest axis R2, because a memoryless 0.05 s model
cannot fully reconstruct acceleration during relief transitions. The controller integrates
DeltaExtension, which is the accurately modeled and directly validated quantity.

## 7. Runtime composition and why it is hybrid

Pure learned paired ID/FD passed fresh 10 mm cylinder steps but failed sustained coupled 80 mm
motion. The learned composite is cross-coupled and was not safe as the sole owner of the PhysX
target increment. The accepted runtime therefore uses the learned pair as a bounded residual
around the intuitive requested cylinder increment:

```text
paired_fd_delta = FD(ID(target)) - FD(ID(zero_target))

applied_delta = 0.8 * requested_cylinder_delta
              + 0.2 * paired_fd_delta
```

The DLS controller inverts this same blended map, so prediction and execution have identical
semantics. When blending is active, the target integrator closes on
`goal - commanded_target`, not measured linkage deflection. This matches direct cylinder-drive
behavior and prevents target-lead windup.

Cylinder-space validation of the accepted hybrid:

- fresh +10 mm and -10 mm on all axes: pass, no backward steps;
- 80 mm four-axis command: worst final error 4.75 mm;
- 80 mm out and back: return error 3.40-3.46 mm;
- 0.1 Hz sine: p95 3.96-5.75 mm, no valve saturation;
- difficult -50 mm cylinder-2 case: within 0.41 mm of the rate-limited direct-drive
  equilibrium, with smoother measured motion.

The final head-down tool-space gate at 0.13 m/s passes with 10.6 mm mean error, 22.2 mm p95,
100% ground-band occupancy, 0.80 degrees maximum head tilt, and 0.66 mm truck drift.

## 8. Reproduction commands

All commands run from the repository root.

```bash
# Rebuild strict 0.05 s transitions from the raw 0.01 s CSV
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO1_ID_FD_PhysX/scripts/build_strict_fd_dataset.py

# Reproduce FD rank fine-tuning
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO1_ID_FD_PhysX/scripts/finetune_fd_rank.py

# Reproduce paired ID/FD control fine-tuning
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO1_ID_FD_PhysX/scripts/finetune_idfd_control.py --epochs 30

# Reproduce the promoted strict FD recipe
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO1_ID_FD_PhysX/scripts/train_fd_strict.py \
  --epochs 150 --lr 3e-4 --w_mat 0.25 --lam_roll 2.0 \
  --lam_kin 10.0 --roll_h 10

# Offline valve-authority report
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO1_ID_FD_PhysX/scripts/validate_fd_authority.py

# Rebuild the head-down direct-PhysX arc
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO1_ID_FD_PhysX/scripts/build_head_down_arc.py --headless --num_envs 64
```

The runtime wrappers now live in `PPO1_ID_FD_PhysX/runtime`, with the original ID/FD architecture
loaders and baseline checkpoints under `PPO1_ID_FD_PhysX/models/vendor_models`. Preserve that
self-contained loader contract when moving the project to another host.

## 9. Artifact checksums

These SHA-256 values identify the exact validated artifacts:

```text
8e26c52e82d89efe193e001d0a8b07a6c2f9ed9bcc33d85d30d03b30e804492a  best_id_cylinder_control.pt
acfce7732161254c867c6b4fb8c9b169bb33e431275a30cd6ef3a836d4766a48  best_fd_cylinder_rank4.pt
09d2e6e8b19ef7484f4134035d0c3aa6dd4a93f05443a227cd0b182a8fa0a31d  best_fd_cylinder_control.pt
1f41e44309a8f320430a3382ca88ce5f06dabfa01470598a1c3e7e43d52b4af3  best_fd_cylinder_strict.pt
8448f9549c35af8b4c7dad49d70452b6052163337da40c17c086c3a1f1c3be26  Excitation_FD_Strict.csv
4c3200bdf2a352fd46e177a288947d5429727a500af13de411c422e57d7af3ed  strict_fd_transitions_0p05.npz
```

Do not promote a retrained checkpoint solely from offline R2 or loss. Promotion requires the
cylinder step/return gates and the complete head-down tool-space acceptance gate.
