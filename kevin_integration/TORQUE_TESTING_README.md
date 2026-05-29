# Mulag Torque Testing

This utility exists because the adaptive RL full-pipeline debug path showed learned FD torques far outside a safe
range:

- `torque_raw_mean` around `9e5` to `3.4e6`
- `torque_raw_max` around `1.5e6` to `8.1e6`
- `torque_clamp_fraction = 1.0`

That means every learned FD output is being clamped before it reaches Isaac. Before increasing effort limits or
using FD torques for real training, validate the robot's normal effort scale and the FD input/output contract.

## Commands

Run from the repository root:

```bash
cd /home/rrlab/rrlab_mulag
```

Sweep small safe efforts through each commanded arm joint:

```bash
./rrlab.sh -p kevin_integration/scripts/test_mulag_torques.py \
  --mode sweep_effort \
  --num_envs 1 \
  --headless
```

Hold the default arm pose with position targets and inspect Isaac's available internal effort/torque fields:

```bash
./rrlab.sh -p kevin_integration/scripts/test_mulag_torques.py \
  --mode hold_position \
  --num_envs 1 \
  --headless
```

Run FD on controlled test inputs without applying FD torque:

```bash
./rrlab.sh -p kevin_integration/scripts/test_mulag_torques.py \
  --mode fd_sanity \
  --num_envs 1 \
  --headless
```

Run FD online, clamp the torque, and apply only the safe clamped effort:

```bash
./rrlab.sh -p kevin_integration/scripts/test_mulag_torques.py \
  --mode compare_fd_vs_applied \
  --num_envs 1 \
  --max_abs_torque 5 \
  --headless
```

Remove `--headless` if you want to watch the robot.

## Outputs

Terminal output prints:

- all robot joint names and IDs
- resolved state and commanded joint IDs
- joint position `q`
- joint velocity `qdot`
- available Isaac effort/torque tensors such as `applied_torque`, when present
- FD raw torque stats
- FD clamped torque stats
- `torque_clamp_fraction`

CSV logs are written by default to:

```text
logs/torque_testing/<timestamp>/torque_test.csv
```

Use `--no_csv` to disable CSV output.

## What To Look For

In `sweep_effort`, check how the arm responds to small torques such as `0.1`, `1`, and `5`. If the robot barely
moves at these values, the low-torque full-pipeline RL smoke may be too conservative. If it moves violently,
keep torque limits low.

In `fd_sanity`, compare:

- `fd_input_raw`
- `fd_input_norm`
- `fd_output_model`
- `fd_output_denorm_torque`

If `fd_output_denorm_torque` is already huge for zero velocity, zero pressure, and zero valve, the issue is
probably not PPO. It is likely in the FD model/scaler/input contract.

In `compare_fd_vs_applied`, `torque_clamp_fraction` near `1.0` means FD is saturating against the chosen clamp.
Do not increase `--max_abs_torque` until the cause is understood.

## Likely Causes

- FD scaler mismatch between `fd.pth` and `fd_scaler.pth`
- FD feature order mismatch: expected `[q5, qdot5, dP4, valve4]`
- FD output denormalization issue
- `dP=0` being out-of-distribution for the learned FD model
- wrong FD checkpoint or wrong scaler file
- runtime joint units/ranges differing from the FD training data

## Safety

Never apply raw FD torque directly. The script clamps effort with `--max_abs_torque`, defaulting to `5`.
Keep this low until FD scale and input contracts are validated.
