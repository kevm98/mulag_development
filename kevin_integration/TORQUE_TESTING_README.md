# Mulag Torque Testing

This utility exists because the adaptive RL full-pipeline debug path showed that learned FD output is not in
Isaac joint-effort units:

- `torque_raw_mean` around `9e5` to `3.4e6`
- `torque_raw_max` around `1.5e6` to `8.1e6`
- `torque_clamp_fraction = 1.0`

Isaac internal torque readings are usually closer to `1e3` to `1e5`, while FD can produce `1e6` to `1e7` even
for zero valve and zero pressure. The FD model was trained from Simscape, so the output is treated as
Simscape-domain actuator/generalized output, not Isaac joint effort.

The safe path is:

```text
FD_simscape_output -> torque_adapter -> tau_fd_adapted
tau_fd_adapted -> fd_residual_alpha -> tau_applied_raw
tau_applied_raw -> low-pass filter -> rate limit -> final clamp -> apply
```

Never apply raw FD output directly.

The first `scale_bias` adapter was structurally correct but still too aggressive: it reduced FD output into
`tau_fd_adapted`, but the adapted torque was often thousands to tens of thousands while `max_abs_torque=5`.
The clamp and rate limiter were therefore doing most of the work.

The current deterministic residual authority rule is:

```text
tau_applied_raw = fd_residual_alpha * tau_fd_adapted
```

The first safe value is `fd_residual_alpha=0.002`. From the diagnostics, values like `268.84`, `1087.11`, and
`2827.66` become roughly `0.54`, `2.17`, and `5.66`.

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
  --torque_adapter_mode scale_bias \
  --torque_adapter_preset conservative \
  --fd_residual_alpha 0.002 \
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
- FD Simscape output stats
- adapted torque, residual-authority torque, filtered torque, and final clamp stats
- `torque_adapter_mode`
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
probably not PPO. It confirms that FD output must be adapted before Isaac effort application.

In `compare_fd_vs_applied`, look at:

- `fd_simscape_output`: may remain huge because it is Simscape-domain output
- `tau_fd_adapted`: may remain hundreds/thousands after conservative `scale_bias`
- `tau_applied_raw`: should be roughly `0.2` to `6` in most early tests with `fd_residual_alpha=0.002`
- `tau_isaac_filtered`: should not be completely dominated by the rate limiter
- `tau_isaac_clamped`: the only value applied to Isaac
- `torque_clamp_fraction`: should drop below `0.5` and ideally near `0`
- `torque_rate_limited_fraction`: should drop below `1.0`

Do not increase `--max_abs_torque` until the adapter scale/input contract and residual authority are validated.

Recommended tuning order:

1. Start with `--torque_adapter_preset conservative`.
2. Try `--torque_adapter_preset moderate` only if conservative is stable and under-active.
3. Use `--torque_adapter_preset aggressive` only if stable and still too weak.
4. If torque is too weak, increase `--fd_residual_alpha` gradually: `0.002 -> 0.003 -> 0.005`.
5. Increase `--max_abs_torque` only after clamp fraction is already low.

Use a custom scale if needed:

```bash
./rrlab.sh -p kevin_integration/scripts/test_mulag_torques.py \
  --mode compare_fd_vs_applied \
  --num_envs 1 \
  --max_abs_torque 5 \
  --torque_adapter_scale "0.0001,0.001,0.0005,0.0001" \
  --fd_residual_alpha 0.002 \
  --headless
```

Fit a simple scale/bias estimate from a torque-testing CSV:

```bash
python kevin_integration/scripts/fit_torque_adapter.py \
  --csv logs/torque_testing/<timestamp>/torque_test.csv
```

## Likely Causes

- FD scaler mismatch between `fd.pth` and `fd_scaler.pth`
- FD feature order mismatch: expected `[q5, qdot5, dP4, valve4]`
- FD output denormalization issue
- `dP=0` being out-of-distribution for the learned FD model
- wrong FD checkpoint or wrong scaler file
- runtime joint units/ranges differing from the FD training data
- Simscape-to-Isaac unit or moment-arm mismatch

## Safety

Never apply raw FD output directly. Start with `--torque_adapter_mode scale_bias`, `--torque_adapter_preset
conservative`, and `--fd_residual_alpha 0.002`; keep `--max_abs_torque 5`, and use the rate limit/low-pass
filtering in the full RL pipeline. Keep this adapter deterministic; later improvements should use explicit
calibrated scale factors or a geometry/moment-arm mapping, not a black-box learned adapter.
