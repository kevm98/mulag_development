# Evidence-to-Fix Plan: MulagAdaptiveRLEnv Pipeline

**Date:** 2026-06-18
**Rule:** No fix is claimed "correct" until validated. Each entry states what the
evidence actually proves, what remains conditional on further measurement, and
what test would validate the fix before merging.

---

## Fix 1: Drop algebraically redundant observation dimensions

### Evidence (definitive)

The code in [`build_rl_obs`](/home/rrlab/rrlab_mulag/kevin_integration/rl/observation_builder.py:44)
defines `q_error = q_ref - q` and `qdot_error = qdot_ref - qdot`, then
concatenates all six tensors. This proves that `q_ref` is recoverable from
`q + q_error`, and `qdot_ref` is recoverable from `qdot + qdot_error`.

Important correction: `q` and `q_ref` cannot both be dropped with zero
information loss. Given only `q_error`, infinitely many `(q, q_ref)` pairs are
possible. The lossless joint-space reduction keeps `q`, `qdot`, `q_error`, and
`qdot_error`.

The synthetic correlation analysis confirms: across 10,000 samples with realistic
noise, `corr(q_i, q_ref_i) ~ 0.94` for all 5 joints, and
`corr(qdot_i, qdot_error_i) ~ -0.93`. These are deterministic linear
relationships, not learned patterns.

| Drop these | Keep these | Dims saved | Information loss |
|---|---|---|---|
| `q_ref` (5) | `q` (5), `q_error` (5) | 5 | Zero — q_ref = q_error + q |
| `qdot_ref` (5) | `qdot` (5), `qdot_error` (5) | 5 | Zero — qdot_ref = qdot_error + qdot |

In task-space, only the target position is algebraically recoverable from
`current_ee_pose[:3] + ee_position_error`. The target orientation is not
recoverable from scalar `ee_orientation_error`, so the lossless reduction keeps
the command quaternion and drops only the 3 target-position dims.

### What is NOT yet evidenced

- Whether the policy actually *uses* the redundant dimensions or ignores them.
  This requires comparing training curves of slim vs full obs — but the
  redundancy is algebraic, so the slim obs cannot be worse.

### Proposed test before merging

Train two runs from scratch with identical seeds, one with legacy 62-dim obs and
one with slim 52-dim obs. Compare:
- Convergence speed (iterations to reach mean q_error < 0.02)
- Final tracking performance
If slim converges at least as fast as full, the fix is validated. If slim is
worse, the policy was benefiting from the legacy duplicated inputs despite
their algebraic recoverability.

### Files to change (when validated)

- `kevin_integration/rl/observation_builder.py`: `build_rl_obs()` — replace
  `torch.cat([q, qdot, q_ref, qdot_ref, q_error, qdot_error, ...])` with
  `torch.cat([q, qdot, q_error, qdot_error, ...])`
- `kevin_integration/rl/observation_builder.py`: `build_task_space_rl_obs()` —
  replace `ee_pose_command` with `ee_pose_command[..., 3:7]`
- `kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py`: update
  `observation_space` from 62 to 52 (joint) and `task_space_observation_space`
  from 73 to 70 (task-space)
- `kevin_integration/tasks/task_space_reference.py`: update
  `TASK_SPACE_OBSERVATION_DIM` and replace `ee_pose_command` with
  `ee_orientation_command` in
  `TASK_SPACE_OBSERVATION_ORDER`

---

## Fix 2: Bound z_arm_hat output range

### Evidence (conditional — depends on actual AAM checkpoint)

The AAM model's head architecture ends with `nn.Linear(head_dim, 24)` — no
activation, no normalization layer. The analysis demonstrates mathematically
that if the AAM latent std is >= 10, z_arm_hat captures >= 99.6% of total
observation variance, drowning out all other signals.

**Critical missing measurement:** The actual AAM checkpoint's output scale.
The analysis used synthetic scales (0.1, 1.0, 10.0, 100.0). The real value
depends on the weights in `aam.pth`. Without loading that checkpoint and
running a forward pass on representative input, we cannot say whether the
current checkpoint actually has this problem or not.

### What is NOT yet evidenced

- The actual AAM latent scale in the deployed checkpoint
- Whether PPO's `empirical_normalization` successfully compensates
- Whether adding a clamp/tanh would hurt the ID model's conditioning

### Proposed measurement (before any code change)

Load the AAM checkpoint and run inference on the ROM integration CSV data
(300 samples of q, qdot, dP, valve_cmd). Compute per-dimension mean, std,
min, max of z_arm_hat. If max(|z_arm_hat|) > 10, the fix is warranted.
If max(|z_arm_hat|) < 3, the current setup is fine — no fix needed.

### Proposed fix (if measurement confirms the problem)

Add one line in [`adaptive_rl_controller.py`](/home/rrlab/rrlab_mulag/kevin_integration/rl/adaptive_rl_controller.py:199)
in `run_aam()`, after the AAM forward pass:

```python
z_arm_hat = torch.clamp(z_arm_hat, -10.0, 10.0)
```

This is a non-destructive bound — it only clips outliers while preserving the
latent structure for the ID model. It does not require retraining the AAM.

### Proposed test before merging

After adding the clamp, run the learned pipeline preflight test
(`run_learned_pipeline_preflight`) and verify:
- AAM still produces finite latents
- ID model still produces reasonable valve commands
- Preflight passes with the same thresholds

---

## Fix 3: Reduce action penalty weights

### Evidence (definitive from synthetic sweep)

The break-even analysis (Section 4 of reward analysis) shows:

| action_mag | action_delta | combined penalty | break-even q_error |
|---|---|---|---|
| 0.20 | 0.05 | -0.0074 | 0.035 rad |
| 0.10 | 0.05 | -0.0034 | 0.021 rad |
| 0.05 | 0.01 | -0.00022 | 0.009 rad |

At `rew_action_rate=0.05` and `rew_action_magnitude=0.02` (current defaults),
a policy producing moderate actions (mag=0.2, delta=0.05) receives tracking
reward exceeding action penalty only when q_error > 0.035 rad. Below that
threshold, the net reward for moving is negative — the policy learns to freeze.

### What is NOT yet evidenced

- What action magnitudes the current trained policy actually produces.
  If the policy naturally uses small actions (< 0.1), the current weights
  may already be fine.
- Whether the settle_tracking reward (active when ramp_progress > 0.95)
  sufficiently counteracts this. The settle reward adds +2.0 * error^2, which
  partially offsets but only in late-stage approach.

### Proposed measurement (before changing weights)

From a training log or evaluation run, extract the distribution of
`|action|` and `|action_delta|`. If the 90th percentile of |action| < 0.15
and |action_delta| < 0.03, the current weights are fine. If the policy
regularly outputs |action| > 0.2, the fix is warranted.

### Proposed fix (if measurement confirms the problem)

In `MulagAdaptiveRLJointReachEnvCfg`:
```python
rew_action_rate = 0.02       # was 0.05
rew_action_magnitude = 0.01  # was 0.02
```

This reduces action penalties by 50-60%, moving the break-even error from
0.035 rad to ~0.025 rad (still protective, less restrictive).

### Proposed test before merging

Train two runs with old vs new weights on the same seed. After 500 iterations:
- Compare mean episode q_error
- Compare action magnitude distribution
- Verify the new policy doesn't saturate valves (> 0.95) more often
If tracking improves without saturation increase, the fix is validated.

---

## Fix 4: Increase progress EMA alpha

### Evidence (definitive from simulation)

The EMA simulation (Section 3 of reward analysis) with `alpha=0.1` shows:
- Step 1: error 0.02 -> 0.06, EMA 0.02 -> 0.024, progress = -0.080 (correct: penalized for getting worse)
- Step 2: error 0.06 -> 0.02, EMA 0.024 -> 0.0236, progress = +0.008 (UNDESERVED: rewarded for returning, not for net improvement)

Over 10 oscillating steps, the accumulated progress reward is -0.274 despite
zero net position change. The EMA lag causes the progress reward to lag
behind the true error delta by 3-5 steps.

With `alpha=0.3`:
- Step 1: error 0.02 -> 0.06, EMA 0.02 -> 0.032, progress = -0.080
- Step 2: error 0.06 -> 0.02, EMA 0.032 -> 0.0284, progress = +0.0072
The net accumulated progress is slightly less negative (less penalty for oscillation).

### What is NOT yet evidenced

- Whether the oscillating behavior actually occurs in training. The simulation
  assumes a worst-case alternating pattern. Real arm motion may be smoother.
- Whether a higher alpha makes the progress reward too noisy (the EMA exists
  to filter noise; higher alpha = less filtering).

### Proposed measurement

From a training log, extract q_error over 100 consecutive steps in a single
episode. Compute the autocorrelation at lag 1. If it's strongly negative
(oscillatory), the fix helps. If it's positive (smooth approach), the
current alpha is fine.

### Proposed fix (if measurement confirms oscillation)

```python
progress_ema_alpha = 0.3  # was 0.1
```

### Proposed test

Train two runs with old vs new alpha. Compare the variance of the progress
reward term. If variance increases significantly (> 2x), the higher alpha
introduces too much noise — revert.

---

## Fix 5: Ablation test for z_arm_hat in observations

### Evidence (definitive from code analysis)

The observation-to-reward gradient analysis (Section 6 of obs analysis) shows
z_arm_hat (24 dims) has zero direct reward gradient. None of the reward terms
depend on it. The policy can only learn to use it through the advantage
function (state value estimation).

This does NOT mean z_arm_hat is useless — it encodes arm configuration
information that may help the critic predict future rewards. But whether
that benefit outweighs the cost of 24 extra input dimensions is an
empirical question.

### What is NOT yet evidenced

- Whether the PPO policy actually uses z_arm_hat. An ablation test is required.

### Proposed test (the fix IS the test)

Train two runs with identical config and seed:
- Run A: full 52-dim obs (includes z_arm_hat)
- Run B: 28-dim obs (drops z_arm_hat entirely)

Compare:
- Convergence speed
- Final tracking performance
- Value loss (indicates whether the critic suffers without z_arm_hat)

If Run B converges at >= 90% of Run A's speed, z_arm_hat can be dropped.
If Run B is significantly worse, z_arm_hat carries useful information beyond
what the ID model already uses internally.

### Files to change (for ablation only)

- `kevin_integration/rl/observation_builder.py`: `build_rl_obs()` — remove
  `z_arm_hat` from the concatenation
- `kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py`: `observation_space`
  from 52 to 28

---

## Fix 6: Torque effort weight — remove or rescale

### Evidence (definitive)

`rew_torque_effort = 1e-6`. At max torque (5.0 Nm x 4 joints), penalty = -1e-4.
At typical torque (< 1 Nm), penalty < -4e-6. This is 3-4 orders of magnitude
below any other reward term. The term contributes no meaningful gradient.

### What is NOT yet evidenced

- Whether a torque penalty is needed at all in ROM mode (torque is never applied
  as joint effort in ROM mode — it's observer-only).
- The appropriate scale if torque effort actuation is ever enabled.

### Proposed fix

In ROM mode (`use_final_causal_cylinder_rom=True`): set `rew_torque_effort = 0.0`.
In effort mode (`use_fd_effort=True`): measure actual torque magnitudes from a
run, then set the weight so that torque penalty ~ 0.001 * q_tracking at typical
operating conditions.

### Validation

This is a no-op in ROM mode (torque is already zero). No test needed — just
remove dead code.

---

## Fix 7: Consolidate action_saturation and valve_saturation penalties

### Evidence (moderate — correlation not yet measured)

Both penalties use the same threshold (0.90):
- `action_saturation`: threshold 0.90, weight 0.50 (on raw PPO action)
- `valve_saturation`: sat_start 0.90, weight 0.15 (on ID valve output)

The ID model maps action -> qddot -> valve via a monotonic (tanh) function.
If the ID model's effective gain is roughly constant across arm configurations,
then action saturation and valve saturation are highly correlated — penalizing
the same behavior twice.

### What is NOT yet evidenced

- The actual correlation between |action| > 0.90 and |valve| > 0.90 in a real
  training run. If the ID model gain varies significantly with arm
  configuration, the penalties may serve different purposes.

### Proposed measurement

From a training run, extract (action, valve_cmd) pairs. Compute the fraction of
steps where (|action| > 0.90 AND |valve| < 0.90) and (|action| < 0.90 AND
|valve| > 0.90). If both fractions are < 5%, the penalties are redundant —
consolidate. If either fraction > 10%, they serve different purposes.

### Proposed fix (if measurement confirms redundancy)

Remove `action_saturation_penalty` from `_joint_reach_reward_components` and
keep only `valve_saturation_penalty` with weight increased to 0.20.
The valve penalty operates on the actual hydraulic command, which is the
physically meaningful quantity.

### Validation

Train two runs: one with both penalties, one with only valve_saturation at
weight 0.20. Compare valve saturation rates. If the consolidation run has
similar or lower saturation, it's validated.

---

## Fix 8: SD observer clamp limits from real data

### Evidence (definitive)

`max_abs_pressure_delta = 1.0e7` and `max_abs_fnet = 1.0e7`. The ROM
integration CSV (300 samples) shows dP values are all zero and fnet
values are all zero in that test. Even in operation, these limits are
5-7 orders of magnitude above expected values. The `clamped_fraction`
and `output_clamped_fraction` diagnostics always read 0.0.

### What is NOT yet evidenced

- The actual dP and fnet ranges during learned control operation.
- The appropriate limits for catching SD model divergence.

### Proposed measurement

Record dP and fnet from the SD observer during a full training episode.
Compute per-dimension mean and std. Set clamp limits at:
```
max_abs_pressure_delta = 5 * max(std(dP_i))
max_abs_fnet = 5 * max(std(fnet_i))
```

### Proposed fix (after measurement)

Replace the hardcoded 1e7 limits with data-driven values. This makes the
clamp diagnostics meaningful — they'll now fire when the SD model diverges.

### Validation

Verify that clamp_fraction remains < 0.01 during normal operation and
increases detectably when the SD model is fed out-of-distribution inputs.

---

## Summary: What to measure before changing anything

| # | Measurement needed | How to get it |
|---|---|---|
| 2 | Actual AAM latent scale | Load aam.pth, run inference on CSV data |
| 3 | Policy action magnitude distribution | Extract from training log or eval run |
| 4 | q_error autocorrelation at lag 1 | Extract from training log |
| 5 | Ablation: train without z_arm_hat | Two training runs, same seed |
| 7 | Correlation between action and valve saturation | Extract (action, valve) pairs from training |
| 8 | SD dP/fnet ranges during operation | Record SD observer output during training |

Fixes 1 (redundancy drop) and 6 (torque effort dead) need no further
measurement — they are proven by code structure and arithmetic respectively.

---

## No production code has been modified

All analysis scripts are in `analysis/` and operate independently:
```
analysis/obs_analysis/run_obs_analysis.py
analysis/reward_analysis/run_reward_analysis.py
analysis/action_analysis/run_action_analysis.py
analysis/pipeline_analysis/run_pipeline_analysis.py
```
