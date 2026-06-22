# MulagAdaptiveRLEnv Pipeline Analysis Report

**Date:** 2026-06-18
**Scope:** Actions, observations, rewards, and adaptive RL controller pipeline
**Goal:** Identify what helps/hurts learnability for end-effector position tracking

---

## 1. Observations (52-dim joint-space / 70-dim task-space)

### 1.1 Redundancy: 13 dims are algebraically derivable and removable

The legacy joint-space observation concatenated `q` (5), `q_ref` (5), and `q_error` (5), plus `qdot` (5), `qdot_ref` (5), and `qdot_error` (5). Because `q_error = q_ref - q`, `q_ref` is recoverable from `q + q_error`; because `qdot_error = qdot_ref - qdot`, `qdot_ref` is recoverable from `qdot + qdot_error`.

Correction to the first draft: `q` and `q_ref` cannot both be dropped with zero information loss. The lossless reduction keeps `q` and `q_error`, and drops only `q_ref`.

| Redundant group | Relationship | Wasted dims |
|---|---|---|
| q, q_ref, q_error | q_ref = q + q_error | 5 |
| qdot, qdot_ref, qdot_error | qdot_ref = qdot + qdot_error | 5 |
| task target position, current position, position error | target_pos = current_pos + position_error | 3 |

**Fix:** Drop `q_ref`; drop `qdot_ref`; in task-space, replace `ee_pose_command[7]` with `ee_orientation_command[4]`. Saves 13 dims with no known information loss.

The active contract is now 52D joint-space and 70D task-space.

### 1.2 z_arm_hat normalization gap (HIGH RISK)

The AAM model's head ends with bare `nn.Linear(hidden_dim, 24)` — no activation, no bounds. At latent_scale=10.0, z_arm_hat captures **99.7% of total observation variance**, drowning out all other signals. At latent_scale=100.0, it's 100%.

PPO's `empirical_normalization` eventually adapts, but early training instability is likely.

**Fix:** Add `torch.clamp(z_arm_hat, -10.0, 10.0)` after the AAM forward pass in `run_aam()`, or add a `tanh` layer to the AAM head during inference.

### 1.3 37 of 52 joint dims have zero direct reward gradient

| Component | Dims | Used in any reward term? |
|---|---|---|
| prev_action | 4 | No |
| prev_valve_cmd | 4 | No (valve_effort uses current, not prev) |
| qdot_error | 5 | No |
| z_arm_hat | 24 | **No — the largest component has no reward signal** |

These 37 dims only help via the value function (state representation for advantage estimation). The policy must learn to extract useful features from them indirectly.

**Fix:** Ablation test — train one run with and one without z_arm_hat in observations. If performance is similar, drop it and save 24 dims.

### 1.4 AAM version-tracking hazard

The mechanism enforcing "ID latent == PPO observation latent" uses an integer version counter rather than tensor comparison. If `_get_observations()` is called twice between control steps (e.g., during evaluation), both calls see the same version and pass the check — but the ID would receive a stale latent.

**Fix:** Store the actual `z_arm_hat` tensor snapshot and compare via `torch.equal()`.

---

## 2. Rewards

### 2.1 The "alive" bonus dominates all regimes

At every error level from large (0.15 rad) to settled (0.002 rad), the alive bonus (+0.1) is the dominant positive term. Q_tracking is always negative (penalty for being away from target), and progress depends on error delta.

| Regime | Pos Sum | Neg Sum | Dominant+ | Dominant- |
|---|---|---|---|---|
| Large error (0.15 rad) | +0.10 | -0.20 | alive | qdot_penalty |
| Near convergence (0.01 rad) | +0.10 | -0.005 | alive | qdot_penalty |
| Settled (0.002 rad) | +0.10 | -0.001 | alive | qdot_penalty |

This means the policy learns primarily "stay alive" rather than "reach the target." The positive tracking signal is entirely from progress and settle rewards, both of which are conditional.

### 2.2 Action penalties exceed tracking reward for errors < 0.035 rad

At action_magnitude=0.20 and action_delta=0.05 (moderate movement), the combined action penalty (-0.0074) exceeds q_tracking (-0.0037) for errors below 0.035 rad. The policy has a **perverse incentive to stop moving** before reaching the target.

The settle_tracking reward partially counteracts this, but only activates when `ramp_progress > 0.95` — a late-stage condition.

**Fix:** Reduce `rew_action_rate` from 0.05 to 0.02, and `rew_action_magnitude` from 0.02 to 0.01.

### 2.3 Progress EMA creates oscillation exploit

With `progress_ema_alpha=0.1`, the EMA baseline lags 5+ steps behind actual error. If the arm oscillates between two error values, the progress reward can be **positive even when error is increasing** — the policy extracts reward from oscillation.

At alpha=0.1, an alternating pattern [0.02, 0.06, 0.02, 0.06, ...] accumulates -0.27 net progress reward over 10 steps, demonstrating the EMA follows the average.

**Fix:** Increase `progress_ema_alpha` to 0.3 or use a windowed-min baseline instead of EMA.

### 2.4 Torque effort reward is dead

`rew_torque_effort = 1e-6`. At max torque (5.0 Nm across 4 joints), penalty = -1e-4. This is 3-4 orders of magnitude below any other term. The term contributes nothing to learning.

**Fix:** Either remove it or increase to 1e-4.

### 2.5 Valve saturation penalty is well-designed

Smooth `tanh(square(excess))` onset, activates gradually from `sat_start=0.90`. At |valve|=0.92: penalty=-0.006. At |valve|=1.00: penalty=-0.095. This is correctly calibrated.

---

## 3. Actions

### 3.1 Scale mismatch: "id_qddot_std" vs fixed 0.5

The `action_to_qddot_scale` is `0.5` in joint-reach config but `"id_qddot_std"` (~3.575 with synthetic data) in hydraulic ROM config. This means the same PPO action produces **~7x larger qddot commands** in ROM mode.

| Config | effective_scale | action=1.0 -> qddot |
|---|---|---|
| Joint reach (fixed) | 0.50 | 0.50 rad/s^2 |
| Hydraulic ROM (id_qddot_std) | ~3.58 | 3.58 rad/s^2 |

The ID model was trained on supervised data with specific qddot distributions. Using a different scale shifts the input distribution, potentially out of the ID model's training range.

### 3.2 Double saturation penalties

`action_saturation` (weight=0.5, threshold=0.90) and `valve_saturation` (weight=0.15, sat_start=0.90) penalize correlated behavior. The action penalty is 3.3x stronger and operates on the raw PPO output; the valve penalty operates on the ID output.

**Fix:** Consolidate to one penalty at weight ~0.2.

### 3.3 Torque adapter is effectively disabled

The "conservative" preset with `fd_residual_alpha=0.002` produces effective scales of ~[2e-7, 2e-6, 1e-6, 2e-7]. Even with FD outputs of 1e7, torques are < 2 Nm. For typical FD outputs (~1e5), torques are < 0.02 Nm. The MulaG arm weighs hundreds of kg — this is negligible.

### 3.4 Clipping diagnostics never fire

SD clamp limits of 1e7 (for dP and fnet) are 7+ orders of magnitude above observed values (~0 in ROM integration test). Torque clamp of 5.0 Nm with the conservative adapter is never approached. These diagnostics are dead.

---

## 4. Control Pipeline

### 4.1 Torque path is dead in ROM mode

When `use_final_causal_cylinder_rom=True` (the primary training config), the FD-derived torque is **never applied as joint effort**. The FD is observer-only, serving only to provide `fd_tau_hat` for observations and `fd_load_excess_penalty` in the reward.

The entire torque adapter, lowpass, rate limit, and torque clamping pipeline is dead code in ROM mode.

### 4.2 Joint 1 drive authority gap

ROM integration data shows joint 1 (Drehzapfen/base rotation) moves only ~0.003 rad per step while joints 2-4 move 0.02-0.05 rad. This is consistent with the `q_tracking_reward_joint_weights = [1.5, 1.0, 1.0, 1.0, 0.0]` — joint 1 has 1.5x reward weight but the ROM may not have sufficient drive authority.

### 4.3 Torque filter interaction

The lowpass + rate limit combination produces an effective filter:
```
tau_new = prev + clamp(alpha * (target - prev), -rate_limit, +rate_limit)
```
With `alpha=0.2` and `rate_limit=1.0`, the effective rate limit on raw delta is `1.0/0.2 = 5.0`. Step response reaches 90% of target in ~11 steps (110 ms at 100 Hz).

---

## 5. Fix Priority (ordered by expected impact)

| # | Fix | Impact | Effort |
|---|---|---|---|
| 1 | Drop q_ref and qdot_ref from observations; keep q, q_error, qdot, qdot_error | HIGH | Done in `observation_builder.py` |
| 2 | Clamp z_arm_hat to [-10, 10] after AAM forward pass | HIGH | 1 line: `adaptive_rl_controller.py` `run_aam()` |
| 3 | Reduce action_rate weight 0.05 -> 0.02, action_magnitude 0.02 -> 0.01 | MEDIUM | 1 file: `mulag_adaptive_rl_env_cfg.py` |
| 4 | Increase progress_ema_alpha 0.1 -> 0.3 | MEDIUM | 1 line: `mulag_adaptive_rl_env_cfg.py` |
| 5 | Consolidate action_saturation + valve_saturation into single penalty at weight 0.2 | LOW-MEDIUM | 2 files: reward_terms + env_cfg |
| 6 | Run ablation: train with and without z_arm_hat in observations | MEDIUM (investigation) | Config change + 2 training runs |
| 7 | Increase torque_adapter_preset to "moderate" (only if using effort mode) | LOW | 1 line: `mulag_adaptive_rl_env_cfg.py` |

## 6. Analysis Scripts

All analysis scripts are in `analysis/` and can be run independently:

```
python analysis/obs_analysis/run_obs_analysis.py
python analysis/reward_analysis/run_reward_analysis.py
python analysis/action_analysis/run_action_analysis.py
python analysis/pipeline_analysis/run_pipeline_analysis.py
```

No production code was modified.
