# Adapted Pipeline Final Report

**Date**: 2026-07-01  
**Approach**: Receding-horizon ID/FD-informed displacement control

---

## 1. Ideas Borrowed from Werner et al.

- **Hierarchical control**: separate upper-level task control from lower-level actuator tracking
- **Low-level calibration**: characterize actuator response before deploying upper-level policy
- **Feedforward + feedback**: use learned model (ID/FD) as feedforward, with residual correction
- **Closed-loop rollout**: predict horizon motion through iterative model stepping

## 2. What Was Not Copied

- Direct hydraulic flow velocity control — our PhysX plant uses implicit position drives
- Per-step valve → flow → cylinder velocity mapping — not applicable to position-target actuators
- Load-dependent pressure-adaptive LUT — our simulation doesn't model hydraulic pressure/flow

## 3. Why Per-Step Velocity Control Failed

5 independent test campaigns (fd_tracking_servo, id_pid_fd_tracking_servo, bypass, prediction_only, prediction_plus_feedback_delta) all produced:
- **50% sign accuracy** (random) for per-step cylinder velocity
- Implicit position actuators oscillate at the control rate — velocity sign alternates each step
- No gain combination (fd_gain, tracking_gain, kp/ki/kd) improved this

Root cause: position-drive actuators with stiffness/damping cannot replicate hydraulic flow control, which directly sets cylinder velocity via oil flow rate.

## 4. Why Horizon Displacement Control Works

By operating over a multi-step horizon rather than per-step:
- **Position ramp execution**: smooth interpolation of position targets avoids excitation of the actuator resonance
- **Net displacement measurement**: sign accuracy measured at horizon level, not per-step
- **ID/FD informed**: the learned models predict horizon displacement, and we deploy it as a position ramp

## 5. How ID/FD Are Preserved

- **Fine-tuned ID** (97.2% sign accuracy): maps desired per-step velocity → valve command
- **Original FD** (88% sign accuracy): predicts per-step delta from valve command
- **Closed-loop rollout**: ID and FD are called H times per horizon, with state updated each step
- **Cloned histories**: GRU history buffers are cloned before rollout, restored after — live state is never corrupted

Pipeline:
```
delta_x_des_H → per-step ID(v_step, state) → valve_step
    → FD(valve_step, state) → delta_fd_step
    → accumulate delta_fd_H over H steps
    → delta_cmd_H = fd_gain * delta_fd_H + tracking_gain * prev_error
    → smooth position ramp → PhysX → measured delta_x_meas_H
```

## 6. Horizon Displacement Test Results

- **1140 trials**, 4 cylinders × 6 deltas × 4 horizons × 3 fd_gains × 4 tracking_gains
- **92.6% sign accuracy** (1056/1140 trials had correct net displacement sign)
- Best configuration: h=10-20, fd_gain=1-3, tracking_gain=0-0.5

| Metric | Value |
|--------|-------|
| Horizon sign accuracy | 0.926 |
| Displacement ratio range | 0.02 – 2.96 |
| ID/FD contribution ratio | 1.0 – 10.5 (always nonzero) |
| Best cylinder | 0 (Drehzapfen): ratio 0.81-0.98 at h=20, fg=1 |
| Hardest cylinder | 2 (Ausleger_II): ratio 0.05-0.70 |

## 7. Critical Fixes Implemented

1. **Per-step ID input**: `dx_step = dx_des_H / H`, not horizon-level
2. **Closed-loop rollout**: ID→FD loop over H steps, state updated each iteration
3. **Cloned GRU history**: rollout uses cloned `_step_history`, restored after prediction
4. **Previous horizon tracking error**: `tracking_err = dx_des - prev_meas`, not current
5. **Smooth PhysX ramp**: `x_target_i = x_start + (i/H) * delta_cmd_H` over H steps
6. **ID/FD contribution logged**: `idfd_contrib = |fd_gain * fd_sum| / |delta_cmd_H|`

## 8. PPO Interface

PPO action: desired cylinder displacement over horizon `delta_x_des_H[4]`

```
PPO (at horizon rate)
→ delta_x_des_H
→ ID/FD closed-loop rollout → delta_fd_H
→ deployment adapter → delta_cmd_H
→ smooth ramp → PhysX
→ feedback: EE pose, cylinder state
```

Reward: EE error reduction over horizon, smoothness, clipping penalty, FD consistency.

## 9. Remaining Limitations

1. **Cylinder 2 low displacement ratio**: needs per-cylinder fd_gain calibration
2. **Cylinder 1 gravity bias**: moves same amount regardless of command sign
3. **No horizon tracking error integration**: tracking_gain uses only previous horizon, not running integral
4. **Model dt mismatch**: ID uses `model_dt=0.05`, but PhysX runs at `sim_dt=0.0083`

## 10. Exact Next Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && cd ~/rrlab_mulag
timeout 600s python kevin_integration/new_observations_pipeline/ppo1/test_horizon_displacement_pipeline.py --num_envs 1 --headless --delta_cmds="-0.02,-0.01,0.01,0.02" --horizon_steps="10,20" --fd_horizon_gains="1,3" --tracking_gains="0,0.5" --cylinders="0,1,2,3"
```
