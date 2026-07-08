# PPO1 Adapter Pipeline - Critical Failure Analysis & Solutions

**Date:** 2026-07-04
**Status:** Gate 2 FAILED - Adapter does not work

---

## ROOT CAUSE ANALYSIS

### Discovery: Zero Correlation in Training Data

```
Correlation between cylinder commands and EE motion (single-step):
  EE_Y vs cyl1-4: 0.006, 0.001, -0.004, 0.001
  EE_Z vs cyl1-4: -0.002, -0.007, 0.002, -0.002
```

**All correlations < 0.01 → NO LEARNABLE SIGNAL**

### Why This Happened

1. **Single timestep too short**: 16.67ms is insufficient for hydraulic system to respond
2. **System dynamics dominate**: PID controller + inertia + damping create delays
3. **Random commands don't accumulate**: Need sustained motion in one direction to see EE displacement

### Adapter Behavior

The trained adapter outputs **constant values** regardless of input:
```
Desired Motion → Adapter Output (mm)
[+5.0, 0.0]   → [+2.36, -2.46, -2.50, -2.50]
[-5.0, 0.0]   → [+2.27, -2.46, -2.50, -2.50]  (0.09mm difference!)
[0.0, +5.0]   → [+2.33, -2.46, -2.50, -2.50]
[0.0, -5.0]   → [+2.31, -2.46, -2.50, -2.50]
```

Cylinders 2-4 are saturated, cylinder 1 shows tiny variation. **The network learned nothing.**

---

## SOLUTION OPTIONS

### Option 1: Multi-Step Trajectory Data (NEW - RECOMMENDED)

**Approach**: Instead of random single-step commands, collect data where system actually achieves EE goals.

**Implementation**:
1. Pick random EE target (e.g., +30mm Y)
2. Apply proportional control for 20 steps
3. Record (target, average_cylinder_command, achieved_motion)
4. Only keep successful attempts (>50% of target achieved)
5. Train adapter on: desired_EE_delta → avg_cylinder_command

**Advantages**:
- ✅ Creates actual cause-effect relationship in data
- ✅ Accounts for multi-step accumulation
- ✅ Filters for successful motions only

**Script**: `collect_trajectory_dataset.py` (just created)

**Estimated time**: ~1 hour (collection: 30min, training: 10min, validation: 20min)

---

### Option 2: Analytical Jacobian Approximation (FASTEST)

**Approach**: Skip learned adapter, use simple kinematic approximation.

**Implementation**:
```python
def simple_jacobian(desired_ee_yz, scale=2.0):
    # Assume differential cylinders
    dy, dz = desired_ee_yz
    return torch.tensor([
        dy * scale / 2,   # cyl1: +Y contribution
        -dy * scale / 2,  # cyl2: -Y contribution
        dz * scale / 2,   # cyl3: +Z contribution
        -dz * scale / 2,  # cyl4: -Z contribution
    ])
```

**Advantages**:
- ✅ Works immediately (no training)
- ✅ PPO can learn to compensate for approximation errors
- ✅ Good enough for initial experiments

**Disadvantages**:
- ⚠️ Ignores configuration-dependent kinematics
- ⚠️ Fixed scale parameter needs tuning

**Estimated time**: <10 minutes to implement and test

---

### Option 3: Direct 4D PPO (SIMPLEST ARCHITECTURE)

**Approach**: Train PPO directly on cylinder actions, skip adapter entirely.

**Changes needed**:
- PPO observation: target_error_yz (2D) + robot_state (N-D)
- PPO action: cylinder_delta (4D) instead of ee_delta (2D)
- Remove adapter wrapper completely

**Advantages**:
- ✅ No adapter training needed
- ✅ Simpler architecture (fewer components to fail)
- ✅ PPO learns end-to-end mapping

**Disadvantages**:
- ⚠️ Higher-dimensional action space (4D vs 2D)
- ⚠️ May take longer for PPO to converge
- ⚠️ Less interpretable (no explicit task-space control)

**Estimated time**: ~2 hours (modify env: 30min, train PPO: 60min, evaluate: 30min)

---

### Option 4: Optimization-Based Inverse (MOST ROBUST)

**Approach**: Use forward model to find cylinder commands via gradient descent at runtime.

**Implementation**:
```python
def find_cylinder_command(desired_ee_delta, state, forward_model, steps=10):
    cyl_cmd = torch.zeros(4, requires_grad=True)
    optimizer = torch.optim.Adam([cyl_cmd], lr=0.01)

    for _ in range(steps):
        pred_ee = forward_model(state, cyl_cmd)
        loss = F.mse_loss(pred_ee, desired_ee_delta)
        loss.backward()
        optimizer.step()

    return cyl_cmd.detach()
```

**Advantages**:
- ✅ Uses accurate forward model
- ✅ Handles configuration-dependent kinematics
- ✅ No adapter training needed

**Disadvantages**:
- ⚠️ Slow (optimization at every step)
- ⚠️ May not converge for all states

**Estimated time**: ~30 minutes to implement and test

---

## RECOMMENDED PATH FORWARD

**Immediate (next 30 minutes)**: Try **Option 2 (Analytical Jacobian)**
- Fastest to implement and test
- Will immediately show if PPO can work with approximate inverse
- If it works, we're done! If not, proceed to Option 1

**If Option 2 fails (next 2 hours)**: Implement **Option 1 (Trajectory Data)**
- Collect multi-step trajectory dataset
- Retrain adapter with meaningful data
- Validate with direction tests

**Last resort**: **Option 3 (Direct 4D PPO)**
- Skip all adapter complexity
- Accept higher-dimensional action space

---

## LESSONS LEARNED

1. **Single-step dynamics don't work**: Hydraulic systems need multi-step horizons
2. **Random exploration is insufficient**: Need goal-directed data collection
3. **Validate data before training**: Check correlations to ensure learnable signal exists
4. **Forward-consistency loss can be harmful**: If forward model is inaccurate, it pushes adapter to conservative (zero) outputs

---

## NEXT COMMAND

**Test analytical Jacobian** (Option 2 - fastest validation):
```bash
# I'll create a simple adapter wrapper with analytical Jacobian and test it
```
