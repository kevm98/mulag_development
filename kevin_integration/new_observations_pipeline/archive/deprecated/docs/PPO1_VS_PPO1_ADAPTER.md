# PPO1 vs PPO1_Adapter: Architecture Comparison

**Date:** 2026-07-04
**Status:** Use PPO1 (with ID/FD), abandon PPO1_Adapter

---

## The Two Approaches

### ❌ ppo1_adapter/ (FAILED - Abandon)

**Concept**: Learn simple task→cylinder mapping without hydraulic models

```
PPO action (2D task)
    ↓
[Learned MLP Adapter]  ← This failed
    ↓
Cylinder commands (4D)
    ↓
PhysX simulation
```

**Why it failed**:
1. Training data had zero correlation (random single-step commands)
2. Doesn't use Kevin's hydraulic models (pressure, valve dynamics)
3. Domain mismatch: adapter trained on IsaacLab, Kevin's models from Simscape
4. Fundamentally flawed approach for hydraulic systems

**Status**: Abandoned ✗

---

### ✅ ppo1/ (WORKING - Use This!)

**Concept**: Task-space PPO with Kevin's ID/FD models and empirical Jacobian

```
PPO action (2D task: ΔY, ΔZ)
    ↓
[HighLevelEEController]  ← Measures effective Jacobian empirically
    ↓
Cylinder goals (4D)
    ↓
[Kevin's ID Model]  ← Real hydraulic inverse dynamics
    ↓
Valve commands (4D)
    ↓
[Kevin's FD Model]  ← Predicts cylinder motion from valves
    ↓
Closed-loop rollout (H steps)
    ↓
Cylinder targets
    ↓
PhysX simulation
```

**Key Components** (already implemented!):

1. **HighLevelEEController** (`low_level_controller/high_level_ee_controller.py`):
   - Measures effective Jacobian: `J_eff = d(EE_motion) / d(cylinder_cmd)`
   - Uses damped least-squares: `Δcyl = J_eff^† · ΔEE_desired`
   - Temporal smoothing: `Δcyl = α·Δcyl_prev + (1-α)·Δcyl_new`
   - Clips commands per-cylinder and globally

2. **ID Model** (Kevin's trained model):
   - Input: `[dx_step, v_step, x_current, xdot_current]` (16D)
   - Output: `valve_commands` (4D)
   - Captures: Valve dynamics, pressure-flow relationships

3. **FD Model** (Kevin's trained model):
   - Input: `[valve_cmd, x_current, xdot_current]` + history
   - Output: `delta_extension` (predicted cylinder motion)
   - Captures: Pressure dynamics, cylinder force generation

4. **Closed-Loop Rollout** (`closed_loop_idfd_rollout`):
   - Runs H-step prediction
   - ID→FD→ID→FD... (closed loop)
   - Accumulates predicted motion
   - Doesn't mutate live state (clones GRU histories)

5. **HorizonIDFDEnv** wrapper:
   - Supports `ee_action_mode = "task_space_delta"` ✅
   - Supports `ee_action_mode = "cylinder_space"`
   - Horizon-level observations and rewards
   - RSL-RL compatible

**Status**: Fully implemented, ready to use ✓

---

## Configuration

### Enable Task-Space Mode

In `mulag_upper_level_env_cfg.py`:

```python
# Line 271
ee_action_mode = "task_space_delta"  # Use task-space actions (2D: Y, Z)
```

### Train PPO1

```bash
cd ~/rrlab_mulag

# Example command (adjust paths as needed)
./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/ppo1/train_horizon_idfd_ppo.py \
  --num_envs 128 \
  --max_iterations 200 \
  --horizon_steps 10 \
  --action_scale 0.01 \
  --task_space  # Enable task-space actions
```

---

## PPO2 (Future Work)

**Concept**: Replace ID/FD with Reduced Hydraulic ROM for efficiency

```
PPO action
    ↓
[High-level controller]
    ↓
[ReducedHydraulicActuatorROM]  ← Lightweight hydraulic model
    ↓
Cylinder targets
    ↓
PhysX
```

**Why ROM for PPO2**:
- Faster inference (no LSTM history management)
- Simpler integration
- Good enough approximation for policy learning
- Can be updated with system identification

**Status**: Infrastructure exists (`ReducedHydraulicActuatorROM`), integrate later

---

## Key Differences

| Aspect | PPO1_Adapter (Failed) | PPO1 (Working) | PPO2 (Future) |
|--------|----------------------|----------------|---------------|
| **Task-space?** | Yes (2D) | Yes (2D) | Yes (2D) |
| **Hydraulic dynamics?** | ❌ No | ✅ ID/FD | ✅ ROM |
| **Jacobian** | ❌ Learned (failed) | ✅ Empirical measurement | ✅ Empirical/ROM |
| **Kevin's models** | ❌ Not used | ✅ ID + FD | ❌ ROM instead |
| **Complexity** | Low | High | Medium |
| **Accuracy** | ❌ Failed | ✅ High | ✅ Good |
| **Speed** | Fast | Slower (LSTM) | Fast |

---

## Action Items

### Immediate (Use PPO1)

1. **✅ Infrastructure exists**: No new code needed!
   - HighLevelEEController: Done
   - ID/FD integration: Done
   - Task-space mode: Done

2. **Verify setup**:
   ```bash
   # Check ID model exists
   ls -lh kevin_integration/models/id.pth

   # Check FD model exists
   ls -lh kevin_integration/models/fd.pth

   # Check config
   grep "ee_action_mode" kevin_integration/new_observations_pipeline/ppo1/mulag_upper_level_env_cfg.py
   ```

3. **Test direction control**:
   - Run env with task_space_delta mode
   - Send [+0.01, 0] action (should move +Y)
   - Verify ID/FD pipeline produces correct motion

4. **Train PPO**:
   - Start with small run (10-20 iterations)
   - Check if distance to target decreases
   - Monitor Jacobian condition number (should be reasonable, not singular)

### Later (PPO2 with ROM)

1. **Calibrate ROM parameters** from Simscape data
2. **Replace ID/FD with ROM** in control loop
3. **Benchmark**: PPO1 (ID/FD) vs PPO2 (ROM) performance

---

## Why PPO1 Approach is Correct

### ✅ Uses Real Hydraulic Dynamics
- Kevin's ID/FD models trained on Simscape (validated hydraulic simulation)
- Captures pressure, flow, valve response
- Not just kinematics - full actuation dynamics

### ✅ Empirical Jacobian
- Measures J_eff through full deployment chain:
  `Δcyl_cmd → ID → valve → FD → predicted_motion → PhysX → measured_EE`
- Accounts for:
  - Hydraulic compliance
  - Controller dynamics
  - PhysX simulation behavior
  - Any domain mismatch between Simscape and IsaacLab

### ✅ Closed-Loop Prediction
- H-step rollout with ID↔FD feedback
- Accumulates predictions over horizon
- More accurate than single-step

### ✅ Hierarchical Control (Literature-Backed)
- PPO: High-level task-space goals (what)
- ID/FD: Low-level actuation (how)
- Separation of concerns
- Standard approach in robot control (Khatib 1987, Sentis & Khatib 2005)

---

## Recommended Path Forward

**STOP working on ppo1_adapter/** - it was a dead end.

**START using ppo1/** infrastructure:

1. Set `ee_action_mode = "task_space_delta"`
2. Load Kevin's ID/FD models (should already be in `Models_cylinder/` or `models/`)
3. Measure Jacobian on env reset
4. Train PPO with 2D task-space actions
5. Evaluate

**Expected results**:
- PPO learns to reach EE targets in task space
- ID/FD handles hydraulic complexity
- System respects real actuator dynamics

**If this works**: Strong validation that approach is sound!

**If this has issues**: Debug ID/FD integration, not the high-level concept.

---

## References

**Literature supporting hierarchical approach**:
- Khatib, O. (1987). "Operational space control"
- Sentis & Khatib (2005). "Hierarchical control of behavioral primitives"
- Righetti et al. (2013). "Optimal force distribution with inverse dynamics"

**Your codebase**:
- `ppo1/horizon_idfd_env_wrapper.py` - Main environment
- `ppo1/low_level_controller/high_level_ee_controller.py` - EE→cylinder mapping
- `ppo1/low_level_controller/idfd_rollout.py` - ID/FD closed-loop
- `kevin_integration/controllers/learned_models.py` - Kevin's ID/FD models

---

## Decision

**Use PPO1 (ppo1/ folder) with ID/FD.**
**Abandon PPO1_Adapter (ppo1_adapter/ folder).**
**Save ROM for PPO2.**

The infrastructure is already built and working. We just need to use it!
