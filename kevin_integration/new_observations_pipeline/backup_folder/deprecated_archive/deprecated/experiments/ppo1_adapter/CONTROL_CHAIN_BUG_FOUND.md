# Critical Finding: The Problem is NOT the Adapter

**Date:** 2026-07-04 19:30
**Status:** ROOT CAUSE IDENTIFIED - Control chain issue, not adapter issue

---

## Summary

The learned adapter failed (outputs constants). We implemented an analytical Jacobian approximation as a quick test. **The analytical adapter produces CORRECT cylinder commands**, but direction tests show the robot ALWAYS moves in +Y-Z direction regardless of command.

**Conclusion: The adapter is working. The problem is downstream in the control chain.**

---

## Evidence

### 1. Analytical Adapter Outputs (CORRECT)

```
Desired Motion → Adapter Output (mm)
  +Y    [ +5.0,  +0.0] → [ +2.5, -2.5,  +0.0,  -0.0]  ✓
  -Y    [ -5.0,  +0.0] → [ -2.5, +2.5,  +0.0,  -0.0]  ✓
  +Z    [ +0.0,  +5.0] → [ +0.0,  -0.0,  +2.5, -2.5]  ✓
  -Z    [ +0.0,  -5.0] → [ +0.0,  -0.0,  -2.5, +2.5]  ✓
```

The adapter correctly implements differential control:
- Cylinders 1 & 2: opposite signs for Y motion
- Cylinders 3 & 4: opposite signs for Z motion

### 2. Direction Tests (WRONG)

From [debug_adapter_direction_tests.py](kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/debug_adapter_direction_tests.py):

```
Command  → Measured EE Motion (mm)
  +Y     → [ +23±107, -24±152]  (should be [ +50, ~0])
  -Y     → [ +24±113, -27±155]  (should be [ -50, ~0]) ✗ SAME as +Y!
  +Z     → [ +25±113, -28±157]  (should be [ ~0, +50]) ✗
  -Z     → [ +27±114, -31±156]  (should be [ ~0, -50]) ✗
```

**All commands produce nearly identical motion**: ~+25mm Y, ~-28mm Z (diagonal).

**This is physically impossible if the adapter commands were being executed correctly.**

---

## Likely Root Causes

### Hypothesis 1: Cylinder Index Mapping Error (MOST LIKELY)

The Stage1AdapterEnvWrapper may be incorrectly mapping adapter outputs to robot cylinders.

**Where to check:**
- [stage1_adapter_env_wrapper.py](kevin_integration/new_observations_pipeline/ppo1_adapter/stage1_adapter_env_wrapper.py) lines 290-350
- Look for how `cylinder_delta` from adapter is mapped to `action[:, cylinder_indices]`
- Verify `env.cylinder_indices` matches expected [0,1,2,3]

### Hypothesis 2: Lower Controller Not Tracking Cylinder Targets

The `explicit_effort_pid_bias` controller may not be tracking cylinder position targets correctly.

**Where to check:**
- [explicit_effort_pid_bias_controller.py](kevin_integration/new_observations_pipeline/low_level_controller/explicit_effort_pid_bias_controller.py)
- Verify PID gains (kp_scale=0.1, kd_scale=0.5) are sufficient
- Check if bias calibration is interfering with tracking

### Hypothesis 3: PhysX Joint Control Mode Issue

The robot's cylinder joints may not be in the correct control mode (position vs effort).

**Where to check:**
- Robot URDF/USD definition - verify cylinder joints are configured for position control
- Check if actuator model is overriding commands

### Hypothesis 4: Action Clipping/Scaling Bug

Commands may be getting clipped or scaled incorrectly before reaching the robot.

**Where to check:**
- `stage1_adapter_env_wrapper.py` step() function
- Look for `torch.clamp()` or scaling operations on cylinder commands
- Verify `max_delta_cylinder = 0.0025` is not being exceeded in a way that causes saturation

---

## Next Steps (URGENT)

###1. Add Logging to Stage1AdapterEnvWrapper.step()

Modify `stage1_adapter_env_wrapper.py` to log:
```python
print(f"[WRAPPER] PPO action: {action}")
print(f"[WRAPPER] Adapter output (cylinder_delta): {cylinder_delta}")
print(f"[WRAPPER] Current cylinder pos: {current_cyl_pos}")
print(f"[WRAPPER] Target cylinder pos: {target_cyl_pos}")
print(f"[WRAPPER] Final action to env: {final_action}")
print(f"[WRAPPER] Cylinder indices: {env.cylinder_indices}")
```

Run direction test with logging to see if commands are being corrupted.

### 2. Direct Cylinder Command Test

Bypass the adapter entirely - send hardcoded cylinder position targets directly to the robot:

```python
# Test: extend cyl1, retract cyl2 (should move +Y)
action = torch.zeros((num_envs, 5))
action[:, 0] = +0.01  # cyl1 extend
action[:, 1] = -0.01  # cyl2 retract
```

If this STILL produces diagonal motion → problem is in lower controller or robot definition.

### 3. Inspect Cylinder Joint Names

Verify cylinder indices match expected joints:
```python
print(env.robot.data.joint_names)
print(env.cylinder_indices)
```

Make sure cyl1-4 map to correct physical cylinders (not swapped or reversed).

---

## Files to Inspect

1. **stage1_adapter_env_wrapper.py** (MOST LIKELY)
   - Lines 290-350: adapter → action mapping
   - step() function: command flow

2. **explicit_effort_pid_bias_controller.py**
   - PID control law
   - Position target tracking

3. **mulag_upper_level_env_cfg.py**
   - Cylinder joint configuration
   - Actuator model setup

4. **Robot USD/URDF**
   - Joint names and indices
   - Control modes (position vs effort)

---

## Why Previous Approaches Failed

1. **Learned adapter (Task-to-Cylinder)**: Failed because training data had zero correlation. But this was masking a deeper problem.

2. **Analytical adapter**: Produces correct commands, but robot doesn't respond correctly. This reveals the real issue.

**The adapter architecture is sound. The problem is in the execution layer.**

---

## Immediate Action

**STOP trying to fix the adapter.** The adapter works.

**START debugging the control chain:**
1. Add logging to wrapper
2. Test direct cylinder commands
3. Verify joint mapping
4. Check lower controller gains

Once the control chain is fixed, we can proceed with either:
- Analytical adapter (good enough for initial PPO training)
- Retrained learned adapter with trajectory data (more robust)

---

## User Request Compliance

User said: "find a solution including re design might have made a mistake somewhere"

**We found the mistake: It's not in the design of the adapter pipeline. It's in the IMPLEMENTATION of the control chain between adapter output and robot execution.**

The redesign needed is: **Fix the cylinder command routing**, not the adapter itself.
