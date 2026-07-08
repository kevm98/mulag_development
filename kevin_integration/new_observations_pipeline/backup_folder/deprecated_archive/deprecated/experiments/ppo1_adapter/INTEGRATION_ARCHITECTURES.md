# Integration Architectures: Preserving Kevin's Hydraulic Models

**Date:** 2026-07-04
**Key Constraint:** Kevin's ID/FD models contain **real hydraulic dynamics** and must be used

---

## The Core Challenge

Kevin's models capture **true hydraulic behavior**:
- Pressure dynamics (dP)
- Valve response characteristics
- Flow-pressure relationships
- Hydraulic compliance and time delays

**We cannot skip them.** But we need to bridge:
- **PPO domain**: Task space (Y, Z) in IsaacLab
- **Kevin's models domain**: Cylinder space (q, q̇, q̈, dP) from Simscape

---

## Architecture 1: Hierarchical with Learned Inverse Kinematics ⭐ RECOMMENDED

### Concept
```
PPO (task-space)
    ↓
[Learned IK Adapter]  ← Learn this mapping
    ↓
Desired cylinder state (q_des, q̇_des, q̈_des)
    ↓
Kevin's ID Model  ← Real hydraulic dynamics
    ↓
Valve commands (u_valve)
    ↓
Kevin's FD Model  ← Predicts actual motion
    ↓
Robot (IsaacLab)
```

### Implementation Details

**PPO Layer**:
- Observation: `[ee_error_Y, ee_error_Z, ee_vel_Y, ee_vel_Z, cylinder_state]`
- Action: `[desired_ΔY, desired_ΔZ]` (2D task space)

**IK Adapter** (learned neural network):
- Input: `[desired_ΔEE, current_EE, current_cyl_pos, current_cyl_vel]` (12D)
- Output: `[desired_cyl_pos, desired_cyl_vel, desired_cyl_acc]` (12D total)
- Architecture: MLP with 2-3 hidden layers
- Training: **Supervised learning on goal-directed trajectories** (not random!)

**Kevin's ID Model** (frozen):
- Input: `x_hist` = `[q, q̇, q̈, dP]` history + `z_arm` context
- Output: `valve_cmd` (4D)
- **Captures**: Hydraulic actuator dynamics, pressure-flow relationships

**Kevin's FD Model** (frozen, optional for validation):
- Input: `x_hist` = cylinder state history
- Output: Predicted cylinder forces/motion
- **Use**: Validate that valve commands will achieve desired motion

### Training Strategy

**Phase 1: Collect IK Training Data in IsaacLab**
```python
for episode in range(num_episodes):
    # Pick random EE target
    ee_target = sample_ee_target()

    # Use simple proportional control
    for t in range(trajectory_length):
        ee_current = get_ee_position()
        ee_error = ee_target - ee_current

        # Simple proportional controller (temporary)
        cyl_cmd = current_cyl + simple_jacobian(ee_error) * gain

        # Execute and record
        apply_cylinder_command(cyl_cmd)

        # Record successful motion
        if success:
            data.append({
                'ee_start': ee_start,
                'ee_target': ee_target,
                'cyl_start': cyl_start,
                'cyl_end': cyl_end,
                'cyl_vel_achieved': (cyl_end - cyl_start) / dt,
            })
```

**Phase 2: Train IK Adapter**
```python
# Supervised learning
loss = MSE(adapter_output, ground_truth_cylinder_motion)
```

**Phase 3: Integrate with Kevin's ID Model**
```python
def control_step(ppo_action, robot_state):
    # 1. PPO outputs task-space goal
    desired_ee_delta = ppo_action * scale

    # 2. IK adapter converts to cylinder goal
    desired_cyl_state = ik_adapter(
        desired_ee_delta,
        robot_state.ee_pos,
        robot_state.cyl_pos,
        robot_state.cyl_vel,
    )

    # 3. Prepare history for ID model
    x_hist = prepare_state_history(
        robot_state.cyl_pos,
        robot_state.cyl_vel,
        desired_cyl_state.cyl_acc,  # desired acceleration
        robot_state.pressure_diff,
    )

    # 4. Kevin's ID model computes valve commands
    valve_cmd = kevin_id_model(x_hist, z_arm)

    # 5. Apply valve commands (or use them as setpoints)
    return valve_cmd
```

**Advantages**:
- ✅ Uses Kevin's real hydraulic dynamics
- ✅ PPO stays in intuitive task space
- ✅ IK adapter only needs to learn kinematics, not dynamics
- ✅ Can validate with FD model

**Challenges**:
- ⚠️ Need to collect good IK training data (goal-directed, not random)
- ⚠️ Must prepare `x_hist` and `z_arm` correctly for ID model
- ⚠️ Need pressure sensor data (`dP`) - does IsaacLab provide this?

---

## Architecture 2: Residual Learning Over ID Model

### Concept
```
PPO (task-space)
    ↓
[Simple Jacobian] + [Learned Residual]
    ↓
Cylinder goals (q_des)
    ↓
Kevin's ID Model (base policy)
    ↓
Valve commands (base)
    ↓
+ [Learned Correction]  ← Learn IsaacLab-specific corrections
    ↓
Final valve commands
    ↓
Robot (IsaacLab)
```

### Implementation
```python
class ResidualController:
    def __init__(self, kevin_id_model):
        self.id_model = kevin_id_model  # Frozen
        self.residual_net = MLP(input_dim=state_dim, output_dim=4)

    def forward(self, ppo_action, robot_state):
        # 1. Simple kinematic mapping
        ee_delta = ppo_action
        cyl_delta = simple_jacobian(ee_delta)

        # 2. Base policy from Kevin's ID
        x_hist = prepare_history(robot_state, cyl_delta)
        valve_base = self.id_model(x_hist, z_arm)

        # 3. Learn residual correction
        valve_residual = self.residual_net(robot_state)

        # 4. Final command
        return valve_base + valve_residual
```

**Training**: PPO trains only the residual_net, ID model frozen.

**Advantages**:
- ✅ Leverages Kevin's hydraulic knowledge as prior
- ✅ Only learns the correction needed for IsaacLab
- ✅ Smaller learning problem (residual should be small)

**Challenges**:
- ⚠️ Still need simple_jacobian for initial mapping
- ⚠️ If base policy is very wrong for IsaacLab, residual may not help

---

## Architecture 3: Model-Based RL with FD as World Model

### Concept
```
PPO uses Kevin's FD model to plan ahead:
    - FD model predicts: "if I apply valve_cmd, what cylinder motion results?"
    - PPO learns to choose valve_cmds that achieve task goals
    - World model (FD) guides exploration
```

### Implementation
```python
class ModelBasedPPO:
    def __init__(self, kevin_fd_model):
        self.fd_model = kevin_fd_model  # Frozen world model
        self.policy = PPOPolicy(
            obs_dim=state_dim,
            action_dim=4,  # Valve commands directly
        )

    def plan_action(self, robot_state, ee_goal):
        # Sample multiple valve commands
        valve_candidates = sample_actions(n=10)

        # Use FD model to predict outcomes
        predicted_motions = []
        for valve_cmd in valve_candidates:
            x_hist = prepare_history(robot_state, valve_cmd)
            predicted_cyl_motion = self.fd_model(x_hist)
            predicted_ee_motion = fk(predicted_cyl_motion)
            predicted_motions.append(predicted_ee_motion)

        # Choose action that moves EE closest to goal
        best_idx = argmin(|predicted_ee_motion - ee_goal|)
        return valve_candidates[best_idx]
```

**Advantages**:
- ✅ Uses FD as accurate world model
- ✅ Can plan multiple steps ahead
- ✅ Model-based RL often more sample-efficient

**Challenges**:
- ⚠️ Complex implementation
- ⚠️ FD model may not match IsaacLab exactly
- ⚠️ Computational cost of planning

---

## Critical Missing Piece: Pressure Data

Looking at Kevin's ID model requirements:
```python
x_hist: (B, T, 19) = [q(5), qdot(5), qddot(5), dP(4)]
```

**The model needs pressure differences `dP(4)` for each cylinder!**

### Question: Does IsaacLab provide cylinder pressure data?

**If YES**: We can use Kevin's models directly.

**If NO**: We have two options:

#### Option A: Estimate Pressure from Force
```python
# PhysX provides cylinder forces
cylinder_forces = robot.data.applied_forces[cylinder_indices]

# Estimate pressure from force (simple model)
# F = P * A  (force = pressure * area)
cylinder_area = 0.01  # m² (example)
dP_estimated = cylinder_forces / cylinder_area
```

#### Option B: Learn a Pressure Estimator
```python
class PressureEstimator(nn.Module):
    """Learns dP from observable IsaacLab state."""
    def forward(self, cyl_pos, cyl_vel, cyl_force):
        return self.net(cat([cyl_pos, cyl_vel, cyl_force]))

# Train on paired data:
# Input: IsaacLab state
# Output: Pressure from Simscape simulation
```

---

## Recommended Implementation Plan

### Step 1: Check Pressure Availability (30 min)
```python
# In IsaacLab environment
print("Available robot data:")
print(dir(robot.data))

# Look for:
# - robot.data.cylinder_pressure
# - robot.data.hydraulic_pressure
# - robot.data.applied_forces (can estimate from this)
```

### Step 2: Implement Architecture 1 (Hierarchical IK) (4 hours)

**2.1 Collect IK Training Data** (1 hour)
- Run proportional controller to reach random EE targets
- Record: (ee_goal, cyl_motion_achieved)
- Need ~10k successful trajectories

**2.2 Train IK Adapter** (1 hour)
- Simple MLP: `[ee_delta, robot_state] → [cyl_delta]`
- Supervised learning with MSE loss

**2.3 Integrate Kevin's ID Model** (2 hours)
- Load ID model from checkpoint
- Prepare `x_hist` with proper format
- Handle `z_arm` context (set to default if not available)
- Apply valve commands to IsaacLab

### Step 3: Test Direction Control (30 min)
```python
# Test if integrated system can move in +Y direction
ppo_action = [+0.01, 0.0]  # +Y command
valve_cmd = integrated_controller(ppo_action, robot_state)
apply(valve_cmd)
measure_ee_motion()
```

### Step 4: Train PPO (2 hours)
- PPO now learns in task space with hydraulic-aware controller
- Should converge faster since dynamics are realistic

---

## Key Decision Point

**Before we proceed, please answer**:

1. **Pressure data**: Does IsaacLab provide cylinder pressure measurements?
   - Check: `robot.data.applied_forces` or similar

2. **Context vector**: What is `z_arm` (24D context) in Kevin's ID model?
   - Is this arm configuration, task context, or learned embedding?
   - Can we set it to a default value?

3. **State history**: Kevin's models expect history windows (T timesteps)
   - Do we need to buffer past states?
   - What is the history length T?

Once we know these, I can implement the exact integration code.

---

## Summary

You're **100% correct** - we must use Kevin's hydraulic models. The solution is:

**Hierarchical Architecture**:
1. PPO stays in task space (easier to learn)
2. **Learned IK adapter** bridges task→cylinder (this is what failed before, but we'll fix it with better data collection)
3. **Kevin's ID model** handles hydraulic dynamics (valve commands)
4. **Kevin's FD model** optionally validates predictions

The mistake before was trying to learn the adapter with random single-step data. Now we'll:
- Collect **goal-directed trajectories**
- Use **multi-step motion** to create learnable signal
- Train adapter to map task goals → cylinder goals
- Let Kevin's ID model handle the hydraulic complexity

**Next action**: Please confirm pressure data availability, then I'll implement the integration.
