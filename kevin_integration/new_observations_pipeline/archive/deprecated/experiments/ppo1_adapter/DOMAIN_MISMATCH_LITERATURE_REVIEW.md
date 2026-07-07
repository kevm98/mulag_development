# Domain Mismatch Problem: Task-Space PPO ↔ Cylinder-Space Learned Models

**Date:** 2026-07-04
**Problem:** PPO (IsaacLab, task-space) → ID/FD models (Simscape, cylinder-space)

---

## Problem Statement

### The Pipeline Gap

```
PPO (IsaacLab)          Kevin's Models (Simscape)
     ↓                         ↓
Task Space (Y, Z)       Cylinder Space (q₁, q₂, q₃, q₄)
     ↓                         ↓
Desired ΔEE            valve_cmd, torques
     ↓                         ↓
   [GAP]  ← MISSING ADAPTER
```

**Kevin's Models Expect (from learned_models.py)**:
- **ID Input**: `x_hist (B, T, 19)` = `[q(5), qdot(5), qddot(5), dP(4)]` + `z_arm (B, 24)`
  - Cylinder positions, velocities, accelerations, pressure differences
  - **Output**: `valve_cmd (B, 4)` - valve commands in cylinder space

- **FD Input**: `x_hist (B, T, 18)` = cylinder state history
  - **Output**: `torque (B, 4)` - cylinder forces

**PPO (IsaacLab) Operates In**:
- **Observation**: Task-space error `[ΔY, ΔZ]` (EE position error)
- **Action**: Task-space command `[desired_ΔY, desired_ΔZ]`

**Critical Issues**:
1. **Coordinate space mismatch**: Task space (2D EE) vs cylinder space (4D joint)
2. **Simulator mismatch**: IsaacLab (PhysX) vs Simscape (MATLAB/Simulink)
3. **Dynamics mismatch**: Even if kinematics match, dynamics differ between simulators

---

## What Literature Says

### 1. Hierarchical Control (Classic Solution)

**Key Papers**:
- Khatib, O. (1987). "A unified approach for motion and force control of robot manipulators: The operational space formulation." *IEEE Journal on Robotics and Automation*.
- Sentis, L., & Khatib, O. (2005). "Synthesis of whole-body behaviors through hierarchical control of behavioral primitives." *International Journal of Humanoid Robotics*.

**Approach**:
- **High-level controller** (PPO): Works in task space, outputs desired EE motion
- **Low-level controller** (ID/FD): Works in joint/cylinder space, computes actuator commands
- **Jacobian mapping**: Converts between task and joint spaces

**Mathematical Framework**:
```
τ = J^T(q) · F_task           (Operational space control)
q̈_desired = J^{-1}(q) · ẍ_EE  (Inverse kinematics)
```

**Problem for Hydraulics**:
- Standard Jacobian assumes rigid-body kinematics
- Hydraulic systems have **compliance**, **valve dynamics**, **pressure dynamics**
- Simple J^{-1} doesn't capture hydraulic actuator behavior

---

### 2. Learned Inverse Models (Neural Network Approach)

**Key Papers**:
- Jordan, M. I., & Rumelhart, D. E. (1992). "Forward models: Supervised learning with a distal teacher." *Cognitive Science*.
- Nguyen-Tuong, D., & Peters, J. (2011). "Model learning for robot control: A survey." *Cognitive Processing*.
- Kalakrishnan, M., et al. (2011). "Learning inverse dynamics models with contacts." *ICRA*.

**Approach**:
- Train neural network: `NN(x_task) → u_cylinder`
- Learn the inverse mapping from task-space goals to cylinder commands
- Can capture nonlinear, configuration-dependent relationships

**Challenges**:
- **Non-uniqueness**: Multiple cylinder configurations can achieve same EE position
- **Data requirements**: Need goal-directed trajectories, not random exploration
- **Generalization**: Model must work across workspace

**This is what we attempted** (task_to_cylinder_adapter.py), but it failed because:
1. Training data had zero correlation (random single-step commands)
2. Didn't account for multi-step dynamics
3. Inverse problem is ill-posed (multiple solutions exist)

---

### 3. Domain Adaptation (Sim-to-Sim Transfer)

**Key Papers**:
- Peng, X. B., et al. (2018). "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization." *ICRA*.
- Tobin, J., et al. (2017). "Domain randomization for transferring deep neural networks from simulation to the real world." *IROS*.
- Christiano, P., et al. (2016). "Transfer from Simulation to Real World through Learning Deep Inverse Dynamics Model." *arXiv*.

**Approach**:
- Train in one simulator (Simscape), deploy in another (IsaacLab)
- Learn **correction model** that bridges the gap
- Use **domain adaptation layers** to match distributions

**Mathematical Framework**:
```
u_IsaacLab = u_Simscape + δ(x, u)     (Additive correction)
u_IsaacLab = T_adapt(u_Simscape, x)   (Learned transformation)
```

**Specific Technique: Residual Learning**:
- Keep Kevin's ID/FD models frozen
- Learn **residual policy**: `π_residual(s) → Δu`
- Final command: `u = u_ID + π_residual(s)`

**Relevant for this problem because**:
- Kevin's models trained on Simscape data
- We're deploying in IsaacLab (different physics engine)
- Dynamics mismatch between simulators

---

### 4. End-to-End Learning (Bypass the Gap)

**Key Papers**:
- Levine, S., et al. (2016). "End-to-End Training of Deep Visuomotor Policies." *JMLR*.
- Haarnoja, T., et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL." *ICML*.
- Rajeswaran, A., et al. (2017). "Learning Complex Dexterous Manipulation with Deep RL and Demonstrations." *RSS*.

**Approach**:
- Train PPO **directly on cylinder actions**: `π(s_task) → u_cylinder`
- Skip the adapter entirely
- Let RL figure out the task→cylinder mapping

**Advantages**:
- No need for Kevin's models (which are Simscape-specific)
- RL learns the mapping end-to-end
- Works natively in IsaacLab

**Disadvantages**:
- Higher-dimensional action space (4D cylinders vs 2D task)
- May take longer to train
- Loses Kevin's prior knowledge from Simscape

---

### 5. Operational Space Control with Learned Dynamics

**Key Papers**:
- Righetti, L., et al. (2013). "Optimal distribution of contact forces with inverse-dynamics control." *International Journal of Robotics Research*.
- Buchli, J., et al. (2011). "Learning variable impedance control." *International Journal of Robotics Research*.
- Schaal, S., & Atkeson, C. G. (2010). "Learning control in robotics." *IEEE Robotics & Automation Magazine*.

**Approach**:
- Use **Operational Space Control** framework (Khatib 1987)
- Replace analytical Jacobian with **learned Jacobian**: `J_learned(q)`
- Replace rigid dynamics with **learned hydraulic dynamics**

**Mathematical Framework**:
```
Given: x_desired (task space goal)
Compute: ẍ_task = Kp(x_desired - x) + Kd(ẋ_desired - ẋ)
Map to joint space: q̈_desired = J^{-1}_learned(q) · ẍ_task
Compute torques: τ = M_learned(q)q̈ + C_learned(q, q̇) + G_learned(q)
Convert to valve commands: valve = ID_learned(q, q̇, τ)
```

**For hydraulics, this becomes**:
```
1. PPO outputs: x_desired (task space)
2. Compute task-space acceleration: ẍ_task
3. Learn J(q): maps task acceleration → cylinder acceleration
4. Learn M(q, valve): hydraulic dynamics (pressure, flow)
5. Solve for valve command
```

---

## Recommended Solutions (Ranked)

### Option 1: End-to-End PPO (SIMPLEST, RECOMMENDED)

**Approach**: Train PPO directly on cylinder actions, ignore Kevin's models.

**Implementation**:
```python
# Modify Stage1AdapterEnvWrapper
class DirectCylinderEnv:
    num_observations = 6  # [ee_error_Y, ee_error_Z, target_vel_Y, target_vel_Z, prev_u[4]]
    num_actions = 4       # [cylinder1, cylinder2, cylinder3, cylinder4]

    def step(self, action):
        # action = cylinder position targets (4D)
        cylinder_target = current_cyl_pos + action * action_scale
        low_level_action = cat([cylinder_target, tool_joint])
        return env.step(low_level_action)
```

**Advantages**:
- ✅ No domain adaptation needed
- ✅ Works natively in IsaacLab
- ✅ RL figures out task→cylinder mapping automatically
- ✅ Fastest to implement (< 1 hour)

**Disadvantages**:
- ⚠️ Doesn't use Kevin's Simscape models (but they may not transfer anyway!)
- ⚠️ 4D action space (vs 2D for task space)

**Time estimate**: 1 hour implementation + 2 hours training

---

### Option 2: Residual Learning Over Kevin's Models (MOST PRINCIPLED)

**Approach**: Use Kevin's ID/FD as base, learn residual correction for IsaacLab.

**Implementation**:
```python
# Wrap Kevin's ID model
id_model = InverseDynamicsModel(checkpoint_path="kevin_integration/models/id.pth")

class ResidualAdapter(nn.Module):
    def forward(self, task_goal, robot_state):
        # 1. Convert task goal to cylinder state change (simple approximation)
        q_desired = current_q + simple_jacobian(task_goal)

        # 2. Run Kevin's ID model
        valve_base = id_model(x_hist, z_arm)

        # 3. Learn residual correction
        valve_residual = self.residual_net(task_goal, robot_state)

        return valve_base + valve_residual
```

**Advantages**:
- ✅ Uses Kevin's prior knowledge
- ✅ Only learns the Simscape→IsaacLab correction
- ✅ Interpretable (base + correction)

**Disadvantages**:
- ⚠️ Requires interfacing with Kevin's models (complex state preparation)
- ⚠️ Kevin's models may not be valid in IsaacLab at all
- ⚠️ More complex architecture

**Time estimate**: 4 hours implementation + 3 hours training

---

### Option 3: Learned Jacobian + Operational Space Control

**Approach**: Learn the task→cylinder Jacobian, use OSC framework.

**Implementation**:
```python
class LearnedJacobian(nn.Module):
    def forward(self, ee_pos, cylinder_pos):
        # Input: current configuration
        # Output: J matrix (2x4) mapping cylinder vel → EE vel
        return self.net(cat([ee_pos, cylinder_pos]))

# At runtime
J = learned_jacobian(ee_pos, cyl_pos)
cyl_vel_desired = J.pinv() @ ee_vel_desired
cyl_target = cyl_pos + cyl_vel_desired * dt
```

**Advantages**:
- ✅ Physically motivated (OSC is well-established)
- ✅ Separates kinematics from dynamics
- ✅ Can handle configuration-dependent behavior

**Disadvantages**:
- ⚠️ Requires learning Jacobian from data
- ⚠️ Still doesn't address hydraulic dynamics (pressure, flow)
- ⚠️ May not be better than end-to-end RL

**Time estimate**: 3 hours implementation + 2 hours training

---

### Option 4: Retrain Kevin's Models in IsaacLab (MOST ROBUST, LONGEST)

**Approach**: Collect cylinder-space data in IsaacLab, retrain ID/FD models.

**Implementation**:
1. Collect data in IsaacLab: `(cylinder_state, valve_cmd) → cylinder_motion`
2. Retrain ID model: `(q, q̇, q̈, dP) → valve_cmd`
3. Retrain FD model: `(q, q̇, valve) → torque`
4. Use retrained models for PPO

**Advantages**:
- ✅ Kevin's architecture proven to work
- ✅ Models will match IsaacLab dynamics
- ✅ Preserves Kevin's control pipeline design

**Disadvantages**:
- ⚠️ Requires extensive data collection in IsaacLab
- ⚠️ May take weeks to collect and train
- ⚠️ Still need task→cylinder adapter for PPO

**Time estimate**: 2-3 weeks (data collection + training + integration)

---

## Immediate Recommendation

**Go with Option 1: End-to-End PPO on Cylinders**

**Why**:
1. **Domain mismatch is fundamental**: Kevin's models were trained on Simscape data. They may not transfer to IsaacLab at all.
2. **Simplest architecture**: No adapter, no domain adaptation, just 4D RL.
3. **RL is good at this**: PPO can learn task→cylinder mapping implicitly through reward shaping.
4. **Fastest validation**: We can test this in < 4 hours and know if it works.

**If Option 1 works**:
- Great! We have a working controller.
- Can publish results without needing Kevin's models.

**If Option 1 fails**:
- Then we know the problem is deeper (reward shaping, env design, etc.)
- Can try Option 2 (residual learning) or Option 3 (learned Jacobian)

---

## Key Literature Takeaways

1. **Hierarchical control** (Khatib 1987): Separate task-level and joint-level control
   - Standard in robotics, but assumes rigid-body kinematics

2. **Learned inverse models** (Jordan & Rumelhart 1992): Learn task→joint mapping
   - Works well when inverse is unique and smooth
   - Fails when inverse is multi-modal (our case!)

3. **Domain adaptation** (Peng et al. 2018): Transfer from one sim to another
   - Use residual learning to correct for dynamics mismatch
   - Relevant because Kevin's models ≠ IsaacLab dynamics

4. **End-to-end RL** (Levine et al. 2016): Learn everything jointly
   - Avoids hand-designing components
   - Works when reward signal is clear

5. **Operational space control** (Khatib 1987, Righetti et al. 2013): Rigorous framework
   - Can incorporate learned dynamics
   - May be overkill for this problem

---

## Action Plan

**Next Steps** (in order):

1. **[30 min] Modify wrapper for 4D cylinder actions**
   - Change num_actions from 2 → 4
   - Remove adapter, send cylinder targets directly

2. **[30 min] Test direction control manually**
   - Hardcode cylinder commands: `[+0.01, -0.01, 0, 0]` should move +Y
   - Verify IsaacLab can execute cylinder commands correctly

3. **[2 hours] Train PPO on 4D cylinders**
   - Run for 50-100 iterations
   - Check if distance to target decreases

4. **[1 hour] Evaluate and compare**
   - Does PPO learn to reach targets?
   - Compare to random baseline

5. **Decision point**:
   - If works → done! Report results.
   - If fails → try Option 2 or investigate env/reward issues.

---

## References

1. Khatib, O. (1987). "A unified approach for motion and force control of robot manipulators: The operational space formulation." *IEEE Journal on Robotics and Automation*, 3(1), 43-53.

2. Jordan, M. I., & Rumelhart, D. E. (1992). "Forward models: Supervised learning with a distal teacher." *Cognitive Science*, 16(3), 307-354.

3. Nguyen-Tuong, D., & Peters, J. (2011). "Model learning for robot control: A survey." *Cognitive Processing*, 12(4), 319-340.

4. Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). "Sim-to-real transfer of robotic control with dynamics randomization." *ICRA*.

5. Levine, S., Finn, C., Darrell, T., & Abbeel, P. (2016). "End-to-end training of deep visuomotor policies." *JMLR*, 17(1), 1334-1373.

6. Righetti, L., Buchli, J., Mistry, M., Kalakrishnan, M., & Schaal, S. (2013). "Optimal distribution of contact forces with inverse-dynamics control." *IJRR*, 32(3), 280-298.

7. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). "Domain randomization for transferring deep neural networks from simulation to the real world." *IROS*.

8. Sentis, L., & Khatib, O. (2005). "Synthesis of whole-body behaviors through hierarchical control of behavioral primitives." *International Journal of Humanoid Robotics*, 2(04), 505-518.
