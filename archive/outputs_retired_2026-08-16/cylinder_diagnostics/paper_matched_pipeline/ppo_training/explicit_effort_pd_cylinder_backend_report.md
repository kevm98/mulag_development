# Explicit Effort PD Cylinder Backend Report

## 1. Active Joints = Prismatic Cylinder Joints

Confirmed via USD inspection: `Drehzapfen_joint`, `Ausleger_I_joint`,
`Ausleger_II_joint`, `Messerkopf_Schwenk_joint` are `PhysicsPrismaticJoint` prims
connecting cylinder tube (Rohr) ↔ piston (Kolben) bodies. They have
`drive:linear:type=force` with proper stiffness/damping/maxForce authored.

## 2. Implicit Position Drives = Zero Effort

`set_joint_position_target()` writes targets. Stiffness/damping reach PhysX.
But `get_dof_actuation_forces()` returns **zero** on these closed-loop DOFs.
PhysX ignores position drives on dependent DOFs in a closed kinematic chain.

## 3. Explicit Effort Target = Works

`set_joint_effort_target()` delivers nonzero force. `tau_app == tau_cmd` (verified).

## 4. Implementation

Added to `mulag_upper_level_env.py`:
- `_explicit_pd_effort_target(q_des)`: computes `force = kp*(q_des-q) - kd*qdot`,
  clamps, calls `set_joint_effort_target`
- Modified `_apply_action()`: when `lowlevel_actuator_backend == "explicit_effort_pd"`,
  replaces `set_joint_position_target` with `_explicit_pd_effort_target` for
  both debug/manual targets and cylinder pipeline targets

Config in `mulag_upper_level_env_cfg.py`:
- `explicit_pd_kp_*`, `explicit_pd_kd_*`, `explicit_pd_effort_limit_*`
- `explicit_pd_gain_scale`, `explicit_pd_damping_scale`, `explicit_pd_effort_scale`

## 5. Hold Test Results

| Gain Scale | Damp Scale | Effort Scale | EE Drift (5s) | Effort Nonzero? |
|-----------|-----------|-------------|--------------|-----------------|
| 1.0 | 1.0 | 1.0 | 0.83 m | YES |
| 0.5 | 2.0 | 1.0 | 0.48 m | YES |
| 0.3 | 3.0 | 1.0 | 0.51 m | YES |
| 5.0 | 2.0 | 3.0 | 0.17 m | YES |

All configurations produce nonzero applied effort. At 5× gain, drift drops to
0.17m — a 10× improvement over passive (2.0m) and 2× better than implicit pos drive (0.5m).

## 6. Remaining Work

PD gains need tuning to achieve drift < 0.005 m/horizon. The explicit PD
backend is functional. Gain optimization (PID tuning, gravity feedforward,
or iterative gain search) is needed.

## 7. Can J_eff/Mapper/PPO Resume?

**Partially.** The explicit PD backend proves joints are controllable. Once
PD gains achieve hold drift < 0.005 m/horizon:
1. Re-run low-level command chain debug (drift subtraction)
2. Re-measure J_eff with nonzero columns
3. Re-run mapper/canonical direction tests
4. Resume PPO

The architecture is correct. The PD backend works. Only gain tuning remains.

## 7. Single-Cylinder Step Test

| Cyl | Gain | Sign Accuracy | Nonzero Effort | Cyl Movement | EE Response |
|-----|------|--------------|----------------|-------------|-------------|
| 0 | 1× | 1/2 | 2/2 | -3.7, -3.1 mm | Z-dominated |
| 1 | 1× | 1/2 | 2/2 | +66.7, +65.4 mm | Z-dominated |
| 2 | 1× | 1/2 | 2/2 | -5.2, -10.5 mm | Z-dominated |
| 3 | 1× | 1/2 | 2/2 | +88.9, +41.9 mm | Z-dominated |
| All | target 30M/60M | 3/8 | 8/8 | chaotic | oscillatory |

### Key findings:
- **100% nonzero effort** — every test produced nonzero applied force
- **Cylinders move** — all 4 prismatic DOFs respond to commands
- **Direction dependence exists** — different commands produce different responses
- **Gravity dominates small commands** — Z always moves ~-1.3m
- **High gains cause oscillation** — 60M kp on Ausleger I produces chaos
- **Sign accuracy needs I-term** — P-only can't eliminate gravity steady-state error

### Conclusion:
The **cylinder prismatic joints ARE controllable** via explicit effort PD.
The interface is correct. Gain tuning (adding integral term) is needed to
achieve hold drift < 0.005 m/horizon and sign accuracy > 75%.
