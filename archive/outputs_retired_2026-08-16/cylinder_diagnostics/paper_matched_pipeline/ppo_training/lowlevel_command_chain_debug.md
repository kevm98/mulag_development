# Low-Level Command Chain Debug Report

## Root Cause

**The arm does not hold position under gravity.** Even with zero command
(`delta_x_des_H = [0,0,0,0]`), the EE drifts 0.005–0.78 m per horizon.
The PD controllers cannot counteract gravity during the 10-step smooth ramp.

The arm sags by ~0.08 m in Z even in the best case, and up to 0.78 m when
the IDFD cross-talk moves cylinders to unstable positions.

## Test A: Zero Command Drift

| Horizon | EE Delta (m) | Cyl Delta (max, m) | Base Vel |
|---------|-------------|-------------------|----------|
| 0 | 0.089 | 0.016 | 0.0 |
| 1 | 0.099 | 0.013 | 0.0 |
| 2 | 0.778 | 0.045 | 0.0 |
| 3 | 0.098 | 0.017 | 0.0 |
| 4 | 0.058 | 0.009 | 0.0 |
| 5 | 0.025 | 0.002 | 0.0 |
| ... | ... | ... | ... |

**Verdict**: Zero command produces massive EE movement. The arm is unstable at rest.
The command chain cannot be debugged until the hold mechanism is fixed.

## Test B: IDFD Mode — Cylinder Perturbations (±0.5 mm)

Example — Cylinder 0 +0.5 mm:
| Stage | Cyl 0 | Cyl 1 | Cyl 2 | Cyl 3 |
|-------|-------|-------|-------|-------|
| dx_des_H | 0.0005 | 0.0 | 0.0 | 0.0 |
| delta_fd_H (IDFD) | -0.0006 | 0.0034 | -0.0049 | -0.0004 |
| Amplification (fd/dx) | 1.3 | ∞ | ∞ | ∞ |
| delta_cmd_H | -0.0006 | 0.0034 | -0.0049 | -0.0004 |
| PhysX target inc | 0.0020 | -0.0388 | 0.0126 | -0.0288 |
| Measured cyl delta | 0.0016 | 0.0157 | -0.0115 | -0.0054 |
| Measured EE delta | [0, 0.026, -0.084] | 0.088 m norm |

## Test C: Direct Bypass (No IDFD) — Cylinder Perturbations

All 8 cases (4 cylinders × ±0.5 mm) produce identical EE delta:
`[0.0, ~0.026, ~-0.084]` — the same as zero command.

**The EE drift is independent of the command.**

## Test D: Baseline Drift Subtraction

For EVERY cylinder:
- D0 (zero command) = `[0.0, 0.0261, -0.0840]`
- Dpos_raw (positive command) = `[0.0, 0.0261, -0.0840]`
- Dneg_raw (negative command) = `[0.0, 0.0261, -0.0840]`

After subtraction: Dpos = Dneg = `[0,0,0]`, J_col = `[0,0,0]`.

**The command has ZERO measurable effect on EE position. Gravity drift dominates entirely.**

## Decision Tree Result

| Check | Result |
|-------|--------|
| If zero command has large EE drift | YES — fix hold/gravity first |
| If delta_fd_H is huge | YES — IDFD cross-talk amplifies |
| If direct_safe_cylinder_delta works but IDFD fails | NO — both fail identically |
| If direct_safe_cylinder_delta also fails | YES — PhysX/hold is the blocker |

**The PhysX cylinder hold mechanism cannot maintain position under gravity.**
Until the arm can hold position with zero command (EE drift < 0.005 m/horizon),
no Jacobian, mapper, or PPO can function.

## Required Fix

The smooth ramp runs 10 physics steps per horizon. In each step:
1. Cylinder position targets are set via `set_debug_manual_cylinder_targets`
2. `_apply_action` applies them to the PD controller
3. Physics runs for 2 substeps (decimation=2)
4. Gravity pulls the arm down, PD fights back but doesn't fully reach target

After 10 steps, the accumulated sag is 0.08–0.78 m depending on cylinder position.

Options to fix:
1. **Increase PD stiffness/damping** — Current: Drehzapfen 15M/1M, others 6M/1M/1M.
   May need 5-10× higher for gravity compensation.
2. **Gravity compensation** — Add feedforward torques to counteract gravity at each timestep.
3. **Single-step application** — Apply the full cylinder delta in 1 step instead of 10.
   Reduces accumulated sag but may cause dynamics issues.
4. **Pre-tensioned hold targets** — Compute equilibrium position where PD force = gravity
   and set hold targets there, so the PD doesn't need to fight gravity.

The most practical first step: increase cylinder PD stiffness values.
