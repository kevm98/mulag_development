# Target Frame Audit Report

## Summary

Frame audit PASSED. All components use the vehicle base frame consistently.
No world/base frame mismatch found.

## Simulation Results (from audit_target_frames.py)

| Metric | Value |
|--------|-------|
| Robot root world position | (-25.40, 4.30, 1.11) |
| EE world position | (-25.40, 0.88, 0.99) |
| EE base-frame position | (-0.00, -3.42, -0.12) |
| Command position (Stage 0) | (0.11, -3.28, -0.12) |
| EE error norm | 0.18 m |
| Target inside workspace | True |
| EE inside workspace | True |
| EE inside cab | False |
| Observation match | True |

## Frame Consistency

| Component | TCP Frame | Target/Guard Frame | Consistent |
|-----------|-----------|-------------------|------------|
| Reward (`compute_tracking_error`) | base | base (`_command_pos_w` as base) | Yes |
| Observation (`ee_pos_error_b`) | base | base | Yes |
| Workspace guard | base | base bounds | Yes |
| Cab guard | base | base bounds | Yes |

## Key Finding

The `_command_pos_w` variable is confusingly named (suggesting world frame) but
stores base-frame values. If it were actually in world frame and transformed to
base, the target would be at (26.45, -7.82, -1.22) — clearly impossible.

The 5.946 m mean EE error in the diagnostic run was caused by unreachable
targets (x=4-6m in base frame), NOT by frame mismatch.

## Stage-0 Fix

After Stage-0 target sampling fix (3D cab exclusion instead of x-only):
- Target distance: 0.18 m (within 0.15 m box, slight overshoot from clamping)
- EE error is now learnable
