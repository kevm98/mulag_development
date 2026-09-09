# Diagnostic CSV Deep Analysis

## 1. Termination Histogram

| Cause | Count | Rate |
|-------|-------|------|
| workspace_violation | 1211 | 97.3% |
| cylinder_limit | 32 | 2.6% |
| timeout | 1 | 0.1% |
| base_flying | 1 | 0.1% |
| success | 0 | 0.0% |
| cab_guard_violation | 0 | 0.0% |
| collision | 0 | 0.0% |

**Conclusion**: Workspace violation dominates entirely. The task as configured is not learnable.

## 2. Workspace Violation Timing Distribution

1245 total episodes, 1211 workspace violations.

| Percentile | Episode Length (horizon steps) |
|-----------|------|
| Min | 1 |
| P10 | 7 |
| P25 | 11 |
| P50 | 13 |
| P75 | 17 |
| P90 | 24 |
| Max | 64 |
| Mean | 14.70 |

- 0-3 steps: 1.5% (rarely immediate)
- 3-7 steps: 2.3%
- 7-10 steps: 14.2%
- 10-15 steps: 44.6% (most common)
- 15-20 steps: 18.7%
- 20-30 steps: 14.1%
- 30+ steps: 4.5%

**Conclusion**: Workspace violations are NOT immediate. Most occur at 10-15 horizon steps. The arm moves toward the unreachable target for ~14 steps before crossing the workspace boundary.

## 3. Mean Episode Length per Done Cause

| Cause | Mean Length |
|-------|-------------|
| workspace_violation | 14.70 |
| cylinder_limit | 38.50 |
| timeout | 5.00 |
| base_flying | 22.00 |

**Conclusion**: Cylinder-limit episodes survive longer (38.5 steps) — these are the episodes where the arm extends deeply but hits cylinder joint limits instead of workspace boundaries. Workspace violations cut episodes short at ~15 steps.

## 4. Mean Final EE Error per Done Cause

| Cause | Mean Final EE Error (m) |
|-------|------------------------|
| workspace_violation | 5.938 |
| cylinder_limit | 6.208 |
| timeout | 6.842 |
| base_flying | 6.437 |

**Conclusion**: Final EE error is consistently 5.9-6.8m across all causes. The arm cannot close the gap to the target.

## 5. Mean Initial EE Error per Done Cause

| Cause | Mean Initial EE Error (m) |
|-------|--------------------------|
| workspace_violation | 6.085 |
| cylinder_limit | 6.884 |
| timeout | 5.351 |
| base_flying | 7.085 |

**Conclusion**: Initial EE error is 5.4-7.1m across all causes. The target is simply too far from the reset TCP position.

## 6. Mean EE Improvement per Done Cause

| Cause | Mean Improvement (m) |
|-------|---------------------|
| workspace_violation | 0.147 |
| cylinder_limit | 0.677 |
| timeout | -1.491 |
| base_flying | 0.648 |

**Conclusion**: For workspace violations, the arm makes very little progress (0.15m). Cylinder-limit episodes show more progress (0.68m) — the arm extends further, survives longer, but still can't reach the target. The negative timeout improvement is from a single outlier.

## 7. Mean Guard Fraction per Done Cause

| Cause | Guard Fraction |
|-------|---------------|
| workspace_violation | 0.0000 |
| cylinder_limit | 0.0070 |

**Conclusion**: The workspace guard is essentially never triggered (guard_fraction = 0). This means the pre-step guard check finds the TCP in-bounds, but the post-step workspace check finds it out-of-bounds. The guard is one step behind — it guards the cylinder target but cannot prevent the arm from moving outside workspace during the smooth ramp.

## 8. Mean Clipping Fraction per Done Cause

| Cause | Clipping Fraction |
|-------|------------------|
| workspace_violation | 0.279 |
| cylinder_limit | 0.449 |

**Conclusion**: About 28% of delta commands are being clipped for workspace violations, and 45% for cylinder limits. This means the FD-predicted delta often exceeds the clip limit — the model wants to move the arm aggressively toward the far target, but the clip constrains it.

## 9. Mean Action Norm per Done Cause

| Cause | Action Norm |
|-------|------------|
| workspace_violation | 1.043 |
| cylinder_limit | 1.188 |

**Conclusion**: Action norms are near 1.0 (the PPO action space bounds). PPO is producing aggressive actions trying to reach the far target, but the clipping and workspace guard prevent effective motion.

## 10. Whether Workspace Violation Happens Immediately After Reset

**No.** Only 1.6% of workspace violations occur within 3 horizon steps. Most (44.6%) occur between 10-15 steps. The arm moves for ~14 steps before violating the workspace boundary.

This means the arm starts in-bounds, extends toward the target, and eventually crosses the boundary. The guard does not prevent this because it only checks before the smooth ramp, not after.

## 11. Whether Final EE Error Is Already Huge at Reset

**Yes.** Initial EE error distribution:
- Min: 4.48m
- P25: 5.61m
- P50: 6.14m
- P75: 6.58m
- Max: 7.68m
- Mean: 6.11m

**Every single episode starts with EE error > 4.48m.** The target (x=4-6m, y=-1.5 to 1.5m, z=0.3m in base frame) is far beyond the reset TCP position (~1-2m in base frame). The arm physically cannot reach these targets.

## Root Cause Summary

The primary root cause is **unreachable targets**. The command ranges (x=4.0-6.0m, y=-1.5 to 1.5m, z=0.3m in base frame) place targets far beyond the arm's maximum reach. The typical reset TCP is at ~1.5-2.5m in base frame, yielding 4-7m initial errors.

Secondary issues:
1. The workspace guard only checks before the smooth ramp, not after — so the arm can drift out of bounds during the ramp and get terminated.
2. The hard termination on workspace violation means the arm never gets a chance to learn recovery.
3. The reward structure penalizes failure but provides no signal for partial progress toward a reachable target.
4. The PPO learns to produce max-magnitude actions (norm=1.0) trying to reach impossible targets.

## Required Fixes

1. Replace target sampling with local-reach curriculum (target near reset TCP).
2. Change workspace violation from hard termination to soft penalty (with persistent-violation timeout).
3. Add post-step workspace guard to clamp cylinders when TCP leaves workspace.
4. Redesign reward to reward progress toward reachable targets.
5. Disable scripted vehicle motion for Stage 0.

