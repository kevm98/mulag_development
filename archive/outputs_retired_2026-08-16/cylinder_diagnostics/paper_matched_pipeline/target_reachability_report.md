# Target Reachability Report

## Summary

The original command ranges (x=4.0-6.0 m, y=-1.5 to 1.5 m, z=0.3 m in base frame) place targets far beyond the arm's reachable workspace. The TCP at reset is at approximately 1.5-2.5 m in base frame, yielding initial EE errors of 4.5-7.7 m.

## Classification

Based on diagnostic CSV analysis of 1245 episodes:
- 0% reachable (initial EE < 1.0 m)
- 0% borderline (initial EE 1.0-3.5 m)
- 100% unreachable (initial EE > 3.5 m)

All targets are beyond the estimated maximum arm reach of ~3-3.5 m.

## Recommendation

Replace target sampling with Stage-0 local reach curriculum. Targets should be sampled within 15 cm of the reset TCP pose and clamped into the safe workspace.
