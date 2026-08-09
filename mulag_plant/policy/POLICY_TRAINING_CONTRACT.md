# PPO2 ROM policy training contract

## Objective

Train one PPO Policy that improves the validated ROM teacher on the exact line,
arc, sine, circle, and sharp square while preserving smooth arm motion, head-down
alignment, and a stationary truck. The policy is conditioned on local reference motion;
it is not told the shape name.

## Fixed ownership

- The deterministic supervisor owns parking, stowing, straight driving, and station changes.
- The reachable trajectory library and command ROM form the hydraulic baseline.
- PPO acts only in `ARM_TRACK`, only on cylinders 1-3, and only within ±2 mm.
- Cylinder 4 owns mower-head pitch. PPO never controls wheels or the truck root.

## Fixed training profile

- Seed: 17.
- Environments: 64; rollout: 128 steps per environment.
- Iterations 0-349: five exact plant-reachable paths.
- Iterations 350-499: 70% exact and 30% plant-gated procedural paths.
- Requested path speed: 0.18 m/s with the library's curvature speed factors.
- Training starts from a new actor, not a legacy residual checkpoint.

## Promotion gate

A checkpoint is eligible only if all exact cycles complete for three seeds, aggregate
mean and p95 tool error improve on the zero-policy ROM baseline, no shape's mean error
regresses by more than 2 mm, measured p95 tool acceleration does not increase, minimum
head alignment is at least 0.98, maximum reference-relative clearance error is at most
0.20 m, and post-constraint truck drift is at most 1 mm.

Passing these gates supports new paths only within the measured workspace and trained
speed/curvature envelope. Every new path must still be plant-gated before deployment.
