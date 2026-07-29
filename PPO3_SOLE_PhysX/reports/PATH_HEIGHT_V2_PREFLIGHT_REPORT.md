# Expanded path-and-height curriculum preflight

Date: 2026-07-29

## Change under test

`path_height_v2` expands the lateral library from seven to eleven smooth path
families, crosses every family with relief scales 0.50, 0.70, 0.85, and 1.00,
changes the terrain seed between resumed PPO blocks, and extends stage 4 from
12 to 16 seconds. The physical verge is 26 m long and the active lateral path
is 16 m. Environment spacing is 28 m, so distinct verge meshes do not overlap.

The actor contract remains 54 observations and four actions. The starting
checkpoint is therefore load-compatible:

```text
logs/rsl_rl/ppo3_sole_terrain/teacher_student_v1_long/model_best.pt
```

## Verification results

- Pure geometry and structural suite: **80 passed**.
- All eleven paths are finite, distinct, strip-bounded, and have maximum
  lateral slope below 0.18.
- Analytic centre-line terrain relief spans approximately 0.28-0.68 m across
  the curriculum, with maximum grade below 0.15.
- A 44-environment, 960-step PhysX run instantiated every path x relief pair.
  It measured 17.7777 m travel at 3.99999 km/h, 0.596 m encountered terrain
  range, 99.16% post-acquisition cutter contact, finite 54-D observations,
  full scan coverage, a stable road-constrained truck, and straight front
  wheels. The old direct-target demonstration did not meet its strict 0.30 m
  maximum lateral-error criterion on all high-relief pairs; it is not used by
  PPO continuation or evaluation.
- Teacher-free evaluation of the current best checkpoint on the same 44-env,
  960-step distribution achieved 0.00229 m mean cut-gap error, 97.77% contact,
  0.0803 m aggregate lateral error, and 0.2567 m worst-family lateral error.
  Every new worst-family gate passed. The checkpoint correctly failed the full
  v2 stop condition because the previously unseen late high-relief section
  caused 100.16 m/s2 p95 acceleration, 0.8992 p01 head alignment, a head flip,
  and 10.25% safety terminations. These are the declared learning targets for
  continued PPO, not hidden as a passing baseline.
- A 512-environment production scene instantiated 745,472 verge triangles,
  exactly 128 environments at each relief scale, finite reset observations,
  scan validity 1.0, and correct PhysX buffer scaling.
- A real 512-environment stochastic PPO update completed at 649 steps/s,
  produced finite losses/rewards, logged all eleven families and four relief
  levels separately, and wrote a resumable checkpoint. No CUDA OOM or dropped
  contact-buffer error occurred.
- An explicit `fourier_meander`, full-relief scene spawned the larger
  collision-free gray grid floor and produced the single desired-versus-actual
  XY tracking plot.

## Training decision

The expanded task is load-compatible and production-width runnable, while the
starting checkpoint is deliberately not labelled converged on it. Extensive
teacher-free PPO should resume the complete actor, critic, observation
normalizer, and optimizer state. The supervisor forces a minimum number of
blocks and then continues until aggregate and worst-family held-out gates pass.
