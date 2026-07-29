# PPO3 current handover

State: final terrain teacher-student PPO completed and held-out converged on
2026-07-29. This file contains only current decisions and reproducible facts;
historical failed experiments were removed from the package.

## Task contract

- Truck: scripted 4 km/h motion on a straight, flat, dark asphalt road.
- Truck wheels remain road-supported; the truck never drives on the verge.
- Cutter: follows a red line on a separate green physical collision terrain.
- Policy: one 54-D actor with four cylinder actions, controlling ROM-A alone.
- Families: straight, sine, S-curve, random spline, offset-return, boom-sweep,
  and chirp.
- Teacher: privileged contact/DLS controller during DAgger only.
- Deployment/evaluation: teacher off; direct target demonstration off.
- Checkpoint schema: `terrain_teacher_student_v1`.

## Converged artifact

```text
logs/rsl_rl/ppo3_sole_terrain/teacher_student_v1_long/model_converged.pt
```

The 128-environment, 720-step held-out result passed every gate with score 0.0:
97.85% contact, 2.0 mm mean contact-gap error, 4.07 cm mean lateral error,
0.9890 p01 head alignment, 0.9522 hard minimum alignment, 61.99 m/s² p95 tool
acceleration, and 0.307% safety-violation rate.

## Important implementation facts

- Physical contact defines zero cut-gap error. The legacy unrotated cutter
  coordinate is only an airborne recovery signal.
- The teacher converts desired learned-cylinder motion into the exact student
  ROM-A action interface and shares its filters/shields. It never directly
  writes PhysX targets during distillation.
- PPO handoff loads actor plus observation normalizer only; critic and optimizer
  start fresh.
- GPU PhysX buffers are resized after the actual `num_envs` is parsed. A real
  512-environment stochastic PPO update verified no collision-stack overflow,
  dropped contacts, CUDA OOM, or non-finite state.
- Held-out convergence uses p01 head alignment plus a separate hard-minimum
  collapse gate, avoiding vectorization-dependent extreme-value thresholds.

## Visualization

`scripts/evaluate.py` accepts `--path_family <family>` and `--plot_tracking`.
Interactive mode adds a non-colliding medium-gray grid floor and dark asphalt
road. Training and headless evaluation do not spawn this visual geometry.

See `README.md` for exact one-family visualization, training, and validation
commands. See `reports/README.md` for retained evidence.
