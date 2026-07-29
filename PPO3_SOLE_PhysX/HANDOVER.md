# PPO3 current handover

State: the seven-family teacher-student PPO converged on 2026-07-29 and is now
the initialization checkpoint for the expanded `path_height_v2` curriculum.
The expanded extensive run has not been launched by Codex.

## Task contract

- Truck: scripted 4 km/h motion on a straight, flat, dark asphalt road.
- Truck wheels remain road-supported; the truck never drives on the verge.
- Cutter: follows a red line on a separate green physical collision terrain.
- Policy: one 54-D actor with four cylinder actions, controlling ROM-A alone.
- Families: the original seven plus double-S, held lane change, Gaussian
  slalom, and Fourier meander (eleven total).
- Heights: every family is crossed with 0.50, 0.70, 0.85, and 1.00 relief
  scales; terrain seeds change between resumed training blocks.
- Horizon: 16 seconds / 960 steps, 17.8 m of travel, 16 m active path, and a
  26 m verge strip.
- Teacher: privileged contact/DLS controller during DAgger only.
- Deployment/evaluation: teacher off; direct target demonstration off.
- Checkpoint schema: `terrain_teacher_student_v1`.

## Best initialization artifact

```text
logs/rsl_rl/ppo3_sole_terrain/teacher_student_v1_long/model_best.pt
```

The earlier 128-environment, 720-step held-out result passed every gate with score 0.0:
97.85% contact, 2.0 mm mean contact-gap error, 4.07 cm mean lateral error,
0.9890 p01 head alignment, 0.9522 hard minimum alignment, 61.99 m/s² p95 tool
acceleration, and 0.307% safety-violation rate.

## Important implementation facts

- Physical contact defines zero cut-gap error. The legacy unrotated cutter
  coordinate is only an airborne recovery signal.
- The teacher converts desired learned-cylinder motion into the exact student
  ROM-A action interface and shares its filters/shields. It never directly
  writes PhysX targets during distillation.
- Initial DAgger handoff loads actor plus observation normalizer only. V2
  continuation from the best PPO loads actor, critic, normalizer, and optimizer.
- GPU PhysX buffers are resized after the actual `num_envs` is parsed. A real
  512-environment stochastic PPO update verified no collision-stack overflow,
  dropped contacts, CUDA OOM, or non-finite state.
- Held-out convergence uses p01 head alignment plus a separate hard-minimum
  collapse gate, avoiding vectorization-dependent extreme-value thresholds.
- V2 convergence additionally gates the worst individual path family's cut
  error, lateral error, and contact fraction; aggregate success is insufficient.
- `--min_blocks` prevents an already-passing starting checkpoint from causing
  the extensive supervisor to exit before any training.

## V2 preflight status

The current best checkpoint is compatible but is not declared converged on v2.
On 44 held-out environments for 960 steps it passed contact, cut-height,
lateral, scan, road-support, and every worst-family gate, then failed head
orientation, acceleration, and safety termination in the new late high-relief
section. A real 512-environment stochastic update completed without OOM or
non-finite state. See `reports/PATH_HEIGHT_V2_PREFLIGHT_REPORT.md` for exact
numbers.

## Visualization

`scripts/evaluate.py` accepts `--path_family <family>` and `--plot_tracking`.
Interactive mode adds a larger non-colliding dark-gray grid floor and dark asphalt
road. Training and headless evaluation do not spawn this visual geometry.

See `README.md` for exact one-family visualization, training, and validation
commands. See `reports/README.md` for retained evidence.
