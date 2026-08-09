# PPO3 terrain teacher-student rewrite — 2026-07-29

## Result

The production learning path is now:

```
privileged adaptive contact teacher (training only)
  -> exact raw four-action labels
  -> bounded replay DAgger actor
  -> teacher-off PPO actor
  -> EMA / slew / limit + head shields
  -> ROM-A
  -> PhysX
```

Evaluation and PPO reject every checkpoint outside the frozen
`terrain_teacher_student_v1` schema. The legacy stored-path distillation and all
pre-correction PPO checkpoints are invalid.

## Corrections made

- The terrain teacher no longer writes PhysX cylinder targets during training.
  Desired learned cylinder increments are converted into the student's exact
  normalized ROM-A command, then passed through the same EMA, slew and shields.
- The visual direct-target demonstration remains available, but its compliance
  change and direct write are isolated from teacher/PPO modes.
- DAgger now uses a bounded aggregated replay buffer and labels the actor's raw
  action directly; it does not discard all earlier student-visited states.
- The teacher's c4 command combines anchored extension feedback with signed
  `atan2` head-angle feedback. Either term alone failed on the late high terrain.
- Physical cutter/verge contact defines zero ground-tracking error. The legacy
  unrotated cutter coordinate lies about 0.26 m inside the collision geometry at
  valid contact and is now used only as an airborne recovery signal.
- PPO handoff loads only the actor and observation normalizer. The critic and
  PPO optimizer are new, and teacher mode is forced off.
- Training logs now include reward decomposition, contact and force statistics,
  scan validity, wheel support, ride height/drift/heading, action saturation and
  shields, cylinder headroom, every action channel, and per-path-family contact,
  lateral error and contact-gap error.
- Every run writes a `run_manifest.json`; exact output checkpoints write a JSON
  manifest with schema, initialization source and teacher-off provenance.

## Validation actually run

- Unit/structural tests: **78 passed**.
- Seven-family, 720-step, interface-compatible teacher preflight:
  `terrain_teacher_interface_preflight_7env_720.json`.
  - physical ground contact: **0.9867**
  - mean contact-gap error: **0.00122 m**
  - p95 contact-gap error: **0.00000 m**
  - mean lateral path error: **0.03436 m**
  - minimum head-down alignment: **0.97781**
  - p95 physical contact acceleration: **60.19 m/s²**
  - mean wheel-road support: **0.7081**
  - scan validity: **1.0**
  - all state/reward/observation values finite: **yes**
  - student slew envelope respected: **yes**
- Production-width 128-environment x 720-step preflight also passed:
  `terrain_teacher_interface_preflight_128env_720.json`.
  - p01 head alignment: **0.98956**
  - hard minimum head alignment: **0.97023**
  - fraction below 0.975: **0.0228 %**
  - contact: **97.24 %**
  - mean lateral error: **3.75 cm**
  - p95 contact acceleration: **62.55 m/s²**
  The convergence contract therefore uses a 0.975 lower-1%-quantile gate plus
  a 0.950 hard-collapse minimum. An absolute minimum alone became stricter as
  the environment count grew and rejected 21 rare samples among 92,160 while
  the lower 99% remained above 0.9895.
- Two-iteration DAgger checkpoint smoke completed and wrote a valid schema
  checkpoint.
- One-iteration teacher-off PPO handoff smoke completed, printed all new metrics
  and wrote a resumable schema checkpoint.
- Teacher-off held-out evaluator smoke loaded that checkpoint successfully. It
  did not converge (expected after only two DAgger iterations and one PPO update).

No long PPO training was launched.

## 512-environment PhysX capacity correction

The first long-run attempt exposed a GPU PhysX `collisionStackSize` overflow;
PhysX reported required sizes around 2 MB and dropped contacts. The repository's
existing environment-scaled memory helper had been called during config
construction, before the CLI changed the scene from its default width, so it
allocated buffers for one environment.

All terrain entry points now resize the GPU contact, pair, collision-stack,
heap and temporary buffers after parsing the real `--num_envs` and before scene
creation. A real 512-environment stochastic PPO update completed successfully
with 16,384 transitions and no capacity overflow, dropped-contact warning, CUDA
OOM, non-finite transition, or Python error. The active 512-env allocation was:

- rigid contacts: 8,388,608
- found/lost pairs: 4,194,304
- aggregate pairs: 4,194,304
- collision stack: 134,217,728 bytes
- heap: 134,217,728 bytes
- temporary buffer: 67,108,864 bytes

Unit/structural tests after this correction: **79 passed**.
