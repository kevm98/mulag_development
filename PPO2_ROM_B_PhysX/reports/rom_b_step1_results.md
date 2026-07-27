# Step 1 result: `model_53` on the frame-consistent clearance signal

Run: `reports/evaluation_2026-07-25_00-56-12` (250 rows, 128 envs, 120 steps,
seed 17, `--clearance_frame_consistent`, MC rounds 5, h-sweep rounds 5).
Baseline: `reports/evaluation_2026-07-24_19-28-48` (contaminated signal).

## 1. The frame fix generalized

The single-condition probe result held across all five Monte Carlo line seeds.

| MC shape | old term (mean / max) | new term (mean / max) |
|---|---:|---:|
| line | 0.03438 / 0.05469 | **0.00156 / 0.00781** |
| square | 0.02187 / 0.03125 | 0.01562 / 0.03906 |
| arc | 0.01719 / 0.03125 | 0.01250 / 0.02344 |
| circle | 0.00000 / 0.00000 | 0.00313 / 0.00781 |
| sine | 0.00156 / 0.00781 | 0.00469 / 0.00781 |
| pooled | 0.01500 | **0.00750** |

Line dropped 22x; the other shapes moved within noise. That is the predicted
signature: the frame offset was a Z-axis error, and line is the shape whose
reference height is most exposed to it. Clearance has essentially vanished as a
termination cause — one occurrence in 45 h-sweep conditions (h=0.6 square,
0.00156). It was previously the dominant cause.

## 2. Monte Carlo now passes

`max(pooled Clopper-Pearson upper, seed-group t upper)`, per shape, worst shape
controls, 95 % one-sided against the 5 % gate:

| shape | failures/group (n=128) | rate | CP upper | group-t upper | bound |
|---|---|---:|---:|---:|---:|
| square | 2, 0, 1, 2, 5 | 0.01563 | 0.02636 | 0.02956 | **0.02956** |
| arc | 0, 2, 2, 3, 1 | 0.01250 | 0.02244 | 0.02099 | 0.02244 |
| circle | 0, 1, 0, 1, 0 | 0.00313 | 0.00980 | 0.00720 | 0.00980 |
| sine | 0, 1, 1, 1, 0 | 0.00469 | — | — | <0.02 |
| line | 0, 0, 0, 1, 0 | 0.00156 | 0.00739 | 0.00489 | 0.00739 |

Worst bound 0.0296 < 0.05: **PASS**. Monte Carlo is the independent-sampling
suite, i.e. the actual training distribution. No condition shows significant
overdispersion (worst phi = 1.78, p = 0.13).

## 3. The h sweep fails — but with the sign reversed

6 of 45 (h, shape) conditions exceed the bound: h = 0.8, 1.0, 1.1, 1.2 on square,
h = 1.0, 1.2 on arc. Worst: h = 1.1 square, bound 0.0825.

**Every failure is at h >= 0.8. Not one is at h = 0.4-0.7.**

Mean termination rate by h:

| h | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 |
|---|---|---|---|---|---|---|---|---|---|
| old | .0224 | .0135 | .0172 | .0255 | .0219 | .0182 | .0172 | .0167 | .0266 |
| new | .0134 | .0166 | .0156 | .0156 | .0181 | .0159 | **.0259** | **.0338** | **.0309** |

The old column is flat — no h signal at all. The corrected column has a clean
monotone rise from h = 0.6 to h = 1.1. The contamination was not merely adding
failures; it was **masking the real h-dependence**, which runs opposite to the
assumption the whole ROM-B programme was built on. Degraded hydraulics are the
*easy* end of the envelope.

## 4. Why: a stiffer plant drives a more aggressive arm

Cause attribution, square:

| h | truck_raw_reaction | head_alignment |
|---|---:|---:|
| 0.4-0.9 | 0.008-0.019 | 0.009-0.016 |
| 1.0-1.2 | **0.031-0.036** | 0.008-0.017 |

`truck_raw_reaction` roughly quadruples from the degraded to the stiff end;
`head_alignment` is flat in h and tracks shape (square and arc corners) instead.
Acceleration corroborates it — p95 acceleration rises with h on every shape and
breaches the 40 m/s^2 limit only at the top: 41.09 (h=0.8 circle), 40.07 (h=1.1
arc), 43.00 (h=1.2 circle).

The mechanism is coherent. `_truck_drift_xy` is the *instantaneous* per-step
displacement of the truck base from its anchor before the constraint teleports it
back (`policy/env.py:456-478`) — not an accumulator, so 0.05 m means 5 cm of base
motion in one control step, a genuine reaction impulse. A stiffer, less leaky
cylinder converts valve command into motion faster, so the arm accelerates harder
and delivers a larger reaction into the chassis. Low h damps the arm.

## 5. Defect found and fixed in the evaluator

The run reports top-level `FAIL: ['clearance']` at 0.2343 m while recording
**zero** clearance terminations. That pair is impossible, and it is my bug:
`--clearance_frame_consistent` routed the environment's observation, reward and
termination through `_clearance_signal()`, but `runtime/evaluation.py` kept its
own inline reconstruction (`_compute_cutter_world_z()` minus the reference).
While the flag was off the two agreed; with it on they diverged, so the audit
terminated on the corrected signal and *reported* the contaminated one. No longer
capped by termination, the contaminated metric floated past its own 0.200 m gate.

Fixed: all three consumers (pre-termination trace, per-step sample buffer,
termination-cause attribution) now call `raw._clearance_signal()`. With the flag
off this is bit-exact with the previous code. Regression test added
(`test_evaluator_records_the_same_clearance_signal_that_terminates`); suite is
44/44.

**What this invalidates:** the `max_clearance_error_m` column and the top-level
pass/fail of the 00-56-12 run. **What it does not touch:** all termination rates
and therefore every acceptance statistic above, because those come from the
environment's own `dones`, which used the corrected signal throughout. The
truck/head cause split is also sound — the two masks are clearance-independent,
and they sum to the observed rate (h=1.1 square: 0.036 + 0.017 = 0.053 vs rate
0.0516).

## 5b. Confirmed by re-run `evaluation_2026-07-25_07-36-48`

Diffing all 250 rows against `00-56-12`, **exactly one column changed:
`max_clearance_error_m`**. Every dynamics number is bit-identical, which is the
controlled proof that the evaluator fix touched the metric and nothing else.

- Max clearance over all 250 rows: 0.31127 -> **0.19514 m**, now inside the
  0.200 m gate everywhere, consistent with the zero clearance terminations.
- Monte Carlo top-level verdict: FAIL -> **PASS**, worst-case clearance 0.17361 m.
- Acceptance statistics unchanged, as predicted: MC bound 0.0296 (PASS), h sweep
  6 failed conditions, worst 0.0825 at h = 1.1 square.
- Cause attribution is now clean: `truck_raw_reaction` 0.01434,
  `head_alignment` 0.00503, clearance **0.0** (the residual 3e-05 was itself
  contamination).

## 6. Consequences for the plan

The plant-response observer was designed for the case "degradation is invisible
to a memoryless policy". Degradation is now measurably *not* the binding
constraint, so the observer pilot is no longer the right next experiment. The
binding constraint is reaction impulse and cornering acceleration at nominal-to-
stiff plant on square and arc.

Note also that `truck_raw_reaction` **terminates but is never shaped**. Clearance,
head alignment and acceleration all have bounded margin penalties approaching
their gates; truck reaction has none. The policy is punished for crossing a line
it gets no gradient toward avoiding. That is the most likely single fix, and it
requires no gate change.

## 7. Implemented: truck-reaction shaping (opt-in)

`compute_rom_b_safety_reward` now takes an optional `truck_reaction_m` and emits
a bounded margin penalty over a 0.035 -> 0.050 m shoulder, structurally identical
to the head, clearance and acceleration terms. `rom_b_truck_reaction_hard_m`
mirrors the existing 0.05 m `truck_raw_reaction_termination_m` and does not move
it — only the approach gradient is new, so no gate is relaxed.

The weight defaults to 0.0, so the reward is bit-exact until a run opts in with
`--truck_reaction_shaping <negative weight>`, which is recorded in the run
contract alongside the soft and hard bounds. Two new TensorBoard scalars,
`rom_b_safety/truck_reaction_margin_penalty` and
`rom_b_safety/truck_reaction_violation_rate`, make the effect auditable during
training. Tests: 47/47.
