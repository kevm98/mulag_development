# Frozen-PPO Robustness Envelope Under Plant Degradation

CPU-only. No IsaacLab, no Isaac Sim, no AAM, no privileged PPO, no fine-tuning.
The baseline PPO checkpoint was loaded read-only and never modified.

## 1. Executive verdict

**The CPU surrogate cannot determine the frozen PPO's robustness boundary, and
no boundary was manufactured.** The required ordering

    rho_degradation  <  rho_surrogate_invalid

does not hold. It is inverted: the surrogate is already outside its own input
support at **rho = 0**, and every stress axis makes it *more* valid, not less.

Across 56 sweep points on five stress axes and 3 584 episode rollouts, the
normalized tracking error `D_E = E(rho)/E(nominal)` **never rose above 1.0**.
The pre-declared degradation threshold of 1.5 was never approached on any axis.
Every axis is monotonically *improving*, converging to the error of a
do-nothing null controller.

The cause is not controller robustness. It is that **inside the frozen PPO's
own training support the surrogate task has no control content**:

| quantity | value |
|---|---:|
| reference travel over a 2.4 s episode (median) | **0.17 mm** |
| recorded expert tracking error | 0.05 mm |
| surrogate free-running drift, zero command, at t = 48 | 8.74 mm |
| one PPO action unit = commanded cylinder increment | 4.00 mm |

The reference is effectively stationary, the surrogate drifts ~50x further than
the reference moves, and the smallest meaningful command is ~20x larger than the
entire task. The measured error is dominated by surrogate drift and actuator
granularity, so `J(rho)` is not a function of the controller.

This is a **negative instrument result**, not a robustness result. It says
nothing about whether the frozen PPO is robust in PhysX.

### Correction to the previous experiment

`reports/RMA_PPO_CPU_SURROGATE_EXPERIMENT.md` reported that stressed error was
*lower* than nominal (ratios 0.517, 0.603, 0.455) and read this as "PPO did not
degrade". That reading does not survive inspection. The same monotone-improvement
pattern is reproduced here in full, and its cause is now identified: reducing
actuator authority removes a control signal that was making tracking *worse*.
There was never a functional nominal envelope to leave.

## 2. Definition of robustness used

The manipulated variable is plant degradation; the measured variable is the task
performance of the frozen controller:

    J(rho) = performance of the FROZEN PPO on plant P(rho)

Frozen at every sweep point: policy weights, observation normalizer, action std,
reward definition, episode set, initial conditions, horizon, seeds. Only the
actuator-boundary stress changed. This is **closed-loop controller robustness to
plant variation**, not a plant robustness test.

## 3. Frozen PPO and CPU surrogate

**Controller.** `logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt`,
a 66-D-input / 4-D-output MLP (256-128-64, ELU), fixed std 0.05.

**Observation contract — corrected.** The 66-D input is the *verge-mowing*
observation (`verge_task/terrain_observations.build_verge_observation`, 54 base +
12 model-error): height profile, cut-height error, contact force, head pose. A
tracking-only hydraulic surrogate cannot produce most of it.

The previous surrogate filled those channels with arbitrary constants. Measured
against the checkpoint's own normalizer that was catastrophic — `headroom = 0.10`
against a training mean of `0.762 +/- 0.007` is a standard score of **-99**, and
channel 49 was set to 1.0 against a mean of `-0.004 +/- 0.043`, a score of +23.
The policy saturated 70% of steps before any stress was applied.

This experiment rebuilds the observation properly:

* computed exactly from surrogate state — extensions, velocities, both headroom
  blocks, previous action, applied valve, ROM delta, commanded target lead,
  actuator prediction residual (channels 12:44 and 58:62);
* every other channel held at the frozen policy's **own normalizer mean**, i.e.
  standard score zero — the least-informative in-distribution value.

The headroom reconstruction is verified, not assumed: pushing the checkpoint's
own mean extensions through `learned_min/max` from
`mulag_plant/runtime/cylinder_coordinate_mapping.json` reproduces its channels
20:28 to four decimals on all four cylinders.

Result: saturation 70% -> 7.8%, worst observation score 99 -> 13.8.

**Reference injection.** With no terrain, the task command cannot enter through
the height profile. It is injected through `commanded_target_lead` (40:44), the
channel that already means "where I was told to go, minus where I am". This is a
declared proxy, held identical at every stress level, so it cannot bias the curve.

**Episode set.** Frozen before any sweep: 64 episodes, 48 ticks (2.4 s) at 50 ms,
seed 20260825, drawn one per run across 64 distinct runs, from the 66 876 windows
(of 446 748) whose entire horizon lies within +/-3 sigma of the frozen PPO's
training extension support. Selection uses only the checkpoint's normalizer and
the frozen dataset — never a measured result — and the same 64 (run, start) pairs
are replayed bit-identically at all 56 points.

## 4. Stress reachability checks

31/31 checks pass (`test_stress.py`). Nominal invariance (section 7) and
requested-vs-delivered positive controls (section 8):

| check | result |
|---|---|
| full chain at nominal, 60 steps | max abs difference **exactly 0.0** |
| deadband off / rate limit off | exactly 0.0 |
| authority/lag/delay vs original `DeploymentActuation` | agree to <= 2.4e-7 |
| delay 0,2,6,12,24,40 | delivered FIFO lag == requested, exactly |
| authority 1.0 / 0.5 / 0.1 / 0.01 | delivered gain 1.0000000 / 0.5000000 / 0.1000000 / 0.0100000 |
| lag tau 0.05/0.15/0.40/1.0 s | first step matches `1-exp(-dt/tau)` to 1e-6 |
| rate limit 2.0/0.5/0.1/0.02 | realized max abs du never exceeds the limit |
| deadband 0.05/0.2/0.5 | sub-threshold -> exactly 0; above-threshold passes through |

Chain order: `deadband -> authority -> lag -> FIFO delay -> rate limit`, applied
between the frozen ID model's valve command and the frozen SD transition. Delay
capacity was raised from 8 to 40 ticks (2.0 s). Deadband and rate limit are new;
authority, lag and delay reproduce the existing wrapper.

**Incidental finding.** The original `DeploymentActuation` is *not* exactly the
identity at nominal: it writes the `tau = 0` bypass as the incremental update
with `beta = 1`, which drifts ~2.4e-7 in float32 once the lag state is nonzero.
Its published check tested a single step from a zeroed state, where that update
happens to be exact. The new chain assigns through instead and is exact. The
magnitude is negligible; it is recorded rather than worked around.

## 5. Surrogate validity envelope

Derived from the frozen SD training data before any sweep — the p99.99 standard
score the *recorded* data itself attains:

| bound | value |
|---|---:|
| SD state support `max abs z` | 48.89 |
| SD action (valve) support `max abs z` | 10.41 |
| recorded extension range, cyl 0-3 | [0.2055, 0.4193], [0.4187, 0.7387], [0.3879, 0.5872], [0.5335, 0.7033] m |

An episode is `surrogate_valid` only if it is finite, under 200 MPa, inside the
recorded extension range, and within both support bounds.

**At nominal, only 16/64 episodes are valid.** The failure is specific: 48/64
exceed the SD *action* support (worst 11.59 vs 10.41) — the frozen PPO drives the
ID model to issue valve commands beyond what the SD transition was fitted on.
No episode was non-finite, over-pressure, or outside the recorded extension range.
The overshoot is 11% on a p99.99 threshold, so this is soft invalidity, not a
blow-up — but it is the wrong end of the required ordering.

## 6. Authority sweep

alpha from 1.0 down to 0.0. `D_E` is normalized error, `null` is the do-nothing
controller on the same episodes.

| alpha | median err (mm) | D_E | return | saturation | delivered gain | valid frac | region | null (mm) |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1.00 | 8.321 | 1.00 | -15.55 | 0.078 | 1.000 | 0.25 | SURROGATE_INVALID | 4.738 |
| 0.90 | 5.152 | 0.62 | -7.50 | 0.013 | 0.900 | 1.00 | ROBUST | 4.167 |
| 0.80 | 2.826 | 0.34 | -3.98 | 0.004 | 0.800 | 1.00 | ROBUST | 3.955 |
| 0.70 | 2.700 | 0.32 | -2.45 | 0.002 | 0.700 | 1.00 | ROBUST | 3.583 |
| 0.60 | 2.734 | 0.33 | -1.97 | 0.001 | 0.600 | 1.00 | ROBUST | 3.033 |
| 0.50 | 2.776 | 0.33 | -1.81 | 0.001 | 0.500 | 1.00 | ROBUST | 2.939 |
| 0.40 | 2.887 | 0.35 | -1.79 | 0.000 | 0.400 | 1.00 | ROBUST | 3.008 |
| 0.30 | 3.086 | 0.37 | -1.82 | 0.000 | 0.300 | 1.00 | ROBUST | 2.956 |
| 0.20 | 3.338 | 0.40 | -2.01 | 0.000 | 0.200 | 1.00 | ROBUST | 2.983 |
| 0.10 | 3.338 | 0.40 | -2.66 | 0.000 | 0.100 | 1.00 | ROBUST | 3.136 |
| 0.05 | 3.232 | 0.39 | -3.07 | 0.000 | 0.050 | 1.00 | ROBUST | 3.181 |
| 0.01 | 3.201 | 0.38 | -3.19 | 0.000 | 0.010 | 1.00 | ROBUST | 3.278 |
| 0.00 | 3.299 | 0.40 | -3.18 | 0.000 | 0.000 | 1.00 | ROBUST | 3.299 |

Delivered gain tracks alpha exactly. **At alpha = 0 the plant receives no command
at all and tracking is 2.5x better than at nominal** — and identical to the null
controller, as it must be. The frozen PPO's own commands are the dominant error
source. No onset, no breakdown.

## 7. Delay sweep

0 to 40 ticks (0 to 2.0 s at the 50 ms control tick), well past the previous 3.

| delay (ticks) | delay (ms) | median err (mm) | D_E | return | saturation | valid frac | region |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 8.321 | 1.00 | -15.55 | 0.078 | 0.25 | SURROGATE_INVALID |
| 2 | 100 | 4.125 | 0.50 | -12.53 | 0.040 | 0.28 | SURROGATE_INVALID |
| 4 | 200 | 3.996 | 0.48 | -6.03 | 0.030 | 0.83 | SURROGATE_INVALID |
| 6 | 300 | 4.067 | 0.49 | -5.56 | 0.027 | 0.94 | SURROGATE_INVALID |
| 8 | 400 | 4.299 | 0.52 | -5.10 | 0.031 | 0.95 | SURROGATE_INVALID |
| 10 | 500 | 4.326 | 0.52 | -4.56 | 0.044 | 0.95 | SURROGATE_INVALID |
| 12 | 600 | 4.228 | 0.51 | -4.25 | 0.061 | 0.95 | SURROGATE_INVALID |
| 16 | 800 | 3.726 | 0.45 | -3.91 | 0.089 | 0.97 | SURROGATE_INVALID |
| 20 | 1000 | 3.373 | 0.41 | -3.93 | 0.080 | 1.00 | ROBUST |
| 24 | 1200 | 3.193 | 0.38 | -3.91 | 0.054 | 1.00 | ROBUST |
| 32 | 1600 | 3.193 | 0.38 | -3.50 | 0.015 | 1.00 | ROBUST |
| 40 | 2000 | 3.202 | 0.38 | -3.23 | 0.001 | 1.00 | ROBUST |

Two seconds of pure transport delay does not degrade this controller on this
surrogate — because its commands were not helping. Delay was the strongest a
priori candidate for RMA (different delays should demand different anticipation);
the evidence does not support it here.

## 8. Lag sweep

| tau (s) | median err (mm) | D_E | return | delivered gain | valid frac | region |
|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 8.321 | 1.00 | -15.55 | 1.000 | 0.25 | SURROGATE_INVALID |
| 0.05 | 6.703 | 0.81 | -13.84 | 0.980 | 0.25 | SURROGATE_INVALID |
| 0.10 | 4.962 | 0.60 | -11.70 | 0.946 | 0.30 | SURROGATE_INVALID |
| 0.15 | 3.659 | 0.44 | -8.97 | 0.914 | 0.45 | SURROGATE_INVALID |
| 0.20 | 3.736 | 0.45 | -7.06 | 0.884 | 0.67 | SURROGATE_INVALID |
| 0.30 | 3.701 | 0.44 | -5.01 | 0.829 | 0.88 | SURROGATE_INVALID |
| 0.40 | 3.730 | 0.45 | -4.19 | 0.782 | 0.97 | SURROGATE_INVALID |
| 0.60 | 3.831 | 0.46 | -3.51 | 0.707 | 0.98 | SURROGATE_INVALID |
| 1.00 | 3.838 | 0.46 | -3.26 | 0.595 | 1.00 | ROBUST |
| 2.00 | 3.766 | 0.45 | -3.10 | 0.420 | 1.00 | ROBUST |
| 5.00 | 3.341 | 0.40 | -3.33 | 0.216 | 1.00 | ROBUST |

A 5 s hydraulic time constant against a 50 ms tick is a 100x bandwidth collapse
and still improves tracking.

## 9. Rate-limit and deadband sweeps

Both are new, both verified nominal-invariant to exactly 0.0.

| rate limit | median err (mm) | D_E | valid frac | region | | deadband | median err (mm) | D_E | valid frac | region |
|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| inf | 8.321 | 1.00 | 0.25 | SURR_INVALID | | 0.00 | 8.321 | 1.00 | 0.25 | SURR_INVALID |
| 1.0 | 8.321 | 1.00 | 0.25 | SURR_INVALID | | 0.01 | 7.441 | 0.89 | 0.34 | SURR_INVALID |
| 0.5 | 7.966 | 0.96 | 0.17 | SURR_INVALID | | 0.02 | 7.655 | 0.92 | 0.39 | SURR_INVALID |
| 0.2 | 6.645 | 0.80 | 0.28 | SURR_INVALID | | 0.05 | 6.719 | 0.81 | 0.48 | SURR_INVALID |
| 0.1 | 6.293 | 0.76 | 0.25 | SURR_INVALID | | 0.10 | 5.157 | 0.62 | 1.00 | ROBUST |
| 0.05 | 3.731 | 0.45 | 0.25 | SURR_INVALID | | 0.20 | 3.944 | 0.47 | 1.00 | ROBUST |
| 0.02 | 2.679 | 0.32 | 0.92 | SURR_INVALID | | 0.30 | 3.657 | 0.44 | 1.00 | ROBUST |
| 0.01 | 2.679 | 0.32 | 1.00 | ROBUST | | 0.50 | 3.017 | 0.36 | 1.00 | ROBUST |
| 0.005 | 2.418 | 0.29 | 1.00 | ROBUST | | 0.80 | 3.123 | 0.38 | 1.00 | ROBUST |
| 0.001 | 3.088 | 0.37 | 1.00 | ROBUST | | 1.20 | 3.299 | 0.40 | 1.00 | ROBUST |

`rate_limit = 1.0` is inert: the ID model's realized valve step never exceeds it.
`deadband = 1.2` suppresses every command (delivered gain 0.000) and reproduces
the null controller's 3.299 mm exactly.

## 10. Robustness-boundary table

| stress | rho_onset (D_E >= 1.5) | rho_break | rho_valid_max | max D_E observed | verdict |
|---|---|---|---|---:|---|
| authority | none, down to alpha = 0 | none | alpha <= 0.9 | 1.00 at nominal | no boundary |
| delay | none, up to 40 ticks (2.0 s) | none | >= 20 ticks | 1.00 at nominal | no boundary |
| lag | none, up to tau = 5.0 s | none | >= 1.0 s | 1.00 at nominal | no boundary |
| rate limit | none, down to 0.001 | none | <= 0.01 | 1.00 at nominal | no boundary |
| deadband | none, up to 1.2 | none | >= 0.10 | 1.00 at nominal | no boundary |

`D_E = 1.0` occurs only at the nominal point itself. The maximum over all 56
points is 1.0. Note that the nominal reference `E(1)` is itself a
surrogate-invalid point, so even the normalization has no trustworthy basis.

The asymptote is the signature of the whole result — at maximum severity every
axis lands on the null controller:

| axis | max severity | delivered gain | frozen PPO (mm) | null (mm) |
|---|---:|---:|---:|---:|
| authority | 0.0 | 0.000 | 3.299 | 3.299 |
| delay | 40 | 0.356 | 3.202 | 3.196 |
| lag | 5.0 s | 0.216 | 3.341 | 3.218 |
| rate limit | 0.001 | 0.043 | 3.088 | 3.305 |
| deadband | 1.2 | 0.000 | 3.299 | 3.299 |

Curves: `robustness_curves.svg`, one panel per axis with the D_E = 1 and D_E = 1.5
lines, surrogate-validity marking, and the null-controller reference.

## 11. Best degraded-but-functional region

**None exists.** No axis produced a `DEGRADED` classification at any severity, so
the interval `[rho_onset, rho_break)` is empty on all five. There is no adaptation
opportunity to hand to RMA from this instrument.

Two independent confirmations that this is an instrument limit, not controller
excellence:

1. **Nothing beats doing nothing.** On the same 64 episodes at nominal:

   | controller | median error (mm) | return | saturation |
   |---|---:|---:|---:|
   | null (zero action) | **4.419** | -5.12 | 0.000 |
   | P controller, kp = 0.10 | 5.353 | -5.09 | 0.000 |
   | frozen PPO | 7.170 | -15.55 | 0.078 |
   | P controller, kp = 0.25 | 7.863 | -12.60 | 0.126 |
   | P controller, kp = 0.50 | 12.292 | -21.09 | 0.281 |
   | P controller, kp = 1.00 | 11.675 | -19.75 | 0.432 |

   A proportional controller on the *exact* tracking error, at any gain, is also
   worse than the null, and monotonically worse as gain rises. The surrogate task
   is not closed-loop controllable at this action granularity.

2. **The support/task trade-off cannot be resolved.** Relaxing the PPO
   training-support gate is the only way to get a moving reference, and it
   destroys the controller's validity at the same rate:

   | support gate `abs z <=` | reference travel (mm) | PPO worst obs score | PPO err (mm) | null err (mm) | PPO saturation |
   |---:|---:|---:|---:|---:|---:|
   | 3 | 0.169 | 13.8 | 7.170 | 4.419 | 0.078 |
   | 5 | 0.177 | 16.6 | 6.950 | 4.558 | 0.066 |
   | 10 | 0.406 | 23.5 | 7.096 | 4.474 | 0.137 |
   | 20 | 9.628 | 38.8 | 11.235 | 5.541 | 0.424 |
   | 50 | 8.341 | 48.6 | 14.295 | 6.402 | 0.424 |

   The reference only becomes non-trivial at gate 20, where the policy is already
   ~39 sigma out of distribution and saturating 42% of steps. There is no gate at
   which a real task and an in-distribution controller coexist.

## 12. Recommended future privileged-PPO stress range

**None is recommended from CPU evidence, and none should be frozen on this basis.**

Choosing an RMA training region from these sweeps would mean selecting a band in
which the frozen controller demonstrably does not degrade, on a plant that is
outside its own validity envelope at the nominal point. `Delta_adapt = E_frozen -
E_privileged` cannot meaningfully exist where `E_frozen` never rises.

The stress *mechanisms* are verified and reusable (section 4); the *severities*
must be re-measured in PhysX. No privileged PPO, AAM, or A-RMA work was performed
or inspected, so nothing here is contaminated by adaptive-controller results.

## 13. What this proves

Established:

* The frozen PPO's observation contract is the verge-mowing 66-D vector; a
  tracking-only hydraulic surrogate cannot instantiate it, and the previous
  surrogate's constants put the policy up to 99 sigma out of distribution.
* Even with a corrected, verified, in-distribution observation and a
  support-restricted episode set, the surrogate task carries no control content:
  the reference moves 0.17 mm while the surrogate drifts 8.7 mm and the minimum
  command is 4 mm.
* Five stress mechanisms are implemented, nominal-invariant to exactly 0.0, and
  verified requested-equals-delivered across their full ranges.
* On this instrument `D_E <= 1.0` everywhere, converging to a null controller.
* The previous experiment's "PPO did not degrade under stress" is an artifact of
  the same effect, not evidence of robustness.

Not established, and not claimed:

* Anything about the frozen PPO's true robustness to actuator degradation.
* Any RMA adaptation region, any value of `Delta_adapt`, any PhysX-equivalent
  result.

Per the task's own criterion — degradation must precede surrogate invalidity —
this is the third admissible outcome: **CPU cannot determine the boundary.**

## 14. What remains for IsaacLab

The boundary search must be repeated where the controller's observation contract
can actually be instantiated:

1. Run the identical five-axis sweep in the verge task with real terrain, contact
   and head pose, so all 66 channels are genuine.
2. Keep the frozen-controller protocol unchanged: same weights, normalizer,
   reward, commands, seeds; only plant stress varies.
3. Port `stress.py` directly — the mechanisms and their positive controls carry
   over unchanged; only the severity grids need re-centring.
4. Re-apply the pre-declared criteria (D_E >= 1.5 onset, >= 3.0 or >= 90%
   saturation breakdown) and re-derive the validity envelope from PhysX state.
5. Keep the null-controller reference arm. It is what exposed this failure and it
   costs one extra rollout per point.
6. Only if a genuine `[rho_onset, rho_break)` interval appears should privileged
   PPO be trained inside it.

## 15. Files and results created

Code, `experiments/RMA_PPO/robustness/`:

| file | purpose |
|---|---|
| `stress.py` | five stress mechanisms; exact identity at nominal |
| `observation.py` | verified 66-D verge observation adapter |
| `robustness_env.py` | CPU plant with fixed episodes and validity diagnostics |
| `protocol.py` | pre-declared episode set, thresholds, validity envelope, classification |
| `frozen_ppo.py` | read-only loader for the frozen actor |
| `test_stress.py` | 31 nominal-invariance and positive-control checks |
| `run_sweeps.py` | the five one-factor sweeps |
| `make_curves.py` | robustness curves |
| `support_scan.py`, `task_content.py`, `controllability.py`, `gate_nominal.py` | the four diagnostics behind sections 1 and 11 |

Results, `experiments/RMA_PPO/robustness/results/`:

| file | contents |
|---|---|
| `robustness_sweeps.json` | metadata, frozen episode set, validity envelope, 56 points with per-episode detail |
| `robustness_sweeps.csv` | 3 584 raw episode rows — requested and delivered stress, run id, start row, error, return, action stats, validity diagnostics, region |
| `robustness_curves.svg` | D_E vs severity, five panels |

Nothing was retrained. No PPO, ID/FD, ROM, SD or AAM checkpoint was written.

### Reproducibility note: data loss during this session

Partway through this work, untracked files vanished from `experiments/RMA_PPO/`
on `/Volumes/DATA` (29 GiB FAT32, 87% full) with no delete issued:
`cpu_surrogate_env.py`, `run_cpu_surrogate_experiment.py`, and the
`cpu_surrogate/` results directory (`cpu_surrogate_results.json`,
`sd_z2_rollout.json`, and the two Phase-1 3-context checkpoints). The most likely
trigger is `__pycache__` writes into that directory corrupting its FAT entry.

All git-tracked files were verified intact (`git ls-files -d` empty), as were every
frozen asset this experiment depends on. `cpu_surrogate_env.py` has been restored
verbatim. `run_cpu_surrogate_experiment.py` and the four result artifacts are
**lost** — they were never committed. Their headline numbers survive in
`reports/RMA_PPO_CPU_SURROGATE_EXPERIMENT.md`.

This work was therefore produced on internal storage and copied here once, with
bytecode writing disabled. **Recommendation: move the repository off FAT32 and
commit `experiments/` — an uncommitted, unbacked-up thesis experiment on this
volume is at active risk.**
