# E3 — Mid-episode plant change

**Defends:** *Online Adaptation* · **Covers:** all three stacks
**Status:** CLOSED — thesis §7.9

The causal demonstration the thesis title rests on: the plant changes at a known
time inside the scored window, and the question is whether the adaptive stack
recovers where the non-adaptive ones cannot.

## RESULT — H4 is NOT TESTED, not refuted

Every contrast over 33 (mesh, family) cells is null: online vs frozen +1.2%
(17/33), vs shuffled +16.7% (15/33), vs mean +4.9% (15/33). Every paired
bootstrap interval straddles zero.

**The nulls carry no information about adaptation**, and the sham arm is what
shows it. Three measurements:

1. The perturbation is real — alpha 0.893 -> 0.536, ratio **0.600** exactly as
   configured, at 9.0 s inside a window ending at 23.4 s.
2. It is **inconsequential**: post-change lateral error 95.3 mm perturbed vs
   98.1 mm unperturbed, **-2.9%**, in the wrong direction.
3. The latent's motion is **not detection**: z_hat shifts 0.496 across the step
   when the plant changes and 0.473 when it does not.

A 40% loss of valve effectiveness does not degrade this task, at this speed,
with this policy. The stack absorbs it, so there is nothing to recover from.

### The trap this experiment set, and nearly sprang

Comparing before and after the step **within the perturbed run alone** shows
lateral error rising 40.3 -> 93.0 mm (+131%) and z_hat shifting almost exactly
one within-episode standard deviation. Both look like a strong plant effect and
a clean detection. **Both are artefacts of position along the path**: the error
ramps monotonically 39 -> 128 mm in the perturbed run and 39 -> 138 mm in the
sham. Without the sham arm these two numbers would have been reported as
evidence for online adaptation.

### To actually test H4

Find a perturbation that measurably degrades closed-loop performance FIRST, then
ask whether the adaptive arm recovers faster than the frozen one. Valve
effectiveness is the wrong knob. Note that transport delay is also a poor
choice: the 117 ms plant dead time is present during training, so the policies
have already compensated for it, and injecting more tests robustness to a novel
disturbance rather than adaptation. A step in the COMMANDED line is the
mechanism built for E2 (`REF_STEP_M` / `REF_STEP_TIME_S`) and is a candidate.

## Run

    bash experiments/E3/code/run_e3_plant_change.sh

## Arms

The latent is substituted only in PPO-(AAM)-ID-FD, the only stack that has one.
PPO-ID-FD is scored under the same schedule as a reference row: a shuffled
latent that drops the adaptive stack to PPO-ID-FD's level says something a
within-stack comparison cannot.

| arm | isolates |
|---|---|
| `estimated` | the deployed system |
| `frozen` | latches a PRE-change estimate; identical up to the latch, differs only in that it stops tracking. **Beating this is the adaptation result.** |
| `shuffled` | same latent distribution, wrong plant — rules out "extra input dimensions help" |
| `mean` | in-distribution, no plant information |
| `sham` | schedule fires, plant does NOT change |

## Two things that make this measure anything

- **`FACTOR_DOMAIN=combined_test`.** Under `nominal` the latent is constant and
  every arm collapses to the same thing.
- **The step must land inside the SCORED window**, which ends when the truck
  reaches 26 m — *not* at the 47.3 s episode length. At 4 km/h that is 23.4 s.
  The driver's guards refuse a configuration where the step never fires.

## Traps

- `zero` is **not** the no-information control. On a real trace both latent
  channels are <= 0 with means -1.19/-2.34, so zero sits 2-3 s.d. outside the
  distribution — an out-of-distribution arm, not an ablation. `mean` does that job.
- `sham` is what makes tracking falsifiable: the latent drifts as the trajectory
  evolves anyway, so "z_hat moved across the step" is not evidence without it.
- `FREEZE_TICK` counts **20 Hz model ticks**, not 60 Hz control steps. Getting
  this wrong by 3x latches AFTER the plant change and turns the decisive control
  into a near-copy of the full arm.
- Perturbation is **not the same mechanism across stacks**: ID-FD and AAM share
  the FD plant and take a valve-effectiveness step; PPO-ROM's actuator is the
  command ROM and needs a ROM factor step (`--rom_factor_step seal=...`) instead.
  Rows are matched on task and schedule, not on the physical parameter changed.
