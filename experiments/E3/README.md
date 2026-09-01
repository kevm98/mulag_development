# E3 — Mid-episode plant change

**Defends:** *Online Adaptation* · **Covers:** PPO-AAM-FD, within-stack
**Status:** complete, including the 2026-09-01 correction · thesis §7.5. The
first dose sweep confounded the mid-episode change with static mass; the
uniform-step rerun isolates it and resolves a component on both metrics.

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

### Why the perturbation was also the wrong FACTOR

Found after this run closed. Linear decoding from z_hat gives **seal 0.956**,
compliance 0.448, **flow 0.114**, pressure 0.081, com_z 0.056, mass -0.008. The
latent is a seal latent. Stepping valve effectiveness asks an estimator that
encodes seal to detect a change in flow, which is not the strongest test of the
adaptation path the available factors permit.

The obvious fix — step seal instead — **does not work on the deployed stack**.
`RomFactorPerturbation` drives `_command_rom_system_factors`, read only in the
`command_rom` branch; AIF inherits `cylinder_dynamics_model = "fd"`, and
`CylinderFD.step(valve_cmd, extension, velocity, dP, force)` takes no factor
argument. On the plant AIF actually runs, seal is inert (see E5 README).

So the factor the estimator identifies best cannot be perturbed where the
controller runs, and the surface that can be perturbed there carries the factor
it identifies worst. **That bind is itself a finding** and is reported in thesis
§7.4 and §7.6.5 rather than worked around silently.

### To actually test H4

Find a perturbation that measurably degrades closed-loop performance FIRST, then
ask whether the adaptive arm recovers faster than the frozen one. Valve
effectiveness is the wrong knob. Note that transport delay is also a poor
choice: the 117 ms plant dead time is present during training, so the policies
have already compensated for it, and injecting more tests robustness to a novel
disturbance rather than adaptation. A step in the COMMANDED line is the
mechanism built for E2 (`REF_STEP_M` / `REF_STEP_TIME_S`) and is a candidate.

**Replacement: TWO experiments, in `code/run_e3_plant_change_v2.sh` (not run;
the original `run_e3_plant_change.sh` that produced the results above is kept
unchanged).** They answer different questions and neither carries the other's
claim.

`EXP=A` — **native plant change.** Tool mass steps mid-episode
(`deployment_mass_step_time_s` / `_scale`, new). Mass reaches every stack
unclamped through PhysX, so this is a property that genuinely exists in the
deployed simulation with no emulator anywhere. The AAM decodes mass at
R² = −0.008, so it may well not help — that is a legitimate outcome, and the
question it answers is what happens when a *real* deployed plant property
changes.

`EXP=B` — **AAM-aligned positive control.** A seal-associated **gain proxy**:
the amplitude half of the ROM's seal law applied to realised actuator motion,
`alpha_eq(w) = 1/(1+0.625w)^2` (0.379 at w=1). It is **not** called seal
degradation — it omits the leakage term (1+0.25w)³ and the longer valve time
constant, and is named a proxy wherever it is reported. It asks whether the
adaptation path can exploit a change along the direction the AAM demonstrably
learned.

Six arms on one AIF checkpoint — online, frozen (FREEZE_TICK at the change),
oracle, mean, shuffled, sham — plus IF as an external reference. **Magnitude is
not chosen here:** it comes from the E5 authority curve, measured on IF alone,
and the run warns if `stress_levels.json` is absent. AIF is never inspected
while the magnitude is selected, so the stress cannot have been picked to
flatter it.

**Success is not "AAM wins."** The experiment succeeds if it separates: frozen
degrades + online recovers + oracle recovers (adaptation works); oracle recovers
but online does not (estimator limiting); neither recovers (path lacks
authority); nobody degrades (too weak); everyone terminates (too strong);
α=0.01 still tracks (harness fault).

`code/plot_recovery.py` draws the four-panel time history on one axis with the
intervention marked: lateral error, cut-height error, realised/commanded
actuator motion, and ẑ. Panel 3 keeps it honest — if that ratio does not step at
the intervention, the perturbation never reached the plant and the other three
panels describe something else.

`code/analyze_recovery.py` reports what episode means cannot: IAE over
[t_c, t_c+1], [t_c+1, t_c+3], [t_c+3, t_c+5], peak error within 3 s, and
recovery time (first sustained re-entry into 1.1x the pre-change band, reported
as *not recovered* when it never happens), paired env-for-env across arms.

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
