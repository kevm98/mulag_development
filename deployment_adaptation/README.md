# Hydraulic Deployment Adaptation Pipeline

This document is the stage-by-stage plan for learning online hydraulic
adaptation in Isaac Lab. It supersedes the older exploratory plan.

Current status: **Stages 1–22 are complete and Stage 22 passed.** The next task
is Stage 23: freeze the privileged representation and begin AAM training.

## Fixed scientific definitions

- No new Simscape data is collected for this experiment.
- The deployment plant is the hydraulic command ROM plus PhysX.
- The System Dynamics model (SD) is a learned approximation of that plant.
- `e` is the privileged physical-factor vector.
- `z = mu(e)` is the privileged dynamics latent.
- `z_hat = AAM(history)` is the deployment-time latent estimate.
- New SD, encoder and AAM models are trained from scratch on Isaac Lab data.
- Historical ID/FD models are frozen comparison components unless a later
  stage explicitly adds a small adapter.
- Splits are by complete run and factor configuration. Windows never cross
  runs and privileged information never enters deployment inputs.

## Target online architecture

```text
recent observable state-action history
                |
                v
               AAM
                |
              z_hat
                |
desired cylinder motion -> frozen ID + latent adapter -> valve -> frozen FD -> PhysX
```

The first adaptation target is the ID/actuation layer. Directly conditioning
the PPO policy on `z_hat` is a separate ablation, not the default design.

---

## Stage 0 — Repository audit

Status: complete.

The active ROM, DLS, PPO-ROM, PPO-ID-FD, logging and model paths were separated
from legacy/superseded code. The selected command ROM checkpoint is
`rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt`.

## Stage 1 — Verify the ROM deployment interface

Status: complete.

The ROM receives desired cylinder extension change and velocity, predicts the
valve internally, advances hydraulics at 20 Hz and applies three held increments
to the 60 Hz PhysX cylinder target. It is not externally valve-commanded.

## Stage 2 — Separate observable and privileged signals

Status: complete.

Observable deployment inputs are cylinder extension/velocity, applied command,
joint context and causal history. Physical factors are privileged. The final
deployment mirror withholds `dP[4]`; pressure remains a recorded diagnostic and
prediction target.

## Stage 3 — Select randomization candidates

Status: complete.

Candidates are hydraulic flow, compliance, seal/leakage and pressure, plus
cutter mass scale and COM-z. Emergent delay is measured later, not randomized
as a known factor. Friction and other weak/unsafe parameters are excluded.

## Stage 4 — Measure factor sensitivity

Status: passed.

One-factor ROM and Isaac tests confirmed that all six selected factors create
measurable changes in extension, velocity, pressure or end-effector motion.

## Stage 5 — Define the physical factor vector

Status: complete.

```text
e = [flow, compliance, seal, pressure, mass_scale, com_z]
```

Final ranges are `[-.25,.25]`, `[-.25,.10]`, `[0,1]`, `[-.15,.15]`,
`[.8,1.5]`, and `[-.10,.05]`. Combined draws reject excessive compounded
response gain.

## Stage 6 — Define deployment domains

Status: complete.

Domains are nominal, hydraulic-only, mechanical-only, combined-train and
unseen combined-test. One factor vector remains fixed for the full run.

## Stage 7 — Draw factor configurations

Status: complete.

Use deterministic scrambled Sobol draws with explicit boundary anchors and
fixed per-domain seeds. Train, validation and test factor keys are disjoint.
The final design is 98.3% unique, full rank and covers every factor quartile.

## Stage 8 — Define the master raw dataset

Status: complete.

Each run contains `metadata.json` and `trajectory.npz` with the full state,
command chain, quality flags, timing and applied factor vector. The schema is
implemented in `AAM_module/aam_data/master_schema.py`.

## Stage 9 — Verify the DLS collection path

Status: complete.

```text
task reference -> cylinder DLS -> normalized action -> command ROM -> PhysX
```

DLS closes the loop on measured cylinder state. No direct joint-target bypass
is used. PRBS uses the same action port during identification segments.

## Stage 10 — Design arm motion

Status: passed.

Three postures plus line, sine/coupled and reach-and-return motion give full
rank over all four action channels and varied geometry/load conditions.

## Stage 11 — Run the DLS-only pilot

Status: passed.

The pilot passed rank, conditioning, response-authority and valve-saturation
checks. DLS covers task motion; dedicated PRBS windows provide cleaner system
identification.

## Stage 12 — Add safe independent excitation

Status: complete.

Use four-channel PRBS with amplitude `0.02`, hold 10 control steps and
posture-specific seeds `20260812..20260814`. Probe actions are logged separately
from controller output.

## Stage 13 — Fix the run structure

Status: complete.

Each run has 11 matched segments, 11,304 control steps and 188.4 simulated
seconds: settle, postures, three PRBS windows, DLS task motion,
reach-and-return and stow.

## Stage 14 — Pass pilot quality and safety gates

Status: passed.

Short and production-duration flights verify finite physics, exact ROM grid,
rank, factor-response authority, state coverage and saturation limits. Unsafe
vector slots are discarded and retried alone in environment 0.

## Stage 15 — Collect the final deployment dataset

Status: passed.

The final dataset contains 118 runs: 2 nominal, 40 hydraulic, 12 mechanical,
48 combined-train and 16 combined-test. Four vectorized Isaac environments
were used with per-environment factors and logging. All 118 runs passed the
final collection gate.

Source: `AAM_module/datasets/master_v3_stage22_wide/`.

## Stage 16 — Freeze and split the master dataset

Status: complete.

Frozen source: `AAM_module/datasets/master_v3_stage22_wide_frozen/`.
Checksums protect raw runs. Run-level splits are train 82, validation 20 and
reserved combined-test 16, with zero factor-key leakage.

## Stage 17 — Fix the learning timebase

Status: complete.

Use exact 50 ms ROM transitions at control indices `0,3,6,...`; state row `t`
and command row `t` predict state row `t+3`. No interpolation is allowed.

## Stage 18 — Build the SD dataset

Status: complete.

The 12-D hydraulic state is extension, velocity and dP for four cylinders. The
target is the next-tick state delta. Joint position/velocity are context only.
History is causal and contained within one run.

## Stage 19 — Train the matched no-latent SD

Status: complete.

The final control uses dP-hidden input, history-6 belief, joint context,
unroll 10, normalized state-rollout loss and hold-start joint rollout.
Reserved-test `R2_all = 0.7989`.

Artifact: `outputs/sd_z_stage22_v6_wide/none_nodp_holdq/`.

## Stage 20 — Train the privileged factor encoder

Status: complete.

Train `z = mu(e)` jointly with `SD_z`, using training-run factor normalization
only. The selected bottleneck is `d_z = 2`. The encoder defines the later AAM
target and is never available at deployment.

## Stage 21 — Train latent-conditioned SD

Status: complete.

Use one-step delta loss plus normalized state-rollout loss, a 60-epoch rollout
ramp, dP-hidden history-6 belief, joint context, unroll 10 and hold-start
rollout. Final dz2 reserved-test `R2_all = 0.8186` at best epoch 252.

Artifact: `outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq/`.

## Stage 22 — Verify that the latent matters

Status: **strict pass**.

At H20, true `z` significantly improves extension and velocity over zero,
shuffled and protocol-matched no-latent controls, pooled and on the 16 reserved
test runs. Random and nominal controls also pass. Mean-latent test velocity is
diagnostic-only and is not part of the formal item-55 gate.

Evidence:

- `outputs/sd_z_stage22_v6_wide/PASSED_GATE.json`
- `outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq/gate_verdict.json`

---

## Stage 23 — Freeze the privileged representation

Status: next.

Freeze the selected dz2 `encoder.pt`, `sd.pt`, `belief.pt`, factor/state/input
normalization, input mask and rollout protocol. Record hashes and provenance.
Do not fine-tune these while training the first AAM.

Gate: the frozen copy must reproduce the Stage-22 rollout metrics.

## Stage 24 — Build the AAM dataset

Build causal observable histories from the frozen 118-run dataset. Initial
per-step features are applied action, cylinder extension/velocity and joint
position/velocity. Exclude physical factors, true `z`, ROM hidden state,
future observations, dP and desired task reference.

Each history window receives the stopped-gradient target `z = mu(e)` for its
run. Store run ID, domain, split and valid history bounds.

## Stage 25 — Select AAM history length

Compare a small predeclared set, for example 0.25, 0.5, 1.0 and 2.0 seconds.
Select the shortest history within uncertainty of the best downstream result.
Do not select using the reserved test split.

## Stage 26 — Train and evaluate AAM

Train `z_hat = AAM(H_t)` from scratch on training runs. Evaluate latent RMSE,
temporal stability and held-out behavior, but select primarily by downstream
`SD_z(s,u,z_hat)` rollout quality.

Required comparisons: true `z`, `z_hat`, zero, shuffled, mean and no-latent.

## Stage 27 — Validate the complete adaptation chain

On validation and reserved test runs compare:

```text
SD_0(s,u)
SD_z(s,u,z_true)
SD_z(s,u,z_hat)
```

Pass only if `z_hat` retains a useful, statistically supported part of the
true-latent advantage without privileged deployment input.

## Stage 28 — Test frozen historical ID transfer

Evaluate the existing Simscape-trained ID without adaptation on nominal,
hydraulic, mechanical, combined and held-out deployment domains. Report
cylinder tracking, action/valve behavior, saturation, overshoot, settling and
tool-path error. This measures the real transfer gap before adding an adapter.

## Stage 29 — Train the latent-conditioned ID adapter

If Stage 28 shows a meaningful gap, keep ID and FD frozen and train only a
small residual adapter:

```text
u_base  = ID_frozen(state, desired_motion)
u_final = clip(u_base + adapter(state, desired_motion, u_base, z))
```

Train first with privileged `z_true`, then replace it with `z_hat`. Compare
frozen ID, true-latent adapter and AAM-latent adapter.

## Stage 30 — Integrate online AAM with ID–FD

Run AAM causally at deployment, update `z_hat` from recent observable history,
condition the ID adapter and pass its action through the same frozen FD and
PhysX interface. Check update rate, warm-up behavior, state resets, latency,
bounded residual action and fail-safe fallback to frozen ID.

## Stage 31 — Freeze the three PPO comparison stacks

All final experiments use these three systems:

| Name | Actuation path |
|---|---|
| **PPO–ROM** | PPO -> desired cylinder motion -> command ROM -> PhysX |
| **PPO–ID–FD** | PPO -> desired motion -> frozen ID -> frozen FD -> PhysX |
| **PPO–ID(AAM)–FD** | PPO -> desired motion -> frozen ID + adapter(`z_hat`) -> frozen FD -> PhysX |

Use the same task, observations, action contract, terrain families, evaluation
seeds, factor configurations, episode lengths and gates. Training budgets must
match. If policies require separate training, report that explicitly and keep
architectures/budgets equal. Only the actuation/adaptation path may differ.

---

# Final comparative experiments and ablations

Run the following studies only after Stages 23–31 pass. Every study evaluates
all three PPO stacks on the same paired episodes. Report each domain separately
and include per-seed uncertainty, not only pooled averages.

## Stage 32 — System-response experiment

Apply matched steps, ramps, PRBS and task commands. Measure per cylinder:

- rise time, settling time and overshoot;
- steady-state error and gain;
- extension, velocity and dP response;
- valve/action saturation and cross-axis coupling.

This establishes how ROM, frozen ID–FD and AAM-adapted ID–FD differ dynamically.

## Stage 33 — Delay experiment

Estimate `command -> dP`, `command -> velocity`, `dP -> velocity` and, where
meaningful, `cylinder motion -> joint/tool motion`. Use cross-correlation,
controlled response and frequency/phase analysis. Report delays per cylinder,
domain and PPO stack. Do not label a kinematic mapping as temporal delay.

## Stage 34 — Nonlinearity experiment

Test response versus command magnitude/sign, dP, extension, velocity, payload,
COM and arm posture. Quantify extension/retraction asymmetry, dead-zone,
saturation, pressure-dependent gain, load dependence, hysteresis and coupling.
Only claim effects supported by repeated paired measurements.

## Stage 35 — End-effector tracking experiment

Use the same feasible path families and held-out terrain meshes for all three
PPO stacks. Report:

- lateral and vertical/cut-height RMSE, MAE and p95;
- along-track error, maximum error and settling after path changes;
- ground-contact fraction, head alignment and hard impacts;
- per-path-family and worst-mesh results.

The current eight-gate contract remains the minimum safety/task criterion.

## Stage 36 — Robustness and generalization experiment

Evaluate nominal, hydraulic-only, mechanical-only, combined and held-out
factors, including in-range boundary anchors and declared out-of-range stress
tests. Also test observation noise, action delay, sensor dropout, terrain shift
and payload/COM changes. Separate interpolation from extrapolation and stop any
unsafe episode.

Report success rate, tracking degradation, saturation, safety terminations and
adaptation recovery time relative to nominal.

## Stage 37 — Adaptation ablations

For **PPO–ID(AAM)–FD**, compare:

| Ablation | Purpose |
|---|---|
| `z_hat` | deployable online adaptation |
| `z_true` | privileged oracle ceiling |
| zero latent | removes adaptation content |
| mean latent | fixed in-distribution context |
| shuffled/random latent | tests configuration specificity |
| frozen ID without adapter | isolates adapter benefit |
| adapter with no AAM history | isolates history inference |
| shorter/longer history | history-length sensitivity |
| slower latent update | adaptation-rate sensitivity |
| optional PPO conditioned on `z_hat` | separates policy adaptation from ID adaptation |

Do not claim online adaptation unless deployable `z_hat` beats frozen
PPO–ID–FD and approaches the true-latent oracle on held-out configurations.

## Stage 38 — Statistics and final reporting

Predeclare primary metrics and seeds. Use paired episodes, bootstrap confidence
intervals and per-domain/per-path tables. Report failures and worst-case results.
Keep validation decisions separate from the untouched final test evaluation.

The final thesis comparison is:

```text
PPO–ROM
vs PPO–ID–FD
vs PPO–ID(AAM)–FD
```

across system response, delay, nonlinearity, end-effector tracking, robustness
and the adaptation ablations above.

## Rules that remain mandatory

- Do not collect new Simscape data for the new AAM/SD stack.
- Do not load old AAM or SD weights into the primary experiment.
- Do not use PPO for identification-data collection.
- Do not use privileged factors, true latent, future state or hidden ROM state
  as AAM deployment inputs.
- Do not split windows from one run across train and test.
- Do not change several PPO-stack components in a comparison and attribute the
  result only to adaptation.
- Do not claim AAM improves control until the ID/controller actually consumes
  `z_hat` and beats matched controls.
