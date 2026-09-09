# Hydraulic Deployment Adaptation Pipeline

This document is the stage-by-stage plan for learning online hydraulic
adaptation in Isaac Lab. It supersedes the older exploratory plan.

Current status: **Stage 31 is active.** Historical Stage-22/30 artifacts are
audit references only. The production chain must first collect fresh data from
the final ROM plant with DLS/PRBS excitation, then derive both SD and AAM data
from that one frozen collection. The ID-FD plant is used only for deployment
and paired evaluation; no ID-FD identification dataset is collected.

## Fixed scientific definitions

- No new Simscape data is collected for this experiment.
- Fresh Isaac data is collected from the final hydraulic command-ROM plant plus
  PhysX (`RRLAB-Mulag-Verge-ROM-v0`).
- The same frozen ROM collection is the source for the derived SD and AAM
  datasets.
- The ID-FD plant is a deployment/evaluation target only; it is never a data
  collection source in this workflow.
- The System Dynamics model (SD) is a learned approximation of that plant.
- `e` is the privileged physical-factor vector.
- `z = mu(e)` is the privileged dynamics latent.
- `z_hat = AAM(history)` is the deployment-time latent estimate.
- New SD, encoder and AAM models are trained from scratch on Isaac Lab data.
- Historical ID/FD models are frozen comparison components unless a later
  stage explicitly adds a small adapter.
- Splits are by complete run and factor configuration. Windows never cross
  runs and privileged information never enters deployment inputs.
- The AAM row is `[joint_q(5), joint_qdot(5), dP(4), previous_valve(4)]`.
  Desired references are logged for ID fine-tuning, but are not AAM inputs.

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

Status: Stage 31 implementation.

The active ROM, DLS, PPO-ROM, PPO-ID-FD, logging and model paths were separated
from legacy/superseded code. The selected command ROM checkpoint is
`learned_models/rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt`.

## Stage 1 — Verify the ROM deployment interface

Status: complete for the Stage 31 contract.

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

Use four-channel PRBS with amplitude `0.01`, hold 10 control steps and
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

## Stage 15 — Collect the final ROM-plant dataset

Status: **fresh Stage 31 collection required.**

The old `master_v3_stage22_wide` collection was made for an obsolete task
identity and is not a production input. Stage 31 replaces it with a fresh
classical-controller collection from `RRLAB-Mulag-Verge-ROM-v0`. No PPO policy
is loaded or queried during collection.

## Stage 16 — Freeze and split the master dataset

Status: rerun for Stage 31.

Frozen source: `AAM_module/datasets/rom_deployment_stage31_v1_frozen/`.
Checksums protect raw runs. Splits are created only after the fresh collection
passes validation.

## Stage 17 — Fix the learning timebase

Status: complete.

Use exact 50 ms ROM transitions at control indices `0,3,6,...`; state row `t`
and command row `t` predict state row `t+3`. No interpolation is allowed.

## Stage 18 — Build the SD dataset

Status: rerun from the fresh ROM collection.

The 12-D hydraulic state is extension, velocity and dP for four cylinders. The
target is the next-tick state delta. The SD action is
`[previous_valve(4), current_valve(4)]`; joint position/velocity are context
only. History is causal and contained within one run.

## Stage 19 — Train the matched no-latent SD

Status: historical reference; rerun for Stage 31.

The Stage 31 control uses history-6 belief, joint context,
unroll 10, normalized state-rollout loss and hold-start joint rollout.
Reserved-test `R2_all = 0.7989`.

Artifact: `outputs/sd_z_stage22_v6_wide/none_nodp_holdq/`.

## Stage 20 — Train the privileged factor encoder

Status: historical reference; rerun for Stage 31.

Train `z = mu(e)` jointly with `SD_z`, using training-run factor normalization
only. The selected bottleneck is `d_z = 2`. The encoder defines the later AAM
target and is never available at deployment.

## Stage 21 — Train latent-conditioned SD

Status: historical reference; rerun for Stage 31.

Use one-step delta loss plus normalized state-rollout loss with a short rollout
ramp, dP-hidden history-6 belief, joint context, unroll 10 and hold-start
rollout. Final dz2 reserved-test `R2_all = 0.8186` at best epoch 252.

Artifact: `outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq/`.

## Stage 22 — Verify that the latent matters

Status: fresh gate required.

At H20, true `z` significantly improves extension and velocity over zero,
shuffled and protocol-matched no-latent controls, pooled and on the 16 reserved
test runs. Random and nominal controls also pass. Mean-latent test velocity is
diagnostic-only and is not part of the formal item-55 gate.

Evidence:

- `outputs/sd_z_stage22_v6_wide/PASSED_GATE.json`
- `outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq/gate_verdict.json`

---

## Stage 23 — Freeze the privileged representation

Status: performed only from the new Stage 31 SD candidate.

Freeze the selected dz2 `encoder.pt`, `sd.pt`, `belief.pt`, factor/state/input
normalization, input mask and rollout protocol. Record hashes and provenance.
Do not fine-tune these while training the first AAM.

Gate: the frozen copy must reproduce the fresh Stage-31 rollout metrics. The
old Stage-23 copy and its evidence remain audit artifacts only.

## Stage 24 — Build the AAM dataset from the ROM-plant collection

Status: historical only; Stage 31 reruns this gate on the fresh collection.

Build causal observable histories from the frozen ROM-plant dataset. The
deployment feature row is 18-D:
`[joint_q(5), joint_qdot(5), dP(4), previous_valve(4)]`. Exclude physical
factors, true `z`, desired references, ROM hidden state and future observations.

Each history window receives the stopped-gradient target `z = mu(e)` for its
run. Store run ID, domain, split and valid history bounds. The AAM dataset is
derived from the ROM collection; there is no separate ID-FD raw dataset.

## Stage 25 — Select AAM history length

Status: fresh rerun required. The initial candidate is **L5 (0.25 s)** under
the shortest-history rule. Any smoother must remain strictly causal.

Compare a small predeclared set, for example 0.25, 0.5, 1.0 and 2.0 seconds.
Select the shortest history within uncertainty of the best downstream result.
Do not select using the reserved test split.

## Stage 26 — Train and evaluate AAM

Status: fresh retraining required. The candidate uses a causal adjacent-window
temporal-consistency loss and no full-run statistics.

Train `z_hat = AAM(H_t)` from scratch on training runs. Evaluate latent RMSE,
temporal stability and held-out behavior, but select primarily by downstream
`SD_z(s,u,z_hat)` rollout quality.

Required comparisons: true `z`, `z_hat`, zero, shuffled, mean and no-latent.

## Stage 27 — Validate the complete adaptation chain

Status: fresh Stage-31 evidence required.

On validation and reserved test runs compare:

```text
SD_0(s,u)
SD_z(s,u,z_true)
SD_z(s,u,z_hat)
```

Pass only if `z_hat` retains a useful, statistically supported part of the
true-latent advantage without privileged deployment input.

Latest verdict: `AAM_module/results/stage27_chain_verify_temporal/verdict.json`.
At H20 hold-start on 36 held-out runs, z_hat beats no-latent with pooled 95%
CIs excluding zero for extension and velocity, is not significantly worse than
z_true, and retains 0.825 extension / 1.101 velocity of the privileged
advantage. Earlier noncausal and pre-retraining causal verdicts are retained
as historical audit artifacts.

## Stage 28 — Test frozen historical ID transfer

Status: **complete; meaningful transfer gap confirmed.** The frozen historical
Simscape-trained ID plus strict FD was evaluated on all 118 Stage-22 runs. It
has millimetre-scale teacher-forced extension error, but the autoregressive
rollout reaches approximately 89% historical-ID preclip saturation and diverges
past |cylinder extension| > 1 m after about 31 s on held-out combined runs.
Full details: `AAM_module/stage28_id_transfer.md`; artifacts are under
`outputs/stage28_id_transfer/`. The Stage-22 collection itself remains healthy
(about 0.807 mm measured cylinder tracking RMSE on held-out runs).

## Stage 29 — Train the AAM-conditioned ID fine-tuning layer

Train a zero-initialized bounded residual layer around the pretrained ID with
the fresh ROM recordings:

```text
u_base  = ID_frozen(state, desired_motion)
u_final = clip(u_base + adapter(state, desired_motion, u_base, z_hat))
```

The loss is L2 between `u_final` and the recorded ROM valve command at the
same exact 50 ms tick. The pretrained ID remains the safety anchor.

## Stage 30 — Integrate online AAM with ID–FD

Run AAM causally at 20 Hz from the ROM-observable history, condition the ID
fine-tuning layer and pass its action through the same frozen FD and PhysX
interface. Check warm-up, resets, latency, bounded residuals and fallback.

## Stage 31 — Freeze the three PPO comparison stacks

Status: **active hard gate.**

All final experiments use these three systems. Only the ROM plant is sampled
for SD/AAM data; ID-FD is evaluated online after AAM promotion:

| Name | Actuation path |
|---|---|
| **PPO–ROM** | PPO -> desired cylinder motion -> command ROM -> PhysX |
| **PPO–ID–FD** | PPO -> desired motion -> frozen ID -> frozen FD -> PhysX |
| **PPO–ID(AAM)–FD** | PPO -> desired motion -> frozen ID + adapter(`z_hat`) -> frozen FD -> PhysX |

## Stage 31 runbook

The active Stage-31 runbook uses the selected one-second causal CNN (`L20`)
and the matching latent adapter. Older Stage-27/30 L5 artifacts remain in the
history below for provenance only and must not be used for promotion.

The one physical collection is the final ROM plant. Run the stages in order:

```bash
deployment_adaptation/stage31_rom_pipeline.sh collect
deployment_adaptation/stage31_rom_pipeline.sh freeze
deployment_adaptation/stage31_rom_pipeline.sh sd
deployment_adaptation/stage31_rom_pipeline.sh freeze-sd
deployment_adaptation/stage31_rom_pipeline.sh aam
deployment_adaptation/stage31_rom_pipeline.sh adapter
deployment_adaptation/stage31_rom_pipeline.sh gates
```

The collector uses DLS plus bounded PRBS on `RRLAB-Mulag-Verge-ROM-v0` with
six environments. `sd`, `aam` and ID fine-tuning all consume that frozen ROM
collection. The ID and FD checkpoints enter only after collection. An
interrupted collection can be continued with `collect --resume`; do not use
`--force` unless replacing a deliberate stage output.

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
- Do not collect an ID-FD dataset. The only raw collection source is the final
  ROM plant; ID-FD is used for online deployment and paired evaluation.
- Do not load old AAM or SD weights into the primary experiment.
- Do not use PPO for identification-data collection.
- Do not use privileged factors, true latent, future state or hidden ROM state
  as AAM deployment inputs.
- Do not split windows from one run across train and test.
- Do not change several PPO-stack components in a comparison and attribute the
  result only to adaptation.
- Do not claim AAM improves control until the ID/controller actually consumes
  `z_hat` and beats matched controls.
