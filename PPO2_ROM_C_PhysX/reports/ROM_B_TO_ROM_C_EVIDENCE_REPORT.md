# ROM-B robust PPO to ROM-C online adaptation: evidence report

Date: 2026-07-26

## Executive answer

The simple, defensible design is:

```text
ROM-B: hidden reset-fixed plant factors -> factor-blind robust PPO
                                      |
                                      +-> locked certified baseline

same five-factor plant -> causal histories -> AAM estimates
                                             |
ROM-C: locked ROM-B + small estimate adapter <-+
```

ROM-B should randomize only the five factors that have an explicit, tested
effect in its hydraulic equations:

```text
e_comp ~ Uniform(-0.125 c, 0.020 c)
d_seal_j ~ Uniform(0, c), independently for j=1..4
c = 0.0, 0.4, 0.7, 1.0 through the curriculum
```

Each environment receives one draw at reset and keeps it for that episode. The
factor values are hidden from the actor and critic. The inner DLS/cascade
controller and the deterministic parked-truck supervisor remain unchanged.
This is the current ROM-B contract.

The old ROM-C contract was not a valid continuation of ROM-B. It started from
PPO2-A, introduced seven factors, and used a different plant branch containing
flow, supply-pressure, valve attenuation, and direct output scaling. Those
changes have now been removed from the active ROM-C path. ROM-C now inherits
the exact ROM-B environment and plant, starts from ROM-B `model_53`, and adds
only five causal AAM estimates.

However, **ROM-C adaptive training must not start yet**. Two upper-bound checks
currently fail to justify it:

1. The corrected privileged oracle achieved a worst Monte Carlo termination
   bound of 0.0420 versus ROM-B's 0.0437. True factors did not produce a useful
   improvement on this reset-fixed task.
2. The current AAM fails to identify any of the five ROM-B factors better than
   a constant population-mean predictor on a 512-plant compatibility screen.

Long training cannot repair either missing premise. It would only teach a PPO
to exploit estimator bias, with no evidence of online adaptation.

## What “robust ROM-B” means

Robustness is not “the policy saw some noise.” It is a concrete experiment:

- Define the plant family before training.
- Sample the full Cartesian product, not only a one-dimensional degradation
  curve. Independent draws expose mixed cases.
- Keep each plant fixed within an episode. This matches the present evidence:
  different plants, not measured within-episode failures or repair.
- Hide the factors from the baseline policy. Otherwise it is an oracle, not
  robust factor-blind PPO.
- Increase range gradually only to stabilize optimization; certify on the full
  range.
- Evaluate deterministic actor means on held-out seeds and fresh process-
  isolated conditions.
- Report the worst predeclared arm metric and safety condition, not a pooled
  average that can hide a bad shape.

The factor equations used by ROM-B are deliberately narrow:

```text
bulk'    = bulk * exp(e_comp)
leakage' = leakage * (1 + 0.25 d_seal_j)^3
pressure_rate = (4 bulk' / volume)
                * (flow - area*velocity - leakage'*pressure)
```

Motion then follows through the existing pressure, force, velocity, extension,
and PhysX path. ROM-B does not directly scale final velocity or extension.
There is no ROM-B flow factor: the ROM-B tensor is now literally five columns
at the runtime and model boundaries. Flow remains the fitted nominal ROM
quantity.

These ranges are an audited software uncertainty set, not a claim that the
endpoints are a new law of hydraulics. Before real-system claims, their physical
coverage still needs provenance from measurements, manufacturer tolerances, or
identified parameter uncertainty. Expanding ranges without such evidence would
be invented physics.

## Why ROM-C cannot be forced to improve

The unchanged inner loop is a feedback controller. Its job is to reject plant
differences. If it already makes the observable task nearly identical across
the five reset-fixed factors, then knowing those factors has little remaining
value. That is exactly what the privileged oracle tested: it gave PPO the true
answers and still did not materially beat the factor-blind policy.

An AAM can help only if all three links are real:

```text
factor changes history -> AAM identifies the individual plant
history/factor changes the best residual action -> adapted action improves task
```

The current evidence breaks both links. Therefore a positive ROM-C conclusion
must not be created by adding unsupported time variation, weakening gates, or
changing the inner controller. If future hardware or Simscape data demonstrate
within-episode parameter drift, that can become a separately declared benchmark.
It should not be invented merely to make adaptation look useful.

## Cheapest compatibility experiment performed

The current AAM was trained from a different Simscape data distribution and has
seven outputs. I generated causal histories using 512 independently randomized
ROM-B plants for 240 control steps, ran the frozen AAM on the same 10 x 20
history interface used at runtime, selected only `e_comp` and the four seals,
and compared each prediction with its true per-environment ROM-B label.

The meaningful baseline is the population mean. A model that predicts the mean
can look much better than nominal zero while identifying no individual plant.

| factor | AAM RMSE | constant-mean RMSE | correlation | R2 | result |
|---|---:|---:|---:|---:|:---:|
| e_comp | 0.0607 | 0.0413 | -0.036 | -1.160 | fail |
| d_seal_1 | 0.2841 | 0.2825 | +0.062 | -0.012 | fail |
| d_seal_2 | 0.3369 | 0.2898 | +0.047 | -0.351 | fail |
| d_seal_3 | 0.3045 | 0.2846 | -0.076 | -0.145 | fail |
| d_seal_4 | 0.2982 | 0.2942 | -0.079 | -0.027 | fail |

All five R2 values are negative: on these histories, the estimator is worse
than using one constant mean. The audit also found zero inputs beyond five
training standard deviations, so this failure is not explained by a few gross
input outliers. This is a pre-PPO screening result, not a policy-performance
result.

Reproduction:

```bash
cd /home/rrlab/rrlab_mulag
PYTHONPATH=/home/rrlab/rrlab_mulag:/home/rrlab/rrlab_mulag/ROM-cylinders_pipeline \
  /home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_C_PhysX/scripts/audit_aam_rom_b_compatibility.py
```

Artifact: `PPO2_ROM_C_PhysX/reports/aam_rom_b_compatibility.json` records the
exact AAM and command-ROM SHA-256 hashes, seed, sample size, metrics, and failed
checks.

## Evidence already established for ROM-B

The following statements have distinct scopes and should not be merged:

- The five-factor plant is non-vacuous. Controlled open-loop recurrent rollouts
  previously showed independent-randomization mean divergence around 13 mm,
  p95 around 25 mm, with larger seal corner effects. Therefore factors really
  alter the ROM trajectory.
- Under process-isolated evaluation, `model_53` passed the predeclared
  termination statistic in all 50 Monte Carlo/h conditions. The worst bound was
  0.0437 against 0.05.
- It is **not fully certified** because 13 of 250 evaluated rows exceeded the
  40 m/s2 p95 acceleration gate; all were circle rows.
- The exact current five-factor distribution also needs a fresh certification
  run because the last completed report predates removal of the legacy sampled
  flow column. The old flow perturbation was measured as effectively inert, but
  formal evidence must match the exact implemented contract.
- The clearance defect was a measurement-frame error. A controlled 250-row
  rerun changed exactly the clearance column.
- Reusing simulator process state fabricated failures. Process isolation changed
  truck failures from 100 to 14 while head failures stayed near 50 to 47. The
  leaking simulator state has not yet been localized, so isolation is the only
  verified evaluation protocol.

## Code changes made

### ROM-B

- The supported factor list is exactly five: `e_comp`, `d_seal_1..4`.
- The ROM-B model boundary now accepts five columns. No hidden zero flow column
  is constructed.
- Randomization remains independent, uniform, reset-fixed, and private.
- The inner controller and deterministic truck supervisor were not changed.

### ROM-C

- `PPO2CAAMPolicyEnv` now subclasses the exact ROM-B environment and does not
  replace its plant runtime.
- ROM-C uses observation size 47: the unchanged 42 ROM-B observations plus five
  normalized AAM estimates.
- The seven-output legacy checkpoint is read only to select outputs 0,3,4,5,6.
  Its flow and pressure predictions are discarded before the policy interface.
- The source checkpoint is ROM-B `model_53`, not PPO2-A `model_607`.
- Checkpoint transfer widens 42 to 47 exactly. New actor columns initialize to
  zero, preserving the neutral ROM-B mapping.
- Fine-tuning freezes the base actor mapping, actor normalizer, exploration
  standard deviation, and downstream layers. Only the five new input columns in
  the two actor branches and the critic are trainable.
- Full adaptive training is blocked before Isaac Sim launches unless the chosen
  AAM and command-ROM hashes have a passing compatibility report.

This is safe scaffolding, not evidence that the current AAM works.

## Literature and how it maps to this design

Only primary research sources are used here.

- Peng et al., [Sim-to-Real Transfer of Robotic Control with Dynamics
  Randomization](https://arxiv.org/abs/1710.06537), randomize dynamics during
  training and show that a recurrent policy can exploit history better than a
  memoryless randomized policy. This supports per-episode plant randomization
  and history, but does not prove that this arm's factors are identifiable.
- Yu et al., [Preparing for the Unknown: Learning a Universal Policy with Online
  System Identification](https://roboticsproceedings.org/rss13/p48.pdf), train a
  policy conditioned on dynamics parameters and infer them from recent
  state-action history. This is the direct oracle -> identifier -> conditioned
  policy decomposition.
- Kumar et al., [RMA: Rapid Motor Adaptation for Legged
  Robots](https://www.roboticsproceedings.org/rss17/p011.html), first train a
  privileged base policy, then train an adaptation module from on-policy
  histories. This supports collecting AAM data from the same policy and plant
  distribution rather than relying on unrelated histories.
- Kumar et al., [Adapting Rapid Motor Adaptation for Bipedal
  Robots](https://arxiv.org/abs/2205.15299), add a final phase that adapts the
  policy to errors from the fixed estimator. This supports the narrow ROM-C
  adapter fine-tune after the estimator itself passes.
- Rajeswaran et al., [EPOpt: Learning Robust Neural Network Policies Using Model
  Ensembles](https://arxiv.org/abs/1610.01283), optimize across a model ensemble
  with emphasis on poor-return models. This motivates inspecting tails and
  worst plants, but tail training should be added only if uniform PPO actually
  fails a measured factor-dependent tail.
- Muratore et al., [Assessing Transferability from Simulation to Reality for
  Reinforcement Learning](https://proceedings.mlr.press/v87/muratore18a.html),
  formalize simulation-optimization bias and stopping over randomized models.
  This supports held-out model/seed evaluation rather than repeatedly selecting
  on the certification suite.

None of these papers licenses arbitrary factors or guarantees that adaptation
helps under a strong fixed feedback controller. The local oracle remains the
decisive experiment.

## Required go/no-go sequence

### Gate 1: finish the ROM-B task contract

1. Diagnose and replace the discontinuous circle reference only if a controlled
   reference experiment lowers acceleration without worsening tracking,
   clearance, head alignment, or termination.
2. Run a short full-scale preflight.
3. Only if preflight says the policy needs learning, resume ROM-B at `c=1.0`.
4. Certify the exact five-factor build with process-isolated Monte Carlo and h
   suites on untouched seeds.

### Gate 2: prove adaptation has value

Train/evaluate a privileged five-factor oracle from the locked certified ROM-B
mapping. Use the same plants, shapes, seeds, isolation, and metrics. Continue
only if the oracle produces a predeclared practically meaningful improvement
without a safety regression. The current experiment says no.

### Gate 3: train a compatible five-factor AAM

If Gate 2 passes in a physically evidenced task regime:

1. Roll out the locked ROM-B policy over the same five-factor plant distribution.
2. Save causal `(valve, extension, velocity, pressure, force)` histories with
   true five-factor labels.
3. Split by plant draw and rollout seed, not random adjacent time windows.
4. Train only five outputs with the exact ROM-B label equations.
5. Require every output to beat training-set-mean and held-out-set-mean baselines;
   also report correlation, R2, calibration plots, and shuffled-history controls.
6. Test closed-loop task sensitivity by substituting true, estimated, zero, and
   shuffled factors before any PPO fine-tuning.

### Gate 4: fine-tune ROM-C

With the estimator fixed, initialize the zero-column 42 -> 47 adapter and train
only the adapter columns plus critic at low learning rate. Then compare on
identical held-out conditions:

1. locked ROM-B;
2. ROM-C with AAM on;
3. ROM-C with AAM inputs zero;
4. ROM-C with AAM estimates shuffled across environments;
5. true-factor oracle.

ROM-C demonstrates online adaptation only if AAM-on beats zero and shuffle,
improves the predeclared arm metric, passes every safety gate, and remains below
the oracle upper bound.

## Commands to use now

Safe offline AAM audit:

```bash
cd /home/rrlab/rrlab_mulag
PYTHONPATH=/home/rrlab/rrlab_mulag:/home/rrlab/rrlab_mulag/ROM-cylinders_pipeline \
  /home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_C_PhysX/scripts/audit_aam_rom_b_compatibility.py
```

Safe ROM-C construction/transfer smoke test; this does not train:

```bash
cd /home/rrlab/rrlab_mulag
timeout 120s /home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_C_PhysX/scripts/train.py \
  --headless --device cuda:0 --num_envs 16 --transfer_only
```

There is intentionally **no recommended overnight training command yet**.
ROM-B still has a circle-acceleration defect, and ROM-C fails both the oracle
value test and AAM compatibility test. The ROM-C trainer will reject the current
failing report before launching Isaac Sim. The next useful work is the cheap
circle-reference diagnostic and a five-factor recertification, not a long PPO
run.

## Verification performed

- AAM/ROM-B compatibility audit: reproducible, current result fail.
- ROM-C transfer-only Isaac/PhysX smoke test: succeeded with 16 environments;
  source iteration 53, observation 47, exact zero new columns, PPO not started.
- ROM-B and ROM-C pure contract tests: 58 passed after the final five-column
  boundary cleanup.
- ROM-C full-training guard: the failing compatibility evidence was rejected
  before Isaac Sim launched.
- No long PPO training was launched.
- No inner-loop controller or truck-motion behavior was modified.

## Remaining uncertainty

The biggest experimental uncertainty is the cross-rollout simulator-state leak.
The isolation protocol fixes the observed outcome, but the exact leaked state is
still unidentified. The first rollout is bit-identical and divergence begins on
rollout 2, so the next mechanism experiment is to serialize and diff all
environment/simulator state after rollout 1 and before rollout 2 across the two
modes. Until that mechanism is found, certification reports must keep process
isolation and state that limitation explicitly.
