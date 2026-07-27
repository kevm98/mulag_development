# ROM-B circle acceleration problem: complete technical handoff

Date: 2026-07-26  
Repository: `/home/rrlab/rrlab_mulag`  
Current working trajectory: `plant_reachable_trajectory_library_v3_circle_c7_retime.json`  
Current PPO checkpoint: `model_53_ppo2_rom_b_final.pt`

> **Pre-training correction:** the plan in Sections 6-10 has been audited
> against the 42-channel observation, the exact acceleration implementation,
> the max-over-32 audit structure and a path-weighting cross-check. Training is
> now gated by the zero-residual 32-row sweep described in Section 13. Section
> 13 is authoritative wherever it tightens an earlier planning statement.
>
> **Zero-baseline result:** the seed-41 sweep is complete. Twenty of 32 rows
> exceed 38 m/s2 and six exceed the formal 40 m/s2 gate with PPO identically
> zero. PPO remediation is therefore paused. Section 14 is the current decision.

## Executive conclusion

ROM-B is **not certified** for the circle task.

The independent seed-29 audit is valid and complete: it contains all 32 expected
circle rollouts, uses a fresh process for every rollout, uses the corrected
clearance measurement, binds itself to the exact PPO and trajectory hashes, and
keeps the actor factor-blind. It nevertheless fails the predeclared 40 m/s2 p95
acceleration gate in two Monte Carlo rows:

- seed 300037: 40.8448 m/s2;
- seed 300042: 41.7117 m/s2.

Everything else passes:

- all nine correlated `h` conditions pass every point gate;
- Monte Carlo and `h` termination confidence tests pass;
- mean tracking, p95 tracking, head alignment, corrected clearance, truck
  post-constraint drift, completion and point termination gates pass.

The remaining problem is therefore narrow but real. It is not a general loss of
control and it is not a robustness collapse across the ROM factor range. It is a
configuration-dependent closed-loop acceleration tail on the circle. The
reference/inner-loop/plant system already operates too close to the 40 m/s2
limit on some phases. The existing PPO residual adds approximately 1-2 m/s2 on
the two independent failures and pushes them over the gate.

This is the measured mechanism:

```text
circle configuration / phase
          ↓
baseline closed-loop acceleration ≈ 39.6–39.7 m/s2
          ↓
PPO residual adds a seed/phase-dependent tail
          ↓
observed p95 = 40.84 or 41.71 m/s2 > 40 gate
```

The next justified intervention is a small, circle-focused, factor-blind PPO
remediation from `model_53`, using c7 unchanged and the acceleration reward that
already exists. Further timing changes based on the current held-out failures
would be test-set overfitting. ROM-C/AAM should not be used to conceal this ROM-B
baseline defect.

---

## 1. What ROM-B is supposed to prove

ROM-B is the robust, factor-blind PPO branch. It must control the arm when the
reset-fixed hydraulic ROM plant varies over exactly five hidden factors:

1. `e_comp`;
2. `d_seal_1`;
3. `d_seal_2`;
4. `d_seal_3`;
5. `d_seal_4`.

`e_flow` has been removed from the ROM-B model boundary and should remain zero
and absent. The policy does not receive the five true factors. The deterministic
truck is intentionally outside PPO because the research target is the arm in
isolation. The inner DLS/cascade controller must remain unchanged.

ROM-B has two robustness tests:

- **Monte Carlo:** factors are sampled independently per environment. This
  explores off-diagonal combinations that cannot be represented by one scalar
  health value.
- **Correlated h sweep:** a scalar `h` maps to a declared correlated factor
  vector. The tested values are 0.4 through 1.2. This is a readable one-dimensional
  slice through factor space, not a replacement for Monte Carlo.

The circle acceptance gates relevant here are:

| Metric | Gate |
|---|---:|
| mean tracking error | <= 0.04 m |
| p95 tracking error | <= 0.08 m |
| p95 measured tool acceleration | <= 40 m/s2 |
| minimum head alignment | >= 0.98 |
| maximum corrected clearance error | <= 0.20 m |
| maximum post-constraint truck drift | <= 0.001 m |
| row termination fraction | <= 0.05 |

Termination also has a confidence-based acceptance statistic. It takes the
maximum of a one-sided pooled Clopper-Pearson upper bound and a one-sided
seed-group t upper bound. It never pools across different shapes or `h` values.

### Meaning of `certifiable: true`

The JSON report field `certifiable: true` means the protocol is eligible to
produce a certification result: factor-blind policy, corrected measurement and
fresh-process isolation. It does **not** mean that every gate passed. The c7
seed-29 report is certifiable in protocol and failed in result.

---

## 2. What is working correctly

It is important not to describe the whole system as broken.

### Tracking works

In the independent c7 audit, the worst Monte Carlo values were:

- mean error: 0.02681 m against 0.04 m;
- p95 error: 0.05023 m against 0.08 m.

The policy therefore follows the circle with substantial tracking margin.

### Head alignment works

Worst Monte Carlo head alignment was 0.98285 against the 0.98 gate. This margin
is small but positive. Every `h` aggregate also passed.

### Corrected clearance works

Worst Monte Carlo clearance was 0.11313 m against 0.20 m. Worst `h` clearance
was 0.12427 m. The earlier clearance failure narrative was largely a measurement
frame defect, described later.

### Termination robustness works

For the independent c7 Monte Carlo audit:

- failures per seed group: `[0, 1, 0, 0, 1]`;
- total: 2/640;
- observed rate: 0.003125;
- one-sided acceptance upper bound: 0.00980;
- threshold: 0.05;
- result: pass.

For the c7 `h` sweep, the worst confidence condition was `h=1.0`:

- failures per group: `[0, 2, 1]`;
- total: 3/384;
- upper bound: 0.02098;
- threshold: 0.05;
- result: pass.

Some individual environments still cross `truck_raw_reaction`, but this is not
the binding certification defect. Seed 300042 fails acceleration at 41.71 m/s2
with zero terminations.

### The complete h sweep works

All nine `h` values pass. Their worst p95 accelerations are:

| h | worst p95 acceleration |
|---:|---:|
| 0.4 | 38.64 m/s2 |
| 0.5 | 37.31 m/s2 |
| 0.6 | 38.20 m/s2 |
| 0.7 | 38.25 m/s2 |
| 0.8 | 38.71 m/s2 |
| 0.9 | 38.77 m/s2 |
| 1.0 | 37.83 m/s2 |
| 1.1 | 38.53 m/s2 |
| 1.2 | 38.42 m/s2 |

This is strong evidence that the remaining failure is not monotonic in the
correlated health coordinate and cannot be explained as simply “the weakest” or
“the stiffest” plant.

---

## 3. Measurement and evaluation defects found before the real circle defect

Two measurement defects initially made ROM-B look much worse than it is. A third
protocol defect was found during the retiming work.

### 3.1 Clearance frame contamination

The historical clearance signal mixed the cutter reference with truck ride
height. Approximately 0.164 m of the reported clearance was frame contamination,
which was about 82% of the 0.20 m gate.

The decisive experiment was a 250-row comparison in which the corrected and
historical evaluators differed in exactly one output column: clearance. Tracking,
acceleration, head and truck values did not change. This proved the issue was the
measurement frame, not policy behavior.

The final evaluator uses `--clearance_frame_consistent`, comparing an
orientation-aware cutter signal against a live-truck-pose-consistent reference.

Source: `reports/rom_b_frame_decomposition.md`.

### 3.2 Sequential simulator-state leakage

The earlier evaluator ran many rollouts sequentially in one Isaac/PhysX process.
The first rollout was reproducible, but later rollouts inherited hidden state.
Across six conditions, condition-level process isolation reduced
`truck_raw_reaction` failures from 100 to 14 while head failures remained roughly
stable at 50 to 47. That proved the truck failure count was largely an evaluator
artifact.

Condition-level isolation was still insufficient. It launched one process per
`(suite, h, shape)` but ran several seed rounds inside that process. A matched
circle check showed:

- `h=1.0`, circle, seed 400030 after two earlier rounds: 44.0805 m/s2;
- the same seed as the first rollout of a fresh process: approximately 39.99
  m/s2.

The certification evaluator now launches one fresh process per individual
rollout seed. The c7 seed-29 audit has `rollout_process_isolation: true`.

### 3.3 Initial-shape construction leakage

A circle diagnostic initially produced about 40.50 m/s2 while the audit produced
44.08 for what appeared to be the same seed. The difference was environment
construction:

- the full audit constructed on canonical `line`, then switched to `circle`;
- the diagnostic constructed directly on `circle`.

That changed the subsequent physical rollout despite resetting with the same
seed. Diagnostics and shape-subset audits now construct on canonical `line`
before selecting the requested shape.

### Consequence for old reports

The old full report in `reports/evaluation_isolated` does not meet the current
per-rollout isolation contract and does not bind itself to a trajectory-library
hash. It remains useful diagnostic history but is not the current certification
authority.

---

## 4. The genuine acceleration problem after fixing measurement

After fresh-process isolation, most apparent circle failures disappeared. A
small real failure remained. With c3 and a fresh process:

- `h=0.4`, seed 400030: 40.98 m/s2;
- high-`h`, Monte Carlo and arc failures that had appeared in sequential reports
  often fell below 40 when run first in a fresh process.

Path-bin diagnostics showed a repeatable circle hotspot:

| Geometric bin | Path range | Representative local p95 acceleration |
|---:|---:|---:|
| 2 | 0.691-1.037 m | 35-52 m/s2 depending on seed |
| 3 | 1.037-1.382 m | 56-65 m/s2 |
| 4 | 1.382-1.728 m | 33-42 m/s2 |

Bin 3 is consistently the strongest local hotspot. This is a local p95 within
that phase bin; the formal row p95 is calculated over all scored samples and is
therefore lower.

The c3 cylinder path in this region has nontrivial path derivatives. For the
original c4 construction diagnostic:

- bin-3 p95 `|dq/ds|`: 0.1976;
- bin-3 p95 `|d2q/ds2|`: 1.9004.

This is consistent with a mechanically demanding region, but cylinder curvature
alone is not a complete predictor: some other circle bins have larger command
second derivatives without producing the largest measured tool-acceleration
tail. Closed-loop plant and policy response matter.

---

## 5. Fixes and experiments that failed or were insufficient

The failed fixes are scientifically useful because they constrain the mechanism.

### 5.1 Globally reducing circle speed

Tested on the original binding `h=0.4`, seed 400030 condition:

| configured base speed | effective circle speed | p95 acceleration | mean error | p95 error |
|---:|---:|---:|---:|---:|
| 0.18 | 0.126 m/s | 40.72 | 0.02646 m | 0.05029 m |
| 0.16 | 0.112 m/s | 40.67 | 0.02578 m | 0.04793 m |
| 0.14 | 0.098 m/s | 40.31 | 0.02506 m | 0.04649 m |

A 22% speed reduction improved tracking but reduced acceleration by only 0.41
m/s2. The bin-3 local p95 remained approximately 56-59 m/s2. Therefore constant
global speed was not the dominant mechanism.

### 5.2 Disabling PPO residual action globally

The zero-residual ablation was mixed:

- `h=0.4`, seed 400030: 40.72 -> 38.51 m/s2, an improvement;
- `h=0.9`, another seed: 35.66 -> 37.46 m/s2, a regression.

Half residual also failed the first condition at approximately 40.53 m/s2.

Conclusion: PPO contributes to some tails, but globally disabling or shrinking
it is not safe. The policy is useful in other plant/phase combinations.

### 5.3 Filtering inverse-reference cylinder velocity

An EMA with beta 0.8 was applied only to inverse-reference cylinder velocity:

- binding low-`h` condition: 40.72 -> 39.33 m/s2;
- second condition: 35.66 -> 36.04 m/s2.

This was promising in two probes but failed the broader preflight:

- beta 0.8 worst: 40.42 m/s2;
- failing conditions remained at `h=0.6` and `h=1.2` in that screen.

The filter was not adopted. Production `dls_velocity_filter_beta` remains zero.

### 5.4 c2/c3 inverse-cylinder continuity smoothing

The original inverse circle reference had a cylinder-command curvature spike
near path fraction 0.321, aligned with the measured bin-3 hotspot. c2/c3 used
periodic Savitzky-Golay smoothing followed by projection through the measured
cylinder-to-tool model. Cartesian circle geometry remained exact.

c3 passed small preflights but did not generalize. In the old 32-row comparison:

- previous circle: 11 rows above 40;
- c3: 15 rows above 40;
- mean acceleration changed by +0.193 m/s2;
- 16 rows improved and 16 worsened.

The c3 full-preflight success was therefore an undersampled conclusion. It did
not include the later binding seed groups and was later retracted.

### 5.5 First retiming implementation: invalid start-phase comparison

The first time-law implementation replaced geometric arc length with a stretched
timing coordinate and then sampled start phase uniformly in that timing
coordinate. This caused more environments to start in slow/stretched regions.
The baseline and candidate no longer had matched physical start phases.

That invalid candidate appeared to regress acceleration from 40.98 to 42.50.
The runtime was corrected to maintain two coordinates:

- geometric `s` for physical phase sampling and diagnostics;
- timing `u` for speed progression.

Start phases are now sampled uniformly in geometric `s` and mapped to `u`. This
was a measurement/design error in the experiment, not evidence against proper
path retiming.

### 5.6 c4: fixed known seeds but failed its complete audit

c4 slowed the measured difficult region by at most 18.6% while preserving every
Cartesian and cylinder sample. It improved the known low-`h` failure:

- acceleration: 40.98 -> 37.88 m/s2;
- mean tracking: 0.02610 -> 0.02470 m;
- clearance: 0.11795 -> 0.09955 m;
- termination: 1/128 -> 0/128.

Five adversarial development shards passed, but the full 32-row circle audit
failed:

- Monte Carlo round 1: 41.90 m/s2;
- Monte Carlo round 4: 41.46 m/s2;
- `h=0.5`: 40.32 m/s2;
- `h=1.2`: 40.33 m/s2.

c4 was tuned to one known seed family and did not cover the full phase/plant
distribution.

### 5.7 c5: stronger but broader timing was worse

c5 lowered the nominal target from 38 to 36 and broadened slowing into additional
regions. It regressed:

- `h=1.0`: 39.75 m/s2;
- clearance: 0.19131 m, close to the 0.20 gate;
- `h=0.4`: 38.15 m/s2 with a termination.

c5 demonstrated that “slower” is not monotonically “better” in this closed-loop
system. Changing timing changes reference previews, cylinder feed-forward and
the policy trajectory, so the response is nonlinear. c5 was deleted.

### 5.8 Two-update circle PPO pilot

A minimal remediation pilot warm-started from `model_53`, used a fresh optimizer
and ran two PPO updates with the existing acceleration reward. Its internal
candidate regressed relative to the accepted state. The training code correctly
rolled it back. No new PPO checkpoint was accepted.

This does not prove PPO fine-tuning cannot work. Two updates and one internal
evaluation seed are too small to answer that question. It proves only that an
unstructured tiny update should not be accepted automatically.

### 5.9 c6: improved all binders but still missed two

c6 was derived from the worst phase-bin profiles of four c4 failures. It added a
capped incremental slowdown over bins 2-4. Results:

- MC round 1: 41.90 -> 38.86;
- MC round 4: 41.46 -> 40.27;
- `h=0.5`: 40.32 -> 36.20;
- `h=1.2`: 40.33 -> 40.21.

It confirmed the common phase mechanism but still failed two gates by about
0.2 m/s2. c6 was deleted.

### 5.10 c7: passed development seeds, failed independent seeds

c7 deepened only the bin-3 slowdown on top of c4. It preserved all geometry and
cylinder commands. On the six seed-17 development/regression shards:

| Condition | c7 acceleration | Result |
|---|---:|:---:|
| MC round 1 | 39.29 | pass |
| MC round 4 | 39.76 | pass |
| h=0.5 round 1 | 37.50 | pass |
| h=1.2 round 3 | 37.51 | pass |
| h=0.4 round 3 | 39.88 | pass |
| h=1.0 round 3 | 38.52 | pass |

Because seed 17 had influenced c7, seed 29 was predeclared as a new independent
audit family. That audit failed:

- MC seed 300037: 40.8448 m/s2;
- MC seed 300042: 41.7117 m/s2.

This is the decisive evidence that timing-only remediation is not robust enough.
Further timing changes using seed 29 would convert the independent test into
more development data and repeat the same overfitting cycle.

---

## 6. Root-cause analysis

The root-cause statement must distinguish established facts from inference.

### 6.1 Established: phase/configuration dependence

Both independent failures retain strong local circle-phase structure.

For seed 300037:

- bin 0 local p95: 44.90 m/s2;
- bin 3 local p95: 57.32 m/s2.

For seed 300042:

- bin 2 local p95: 52.05 m/s2;
- bin 3 local p95: 56.49 m/s2;
- bin 4 local p95: 41.93 m/s2;
- bin 6 local p95: 40.27 m/s2.

The dominant bin remains 3, but the tail moves into neighboring and occasional
other bins. A single hand-shaped timing pocket cannot guarantee the global p95
for unseen phase populations.

### 6.2 Established: no single factor explains the failures

The diagnostic computes each environment's p95 acceleration and compares it
with its hidden factors. Across both independent failures, the absolute Pearson
correlation for every factor is at most 0.09.

The complete correlated `h` sweep also passes. Therefore there is no evidence
that one factor, or the correlated health coordinate, directly controls the
tail.

This does not prove factors have exactly zero effect. It shows that factor main
effects are weak compared with phase/closed-loop variability in these rollouts.
Nonlinear interactions may exist, but the current evidence does not justify a
factor-specific fix.

### 6.3 Established: PPO adds the final failing tail

Matched zero-residual ablations on the two independent failures produced:

| Seed | PPO enabled | PPO residual forced zero | Change |
|---:|---:|---:|---:|
| 300037 | 40.8448 | 39.69 | -1.15 m/s2 |
| 300042 | 41.7117 | 39.57 | -2.14 m/s2 |

Both zero-residual rows pass the formal 40 gate. This is direct causal evidence:
the policy residual action supplies the final increment that crosses the gate on
these seeds.

### 6.4 Established: PPO is not the only physical source

The zero-residual baseline is still 39.57-39.69 m/s2. That is only 0.31-0.43
m/s2 below the formal gate and far above the desired 38 m/s2 development margin.

Therefore the accurate statement is not “PPO causes all acceleration.” The
reference, inner controller and plant already create a high-acceleration
configuration-dependent response. PPO amplifies it selectively.

### 6.5 Established: the PPO effect is not globally harmful

Earlier zero-policy experiments showed the opposite direction on another
condition: 35.66 became 37.46 m/s2 when the residual was disabled. PPO is useful
in some configurations and harmful in others. A global residual shutdown would
trade one failure for another.

### 6.6 Inference: insufficient policy selectivity around the circle

The most likely policy-level interpretation is that `model_53` has not learned
to reduce or reshape its residual specifically when the arm crosses the
high-acceleration circle configurations. It continues applying corrections that
may improve tracking but add acceleration to a baseline already near the gate.

This is an inference from the causal zero-residual ablation and phase profiles.
It is not yet localized to a particular action channel or network feature.

### 6.7 What is explicitly not the current root cause

The independent failure cannot reasonably be attributed to:

- sequential simulator-state leakage: every row owns a fresh process;
- clearance contamination: corrected clearance is used and passes;
- deterministic truck design: truck is intentionally held outside PPO and the
  acceleration failure occurs without required truck termination;
- one hidden ROM factor: correlations are weak and all h conditions pass;
- gross tracking failure: tracking gates have large margin;
- circle geometry discontinuity: the Cartesian circle is exact and closed;
- cylinder endpoint discontinuity: the cylinder circle is closed;
- lack of global slowing: global speed reduction was already falsified.

### Root cause in one sentence

The remaining defect is a configuration-dependent closed-loop circle
acceleration tail with inadequate baseline margin, selectively amplified by the
existing PPO residual on some independently randomized plant and start-phase
combinations.

---

## 7. Why ROM-C/AAM is not the immediate fix

ROM-C is intended to study online adaptation, not to hide a ROM-B reference or
policy defect.

The corrected privileged-oracle experiment supplied the true hidden factors to
the policy and used the corrected warm-start design. It achieved approximately
0.0420 compared with ROM-B's approximately 0.0437 on the relevant tracking
metric. Knowing the true factors bought essentially nothing.

That result is structurally consistent with the present circle evidence:

- ROM-B factors are reset-fixed;
- the feedback loop has time to converge to a constant plant;
- all correlated h conditions pass;
- factor/acceleration association in the failing Monte Carlo rows is weak.

An online estimator cannot outperform an oracle that already knows the truth.
Therefore AAM should not be expected to remove the current circle acceleration
tail on a reset-fixed plant.

For ROM-C to test meaningful adaptation later, the plant must change within an
episode in a declared, physically defensible way. The inner controller and truck
design can remain unchanged. Examples must come from measured/declared ROM
parameter changes, not invented physics.

The repository's current architectural contract also treats ROM-B and ROM-C as
parallel branches from PPO2_A `model_607`, not automatically sequential stages.
Changing ROM-C to start from a remediated ROM-B checkpoint would be an explicit
experimental-design change and must be documented, budget-matched and compared
fairly.

---

## 8. Planned ROM-B remediation

Timing development should stop at c7 for now. Seed families 17 and 29 are
development/diagnostic data and must not be reused for final acceptance after a
policy change.

### Step 1: freeze everything except PPO weights

Keep fixed:

- c7 Cartesian geometry and timing law;
- c7 SHA-256;
- inner DLS/cascade controller;
- deterministic truck behavior;
- command hydraulic ROM checkpoint;
- five-factor ROM-B boundary;
- factor-blind 42-dimensional policy contract;
- corrected clearance measurement;
- acceleration definition and 40 m/s2 gate;
- 128-environment validated scene scale;
- `e_flow` absent.

The intervention should change only PPO weights. This makes the experiment
interpretable.

### Step 2: use the existing reward before inventing another one

ROM-B already includes:

- an acceleration soft threshold at 35 m/s2;
- a hard threshold at 40 m/s2;
- quadratic acceleration-margin weight -1.0;
- acceleration-event weight -5.0;
- residual action-rate penalty;
- tracking and safety terms.

The first remediation should use this existing reward unchanged. Adding new
coefficients before testing whether the current gradient works would repeat the
earlier pattern of proposing fixes before measuring the mechanism.

### Step 3: short, warm-started, circle-focused PPO run

Warm-start from accepted `model_53` actor and critic weights with a fresh
optimizer. Do not restart from PPO2_A and do not train from scratch.

The run should be deliberately small and should include:

- full ROM-B randomization, not only h-correlated plants;
- randomized geometric start phase;
- circle-focused experience;
- a small replay allocation for the other four shapes, or immediate all-shape
  regression checks, to detect forgetting;
- frozen exploration standard deviation if that remains the accepted joint
  fine-tuning contract;
- no privileged factors and no AAM observer;
- no final-test seeds in training or promotion.

The policy needs to learn a selective behavior: preserve useful residuals in
easy configurations and reduce/reorient them where baseline circle acceleration
is already near the gate.

### Step 4: cheap development gates

Use only designated training/development seed families. Do not use seeds 17 or
29 for checkpoint selection.

Require the candidate to achieve, on development data:

- worst p95 acceleration <= 38 m/s2, providing 5% margin to the 40 gate;
- mean tracking <= 0.04 m;
- p95 tracking <= 0.08 m;
- head >= 0.98;
- clearance <= 0.20 m;
- no worse termination confidence trend;
- no regression on line, arc, sine or square smoke tests.

The 38 target is a development margin, not a new certification threshold.

### Step 5: causal ablations on the candidate

Before a long audit, compare:

1. accepted `model_53` on c7;
2. fine-tuned policy on c7;
3. zero residual on c7.

Use matched development conditions. The candidate should improve over
`model_53` without merely collapsing toward zero residual everywhere. Action
profiles by circle phase should show selective suppression around the difficult
configurations.

If possible, report per-cylinder residual magnitude/rate by geometric bin. This
will identify whether one cylinder channel drives the acceleration tail.

### Step 6: independent circle certification

After freezing the checkpoint, predeclare a completely new base seed family.
Seeds 17 and 29 are no longer held out. A future seed such as 43 may be used only
if it is declared before the run and never used for further tuning.

Run the complete 32-row circle audit:

- five independent Monte Carlo seed groups;
- three seed groups at each of nine h values;
- 128 environments;
- 120 scored steps;
- one fresh process per rollout;
- corrected clearance;
- exact checkpoint and trajectory hashes.

The circle is certified only if all point gates and both termination confidence
families pass.

### Step 7: complete all-shape certification

Circle-only success is necessary but not sufficient for ROM-B as a whole.
Run the complete five-shape audit with another untouched seed family after the
circle result is frozen. This protects against catastrophic forgetting caused by
circle-focused training.

Do not claim full ROM-B certification from a circle-only report.

---

## 9. What happens if ROM-B passes

If the fine-tuned checkpoint passes the independent circle audit and the final
all-shape audit:

1. Freeze and hash:
   - PPO checkpoint;
   - c7 trajectory library;
   - command ROM checkpoint;
   - factor ranges/order;
   - evaluator code and thresholds.
2. Write the final ROM-B certification report with no diagnostic/test seeds used
   for selection.
3. Treat that artifact as the robust factor-blind baseline.
4. Start the ROM-C experimental design from the declared branch point.
5. Keep the inner controller and deterministic truck unchanged in both branches.
6. Use the same trajectory library, task distribution, evaluation seeds and
   training budget for fair ROM-B/ROM-C comparison.
7. Do not present the privileged oracle as a ROM-B result. It deliberately
   violates factor blindness and is only an adaptation upper bound.

### Meaningful ROM-C experiment

The current reset-fixed plant offers little reason for online identification.
If ROM-C is intended to demonstrate AAM benefit, introduce a measured and
declared within-episode plant change. The change schedule must be identical
between ROM-B and ROM-C evaluations and must remain within the validated ROM
factor envelope unless new physical evidence justifies extending it.

A fair ROM-C study should report:

- pre-change performance;
- transient immediately after the plant change;
- adaptation settling time;
- post-adaptation tracking and acceleration;
- safety/termination confidence;
- comparison against factor-blind ROM-B;
- privileged-oracle upper bound;
- observer/AAM ablations such as active, zeroed and shuffled estimates.

ROM-C counts as an improvement only if it beats the certified ROM-B baseline on
the nonstationary task without degrading safety or exploiting privileged factor
access.

---

## 10. Stop conditions and decision rules

To avoid another expensive loop of inconclusive runs:

### Stop policy fine-tuning if

- acceleration improvement requires tracking or clearance regression;
- the candidate simply drives all residuals to zero;
- easy shapes regress;
- three short, predeclared remediation attempts fail to create <=38 development
  margin;
- results improve only on seeds already used for selection.

### Revisit the baseline/controller only if

- zero-residual c7 remains near or above 40 on new development seeds;
- PPO fine-tuning cannot improve margin selectively;
- action-channel diagnostics show the policy is not responsible for the tail.

At that point, the proper question is no longer “how should PPO compensate?” It
is whether the c7 reference and unchanged inner loop can physically provide the
required acceleration margin. Any controller modification would be a new scoped
experiment and must be compared against the current unchanged-controller
baseline.

### Proceed to ROM-C only if

- ROM-B circle passes an independent audit;
- ROM-B passes the complete all-shape audit;
- the final checkpoint and libraries are frozen;
- ROM-C's nonstationary plant task is declared before training;
- comparison budget and seeds are fixed.

---

## 11. Reproducibility and authoritative artifacts

### Current hashes

- `model_53` SHA-256:
  `6731bac46e083d7ac99074e3a7b34752b64117a5003755e3d8296747353a69e9`
- c7 SHA-256:
  `c5fcfb249cc7d0be29e6c13340b74953f64ca5ed090472989ab798ea54aa603b`

### Independent c7 audit

- `reports/circle_c7_seed29_audit/rom_b_held_out_report.json`
- `reports/circle_c7_seed29_audit/isolated_rows.json`
- `reports/circle_c7_seed29_audit/rom_b_held_out_rows.csv`
- `reports/circle_c7_seed29_audit/rom_b_final_robustness.png`

### Root-cause diagnostics

- `reports/circle_c7_seed29_profile_mc_r2/rom_b_diagnostics.json`
- `reports/circle_c7_seed29_profile_mc_r3/rom_b_diagnostics.json`
- `reports/circle_c7_seed29_zero_mc_r2/rom_b_diagnostics.json`
- `reports/circle_c7_seed29_zero_mc_r3/rom_b_diagnostics.json`
- `reports/circle_c7_seed29_root_cause.md`

### Earlier experimental summaries

- `reports/circle_cheap_experiment_2026-07-26.md`
- `reports/circle_retiming_report_2026-07-26.md`
- `reports/circle_c7_revision_report_2026-07-26.md`
- `reports/rom_b_frame_decomposition.md`
- `reports/rom_b_step1_results.md`
- `reports/rom_b_root_cause.md`

### Code

- retiming generator:
  `PPO2_ROM_PhysX/scripts/build_circle_retiming_candidate.py`
- geometric/timing-coordinate runtime:
  `PPO2_ROM_PhysX/policy/env.py`
- per-rollout isolated evaluator:
  `PPO2_ROM_B_PhysX/scripts/evaluate.py`
- isolation parent:
  `PPO2_ROM_B_PhysX/scripts/evaluate_isolated.py`
- diagnostic evaluator:
  `PPO2_ROM_B_PhysX/scripts/diagnose.py`
- factor/phase summaries:
  `PPO2_ROM_B_PhysX/runtime/evaluation.py`

### Tests

The current relevant suites pass 78 tests:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  PPO2_ROM_B_PhysX/tests \
  PPO2_ROM_C_PhysX/tests \
  PPO2_ROM_PhysX/trajectories/tests
```

---

## 12. Literature basis for the approach

The useful literature lesson was to separate a fixed geometric path from its
timing law, rather than asking PPO to compensate for every aggressive reference.

- Pham and Pham, TOPP-RA, reachability-based time parameterization under
  constraints: https://arxiv.org/abs/1707.07239
- Pham and Pham, TOPP3, third-order/jerk-constrained path parameterization:
  https://arxiv.org/abs/1609.05307
- Berscheid and Kroger, Ruckig, jerk-limited real-time trajectory generation:
  https://arxiv.org/abs/2105.04830

The c4-c7 work correctly implemented geometric/timing separation, but the
independent audit shows why literature structure is not a substitute for system
identification: the closed-loop acceleration response is nonlinear, depends on
phase and policy action, and does not decrease monotonically with hand-shaped
speed changes. PhysX acceptance remains the authority.

---

## Final plain-language explanation

The arm can draw the circle accurately and safely. The problem is that in a few
parts of the circle it is already moving with almost the maximum allowed
acceleration. The PPO then adds a small corrective movement. Most of the time
that correction is useful, but for a few combinations of starting phase and
randomized plant it adds just enough acceleration to cross the 40 m/s2 limit.

We repaired broken measurements, isolated every simulation rollout, smoothed the
cylinder path and made the circle slow down in difficult regions. Those changes
removed many false and real failures, but they did not create enough margin for
unseen Monte Carlo seeds. Making the circle slower again and again is now
overfitting the test cases.

The next step is to teach the PPO a more selective correction: help where help
is needed, but back off when the arm is already near the acceleration limit. If
that policy passes completely new circle seeds and then all five shapes, ROM-B
becomes the frozen robust baseline. Only then should ROM-C test whether online
adaptation helps when the plant actually changes during operation.

---

## 13. Pre-training gate audit and corrections

This section responds to five load-bearing concerns identified after the first
complete handoff. It supersedes any weaker wording in the earlier plan.

### 13.1 The 38 m/s2 development target requires active cancellation

On the two independent failures, forcing PPO residual action to zero gives:

- seed 300037: 39.69 m/s2;
- seed 300042: 39.57 m/s2.

Therefore “back off to zero residual” cannot achieve the proposed <=38 m/s2
development target on these seeds. A policy reaching 38 must actively command a
counteracting residual that reduces approximately 1.6-1.7 m/s2 of acceleration
created by the reference/inner-loop/plant response.

This is materially harder than selective suppression. The policy must predict a
future high-acceleration configuration and use its limited cylinder-goal
authority to reshape the motion without increasing tracking error, clearance or
head error.

The 38 target remains useful as a desired margin, but it must not be treated as
automatically reachable by PPO. Its feasibility is now gated by a complete
zero-residual population sweep.

#### Mandatory next experiment

Run all 32 circle rows with PPO residual forced to zero using a fresh development
seed family. This is a diagnostic baseline, never a certification result.

Interpretation:

- If most rows cluster around 36-37 and only a few approach 39.5, selective PPO
  remediation may be worth funding.
- If a meaningful fraction exceed 38, the baseline/controller revisit condition
  is already satisfied. PPO would have to actively cancel the baseline over a
  broad population, making policy-only remediation substantially less likely.
- If any zero-residual rows exceed 40, the unchanged baseline itself fails the
  formal gate and PPO training should pause until the reference/inner-loop
  mechanism is revisited.

The predeclared zero-baseline development seed is 41. The evaluator records
`force_zero_policy: true`, labels the policy `forced_zero_residual`, and sets
`certifiable: false` by construction.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/evaluate.py \
  --headless --device cuda:0 --num_envs 128 --seed 41 \
  --trajectory_shapes circle --monte_carlo_rounds 5 \
  --h_sweep_rounds 3 --eval_steps 120 \
  --clearance_frame_consistent --isolate_conditions \
  --force_zero_policy \
  --trajectory_library_path \
    PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_circle_c7_retime.json \
  --output_dir PPO2_ROM_B_PhysX/reports/circle_c7_zero_seed41
```

Expected runtime is approximately 35-40 minutes on the measured machine.

### 13.2 Complete 42-channel observation audit

The policy actor is a feed-forward MLP. It has no recurrent hidden state and no
stack of previous tool velocities. The four-action observation layout is:

| Indices | Width | Signal | Relevance to acceleration control |
|---:|---:|---|---|
| 0:3 | 3 | tool position error in truck frame | current tracking state |
| 3:6 | 3 | measured tool velocity in truck frame | instantaneous motion, but not acceleration |
| 6:9 | 3 | reference tool velocity | commanded motion |
| 9:12 | 3 | reference tool acceleration | tangential timing acceleration only |
| 12:16 | 4 | measured cylinder extensions | arm configuration; indirectly identifies circle phase |
| 16:20 | 4 | corrected cylinder goal error | instantaneous plant/controller lag |
| 20:24 | 4 | measured cylinder velocities | actuator motion and phase-dependent dynamics |
| 24:28 | 4 | previous filtered PPO action | one-step policy-action memory |
| 28:32 | 4 | current valve commands | current inner-loop effort |
| 32:36 | 4 | corrected ROM increments | predicted/achieved actuator evolution |
| 36:38 | 2 | 0.25-second XY reference preview | local path direction/curvature cue |
| 38 | 1 | trajectory direction | forward/reverse leg |
| 39 | 1 | corrected clearance | safety/configuration cue |
| 40 | 1 | head alignment | safety/configuration cue |
| 41 | 1 | head-alignment rate | one temporal derivative |

#### What is absent

The policy does not receive:

- measured tool acceleration;
- previous measured tool velocity;
- explicit geometric path coordinate or phase;
- explicit circle curvature;
- hidden ROM factors, by design.

The policy therefore cannot directly implement a rule such as “my measured
acceleration is 39 m/s2, reduce it.” It can only predict a risky future state
from cylinder configuration, cylinder/tool velocities, goal error, valve/ROM
signals and the XY preview.

The desired behavior is **indirectly representable** as phase/configuration
conditioned feedback because cylinder extensions and preview identify the arm's
location on the circle. It is not directly representable as acceleration-limit
feedback.

There is another important limitation: `reference_tool_acceleration_truck` is
formed as signed path acceleration multiplied by the unit tangent. During
constant-speed circle travel, signed path acceleration is near zero, so this
channel omits the centripetal/curvature component of reference acceleration. The
preview contains a curvature cue, but the nominal reference-acceleration channel
does not explicitly describe circular acceleration.

#### Decision consequence

Do not fund a long fine-tune on the assumption that the actor observes the gate
state. After the zero-baseline sweep, the cheapest representability experiment
is a short, checkpoint-widened pilot that adds either:

- previous tool velocity, allowing one-step acceleration reconstruction; or
- measured tool acceleration plus a properly defined curvature/reference-
  acceleration preview.

This would change the observation contract and requires zero-padded warm-start
transfer, explicit ablations and a new checkpoint family. It must not be slipped
into ROM-B silently. The existing 42-channel policy should still receive one
short fair attempt first only if the zero-baseline distribution shows that
phase-conditioned cancellation is plausible.

### 13.3 Acceleration definition and independent recomputation

The formal acceleration signal is computed as follows:

1. Read the orientation-aware cutter point in world coordinates.
2. Transform it into the live truck-chassis frame.
3. At each environment control step, compute
   `velocity[t] = (position[t] - position[t-1]) / dt`.
4. Compute
   `acceleration[t] = (velocity[t] - velocity[t-1]) / dt`.
5. Take the Euclidean norm of the 3-D acceleration.
6. Pool finite scored samples from completed environments and take `torch.quantile(..., 0.95)`.

Simulation physics runs at 120 Hz with decimation 2, so the environment control
step is `dt = 1/60 s = 0.0166667 s`. No filtering is applied to position,
velocity or acceleration before the gate.

#### Sanity check

For the independent binding seed 300042, the diagnostic recorded every scored
cutter position and independently recomputed the second finite difference:

| Quantity | Result |
|---|---:|
| matched samples | 15,104 |
| online aligned p95 | 41.9212265 m/s2 |
| offline position-difference p95 | 41.9212341 m/s2 |
| mean absolute disagreement | 0.00000227 m/s2 |
| p95 absolute disagreement | 0.00000763 m/s2 |

The online implementation is therefore internally correct and reproducible from
the position trace.

This check does **not** prove that an unfiltered second difference is the best
physical acceptance signal. Twice differentiating position amplifies 60-Hz
position jitter and PhysX/contact motion. Because the baseline is only 0.3-0.4
m/s2 below the gate, a second independent physical comparison remains valuable:
derive cutter-point velocity from PhysX body linear/angular velocity and compare
its derivative, and report a separately labelled filtered engineering signal.
The predeclared formal gate must not be changed after seeing the result; any new
signal is initially diagnostic.

The evaluator also excludes terminated environments from continuous p95 metrics
and handles their failures through termination metrics. This should remain
documented because it affects the exact population entering the acceleration
quantile.

### 13.4 Retiming and time-uniform order statistics

The formal row p95 weights every scored time sample equally. A slower path region
receives more time samples. Therefore retiming changes both:

1. local acceleration magnitudes;
2. the fraction of the row distribution contributed by each geometric region.

Those effects can oppose one another and were not documented in the original
retiming argument.

Raw per-step traces from deleted c5/c6 candidates were not retained, so an exact
offline c5/c6/c7 reranking cannot be reconstructed honestly. Future diagnostics
now report both the formal time-uniform p95 and an equal-geometric-bin weighted
p95.

For independent c7 seed 300042:

- formal time-uniform p95: 41.7117 m/s2;
- equal-geometric-bin weighted p95: 42.3065 m/s2.

On this binding row, time weighting slightly **reduces** the reported p95 by
about 0.59 m/s2. It does not create the failure. Thus the proposed occupancy
mechanism is real in principle but has the opposite direction on the checked
c7 row. The c4-c7 nonmonotonicity cannot currently be attributed primarily to
time weighting; closed-loop trajectory/policy changes remain necessary to
explain it.

### 13.5 Development protocol must reflect max over 32 rows

The six-shard screens that accepted c4 and c7 were structurally underpowered.
Certification takes the maximum over 32 rows:

- five Monte Carlo rows;
- 27 h-sweep rows.

A maximum over six hand-selected rows is systematically less extreme and cannot
serve as an acceptance test for a max-over-32 specification. The earlier cycle
demonstrated this twice: six shards passed, then the complete audit failed.

The corrected protocol is:

1. A small adversarial shard set may **reject** a bad candidate cheaply.
2. It may never accept a candidate.
3. Development acceptance requires a full 32-row circle suite using development
   seeds.
4. Final acceptance requires another full 32-row suite using a seed family not
   used for training, checkpoint selection, timing design or development gates.
5. After circle acceptance, full ROM-B still requires the five-shape audit on
   untouched seeds.

No simple Gaussian multiple-comparison correction is justified because row
metrics are dependent through shared checkpoint, shape and simulator structure.
Matching the actual row count and max operator is clearer and safer.

### 13.6 Revised decision sequence

The new sequence before any substantial PPO compute is:

1. **Observation audit:** complete; desired cancellation is only indirectly
   representable in the 42 channels.
2. **Acceleration-definition audit:** online/offline finite differences agree;
   metric remains intentionally unfiltered and noise-sensitive.
3. **Path-weighting audit:** checked on a binding c7 seed; weighting does not
   explain away the failure.
4. **Full zero-residual seed-41 sweep:** next mandatory experiment.
5. **Decision:**
   - broad zero-baseline >38 or any >40: revisit baseline/reference/controller
     before PPO funding;
   - zero baseline mostly 36-37 with sparse near-40 rows: fund a short,
     warm-started PPO representability pilot;
   - existing 42-channel pilot cannot create margin: evaluate an explicitly
     widened observation contract before longer training.
6. **Development evaluation:** full 32 rows, not six shards.
7. **Final independent circle and all-shape audits:** untouched seed families.

This sequence is designed to falsify the expensive plan as early as possible.

---

## 14. Zero-residual seed-41 result: PPO training is paused

The mandatory 32-row baseline sweep completed with the expected diagnostic
contract:

- policy label: `forced_zero_residual`;
- `force_zero_policy: true`;
- `certifiable: false` by design;
- 32 rows: five Monte Carlo plus 27 h-sweep;
- one fresh process per rollout;
- c7 SHA-256 bound correctly;
- corrected clearance enabled;
- 128 environments and 120 scored steps.

### Acceleration distribution

| Statistic | Zero-residual row p95 acceleration |
|---|---:|
| minimum | 37.0741 m/s2 |
| 25th percentile | 37.5132 m/s2 |
| median | 38.2736 m/s2 |
| mean | 38.6676 m/s2 |
| 75th percentile | 39.6507 m/s2 |
| 90th percentile | 40.8023 m/s2 |
| 95th percentile | 40.9045 m/s2 |
| maximum | 41.6825 m/s2 |

- 20/32 rows, or 62.5%, exceed the desired 38 m/s2 development margin.
- 6/32 rows, or 18.75%, exceed the formal 40 m/s2 gate.

The formal baseline failures are:

| Suite | h | round | seed | acceleration |
|---|---:|---:|---:|---:|
| Monte Carlo | — | 2 | 300049 | 41.6825 m/s2 |
| Monte Carlo | — | 5 | 300064 | 40.9545 m/s2 |
| h sweep | 0.6 | 2 | 400049 | 40.8367 m/s2 |
| h sweep | 0.7 | 2 | 400049 | 40.4927 m/s2 |
| h sweep | 0.9 | 3 | 400054 | 40.8635 m/s2 |
| h sweep | 1.2 | 1 | 400044 | 40.0099 m/s2 |

Monte Carlo and h termination-confidence statistics still pass, as do tracking,
head, corrected-clearance and truck point gates. Acceleration is the isolated
baseline defect.

### Decision

The zero-baseline distribution is the broad high-margin regime that the
pre-training gate was designed to detect. The earlier proposed circle-focused
PPO run is **not funded** at this stage.

Suppressing PPO cannot solve six formal baseline failures. Achieving <=38 would
require active cancellation over most of the 32-row population, despite the
actor lacking direct measured acceleration. That is no longer a reasonable
first intervention.

### Revised root cause

PPO can amplify particular failures, as proven on seeds 300037 and 300042, but
it is not the primary population-level cause. The zero-policy baseline itself
has a broad configuration-dependent acceleration tail.

The current root-cause hierarchy is:

1. **Primary:** c7 reference plus unchanged inner-loop/ROM/PhysX closed-loop
   response produces insufficient acceleration margin across the circle
   population.
2. **Secondary:** PPO residual can add another 1-2 m/s2 on some conditions and
   remove margin, but other conditions previously showed PPO reducing the tail.
3. **Not supported:** one ROM factor, correlated h, clearance error, sequential
   state leakage or gross tracking failure.

### Next work before any PPO training

1. Compare position-difference acceleration against a cutter-point acceleration
   derived from PhysX body linear/angular velocity. The existing offline
   position check proves implementation consistency but not immunity to
   high-frequency PhysX position noise.
2. Report sensitivity to a separately labelled, predeclared low-pass or
   Savitzky-Golay diagnostic. Do not change the formal gate from this analysis.
3. On a small set of zero-baseline failures, log per-cylinder position, velocity,
   goal, valve and ROM increment by geometric phase to localize whether the tail
   enters through reference interpolation, DLS feed-forward, hydraulic ROM or
   rigid-body response.
4. Test reference-side changes only while respecting the standing requirement
   that the inner controller remain unchanged. A candidate must be rejected on
   cheap shards and accepted only on a complete 32-row development suite.
5. If no reference-side candidate creates baseline margin, report that the
   unchanged-controller constraint and the unfiltered 40 m/s2 metric are jointly
   incompatible for this circle. Do not force PPO or AAM to hide that result.

### Authoritative artifact

`reports/circle_c7_zero_seed41/rom_b_held_out_report.json` is a diagnostic
baseline artifact. It is intentionally non-certifiable but decisive for choosing
the next engineering intervention.
