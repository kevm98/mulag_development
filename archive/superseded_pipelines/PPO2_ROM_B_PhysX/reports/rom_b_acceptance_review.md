# ROM-B acceptance statistic: review, corrections, and revised fix plan

Response to the review of `rom_b_root_cause.md` / `rom_b_remediation_contract.md`.
Verdict up front: **your central objection is correct, my headline claim was wrong,
and the underlying reason is stronger than either of us stated.** New evidence
below changes the fix plan in a way that also makes it cheaper.

---

## 1. Conceded without reservation

**1a. Global pooling across `h` does not certify per-`h`.** Correct, and my framing
leaned on it. Pooling 3456 line samples across all nine `h` values estimates the
marginal rate over the tested `h` mixture. A weak `h` can be masked by strong ones
even with shapes kept separate. Table B in my earlier reply should not have been
presented alongside the per-`h` result as if it were corroborating.

**1b. "True rate" was the wrong word.** `0.0376` is an observed pooled estimate
under one test design, not a property of the policy. Section 3 shows this matters
more than a wording quibble.

**1c. The acceleration estimand.** You are right that a pooled p95 cannot be
reconstructed from three per-rollout p95 summaries, that within-environment time
samples are autocorrelated, and that max-of-three-p95 is conservative rather than
wrong. Changing it changes the estimand and would need predeclaration. **I now
recommend not changing it** — see section 5.4.

**1d. Sample-size arithmetic.** `1024` per `(h, shape)` is `2.67x` the current 384,
not `8x`. I was quoting a ratio against a single 128-environment group while
discussing per-condition totals. Section 3 retires that recommendation entirely for
a more fundamental reason, but the arithmetic correction stands.

---

## 2. New evidence: the replicas are not exchangeable on the failing shape

Your remark that "heterogeneity across trajectories and initial conditions should
be acknowledged" is testable from the existing artifacts. I tested it.

Decomposing the variance of the five Monte Carlo seed-group rates per shape, with
`Var(group rate) = p(1-p)/n + sigma_b^2`:

| shape | p | Var observed | Var binomial | sigma_b |
|---|---:|---:|---:|---:|
| **line** | 0.0344 | 5.37e-04 | 2.59e-04 | **0.0167** |
| arc | 0.0172 | 1.34e-04 | 1.32e-04 | 0.0015 |
| sine | 0.0016 | 1.22e-05 | 1.22e-05 | 0.0001 |
| circle | 0.0000 | 0 | 0 | 0 |
| square | 0.0219 | 1.04e-04 | 1.67e-04 | 0 (truncated) |

Line's group rates are `0.0547, 0, 0.0234, 0.0547, 0.0391` — a between-group SD of
about **1.7 percentage points on a mean of 3.4 %**. Every other shape is
binomial-consistent. **The one shape with excess between-group variance is the one
shape that fails.**

Treating the seed group as the unit of analysis (one-sided 95 % t bound on the
mean of the group rates, `m-1` df):

| shape | m | mean | sd | cluster-robust upper | CP95 pooled | |
|---|---:|---:|---:|---:|---:|---|
| **line** | 5 | 0.0344 | 0.0232 | **0.0565** | 0.0487 | **FAIL** |
| arc | 5 | 0.0172 | 0.0116 | 0.0282 | 0.0283 | PASS |
| sine | 5 | 0.0016 | 0.0035 | 0.0049 | 0.0074 | PASS |
| circle | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0047 | PASS |
| square | 5 | 0.0219 | 0.0102 | 0.0316 | 0.0340 | PASS |

**I retract the claim that `model_53` passes a properly constructed test.** It does
not pass even the Monte Carlo suite once the seed group is the unit. My CP95 of
`0.0487` treated 640 outcomes as exchangeable Bernoulli draws; on line they are not.
Your verdict — not accepted — is right, and it holds on the Monte Carlo suite too,
not only per-`h`.

### 2.1 Honest limits of this evidence

* `sigma_b` for line is estimated from `m=5` groups, i.e. **4 degrees of freedom**.
  The point estimate is very uncertain.
* Formally, binomial cannot be *rejected* for line: the quasi-binomial dispersion is
  `phi = 2.07` with `chi2 p = 0.082`. Across all 43 conditions the median `phi` is
  `0.83` and the mean `1.12` — on average unremarkable — though line accounts for
  four of the highest values (`phi = 3.39, 2.50, 2.07, 1.72`).
* Method-of-moments `sigma_b` is truncated at zero, which biases the reported
  values upward slightly.

So the correct statement is not "line is overdispersed" but **"with `m = 3` to `5`
groups we cannot tell, and the failing shape is where the evidence for
heterogeneity concentrates."** For a safety gate, the right default under that
uncertainty is the conservative (cluster-robust) estimand, not the independence
assumption. That is a design choice and it must be predeclared, not selected after
seeing which one passes.

---

## 3. This retires the "more environments" recommendation

Under a random-effects model
`SE = sqrt( p(1-p)/(m*n) + sigma_b^2 / m )`, evaluated at line's `p = 0.0376`,
`sigma_b = 0.0166`:

| design | total n | SE | t(.95, m-1) | 95 % upper | |
|---|---:|---:|---:|---:|---|
| m=3 x 128 | 384 | 0.01364 | 2.92 | 0.0774 | FAIL |
| m=3 x 384 | 1152 | 0.01110 | 2.92 | 0.0700 | FAIL |
| **m=3 x 1024** | **3072** | 0.01018 | 2.92 | **0.0673** | **FAIL** |
| m=5 x 128 | 640 | 0.01057 | 2.13 | 0.0601 | FAIL |
| m=6 x 128 | 768 | 0.00965 | 2.02 | 0.0570 | FAIL |
| m=9 x 128 | 1152 | 0.00788 | 1.86 | 0.0522 | FAIL |
| m=12 x 128 | 1536 | 0.00682 | 1.80 | 0.0498 | (marginal) |

Two conclusions:

* **My `n >= 1024` recommendation was wrong**, and not by the factor you identified.
  With `m` stuck at 3, `1024` environments per condition — 3072 samples, roughly 8x
  the evaluation cost — still yields an upper bound of `0.0673`. `sigma_b^2 / m`
  dominates and is untouched by `n`. Adding environments inside a fixed number of
  seed groups buys almost nothing.
* **Certifying the current policy is effectively infeasible.** Even `m = 12` only
  reaches the threshold, and the `t` penalty at small `m` is severe (2.92 at `m=3`
  versus 1.80 at `m=12`).

At an improved rate of `0.025` with `sigma_b` scaled proportionally:

| design | total n | 95 % upper | |
|---|---:|---:|---|
| m=3 x 128 | 384 | 0.0548 | FAIL |
| **m=5 x 128** | **640** | **0.0418** | **PASS** |
| **m=6 x 128** | **768** | **0.0395** | **PASS** |
| m=9 x 128 | 1152 | 0.0360 | PASS |

**The measurement fix alone cannot rescue `model_53`. Remediation training is
required.** And the `<= 0.025` line target we independently converged on is now
justified for a principled reason: it is the rate at which a feasible design
(`m = 5-6` seed groups at 128 environments) certifies with margin.

---

## 4. Where I disagree with your proposal

Your rule — CP95 upper `<= 0.05`, pooled only across seed groups, separate per
shape, separate per `h`, worst shape controls — is a large improvement on
max-of-rows and I support four of its five elements. My objection is to one:

**Plain Clopper–Pearson pooled over `m = 3` seed groups repeats my error at finer
granularity.** It treats the 384 outcomes in an `(h, shape)` cell as exchangeable
Bernoulli draws. Section 2 shows that assumption is exactly what fails on line, and
per-`h` cells have only three groups — two degrees of freedom for any
heterogeneity check, so the assumption is unverifiable at that granularity.

Concretely, on Monte Carlo line your rule gives `0.0487` (pass) where the
cluster-robust bound gives `0.0565` (fail). The rule would accept a policy the
conservative estimand rejects.

Two changes, both small:

1. **Make the seed group the unit of analysis.** Compute the bound from the `m`
   group rates, not from the pooled Bernoulli count. A one-sided `t` bound on the
   group rates is the simplest defensible choice; a cluster bootstrap over seed
   groups is equivalent in spirit if you prefer it. Keep everything else — 0.05,
   per shape, per `h`, worst shape controls.
2. **Raise `m` from 3 to at least 5** in the h sweep (`--h_sweep_rounds 5`). With
   `m = 3` the `t` multiplier is 2.92 and no realistic policy certifies. This costs
   `5/3` of the current h-sweep runtime — far cheaper than the environment-count
   route, and statistically strictly better.

I am not confident enough in `sigma_b` to insist the cluster-robust bound is the
*only* defensible estimand. If you prefer plain CP, that is arguable — but then the
exchangeability assumption should be stated explicitly as a limitation, and the
`phi` diagnostic reported alongside each condition so a reader can judge it.

---

## 5. Revised fix plan

Your sequence is sound; these are amendments, not a replacement.

### 5.1 Predeclare the statistic — with the cluster unit named

Before any further run, fix in writing:

* Estimand: one-sided 95 % upper bound on the termination rate, **seed group as the
  unit of analysis**, `t` bound on `m` group rates with `m-1` df.
* Granularity: one bound per shape (Monte Carlo), one per `(h, shape)` (h sweep).
  No pooling across `h`, no pooling across shapes.
* Acceptance: worst shape controls. Threshold `0.05`, unchanged.
* `m >= 5` seed groups per condition.
* Report `phi` per condition so the independence assumption is auditable.

### 5.2 Demote the observed seeds to diagnostic — agreed

Base seed 17 (Monte Carlo 300017–300041, h sweep 400017–400031) is burnt. Final
validation uses the predeclared unseen bases **61** and **149**.

### 5.3 Run the `model_607` control arm — agreed, and now more informative

With the revised statistic, compare `model_607` and `model_53` per `(h, shape)` on
the identical diagnostic seeds: point estimate, cluster-robust upper bound,
termination causes, tracking, acceleration. This is the only thing that can support
or refute a robustness claim for ROM-B training, and it needs no training. Given
that the same failure modes appear at `randomization_scale = 0.0`, an honest
negative result remains a live possibility.

### 5.4 Leave the acceleration statistic alone

Reversing my earlier recommendation, for three reasons: the pooled p95 is not
reconstructible from the stored summaries; within-environment autocorrelation means
a naive pooled p95 would need a cluster bootstrap to be honest; and circle sits at
`36.0 +/- 2.1 m/s^2` against a 40 gate, so acceleration is not the binding
constraint. Max-of-groups is conservative and already predeclared. **Change exactly
one statistic — the one that is broken.** If you want the pooled version later, the
diagnostics instrumentation now retains per-step samples and `evaluate.py` would
need to persist them; that is a separate, predeclared decision.

### 5.5 One remediation experiment, one variable

`--train_safety_terminations`, nothing else. Target: line `<= 0.025`. Shaping
changes become experiment #2 only if #1 misses — and section 6 has already narrowed
what that shaping change should be.

### 5.6 Randomization contract and architecture — unchanged

Six factors, independent uniform, one scale parameter, four-stage curriculum,
factor-blind 42-observation policy, 2 mm residual authority, `h` evaluation-only.
Nothing in the diagnosis implicates any of it.

---

## 6. Smoke test, and what it already shows

`diagnose.py` ran end to end under Isaac Lab (headless, `cuda:0`, 32 environments,
60 scored steps, `line`, Monte Carlo seed 300017). Gate metrics were produced
normally and every recorder signal read cleanly from the live environment.

The path profile from that single short run is already informative:

| path bin | s (m) | mean clearance | p95 | max | acc95 |
|---|---|---:|---:|---:|---:|
| **0** | 0.05–0.29 | **0.0620** | **0.0845** | **0.1205** | **34.65** |
| 1 | 0.29–0.54 | 0.0308 | 0.0478 | 0.0795 | 14.42 |
| 2–4 | 0.54–1.27 | 0.015–0.021 | 0.018–0.031 | 0.019–0.035 | 2.8–15.3 |
| **5** | 1.27–1.51 | 0.0201 | 0.0325 | 0.0493 | **33.17** |
| 6–9 | 1.51–2.48 | 0.014–0.021 | 0.020–0.030 | 0.023–0.042 | 17.3–27.7 |

Two things follow, both bearing directly on the fix plan:

* **Line clearance is a startup transient, not a biased operating point.** Mean
  clearance in the first path bin is 3–4x every later bin, with the largest
  acceleration; there is a secondary peak at bin 5, the traverse reversal near the
  midpoint of the 2.5 m path. Steady-state mean clearance is `0.014–0.031 m`, not
  the `0.19 m` that the audit's *max* statistic suggests.
* **This resolves the conditional in the remediation contract §2.1 in favour of the
  excursion branch.** Widening the clearance shaping onset from `0.180` to `0.120`
  would be wrong — it would penalize normal operation, which sits an order of
  magnitude below that. If shaping is touched at all it should be the tail, and the
  first candidate is not a reward change but the entry transient itself: the
  preposition/settle exclusion evidently does not cover the acceleration into the
  path.

Caveat: this run produced zero terminations, so it characterizes the *surviving*
population and the `recorder.capture` path is exercised only by unit tests, not yet
against the live environment. The step-1 diagnostic run on the failing seeds will
exercise it.

### Verification status of the workspace changes

| item | status |
|---|---|
| 19 offline contract tests (4 new) | pass |
| `train.py`, `diagnose.py`, `evaluation.py` byte-compile | pass |
| New argument validation rejects bad input before Isaac launches | verified (`--remediation` without checkpoint; seed offset 300000) |
| `diagnose.py` end-to-end under Isaac Lab | **verified** |
| Gate metrics unchanged when `collect_diagnostics=True` | verified structurally (same five tensors) + live run |
| `recorder.capture` against the live environment | **not yet exercised** — no terminations in the smoke run |
| `--train_safety_terminations` throughput | **not measured** — must be checked in the first rounds |
| `--remediation` end to end | **not exercised** |

The last three should be cleared before the long run, as you said.

---

## 7. Net position

* You are right that global `h` pooling cannot certify per-`h`, and right that
  `model_53` is not accepted. I withdraw the contrary claim.
* The reason is stronger than the one given: on line — the failing shape — the
  seed-group heterogeneity is large enough that `model_53` fails the Monte Carlo
  suite too under a cluster-robust estimand.
* My `n >= 1024` recommendation is withdrawn. The binding quantity is the number of
  seed groups, not environments per group; `m = 3 x 1024` still fails.
* I am asking you to strengthen one element of your own proposal: make the seed
  group the unit of analysis and raise `m` to at least 5. Otherwise your rule
  accepts on line at `0.0487` where the conservative estimand rejects at `0.0565`.
* Everything else in your sequence I support as written, and the randomization
  contract and architecture stay untouched.
