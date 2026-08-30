# Observability gate on the Phase 1 policy

**Verdict at 44 pairs: NULL.** The ordering is exactly as predicted and every
point estimate favours the privileged context, but no confidence interval
excludes zero. Under the standard set for this project that is not a positive
result.

## Subject

`logs/rsl_rl/ppo_id_aam_fd/2026-08-29_15-20-58/model_399.pt` -- the first RMA
Phase 1 policy this project has trained successfully: 400 iterations on a
randomised plant (payload 1.0-1.5), privileged 4-D context live from iteration 0,
stable across five episode cycles (peak reward 19.75 -> 20.05, lateral ~42-46 mm,
ground contact 0.97-0.996).

## Instrument

Block-per-cycle per-env arms in ONE rollout, cycle DERIVED as
lcm(11 path families, 4 height scales) = 44, `MULAG_PAIR_PERIOD=44` so paired
environments draw the same plant. Payload 1.0-2.0.

**A/A control passed.** Four arms all fed the privileged context:

| arm | cut error median |
|---|---|
| privA | 7.439 mm |
| privB | 7.510 mm |
| privC | 9.047 mm |
| privD | 7.832 mm |

Paired differences -0.33 to -1.05 mm, **every CI includes zero**. The instrument
does not manufacture differences. Its noise floor is the thing to note: CI
half-widths ~2 mm on cut error at 44 pairs.

## Result

| arm | cut error median | vs privileged | 95% CI |
|---|---|---|---|
| **privileged** | **3.562 mm** | -- | -- |
| mean | 3.677 mm | -7.1% | [-43.4, +50.0] |
| shuffled | 3.911 mm | **+37.6%** | [-39.4, +88.5] |
| zero | 4.806 mm | **+49.3%** | [-7.6, +101.5] |

Ordering privileged < mean < shuffled < zero is precisely the hypothesis's
prediction, and the point estimates are large. **All CIs include zero**, so this
is a null: the effect may be real and the instrument too coarse -- 44 pairs
against a ~2 mm noise floor cannot resolve a ~1 mm difference -- but absence of
resolution is not evidence of effect.

## Instrument caveat: absolute numbers are not comparable ACROSS runs

The same `privileged` arm scored **7.44 mm** in the A/A run and **3.56 mm** in
the arms run: same policy, same environments, same seed. The payload draw is
seeded by `payload_scale_seed + _deployment_factor_reset_index`, so it is not
reproducible across runs whose reset patterns differ.

Within-run pairing is unaffected (the period-44 tiling holds), so both
comparisons above are valid. Cross-run comparisons of absolute values are not,
and none is made here.

**Fixed.** The payload draw is now seeded by the environment index alone, which
is the convention the other two randomisers already used:

| randomiser | seeding | reproducible |
|---|---|---|
| plant perturbation (alpha, delay) | `manual_seed(cfg.seed)`, drawn once | yes |
| cylinder damping | `manual_seed(seed)`, cached | yes |
| payload (before) | `manual_seed(seed + reset_index)` | **no** |
| payload (after) | `manual_seed(seed)` | yes |

A fresh draw per episode is still available for training diversity via
`MULAG_PAYLOAD_RESAMPLE=1`, opt-in and documented as breaking cross-run
comparability. Fixed per environment is also what RMA assumes when it treats the
environment vector as constant for an episode.

## Next

Pooling six meshes raises the pair count 44 -> 264, which should roughly halve
the CI width. If the effect is real at the measured magnitude it becomes
resolvable; if the CIs still span zero at 264 pairs the honest conclusion is that
the privileged context does not measurably help this policy, and Phase 2 should
not be built on it.

The **null (zero-action) arm** is still outstanding -- the first attempt failed
because `--null` scores a runtime mode and must not be passed a checkpoint. It is
not optional: Stage 0 recorded a do-nothing controller BEATING the frozen PPO
(4.4 vs 7.2 mm) because the selected episodes demanded almost no motion.

---

# Pooled result (264 pairs): NULL, and the reason is upstream

Six meshes, one fresh process each, pooled: **264 paired quadruples**.

| arm | cut height median | vs privileged | 95% CI |
|---|---|---|---|
| privileged | 6.900 mm | -- | -- |
| mean | 6.802 mm | -6.2% | [-20.0, +7.4] |
| **shuffled** | **6.518 mm** | **-13.4%** | [-22.3, +1.2] |
| zero | 7.098 mm | +6.7% | [-10.5, +17.4] |

Lateral: shuffled 50.26 mm also beats privileged 52.89 mm (paired median
+1.696 mm, CI [+0.005, +4.371] -- the only interval excluding zero, marginally,
in the WRONG direction, across six comparisons).

The 44-pair ordering (privileged best) was noise; powering up reversed it.

## Why: the policy is context-blind

Direct probe of the actor -- perturb the context columns, measure the action.

| | phase1_init (before Phase 1) | model_399 (after 400 iterations) |
|---|---|---|
| context weight abs-max | 0.000000 | **0.000367** |
| base column weights | 2.4049 | 2.4047 |
| mean action at zero context | 5.0488 | 5.0472 |
| delta action, context ~N(0,2) | 0.000000 | **0.000281 (0.01%)** |

Context weights are **6500x smaller** than base weights. The context moves the
action by 0.01%. **The policy never learned to use the context**, so
privileged / mean / shuffled / zero are the same input to it and the gate could
not have shown anything.

## The cause is the stability fix

Both settings introduced to stop the Phase 1 collapse also prevented learning:

* `--trajectory_max_reference_kl 0.10` anchors to the **distilled reference,
  which is context-blind by construction** -- its context columns were
  deliberately zeroed. Measured KL to reference: **0.0034**, 3% of the budget.
  The policy was never pushed away from context-blindness.
* `--trajectory_actor_learning_rate 3e-6` moved the context weights from 0 to
  3.7e-4 across 400 iterations.

"Phase 1 is stable" was true; "Phase 1 works" was not. Stability here largely
means the policy barely moved.

## Fix

The context columns are the ONLY new capacity and they start at exactly zero, so
they can have their own parameter group at a much higher learning rate while the
base trunk stays anchored at 3e-6. The policy can then acquire context
dependence without drifting the behaviour the KL anchor protects.

**Run the context-sensitivity probe before any ablation.** It costs seconds,
and it is the necessary condition for the gate to mean anything: if the action
does not move with the context, no arm comparison can.

---

# The context cannot be learned under trajectory regularisation

`--context_lr_multiplier 1000` on the context columns only (trunk left at 3e-6):

| checkpoint | context weight abs-max | delta action, context ~N(0,1) | % of action |
|---|---|---|---|
| model_0 | 0.000021 | 0.000037 | 0.001% |
| model_75 | 0.000100 | 0.000108 | 0.002% |
| model_150 | 0.000162 | 0.000183 | **0.004%** |

A thousand-fold gradient multiplier moved context sensitivity from 0.001% to
0.004% of the action scale.

**The cause is structural.** The reference-KL term penalises deviation from the
distilled reference, and that reference is **context-blind by construction** --
its context columns were deliberately zeroed. Any context-dependent behaviour
increases KL-to-reference, so the penalty gradient directly opposes context
learning. The multiplier scales the task gradient and the opposing penalty
gradient equally, so it moves faster toward the same context-blind equilibrium.

This is not a tuning problem. **The trajectory-regularised recipe is what makes
the fine-tune stable, and stability here means "do not change".** It cannot be
the vehicle for acquiring a new input dependence.

## Consequence

RMA Phase 1 as configured is a dead end on this stack:

* without the anchor, the fine-tune collapses (lateral 0.05 -> 1.25 m while total
  reward rose -- PPO optimising an objective that prefers planting the cutter)
* with the anchor, the policy cannot learn to use the context (0.004%)

The **frozen-policy residual design** avoids both horns: the base policy is
frozen so it cannot collapse, and the residual head is new capacity with no
anchor, so nothing opposes its dependence on the latent.
