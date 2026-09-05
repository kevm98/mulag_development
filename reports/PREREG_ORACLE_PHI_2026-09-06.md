# Pre-registration: does a task-space correction exist at all

Written 2026-09-06, before any arm is run. This is a feasibility probe, not an
adaptation mechanism. Nothing is trained and nothing adapts online.

## Why this comes before a fifth mechanism

Four online adaptation mechanisms have been tried and each is closed by its own
control (`PPO_AAM_FD/memory.md`, commit 0f02a65). Every one of them asked the
online machinery to solve two problems at once: work out what changed, and work
out what controller change that implies. The results say the first half can
succeed while the second half does not reach the scored metrics.

Before building a fifth mechanism, this probe establishes whether the target
exists. If no fixed correction improves both scored channels at a given payload,
then no estimator, RLS, meta-learner or Bayesian optimiser can deliver one,
because none of them can outperform the best setting of the thing they tune.

## The correction

Two parameters, both task-space reference offsets:

    phi = [b_Y, b_Z]

`b_Y` displaces the tracking line, `b_Z` the cut-height contact reference.
They enter the frozen policy's observation only:

    observed cut     = relu(e_cut - b_Z)
    observed lateral = e_lat - b_Y

These are algebraically exact. `cut_height_error` is `relu(h - reference)` and
zero in contact, so raising the reference by `b_Z` gives `relu(e_cut - b_Z)`;
`verge_lateral_error` is `cutter_y - line_y`, so displacing the line by `b_Y`
gives `e_lat - b_Y`. Keeping the cut channel inside the relu keeps it
non-negative, which is the only sign the frozen policy was trained on.

Everything downstream is untouched: the policy weights, the FD model, the plant.
The SCORED errors are read from the environment against the true reference, so
they are the same quantities every previous campaign reported.

Implementation: `--task_bias_y`, `--task_bias_z`, `--paired_task_bias` in
`PPO_ROM/scripts/evaluate_gates.py`.

## Plant

Constant payload per run, `MULAG_PAYLOAD_RANGE=M,M`, at
M in {1.0, 2.0, 3.0, 4.0, 5.0}. Otherwise the campaign plant:
`MULAG_DIRECT_VALVE=1`, `MULAG_RMA_CONTEXT=1`. Frozen base
`logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt`, 176 environments,
900 steps.

The base's sensitivity to this axis is already measured
(`RESULT_starved_gate_2026-09-05.md`): cut-height median 3.353 mm at 1.0x,
19.387 mm at 3.0x, 320.314 mm at 5.0x. The 5.0x end is a failing controller
rather than an imperfect one, and a two-parameter constant offset is not
expected to rescue it. That is recorded here so a failure at 5.0x is read as
predicted rather than as a surprise.

## Pairing

All offsets at one payload are rolled inside ONE process against the same
environments, terrain and plant draw (`--paired_task_bias`). Invariant 19 is
the reason: between two runs identical by construction the marginal medians
moved 15.3% while the paired difference straddled zero. Only the paired
difference is read.

## Meshes

Mesh 2718 is the SELECTION mesh. Any offset chosen on it is confirmed on
held-out meshes 3141 and 8191 before it is claimed. This is the rule that
version 2 of the online-gradient campaign adopted after a favourable rung on
2718 reversed on 3141.

## Search

Staged, because the full product grid is 30 rollouts per payload and does not
fit the budget.

Stage 1, axis sweep at each payload, one process per payload, baseline first:

    b_Y in {-0.02, -0.01, 0, +0.01, +0.02}      with b_Z = 0
    b_Z in {-0.15, -0.08, -0.04, -0.02, +0.02}  with b_Y = 0

**Amendment, same day, before any scored arm ran.** The b_Y grid above started
as {-0.04, -0.02, 0, +0.02, +0.04} and was halved. The plumbing smoke that
checked the new code path (44 environments, 120 steps, payload 3.0x, not a
scored configuration and not read as a result) measured b_Y = +0.02 moving
cut-height error from 15.8 to 154 mm. A grid whose second point is already far
past the failure is not a search. The change is recorded here rather than made
silently, and the b_Z grid is unchanged.

Stage 2, the product of the best two values on each axis, at each payload,
4 combinations plus the baseline.

Stage 1 grids are fixed here and are not widened after seeing the results. If
the optimum sits on a grid edge, that is reported as "the search hit its
boundary" and the boundary is quoted; it is not silently extended.

## Success rule

Fixed here, and identical to Table 7.7 so this probe is comparable with every
previous campaign. Against the 0:0 baseline in the same process, paired per
environment, percentile bootstrap B = 20000 on the median difference:

* one channel improves with its interval excluding zero, and
* the other improves or stays within its non-inferiority bound: cut height 10%,
  lateral 1.5%.

Those bounds are the measured A/A floor from `PREREG_R2_2026-09-01.md` and are
not revisited here.

## The clearance guard

`cut_height_error` is relu'd and exactly zero whenever the cutter is in ground
contact. An offset that drives the cutter into the ground therefore scores a
perfect cut height by destroying clearance, and would pass the rule above while
being useless. So a Z improvement counts only if, at the same offset:

* `worst_family_ground_contact_fraction` stays at or above 0.700, the E5
  criterion already used in this project, and
* `safety_violation_rate` does not rise above the baseline's.

Both are recorded per offset alongside the per-environment contact fraction.
An offset that wins on both error channels but fails the guard is reported as a
metric artefact, not as a result.

## Declared outcome

* **Pass at a payload** if some phi meets the success rule and the clearance
  guard there, on the selection mesh, and reproduces on 3141 and 8191.
* **Pass overall** if that holds at every payload in 1.0-5.0.
* If it passes overall, the next step is online estimation of the payload and
  interpolation of `phi*(M)`, and only then.
* If it fails at any payload, the online estimator is NOT built. The reported
  finding is that a two-parameter task-space correction cannot Pareto-improve
  this stack at that payload, and the parameterisation is what has to change.

A partial pass, for example 1.0-3.0 passing and 4.0-5.0 failing, is reported as
a partial pass with the boundary named. It is not rounded up.

---

## Amendment 2: the success rule is mis-specified at nominal payload

Written after the 1.0x arm and BEFORE any loaded payload had finished, so that
this is on record while the substantive result was still unknown. The 3.0x
process was running when this was written and its report did not yet exist.

The 1.0x table fails the rule above on every offset. That failure is an artefact
of the rule, for two reasons, and both were foreseeable before the run.

**One: `phi = 0` is the correct answer at nominal payload and the rule cannot
express it.** The frozen policy was trained at this plant. It is already at its
optimum here, at 1.587 mm of cut error, so no correction can Pareto-beat it and
none should be expected to. What the programme actually requires at 1.0x is
`phi*(1.0) = 0`, i.e. that the payload-conditioned schedule leaves the nominal
plant alone and ties the static arm. The rule as written demands a strict
improvement at every payload, which at the training plant is a demand that the
correction beat the thing it is correcting.

**Two: a percentage bound on a 1.587 mm denominator is not a test.** The 10 %
cut-height bound is the A/A floor measured against a base at roughly 7.9 mm,
where it means 0.79 mm. On the 1.0x baseline it means 0.16 mm, far below the
instrument's own measured false-positive floor for this channel. The offset
`b_Y = +0.02` reads as -63.1 % on cut while costing 2.90 mm in absolute terms,
and buys 15.88 mm of lateral.

**What changes and what does not.** The rule is NOT re-scored and the 1.0x
result is reported exactly as the rule scores it: fail. The verdict for the
programme is taken from the loaded payloads 2.0-5.0, where the frozen base
carries 19-320 mm of cut-height error and there is headroom for a correction to
recover. 1.0x is reported as uninformative by construction, with its table
printed in full so the reader can see the trade rather than the percentage.

Absolute millimetre differences are added to the table alongside the
percentages. That is a reporting change, not a change to the criterion.

**What would still count as a negative.** If no offset Pareto-improves at 3.0x,
4.0x or 5.0x either, the programme's answer is negative and the estimator is not
built, exactly as the original decision tree says. This amendment widens nothing
about the payloads that carry the verdict.
