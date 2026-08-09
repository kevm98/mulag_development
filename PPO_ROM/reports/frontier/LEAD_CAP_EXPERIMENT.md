# Lead-cap experiment: does the anti-windup clamp bound boom_sweep tracking?

**Date:** 2026-08-06
**Subject:** `logs/rsl_rl/ppo3_1_sole/student_distilled.pt` (distilled student)
**Meshes:** 1337, 17, 909 (held out) · **32 envs · 480 steps** · headless
**Reports:** [`student_lead05.json`](student_lead05.json) (treatment) vs
[`student_family.json`](student_family.json) (baseline)

---

## 1. The question

The 8-gate contract fails `lateral` because of a single path family,
`boom_sweep`. The hypothesis under test:

> `boom_sweep` requires a *sustained* lateral offset. The anti-windup lead cap
> bounds how far a commanded cylinder target may lead the measured position, at
> 0.018 m on c1-c3. If that clamp is the binding constraint, relaxing it should
> reduce boom_sweep's lateral error.

Treatment: `--cylinder_target_max_lead_per_axis_m 0.05,0.05,0.05,0.30`
(c1-c3 relaxed 2.8x; c4 held at its existing 0.30).

**Decision rule as originally stated:** if boom_sweep drops below 0.30, the gate
measures a fixable plant limit; if it stays above, the path is infeasible and
the gate must be re-derived.

## 2. The addition to that decision rule

The decision rule above is necessary but not sufficient, and the reason is
recorded history rather than caution in the abstract:

> boom_sweep dropping below 0.30 shows the lead cap is a binding constraint —
> but the cap exists for a reason: relaxing it 0.018 → 0.026 was previously
> measured to cost ~0.30 of head alignment on unseen meshes, and this run
> relaxes it nearly 3× further. So report `p01_head_alignment` and
> `hard_impact_step_fraction` alongside the lateral numbers. A configuration
> that fixes lateral by breaking the head gate isn't a fix, it's a different
> failure — and that trade is precisely why the 0.026 change was reversed last
> time. The c4 axis is held at 0.30 here, which is the right call given the
> measured head collapse comes from relaxing c4 specifically.

In other words the experiment has **two** ways to fail: boom_sweep may not
improve, or it may improve while the safety gates collapse. Both were checked.
The 0.026 reversal is the precedent: a lateral gain of 0.034 m was abandoned
because it cost ~0.30 of head alignment on meshes other than the pinned one --
a trade only visible because the check was run on varied terrain.

## 3. Result: the hypothesis is falsified, on both counts

### 3.1 Lateral got worse, on every mesh

| mesh | worst-family lateral (0.05) | boom_sweep (0.05) | boom_sweep (0.018) |
|------|------|------|------|
| 1337 | 0.4849 | **0.4849** | 0.2800 |
| 17   | 0.3281 | **0.3281** | 0.3189 |
| 909  | 0.5134 | **0.5134** | 0.4046 |

boom_sweep is the worst family on all three meshes and remains above the 0.300
threshold everywhere. Provenance is recorded in each per-mesh report:
`cylinder_target_max_lead_per_axis_m: [0.05, 0.05, 0.05, 0.3]`,
`lead_cap_overridden: true`.

### 3.2 The safety gates collapsed as well -- the second failure mode

| metric (worst mesh) | lead 0.05 | baseline 0.018 |
|------|------|------|
| p01 head alignment | **0.8814** (gate 0.900: FAIL) | 0.9597 PASS |
| hard impact step fraction | **0.2698** | 0.0678 |
| min cylinder headroom (m) | **0.0000** | 0.0387 |
| worst-family ground contact | **0.4722** | 0.7076 |
| safety termination rate | **0.1008** | 0.0611 |

Failing gates go from 3 to **6 of 8**. Head alignment fails outright, so the
"different failure" the addendum warned about did occur -- it simply arrived
alongside worse lateral rather than instead of better lateral. Note this
happened with **c4 held at 0.30**, confirming that c1-c3 alone are sufficient
to break the head gate; the prior attribution of head collapse to c4 relaxation
is not the whole story.

### 3.3 The family pattern inverts the hypothesis

Mesh 1337, `mean_abs_lateral_error_m`:

| family | 0.018 | 0.05 | |
|--------|-------|------|---|
| boom_sweep | 0.2800 | **0.4849** | sustained offset -- worse |
| lane_change_hold | 0.1102 | **0.1968** | sustained offset -- worse |
| offset_return | 0.0759 | **0.1633** | sustained offset -- worse |
| gaussian_slalom | 0.2818 | **0.1405** | fast reversal -- *better* |

All three families that `rewards.py` identifies as structurally requiring a
sustained non-zero offset degraded, roughly doubling. The one family that
improved is a fast-oscillation path.

That inversion has a direct mechanical reading: **relaxing an anti-windup clamp
reintroduces windup.** Windup harms held commands and is irrelevant to commands
that reverse quickly. So this run is evidence *for* the current 0.018 value
being correctly tuned, not against it.

Contact collapses on mesh 909 corroborate the bottoming-out picture:
`lateral_sine` 0.7868 -> 0.4722, `fourier_meander` 0.8156 -> 0.5427 -- families
that were healthy at baseline lose the ground entirely.

## 4. What this implies about boom_sweep

Given ~2.8x the command authority, the arm does **not** track boom_sweep better.
It runs out of stroke (`min_cylinder_headroom_m` = 0.0000 exactly, on all three
meshes), loses ground contact, and slams into the terrain at 4x the impact rate.

Authority spent without tracking improvement, terminating at the travel limit,
is the signature of a target **beyond kinematic reach** -- not of a command
clamp holding back an otherwise-capable arm. This is positive evidence for
infeasibility, not merely absence of evidence against it.

It is consistent with the independent finding that the privileged DLS teacher
also fails boom_sweep on all three meshes (0.2946 / 0.3253 / 0.3657), and with
the student having reached teacher parity on that family.

## 5. Conclusions and next step

1. **Do not adopt `0.05` for c1-c3.** It degrades lateral tracking and fails
   three additional gates.
2. **The 0.018 cap appears correctly tuned** for the sustained-offset families
   it protects.
3. **A claim in the codebase does not reproduce.** The help text for
   `--cylinder_target_max_lead_per_axis_m` asserts "boom_sweep 0.3745 -> 0.1602"
   from relaxing c1-c3. The same relaxation moves boom_sweep the other way on
   all three meshes here. Whatever configuration produced 0.1602, it is not this
   one, and the figure should not be relied on until reproduced.
4. **Next measurement is geometry, not simulation.** Compute boom_sweep's
   required lateral excursion at amplitude scale 2.2 against the arm's
   kinematic reach. It needs no GPU and settles infeasibility directly rather
   than by inference. If confirmed, the honest fix is to reduce boom_sweep's
   amplitude or remove it from the gate -- not to keep tuning the plant around
   a path it cannot reach.
