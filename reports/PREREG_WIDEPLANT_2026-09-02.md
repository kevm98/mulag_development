# Pre-registration — W1: widened plant distribution (the ceiling hypothesis)

Fixed **2026-09-02 05:55 UTC**, before any W1 arm exists.

## Disclosure of attempts, stated up front

This is the **sixth** configuration tried against the adaptation question:
R1, R2 (abandoned screen), R3, the triplet (full-obs history), the plant-history
variant (stopped before completion, never scored), and now W1. Every one of them
is reported, including the two negatives already in hand.

Because the search is ongoing, this pre-registration splits the meshes:

| role | meshes | use |
|---|---|---|
| **development** | 6006, 4711 | screening, iteration, any number of looks |
| **final test** | 2718, 3141, 8191 | used ONCE, for the final reported table |

No configuration is chosen on the final-test meshes, and the final table is
produced once. This is what keeps a searched-for result meaningful; without it
the sixth attempt would carry no evidential weight at all.

## The hypothesis

The triplet experiment returned: base -> capacity-matched control gains of
+63.9%, +77.2%, +90.9%, +1.3%, +95.0% on cut, and control -> adaptive succeeding
on only 1 of 5 seeds. The residual works; the working part is capacity.

The proposed cause is a **ceiling effect from an over-narrow plant
distribution**:

```
MULAG_MASS_BIMODAL=1.0,3.0     two modes
MULAG_MASS_STEP_TIME_S=20.0    one flip per 47.3 s episode
MULAG_PAYLOAD_RANGE=1.0,1.0    payload fixed
damping                        not randomized
```

Two modes and one switch. A fixed residual need not identify anything: it learns
a single hedge acceptable in both modes, captures most of the available
improvement, and leaves the online estimate nothing to add.

The supporting evidence is seed 42, the only seed where capacity bought nothing
(+1.3%, n.s.) and the only seed where history won decisively (+38.4% cut,
+124.9% lateral). Where the hedge failed, adaptation paid.

## The change

Add a **hidden, strong, per-environment plant axis** the hedge cannot cover:

```
MULAG_DAMPING_RANGE=0.6,1.6    cylinder joint damping, drawn per env index
MULAG_PAYLOAD_RANGE=0.9,1.3    continuous, was fixed at 1.0,1.0
```

Cylinder damping is measured strong in `MECH_SENSITIVITY_2026-08-29.md`, is
drawn once per environment index and cached (`terrain_env.py:423`), and is **not
a context column** -- the 4-wide RMA context carries authority, lag, delay and
mass scale only. It is therefore hidden plant identity that only the history can
reveal. With 256 training envs this replaces 2 plants with 256.

The mid-episode mass flip is retained unchanged: E4 measured the time-varying
component at 13.6% against 0.43% for static identity, so it remains the reason
online rather than reset-time estimation is required.

Arms also use `--plant_history` (encoder sees the 240-number plant signature,
not 1400 mostly-terrain numbers). **Two changes are composed here**, so W1 does
not attribute its result between them. What W1 does preserve is the comparison
that matters: within each triplet the control and the adaptive arm are identical
in every respect except whether the history they read is real or zeros.

## Predictions, fixed in advance

1. **The base -> control gain collapses** relative to the +64..+95% seen on the
   narrow plant. If one hedge can no longer cover the plant distribution, added
   capacity alone should stop capturing most of the improvement.
2. **control -> adaptive opens up**, succeeding on at least 3 of 5 seeds.
3. The frozen base gets WORSE on the widened plant than on the narrow one. If it
   does not, the widening did not actually make the task harder and the whole
   manipulation was inert.

**Falsifier: if the control still gains +60% or more on cut, the ceiling
hypothesis is wrong** and W1 is reported as refuting it.

## Procedure

0. **Base validation** (before any training): score the frozen base on the
   widened plant. If worst-family ground contact falls below the E5 criterion of
   0.700 the plant is past functional breakdown, and the ranges are reduced
   before anything is trained. Reported either way.
1. **Screen** on seeds **7 and 101** -- deliberately the two seeds where
   capacity dominated most (+95.0%, +90.9%), so the screen tests the mechanism
   where it is most adverse, not where it is most favourable. Development meshes
   only.
2. If the screen supports prediction 1, train the remaining seeds (17, 23, 42)
   and produce the five-seed table.
3. The final table is scored ONCE on 2718, 3141, 8191 and reported whatever it
   says.

## Success criterion — unchanged

Identical to `PREREG_TRIPLET_2026-09-01.md`: an arm succeeds if one metric
improves significantly (paired bootstrap CI excluding zero) while the other
improves or stays within its measured non-inferiority bound (cut 10%, lateral
1.5%). The claim holds only if the median of the five seeds succeeds.

## Reporting

The two existing negatives stand and are not withdrawn. If W1 passes, the
reported claim is explicitly conditional: online adaptation beats matched
capacity **on a plant distribution wide enough that a fixed hedge cannot cover
it**, and does not on the narrow one. That conditional is a stronger and more
useful engineering statement than an unqualified pass, and it is the honest
reading of both results together.

---

## RESULT — W1 ABANDONED AT STEP 0, 2026-09-02 06:15 UTC

W1 never reached training. Its step-0 validation refuted prediction 3, and a
re-reading of the plant code showed the design premise was wrong.

**Validation, paired on mesh 6006 against the identical narrow-plant run:**

| configuration | cut | lateral |
|---|---|---|
| damping only, 0.6-1.6 | **+19.8%*** easier | +0.5% |
| damping 0.6-1.6 + payload 0.9-1.3 | **+37.6%*** easier | -1.5% |

Ground contact 0.7983 -> 0.8075, terminations 0.000 throughout. The widened
plant is comfortably inside functional limits — and comfortably EASIER for the
frozen base, which is the opposite of prediction 3. A plant with more headroom
cannot defeat a fixed hedge.

**Why the premise was wrong.** `terrain_env.py:385` already states it:

> "A continuous range does not do this: one compromise profile is decent across
> a narrow continuum, so the value of sensing is small by construction and the
> experiment understates adaptation."

Adding continuous payload and damping ranges is the documented anti-pattern for
this question. Worse, cylinder damping is drawn once per environment INDEX and
cached (`terrain_env.py:423`), so it is a STATIC axis — and E4 measured static
plant identity at 0.43% against 13.6% for the time-varying component, a factor
of 32. W1 widened the weak channel, in the helpful direction.

**What this rules out.** The plant distribution is not the bottleneck. The
existing bimodal flip — half the envs light->heavy, half heavy->light, direction
unguessable, mid-episode — is already the strongest adaptation-favouring design
available here, and it is what the negative triplet result was measured on.

**Cost:** four evaluations, no training. The step-0 gate did its job.

**Where the search goes next.** Since the plant is not the bottleneck, the
remaining candidates are about how the encoder LEARNS, not what it observes:

1. `--plant_history` — 240-number plant signature instead of 1400 mostly-terrain
   numbers. **Never once executed**: the flag was broken until 2026-09-02
   (history_feature_dim computed and never passed). This is the largest untested
   lever and the one the repo's own code recommends.
2. RESeL-style encoder-specific learning rate, lower than the policy head.
3. Amplifying the TIME-VARYING channel (multiple flips per episode) rather than
   the static one — a code change, gated behind a new env var, and the only
   plant-side change consistent with E4's 32x finding.
