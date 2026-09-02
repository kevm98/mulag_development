# Pre-registration — dose–response: does adaptation track the size of the plant change?

Fixed **2026-09-01 09:10 UTC**, before any dose–response arm was scored.

## Question

Not "does the method work on average" — that is the seed table, and it is
answered (2 of 5). This asks a different and more informative question:

> **Does the benefit of the online estimate scale with the size of the
> mid-episode plant change, and does it disappear when there is no change?**

## Design

The dose is the magnitude of the mid-episode mass flip. For dose `X`:

```
MULAG_MASS_BIMODAL=1.0,X    MULAG_MASS_STEP_SCALE=X
```

Doses: **1.0, 1.5, 2.0, 2.5, 3.0**. Everything else — jitter, flip instant,
payload, meshes, seed, env count, steps — is identical across doses.

Arms at each dose: **base** (`--residual_alpha 0.0`) and **adaptive** (seed 17,
`model_399`, `--residual_alpha 0.5`), paired on the same environments.
Two meshes (1337, 17), n=352 per dose point. 20 evaluations.

Seed 17 is used because it is the arm the mechanism claim rests on; this is a
characterisation of that arm, not a fresh test of the median-seed hypothesis,
and it is reported as such.

## The dose = 1.0 point IS the sham

At dose 1.0 both bimodal modes are 1.0: **the flip schedule fires on the same
jittered instants and writes the same mass, but the mass never changes.** This
is exactly the sham control of condition 3, embedded in the curve.

## Predictions, fixed in advance

1. **At dose 1.0 the adaptive–base gap is near zero.** There is nothing to adapt
   to, so a residual that is genuinely adapting has nothing to recover.
2. **The gap grows with dose** over the range where the machine still functions.
3. **The gap may fall at dose 3.0**, past the 2.5–3.0 functional-breakdown band,
   because a residual cannot recover headroom from a base that has already
   failed.

## What would falsify the adaptation claim

**A large adaptive–base gap at dose 1.0.** If the arm beats the base by a
similar margin when the mass never moves, then the advantage is better tracking,
not adaptation, and the mechanism claim fails — regardless of how good the seed
table looks. Under the OLD objective the sham showed +29.3%, so this is a live
risk, not a formality.

This is the primary falsification test and it is declared here before the run.

## Reporting

The full curve is reported with all five dose points for both metrics, whatever
shape it has — including a flat curve, or one that peaks at dose 1.0. A curve
that does not match the predictions above is reported as refuting them.
