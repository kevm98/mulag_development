# ROM authority ladder — provenance and the open Stage 0 question

Ladder run 2026-08-28 on the GCP L4. Result:
`rom_authority_ladder.json`. Raw traces: `outputs/e5_rom_authority_ladder`,
`outputs/e5_authority_stage0`, `logs/e5_run` — all force-added to git and
sha256-verified against the VM (`raw_manifest/e5_ladder_raw.sha256`).

**Read this before citing the curve.** It records what the run actually
reported, including the part that is not a clean pass. Nothing here reinterprets
the numbers; it is the log, transcribed.

## Stage 0: one half passed, one half raised a fault

The authority factor demonstrably reaches the plant. From
`outputs/e5_authority_stage0/e5_authority.log`:

| alpha | applied/raw | mean abs raw (m) | worst lateral |
|---|---|---|---|
| 1.00 | 1.0000 | 0.000778 | 0.4608 |
| 0.01 | 0.0100 | 0.001119 | 0.1322 |

`|delta_applied|/|delta_raw|` equals alpha to four decimals, and the log closes
with `All arms applied the authority factor as configured`. The first failure
mode named in the handover — factor never arrives — is ruled out.

The second is not. The same check printed:

```
ppo_rom: lateral error x 0.3 from alpha=1.00 to 0.01
  -- STILL TRACKING -- harness fault, investigate before interpreting any sweep
```

Cutting authority to 1% made lateral error 3.4x *better*. The handover names
this exact case: "applied and still tracking (ratio = alpha, error flat) — the
second is a harness fault further along, not robustness." **That investigation
has not been done.** The ladder was run before it.

Two further facts from the same log, both relevant to why:

- `mean|raw|` **rises** as authority falls (0.000778 -> 0.001119). The commanded
  displacement is not held fixed across the ladder; the policy asks for larger
  raw increments when starved. Any reading of the curve as "the plant degrades
  under X% authority" assumes a fixed command that this table contradicts.
- Stage 0 ran **single-mesh**. `merge_gates.py` aborted on both arms with
  `acceptance needs >= 3 distinct terrain meshes, got 1`, so the Stage 0 gate
  merge never completed. The ladder itself used meshes 1337/17/909.

## The predeclared criteria, and which one was dropped

The handover predeclared **two** onset criteria: median lateral error >= 1.5x
its own nominal value, **or** ground contact >= 5 pp below nominal. Breakdown:
contact < 0.700, or safety termination > 0.080.

`summarise_rom_ladder.py` reports `primary_metric: ground_contact_fraction` and
`lateral_onset_criterion: "INAPPLICABLE - starved arm is stiller, deviation
metrics improve"`. The onset and breakdown alphas in the JSON are therefore
**contact-only**.

Dropping the lateral criterion may well be physically correct — a starved arm
moves less, so every deviation metric flatters it. But it was dropped *after*
the curve was seen, which is the move the handover predeclared thresholds to
prevent. Cite the contact-only result as contact-only, and say the lateral
criterion was set aside post hoc and why.

## What the curve says

Non-monotonic. Contact rises from 0.791 at nominal to 0.954 at alpha=0.2, then
falls, crossing 0.700 only at alpha <= 0.01. Contact-only onset is alpha=0.005.

The defensible statement today is: **on ground contact, with the lateral
criterion set aside, the ROM stack shows no degradation until authority is cut
below ~3% of nominal, and scores better at a fifth of nominal authority than at
nominal.** The stronger claim — that the plant tolerates a 99% authority cut —
is not yet supported, because Stage 0's STILL TRACKING flag and the rising
`mean|raw|` both point at the harness rather than at the plant.

An earlier commit message in this branch (62b80fa) stated the 99% conclusion
without this caveat. This file is the correction.

## What would settle it

Investigate the STILL TRACKING flag, as Stage 0 asked:

1. Determine why lateral error improves under starvation. The rising
   `mean|raw|` is the first suspect — check whether the command integrator winds
   up when increments are attenuated, so the ladder varies the request as well
   as the authority.
2. Re-run Stage 0 across >= 3 meshes so `merge_gates.py` completes and the gate
   verdict exists rather than aborting.
3. Only then decide whether the lateral criterion is genuinely inapplicable or
   whether the metric is measuring stillness. If it is stillness, say so in the
   thesis and name the replacement metric before re-reading the curve.

## The ladder ran on the nominal plant, not on combined_test

Established 2026-08-28 while auditing the VM for the AAM repair plan. Every
ladder result file records:

```
deployment_factor_enabled = False
factor_domain             = 'nominal'
factor_key                = ''
```

The cause is mechanical, not a choice made at the console:
`AAM_module/datasets/rom_deployment_stage31_v1_frozen/raw/manifest.csv` — the
7.9 KB file carrying the factor draws, read by
`verge_task/deployment_factors.py:30` — was never pushed to the VM. Passing
`--factor_domain` at all sets `deployment_factor_enabled=True` and crashes
without it, so `run_rom_authority_ladder.sh:42` defaults `FACTOR_DOMAIN` to
empty and says so in a comment.

**This does not invalidate the curve, but it renames it.** Holding the plant
fixed and varying authority alone is the cleaner design for an authority axis,
and the script's comment argues exactly that. What the ladder measures is
therefore an **actuator-authority axis on a fixed nominal plant** — not a
plant-variation axis. E5's chapter framing is "plant robustness envelope"; this
curve does not defend that framing on its own, and must not be captioned as
though it varied the plant.

It also means the alpha-1.00 row is a nominal-plant baseline, so the 11.0 pp
across-mesh contact spread at nominal is terrain variation alone, with no
factor draw contributing.

The manifest has since been pushed and sha256-verified on the VM, so a
combined_test re-run is now possible. Whether to re-run the ladder under
combined_test, or to keep it as a declared authority-only axis and add a
separate plant axis, is a design decision for the user — it changes what E5
claims.
