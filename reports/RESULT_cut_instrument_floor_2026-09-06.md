# The cut-height instrument floor is far worse than recorded

Measured 2026-09-06 from an A/A that came free: the frozen base was scored at
payload 3.0x, mesh 2718, 176 environments, 900 steps, in two separate processes
on the same night with identical settings.

## The A/A

| | cut | lateral | contact |
|---|---:|---:|---:|
| Stage A `base_p3.0` | 16.70 mm | 99.66 mm | 0.788 |
| context-only `base_p3.0` | 23.56 mm | 79.48 mm | 0.783 |
| paired difference | **-40.9 %, CI excludes zero** | +7.7 %, ns | - |

Base at payload 3.0x across five nominally equivalent runs tonight:
**15.67, 16.70, 16.91, 23.31, 23.56 mm**. A 1.4x spread between runs that are
identical by construction, and the paired bootstrap calls one such pair
significant.

`PPO_AAM_FD/memory.md` invariant 18 puts the false-positive floor on cut height
at "~10 %", from an A/A that gave -1.7 %. That figure is optimistic. On this
evidence a cut-height effect below roughly 40 % is not distinguishable from the
instrument by a single pair of runs, however tight its interval looks. Lateral
is better but not immune: 99.66 / 79.48 / 93.06 mm across the same three runs.

## What it rescues

The construction check `base -> ctxzero` reported +78.4 % on cut and read as
"the freeze leaked". It did not. `ctxzero` scored **16.91 mm**, against Stage A's
base at **16.70 mm** -- agreement to 1 %. The outlier is the context-only
campaign's own base run at 23.56 mm. The freeze is intact; the reference was
unlucky.

That is the same reasoning error the sweep A/A caught earlier, one level up:
a control that looks like it failed, where what actually moved was the thing it
was measured against.

## What changes

The primary comparison becomes **`ctxzero -> ctxlive`**, not `base -> ctxlive`.

Both are the same policy by construction, but `ctxzero` is the same CHECKPOINT
FILE scored in the same campaign, so it shares the run-level conditions of the
live arm. `base` is a different file from a different run and imports the whole
40 % spread above. The matched control was already in the design; it was merely
ranked second.

`base -> ctxzero` is retained as a printed row and relabelled: it is the
INSTRUMENT floor for that pair of runs, and any effect in the primary row
smaller than it is not readable.

## What this does not excuse

It does not license reading a favourable primary and dismissing an unfavourable
one as noise. The floor is declared here, before the live arm was scored, and it
applies symmetrically: an effect under ~40 % on cut is not a result in either
direction, and the honest report of such an outcome is "below the instrument".

Lateral, contact, hard impacts and the violation rate carry more of the weight
for exactly this reason, and the root-cause report already established that on
this plant those are the mechanism rather than side statistics.


## The consequence for the success rule, and a better channel

The pre-registered rule (Table 7.7, and every campaign in this project) allows
the non-primary channel to worsen by up to **10 % on cut height**. That bound
came from an A/A that measured -1.7 %.

If the instrument's own floor on cut is nearer 40 %, the 10 % bound is not
operable: an arm can breach it by noise alone, in either direction, and no
single pair of runs at n = 176 can verify non-inferiority at a resolution four
times finer than the measurement. Every "trade-off (cut worse)" verdict in this
project that turns on a cut movement under ~40 % is therefore unsafe, including
ones recorded tonight.

### Ground contact is the same physics with a thirteenth of the noise

Across every nominally equivalent frozen-base run at payload 3.0x tonight:

| quantity | values | spread |
|---|---|---|
| cut median | 15.67, 16.70, 16.91, 23.31, 23.56 mm | ~40 % |
| ground contact | 0.776, 0.783, 0.788, 0.795, 0.803 | **~3 %** |

`RESULT_rootcause_contact_2026-09-06.md` already measured that cut-height error
is almost entirely contact loss: corr(contact, log cut) = -0.975 across arms and
-0.909 across environments within one arm. `cut_height_error` is
`relu(h - reference)` and exactly zero in contact, so it is a censored,
heavy-tailed function of the same underlying quantity -- which is precisely why
its median is unstable while the fraction it derives from is not.

**Recommendation.** Score ground contact fraction as the primary vertical
channel and report cut height alongside it. It measures the same mechanism, it
is what the E5 criterion already gates on, and it is roughly thirteen times more
stable. Cut height in millimetres remains the quantity a reader wants to see, so
it should still be quoted -- but a decision rule should not hang on a statistic
whose A/A floor is four times its own bound.

This applies to the thesis's existing tables, not only to tonight's work.
