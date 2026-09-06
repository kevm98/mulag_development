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
