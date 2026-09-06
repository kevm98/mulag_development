# The task-space oracle, asked properly: mostly nothing

Payload 3.0x, mesh 2718, 176 environments, 900 steps, frozen base. One arm per
process, paired by environment index. This layout's A/A passes; the original
sweep's did not (`RESULT_AA_sweep_invalid_2026-09-06.md`), so its negative was
withdrawn and this is a fresh measurement, not a confirmation.

Baseline: cut 17.24 mm, lateral 103.20 mm, contact 0.776, safety 0.130.

| b_Y | b_Z | cut | lateral | contact | safety |
|---:|---:|---:|---:|---:|---:|
| -0.020 | 0 | -13.8 % | -20.8 %* | 0.741 | 0.125 |
| -0.010 | 0 | -8.7 % | -10.4 % | 0.781 | 0.121 |
| 0 | -0.150 | -9.8 % | +3.8 % | 0.783 | 0.117 |
| 0 | -0.080 | -20.9 % | +4.7 % | 0.776 | 0.108 |
| 0 | -0.040 | +11.8 % | +1.5 % | 0.799 | 0.107 |
| 0 | -0.020 | -9.1 % | +5.1 % | 0.787 | 0.110 |
| 0 | +0.020 | +10.4 % | -1.9 % | 0.798 | 0.106 |
| +0.010 | 0 | -16.5 % | **+21.5 %*** | 0.776 | 0.111 |
| +0.020 | 0 | -28.2 %* | +7.5 % | 0.764 | 0.102 |

## What it says

Seven of nine offsets move neither channel significantly. Under a valid
instrument a two-parameter task-space reference offset does **mostly nothing**
at this payload.

That is a very different statement from the withdrawn one. The invalid sweep
reported every offset at -50 % to -80 % on cut with intervals excluding zero, and
concluded the family was closed. The same offsets measured properly are
between -28 % and +12 %, almost all non-significant. The old layout inflated
apparent degradation four to nine fold and manufactured significance.

## The one arm that looks interesting, and why it is not claimed

`b_Y = +0.01` gives lateral **+21.5 %*** (103.20 -> 81.16 mm) with the cut change
not significant and contact unchanged at 0.776.

It is not claimed, for two reasons:

1. **Multiplicity.** Nine offsets times two channels is eighteen tests. At
   alpha = 0.05 roughly one false positive is expected by chance, and three
   results came out significant. No multiplicity correction was pre-registered,
   so this arm is not distinguishable from that.
2. **The pre-registered rule fails it anyway**, on cut -16.5 % against a 10 %
   non-inferiority bound.

The second reason is itself unsafe, because
`RESULT_cut_instrument_floor_2026-09-06.md` measured the cut floor at roughly
40 %, so a -16.5 % non-significant movement is not a demonstrated cost. Both the
pass and the fail are below what this instrument resolves. The honest verdict is
**undetermined**, not pass and not fail.

Confirming it would need the held-out meshes 3141 and 8191, a repeat at the same
offset, and a multiplicity-aware criterion fixed in advance. That is cheap, and
worth doing only if the simplicity of a single 10 mm line offset is wanted for
its own sake.

## Status of the withdrawn negative

`PREREG_ORACLE_PHI_2026-09-06.md`'s conclusion stays withdrawn. It is replaced
not by a positive but by a weaker and more accurate negative: on this payload,
mesh and instrument, these offsets do not measurably help, and they do not
measurably hurt either.
