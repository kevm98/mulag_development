# Root cause: heavy payload fails by impact and rebound, not by lack of authority

Measured 2026-09-06 from the oracle-probe reports on mesh 2718, 176
environments, 900 steps, frozen base, fixed payload per run. No new GPU time was
spent: every number here was already in the reports.

## The chain

| payload | cut median | lateral median | contact | hard impacts | p95 tool accel | safety | action saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0x | 1.59 mm | 47.14 mm | 0.960 | 0.164 | 50.7 m/s2 | 0.018 | 0.0012 |
| 3.0x | 15.67 mm | 91.35 mm | 0.795 | 0.445 | 90.4 m/s2 | 0.105 | 0.0082 |

## Four findings, in the order they matter

**1. The actuator is not saturated.** 0.12 % of steps at 1.0x and 0.82 % at
3.0x, with mean |action| between 0.16 and 0.63 against a limit of 1.0. Command
authority is available and unused. This is the finding that keeps the whole
adaptation programme alive: had the drive been against its stop, no amount of
plant knowledge could help, and the honest conclusion would have been that the
task is infeasible at heavy payload. It is not.

**2. Cut-height error is almost entirely contact loss.** Across the ten offset
arms at 3.0x, corr(mean contact fraction, log median cut) = **-0.975**. Within
the baseline arm alone, across its 176 environments, **-0.909**.
`cut_height_error` is `relu(h - reference)` and exactly zero in contact, so the
channel is nonzero only while the cutter is airborne. At heavy payload it is
largely a measure of time spent off the ground.

**3. The head is bouncing, not sagging.** From 1.0x to 3.0x the hard-impact
fraction goes 0.164 -> 0.445, p95 tool acceleration 50.7 -> 90.4 m/s2 and the
safety-violation rate 0.018 -> 0.105. The heavier head strikes the terrain
harder and rebounds off it.

**4. Pushing down makes it worse.** Every task-space reference offset reduced
contact and raised the violation rate, the downward ones most of all:
0.795 -> 0.710 at `b_Z = -0.02` and 0.621 at `-0.04`, with violations rising
0.105 -> 0.142 -> 0.259. A constant positional bias cannot fix a dynamic
problem; commanding harder into the ground buys more rebound.

## Why this closes the task-space offset family, and what it opens

`PREREG_ORACLE_PHI_2026-09-06.md` found no two-parameter offset that
Pareto-improves at 1.0x or 3.0x. Finding 4 explains why, mechanistically, rather
than leaving it as an empirical null: the offsets act on position and the
failure is in impact dynamics.

The correct response to an impact problem is to modulate approach velocity and
damping, which is state-dependent and mass-dependent. A constant offset cannot
express it. A payload-conditioned policy can, and finding 1 says it has the
command authority to do so.

## The consequence for what is measured

Contact retention, hard-impact fraction and the safety-violation rate are not
side statistics on this plant. They are the mechanism, and they degrade 2.7x,
1.8x and 5.7x respectively over a payload range the frozen base is blind to by
construction. They are added as declared secondary outcomes of the payload
conditioning campaign, before that campaign has produced any arm.
