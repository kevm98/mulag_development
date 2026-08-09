# Is boom_sweep infeasible by construction? — pure geometry, no GPU

**Date:** 2026-08-06
**Script:** [`scripts/probe_path_feasibility.py`](../../scripts/probe_path_feasibility.py)
**Data:** [`path_feasibility.json`](path_feasibility.json)
**Runtime:** ~2 s on CPU. `verge_task/terrain.py` is pure numpy + trimesh,
so the simulator's own path generator can be called directly.

This is the evidence Question 10 of the review brief asks for: what the tracked
path *demands*, measured independently of any policy.

---

## 1. Method

For every family on meshes 1337/17/909 at the shipped stage-6 settings
(amplitude scale 2.2, relief `flat_patches`, strip width 1.8 m, active length
16 m), regenerate the exact path and measure:

* `peak |offset|` — furthest excursion from strip centre [m]
* `peak |dy/dx|` — path slope in the strip frame [m/m]
* `peak |dy/dt|` — **required lateral cutter speed** [m/s], = slope x 1.111 m/s

The truck drives at a fixed 4 km/h and cannot slow for a corner, so path slope
converts directly into a required lateral speed.

**Capability ceiling: 0.200 m/s.** From `cylinder_action_scale` 0.004 m per
control step and a 0.02 s control step (`rom_dt` 0.01 x `decimation` 2). This is
deliberately *optimistic*: it assumes a lateral Jacobian of 1.0 (a metre of
cylinder extension buys a metre of lateral cutter travel) and that the entire
cylinder budget goes to lateral motion rather than holding height and head
angle. True capability is lower.

## 2. Result: demanded RATE predicts error; displacement does not

Pearson correlation against the student's measured per-family lateral error,
across all 33 family x mesh cells:

| geometric quantity | r |
|---|---|
| **peak lateral speed** | **+0.802** |
| p95 lateral speed | +0.780 |
| rms offset | +0.505 |
| peak offset | +0.499 |

Tracking error is driven by how *fast* the path demands lateral motion, not by
how *far* it goes. `offset_return` reaches a larger peak offset (0.362 m) than
`boom_sweep`'s rms (0.289 m) and tracks it at 0.076 m error, because it gets
there slowly (0.081 m/s).

## 3. boom_sweep is rate-limited, and the ratio is constant

| mesh | demanded peak lateral speed | student error | ratio |
|---|---|---|---|
| 1337 | 0.5813 m/s | 0.2800 | 0.482 |
| 17   | 0.6785 m/s | 0.3189 | 0.470 |
| 909  | 0.9138 m/s | 0.4046 | 0.443 |

Demand varies **57 %** across meshes while the error/demand ratio varies **8 %**.
Error accumulating in fixed proportion to demanded rate is the signature of a
rate-limited actuator, not of a policy that has failed to learn something.

boom_sweep demands **2.9x - 4.6x the optimistic ceiling**, and 27-33 % of its
path length is above it. It is the highest-demand family on every mesh, by
1.4x - 2.3x over the next worst.

**Why it is worst on m909:** boom_sweep is the only family whose offset is
coupled to terrain height (`raw_offset = tracking_height_sweep_gain *
(height - reference_height)`). Every other family draws a fixed random
amplitude. So rougher terrain mechanically demands faster lateral motion —
which is exactly why this family, alone, degrades as the mesh gets harder.

Corroboration from the other two experiments:
* the privileged DLS teacher fails boom_sweep on all three meshes
  (0.2946 / 0.3253 / 0.3657);
* granting **2.8x more command authority** (lead cap 0.05) made boom_sweep
  *worse* on every mesh and drove `min_cylinder_headroom` to exactly 0.0000 —
  authority spent without tracking gain, terminating at the stroke limit.

## 4. What would fix it, quantitatively

Fitting `error = 0.465 x demanded_peak_speed` (the ratio above) and re-running
the generator across amplitude scales:

| amplitude scale | peak lateral speed (1337/17/909) | predicted error | verdict |
|---|---|---|---|
| 1.0 | 0.283 / 0.332 / 0.425 | 0.132 / 0.154 / 0.198 | PASS |
| 1.2 | 0.337 / 0.395 / 0.509 | 0.157 / 0.184 / 0.237 | PASS |
| **1.4** | 0.389 / 0.456 / 0.592 | 0.181 / 0.212 / 0.275 | **PASS** |
| 1.6 | 0.440 / 0.515 / 0.673 | 0.205 / 0.239 / 0.313 | FAIL (m909) |
| 1.8 | 0.489 / 0.572 / 0.755 | 0.227 / 0.266 / 0.351 | FAIL |
| 2.0 | 0.536 / 0.626 / 0.835 | 0.249 / 0.291 / 0.388 | FAIL |
| 2.2 (shipped) | 0.581 / 0.678 / 0.914 | 0.270 / 0.315 / 0.425 | FAIL |

**Amplitude scale 1.4 is the largest value at which boom_sweep is predicted to
clear the 0.300 gate on all three meshes.** Reducing
`tracking_height_sweep_gain` instead would work equally well and would leave the
other ten families at full difficulty — arguably the better lever, since only
boom_sweep reads that gain.

## 5. Verdict — and one family this does NOT explain

**boom_sweep: "infeasible by construction" is supported.** Rate demand 2.9-4.6x
an optimistic capability bound, error linear in demand with a ratio stable to
8 % across a 57 % demand range, the privileged teacher fails identically, and
extra authority makes it worse while bottoming out the stroke.

**gaussian_slalom: NOT explained by this, and it should not be lumped in.** It
demands only 0.196-0.209 m/s — near or below the ceiling, second lowest of all
eleven families — yet errors 0.179-0.282 (student) and 0.205-0.353 (teacher).
On m1337 the **teacher is much worse than the student** (0.3528 vs 0.2818). A
family that fails while demanding little rate has a different cause: three
closely-spaced sign reversals (centres 0.23/0.50/0.76, widths 0.105/0.14/0.09)
are the obvious suspect, and the DLS teacher's poor showing suggests the
controller structure rather than the plant. **That needs its own investigation
and no conclusion is drawn here.**

## 6. Honest limits of this analysis

1. The **0.465 constant is fitted on three points from one family.** It predicts
   those three points well by construction. Treat the amplitude table as an
   estimate that requires one confirming simulation run at scale 1.4, not as a
   result.
2. The **0.200 m/s ceiling assumes a lateral Jacobian of 1.0** and is not a
   measured quantity. It is a bound, not the true capability. Note that 9 of 11
   families exceed it at their peak yet track acceptably, so the ceiling alone
   proves nothing — the load-bearing evidence is the linear error-vs-demand
   relation and the r = +0.802 correlation, not the ceiling crossing.
3. This measures **demand**, not reach. A direct kinematic reach computation
   (workspace envelope vs required y at each x) would be stronger still and has
   not been done.
