# Pre-registration: the final seed protocol and its decision rule

Written 2026-09-06, before the final architecture is selected and before any
additional seed is trained.

## What changes from earlier campaigns

**The decision rule is no longer "all seeds must pass both channels".** That
binary was inherited from the previous campaigns and it is the wrong test: it
converts an aggregate effect into a conjunction of per-seed events and fails on
one neutral seed even when the pooled result is unambiguous.

The primary hypothesis is instead

    H1:  E_adaptive  <  E_frozen

under plant conditions where the frozen policy has a MEASURABLE robustness
deficit, with the paired per-environment statistic pooled across seeds, and the
individual seeds reported transparently beside it.

A result of the form "two seeds improve strongly and one is approximately
neutral" supports H1 if the pooled paired result is convincingly better. A
result where one seed improves and two degrade does NOT, because that is
initialisation-dependence rather than a property of the method, and it will be
reported as such.

## Structure

    3 independently trained seeds
      x  held-out evaluation at 3.0x, 4.0x, 5.0x cutter mass
      x  {Frozen, Adaptive, Oracle}

**Frozen is scored once per mass, not once per seed.** With the context zeroed a
trained checkpoint computes `base(x) + W_ctx . 0 = base(x)`, so it IS the frozen
policy independent of which seed trained the context. That is a property of the
architecture, verified by measurement, and it removes six evaluations.

Each evaluation is 176 independent environments -- distinct terrain draws, path
families and plant draws -- so the episode count per condition is already about
ten times the 10-20 that would be asked for, and three meshes give 528. What is
scarce is REPEATED RUNS of the same condition, because the cut channel's
run-to-run floor was measured at roughly 40 % between runs identical by
construction. Spare capacity goes to repeats and to the 4.0x level, not to more
environments per run.

## Scope discipline

**Architecture selection stops before the seeds start.** The FiLM, estimator-refit
and CNN experiments are mechanism selection and run on seed 17 only. Exactly one
adaptation method is carried forward, and no architecture comparison is run
across seeds. Replicating a development sweep across seeds would multiply the
comparisons without adding evidence for any single claim.

## Reporting

Per-seed rows, pooled paired statistic, and the frozen baseline quoted in
millimetres beside the percentage. No seed is dropped, and a neutral or negative
seed appears in the table with the others.
