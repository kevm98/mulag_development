# BLOCKING: the deterministic probe is not a function of its seed

Found 2026-07-24 while running the authorized failing-seed diagnostics. This
supersedes the acceptance-statistic discussion: the statistic is sound, but the
unit it operates on is not stable.

## 1. The measurement

Monte Carlo `line`, round 4, seed `300032`, checkpoint `model_53`, 128
environments, 120 scored steps — the same probe throughout:

| context | failures /128 | rate |
|---|---:|---:|
| audit `evaluation_2026-07-24_19-28-48`, 16th probe in the process | **7** | 0.0547 |
| fresh process, 2nd probe (after round 1) | **13** | 0.1016 |
| fresh process, 1st probe | **11** | 0.0859 |
| same process, 2nd probe (after round 4) | **11** | 0.0859 |
| same process, 3rd probe (after round 4, round 4) | **10** | 0.0781 |
| separate process, 1st probe (repeat) | **11** | 0.0859 |

Round 1 (`seed 300017`) as the first probe reproduces the audit exactly at 7/128,
and two separate processes running round 4 first agree to five decimal places on
every metric including `p95_acceleration = 32.06`.

**The simulator is deterministic. The probe is position-dependent.** The same
seed spans 7–13 failures, i.e. 0.055–0.102, depending only on how many probes ran
before it. The gate is 0.05.

## 2. The mechanism

`_reset_idx` is a soft reset (`PPO2_ROM_PhysX/policy/env.py`):

* `:1373` `self._preposition_start_x[env_ids] = measured_x[env_ids]` — redeployment
  starts from wherever the arm was left, not from a canonical pose.
* `:1385-1390` `self._truck_anchor_xy[env_ids] = truck_base[env_ids, :2]` — the
  parked-truck anchor is re-taken from the current chassis pose.
* No joint state, root pose or root velocity is written back to the simulator.

The probe deliberately does not reset failed replicas — that is the
"preserve failed replicas" property we want — so every probe inherits the terminal
physical state of its predecessor, including replicas left in a violated
configuration.

## 3. Attempted fix, and why it is not shipped as working

`restore_canonical_physical_state` writes `default_root_state`,
`default_joint_pos`, `default_joint_vel` and re-derives the truck anchor, exposed
as `--hard_reset` (default off). It **does not fix the problem**:

| probe | positions 1, 2, 3, 4 |
|---|---|
| `--hard_reset`, rounds 1, 4, 1, 4 | 5, 10, 4, 12 |

Round 1 gives 5 then 4 (against 7 from a fresh process), round 4 gives 10 then 12.
Still position-dependent, and position 1 no longer matches the fresh-process
value — so it is not a faithful restoration either. State is still leaking through
something it does not touch. Remaining candidates, in rough order of likelihood:

1. the command-ROM recurrent history buffer,
2. the DLS / cascade controller internal state,
3. PhysX solver warm-start and contact caches,
4. articulation internal state not covered by the three default tensors.

The flag is kept, default off, with the measurement recorded in its docstring, so
the diagnosis is not lost. **It must not be used for certification.**

## 4. What this invalidates

* **Every per-row number in the audit is conditioned on its position.** Rows are
  not exchangeable replicates.
* **My `sigma_b = 0.0167` for line is confounded.** In the audit, MC line rounds
  1–5 sit at process positions 1, 6, 11, 16, 21. The counts 7, 0, 3, 7, 5 mix seed
  variation with position variation. The between-group variance is real, but I
  attributed it to trajectory and initial-condition heterogeneity; an unknown part
  of it is protocol artifact. The cluster-robust bound is still the *conservative*
  choice — it widens the interval — so the acceptance verdict (line fails) does not
  reverse. The causal story does.
* **Subset diagnostic runs are not comparable to audit rows** unless they occupy
  the same position, which is why my round-4 diagnostics read 13 where the audit
  read 7.

What it does **not** invalidate: the failure mechanism in section 6, which is
qualitatively identical across every run.

## 5. Options

| option | cost | verdict |
|---|---|---|
| **A. Fresh process per probe** | ~90 s startup x 160 rows = ~4 h on top of ~1.75 h of probing | Only verified-correct option. Right for final certification. |
| **B. Finish the state-leak fix** | unknown; my first attempt failed | Correct long-term. Needs the four candidates above ruled out one at a time. |
| **C. Declare the fixed sequence part of the protocol** | free | Defensible *if* no result is ever compared across protocols. The audit is then one deterministic experiment, reproducible end to end — which is what the three byte-identical preflights actually demonstrated. Seed groups stay confounded with position, inflating the group-t bound, which is conservative rather than unsafe. |

Recommendation: **C now for the control arm, A for final certification, B when
there is time.** The control arm is a paired within-protocol comparison — identical
script, seeds and sequence for `model_607` and `model_53` — so the position effect
is largely a shared nuisance. It is not perfectly shared, because inherited state
depends on where the previous probe's replicas ended and therefore on the policy;
that caveat should be recorded with the result.

I did not launch the control arm, because this finding changes what its numbers
mean and that is your call to make.

## 6. The failure mechanism (this part is solid)

From 20 captured failing replicas across MC line rounds 1 and 4:

* **18/20 are clearance; 15/18 of those sit at `path_s` = 0.011–0.048 m of a 2.5 m
  path, at `reference_phase` ~ 0.995–0.997** — the very end of the return leg, i.e.
  decelerating into the A endpoint of the A→B→A traverse.
* **Head alignment at termination is 0.9998–1.0000.** The cutter's vertical error is
  not a tilt effect.
* **Acceleration at termination has median 4.5 m/s².** Quasi-static, not a transient.
* **Clearance is positive in every single case** — the cutter is always too *high*,
  never too low.
* **Residual usage is 0.17 mm of 2.0 mm authority.** The policy is not fighting it.

A representative trace, the last 12 steps before termination (env 21):

```
path_s   phase   clearance   head      acc    residual
0.0271   0.995   +0.1628     0.99995   3.17   0.26 mm
0.0217   0.996   +0.1678     0.99997   4.11   0.27 mm
0.0181   0.996   +0.1854     1.00000   3.73   0.26 mm
0.0147   0.997   +0.1974     1.00000   3.03   0.24 mm
0.0137   0.997   +0.2005     0.99999   3.16   0.24 mm   <- terminates
```

Clearance climbs monotonically and smoothly from 0.163 to 0.200 over twelve steps
while the arm is nearly stationary and perfectly aligned. This is a slow upward
drift of the cutter as the trajectory decelerates into its endpoint — consistent
with a velocity-dependent lag or deadband at low commanded flow, which is exactly
where reduced `e_flow` and increased seal leakage would bite. The failing factor
vectors are consistent with that: `e_comp` mostly negative (−0.03 to −0.12),
`seal_mean` 0.33–0.70.

The two non-endpoint clearance failures are the other sub-mode: `path_s` ~ 0.15
on the outbound leg with acceleration 51–54 m/s², i.e. the startup transient the
path profile flagged earlier.

### 6.1 I was wrong about the 0.120 shaping onset

You said calling it "definitely wrong" was premature. The trace confirms you were
right and I was not. In this mode clearance passes through 0.163 twelve steps
before the limit, so a 0.120 onset would be **active for the entire approach**,
giving the policy a gradient throughout, whereas the current 0.180 onset only
engages in roughly the last five steps. And 0.120 sits above the p95 of normal
operation (steady-state mean clearance 0.014–0.031 m), so it would be largely
inactive in cruise. A lower onset is now positively supported rather than merely
not-disproved. It still should not be changed before the control arm.

## 7. State of the implementation

Delivered and tested (27 offline tests pass):

* `runtime/acceptance.py` — your `max(pooled Clopper-Pearson upper, seed-group t
  upper)` rule, per shape and per `(h, shape)`, worst shape controls, never pooled
  across shapes or `h`. Dispersion `phi` reported for audit, never gating.
  Degenerate cases covered: `0/640` returns 0.00467 rather than 0, uniform groups
  fall back to Clopper-Pearson, `m = 1` falls back to Clopper-Pearson.
* Replayed on the existing audit: **24 of 45 h-conditions bind on the group-t term,
  21 on Clopper-Pearson.** Neither term is redundant — your rule is doing real work
  in both directions.
* `evaluate.py` reports the new statistic alongside the legacy row-wise gate;
  nothing is silently replaced.
* `run_deterministic_probe` now emits exact integer `unsuccessful_environments`
  and `replicas`, so the statistic never rounds a reported fraction.
* Acceptance replay of the current audit: Monte Carlo **FAIL** (worst line,
  upper 0.05647); h sweep **FAIL**, 13 of 45 conditions, 7 of 9 `h` values.

Not done, pending your decision: control arm, `--train_safety_terminations`
smoke test, `--remediation` smoke test. All three are downstream of the protocol
question in section 5.
