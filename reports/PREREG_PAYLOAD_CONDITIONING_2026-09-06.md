# Pre-registration: train the payload channel the base multiplies by zero

Written 2026-09-06, before any arm is trained. Fifth mechanism. It differs from
the previous four in what it changes, and the difference is measured, not
argued.

## The finding this rests on

The frozen base `distilled_v1/phase1_init_70.pt` reads a 70-wide observation:
66 task columns plus the 4-wide RMA context `(alpha, tau, delay, mass_scale)`.
The true normalised payload is in column 69, present every step.

Measured on the checkpoint:

    mlp.0.weight (256, 70)
      base columns  0:66   mean |w| = 0.125463
      context      66:70   mean |w| = 0.000000
        col 69 mass_scale  mean |w| = 0.000000

Exactly zero, not merely small. `PPO_AAM_FD/scripts/zero_context_columns.py`
put them there on purpose, so that Phase 1 starts as an exact identity.

So the base is blind to payload **by construction**, and Phase 1 was never run
on a plant where payload matters. Every campaign so far scored this blind policy
across 1.0-5.0x, where it degrades 95x on cut height, and asked a residual, an
encoder or a per-step gradient to compensate from outside it.

The reason the context was never learned is also on record and is not a mystery:
distillation and Phase 1 both ran on the manifest's own 0.8-1.5 payload range,
which `env_cfg.py` records as costing a fixed policy nothing measurable. A
channel that predicts nothing has no gradient. Measured, a 1000x gradient
multiplier on the context columns moved action sensitivity from 0.001% to
0.004% of action scale. That is not a plasticity failure; there was nothing to
learn. On 1.0-5.0x there is.

## Arms

Three, all 70-wide, all the same architecture and parameter count.

| arm | context value | trained |
|---|---|---|
| `base` | live, ignored (weights are zero) | no, the existing frozen policy |
| `blind` | zeroed every step | yes |
| `adaptive` | live, true normalised payload | yes |

`blind` and `adaptive` both resume from `phase1_init_70.pt`. Because its context
weights are exactly zero, **the two arms are literally the same policy at
iteration 0**, for any context value. A difference at the end is attributable to
the payload channel and to nothing else: not width, not capacity, not seed, not
budget, not objective.

The ablation is on the context VALUE, not the slot
(`MULAG_CONTEXT_ABLATE=zero|shuffle|live`). Dropping the slot
(`MULAG_NO_CONTEXT_SLOT=1`) changes the observation to 66 wide and the arms stop
being comparable.

## Plant and objective

    MULAG_DIRECT_VALVE=1  MULAG_RMA_CONTEXT=1
    MULAG_PAYLOAD_RANGE=1.0,5.0  MULAG_PAYLOAD_RESAMPLE=1

Rebalanced objective, `--lateral_tracking_std_m 0.09 --lateral_tracking_weight
4.0`, applied after the stage curriculum. 256 environments, 400 iterations,
seed 17. Identical for both trained arms.

## The comparison, and the one that must NOT be the headline

**PRIMARY: `blind` -> `adaptive`, paired per environment.** This is the only
comparison that isolates payload information. Both arms get the same extra
training on the same wide-payload plant; only one can see which payload it is
carrying.

**`base` -> `adaptive` is NOT the headline and will not be reported as the
result.** It will be large, because the base is untrained on this plant AND
blind to the payload, so it conflates the two. Reporting it as the effect of
adaptation is precisely the error `PPO_AAM_FD/memory.md` records the previous
campaign making. It is reported as context, labelled as confounded.

**`base` -> `blind` is reported always.** It measures how much of the deficit is
recovered by training on the range alone, with no plant information. If it
recovers nearly all of it, the honest conclusion is that robustness training
beats online adaptation on this stack, and that is the finding.

## Falsification

**Corrected 2026-09-06, before any arm was scored.** The original plan was to
rescore `adaptive` with `MULAG_CONTEXT_ABLATE=shuffle`. That control is
DEGENERATE at a fixed payload: every environment then carries the same
`mass_scale`, so permuting the context rows is the identity. The shuffled arm
would have matched the live arm exactly, and the pre-registered reading of that
match is "the gain is not payload information" -- a false negative manufactured
by the control, on a run where the information was never actually removed.

So the falsification is split to match where each control has content:

* **At fixed payload, feed a WRONG estimate.** `adaptive` rescored with the
  context zeroed, which is the normalised nominal plant, so the policy is told
  it is carrying 1.0x while the plant gives it 3.0x or 5.0x. Same checkpoint,
  same weights. If `adaptive`-live does not beat `adaptive`-wrong, the live
  payload value is not doing the work.
* **On the mixed range, shuffle.** One extra pair at
  `MULAG_PAYLOAD_RANGE=1.0,5.0` resampled per episode, where environments
  genuinely differ so a permutation removes the pairing while keeping the exact
  marginal distribution.

Both are declared here before any scored arm exists. If either control matches
the live arm, the claim fails whatever the primary comparison says.

## Success rule

Fixed here, identical to Table 7.7 and to every previous campaign. Paired
per-environment median difference, percentile bootstrap B = 20000, percentage as
the median log-ratio. One channel improves with its interval excluding zero and
the other improves or stays within its non-inferiority bound (cut 10%, lateral
1.5%, the A/A floors of `PREREG_R2_2026-09-01.md`).

Applied to `blind` -> `adaptive`, and separately at each scored payload.

## Declared secondary outcomes: contact and safety

Added 2026-09-06 BEFORE any arm of this campaign was trained, on the strength of
`RESULT_rootcause_contact_2026-09-06.md`, which was computed from the oracle
probe's existing reports and spent no new GPU time.

The root cause of the heavy-payload deficit is impact and rebound, not lack of
authority. Over 1.0x -> 3.0x the frozen base moves:

    ground contact fraction   0.960 -> 0.795
    hard-impact fraction      0.164 -> 0.445
    p95 tool acceleration     50.7  -> 90.4 m/s2
    safety-violation rate     0.018 -> 0.105

while action saturation stays under 1 %.

These are therefore reported for every arm, at every payload, as declared
secondary outcomes:

* `ground_contact_fraction` and `worst_family_ground_contact_fraction`
* `hard_impact_event_fraction`
* `p95_tool_acceleration_mps2`
* `safety_violation_rate`

**Status of these outcomes.** They are SECONDARY. The primary remains the two
scored task channels under the Table 7.7 rule, and a failure there is a failure
whatever the secondaries do. But a payload-conditioned arm that measurably
reduces hard impacts and safety violations against its capacity-matched blind
twin is a genuine and useful result on this plant, because those quantities are
the mechanism by which the task fails, and it is reported as such rather than
discarded for not being a tracking gain.

They are written down here, with their baseline values, so that neither a
favourable nor an unfavourable movement in them can be selected after the fact.

## Staging

Stage A, one seed, decisive and cheap. Score `base`, `blind`, `adaptive` and
`adaptive`-shuffled at fixed payloads 1.0/3.0/5.0 on mesh 2718 only.

Stage B, only if Stage A passes the primary AND the shuffle control fails to
match: extend to seeds 23, 101, 42, 7 and to held-out meshes 3141 and 8191. The
median-seed bar is 3 of 5, the same bar the starved campaign was held to.

A Stage A pass is NOT the result. It is permission to pay for Stage B.

## Stage 2, the online part, pre-committed in structure only

If Stage B passes, the true payload is replaced by an estimate computed at
runtime from signals already in the observation: normalised cylinder effort
(62:66), the actuator prediction residual (58:62), and cylinder state. Offline
regression, frozen at deployment, feeding the same context slot.

Arms then: `oracle` (true payload), `estimated` (online), `shuffle`, `blind`.
The claim is `estimated` approaching `oracle` while beating `blind`, with
`shuffle` failing. Details are pre-registered separately once Stage 1 has a
ceiling to aim at; nothing about Stage 2 is used to justify Stage 1.

## Declared outcome

* `blind` -> `adaptive` passes at 3.0x and 5.0x, and shuffle fails to match:
  proceed to Stage B.
* `blind` -> `adaptive` null: report that payload information adds nothing on
  top of robustness training on this stack. Given four mechanisms already closed
  by their own controls, that is a reportable finding under the thesis title and
  is not re-rolled into a sixth mechanism.
* `base` -> `blind` recovering most of the deficit while the primary is null:
  report that the deficit was a training-distribution problem, not an adaptation
  problem. This is the most likely null and it is written down here so it cannot
  be presented as a surprise later.
