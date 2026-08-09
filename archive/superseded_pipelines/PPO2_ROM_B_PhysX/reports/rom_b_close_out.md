# ROM-B clearance investigation — close-out

## 0. Retraction

My previous message reported the USD world bound of the `Messerkopf` prim as
"ground truth" and concluded the gate's cutter computation was wrong by 1.64 m on
that basis. **That measurement was invalid.** The two environments returned
*identical* bounds (1.7284 / 2.2859) despite different live poses
(`ee_origin_world_z` 0.8764 vs 0.8806, `head_alignment` 0.99962 vs 0.99997). The
USD stage transforms are the authored spawn pose and are never synced from PhysX
in headless mode, so `ComputeWorldBound` returns spawn geometry, not simulated
state. The corroborating absurdity: the whole-robot USD bound starts at 1.2603 m
while the live truck base body sits at 0.1423 m — a robot whose lowest geometry
were really at 1.26 m would be floating.

The conclusion survives, but on live evidence, not that measurement. Details below.

## 1. What is established, from live data only

**The gate's cutter computation ignores orientation and is wrong by 1.58 m.**

```
_compute_cutter_world_z = ee_origin_z + tool_tip_local_offset_z        (unrotated)
_compute_cutter_pos_w   = ee_pos_w + quat_apply(ee_q, [0,0,offset])    (rotated)
```

With `tool_tip_local_offset_z = -0.79` (`tasks/env_cfg.py:41`) and the live
head-down quaternion, `quat_apply(q, [0,0,-0.79])_z = +0.79`: the EE body Z axis
points *down* in world, so the body-frame offset resolves to +0.79 in world and
the two formulas differ by `2 x |offset| = 1.58 m`. Measured across 84 live
pre-termination samples: **−1.5798 m, sd 0.0003**. Pinned by
`test_unrotated_cutter_formula_is_wrong_when_the_head_points_down`.

**The library's `cutter_world_z` column was generated with the same unrotated
formula.** Live values at the settled start pose: gate 0.0864, library reference
0.1036 — agreeing to 1.7 cm, while the orientation-aware cutter is at 1.6661.
Both sides of the gate are ~1.6 m low, which is why the gate reports a
plausible-looking residual instead of a 1.6 m absurdity.

**The vertical datum is sane and the task is not ground mowing.** Ground plane
`/World/ground` at world Z [−0.02, 0.00] (static geometry, so this reading is
valid). Live articulation root 1.2492, live truck base body 0.1423 — the root sits
1.11 m above the truck base, matching the known MULAG geometry. Commanded path
height = truck base 0.142 + `cutter_truck[2]` 1.55 ≈ **1.69 m world**, and the live
orientation-aware cutter is at 1.666 m. Consistent. The 1.7 m working height is
correct for this task, not an anomaly.

**What does not cancel is the truck's ride height.**

```
reference/frame term = truck_base_world_z + ref_cutter_truck_z - ref_world_z

  settled zero-policy pose : 0.1423 + 1.55 - 0.1036 = +1.5887  ->  net +0.009
  failing probe            : 0.2533 + 1.55 - 0.0600 = +1.7433  ->  net +0.163
                             (cutter-definition term ~ -1.580 in both)
```

The gate compares a **truck-frame-authored** commanded geometry against a **fixed
world-frame number**, so the truck's absolute ride height leaks one-for-one into
what is reported as cutter clearance. A 0.111 m ride-height change between poses
*is* the 0.164 m offset that consumes 82 % of the 0.200 m limit. Pinned by
`test_clearance_gate_leaks_truck_ride_height`.

## 2. Why I did not apply the repair

Two coherent fixes exist and they answer different requirements. Choosing between
them is a specification decision, not a code decision:

**A — absolute height safety.** Gate uses `_compute_cutter_pos_w()[2]`;
`cutter_world_z` is regenerated as a true world height. Ride height then
legitimately counts, because the hazard is world-fixed. Requires re-running the
trajectory-library generation pipeline.

**B — path-relative tracking.** Clearance becomes
`measured_cutter_truck[2] - reference_cutter_truck[2]`, i.e. the vertical component
of truck-frame tracking error. Ride height is excluded. Needs no library change.
**Consequence:** the clearance gate then measures a subset of what the 0.08 m p95
tracking gate already covers, at a 0.20 m threshold — it becomes near-vacuous, a
fourth vacuous gate alongside the three already identified.

There is also a scope constraint: the clearance signal is computed in the shared
parent (`PPO2_ROM_PhysX/policy/env.py:1030, :1062, :1212`), not in ROM-B. Changing
it affects PPO2-A and ROM-C, which this task is not permitted to modify.

On the evidence I lean to **A** — the gate exists to stop the cutter hitting
something, that hazard is world-fixed, and B would quietly delete the gate. But A
costs a library regeneration and re-validation of every prior clearance result,
so it is your call.

## 3. What this means for the ROM-B question

With the frame offset removed, the seven captured failing replicas would read
roughly 0.027 m rising to 0.05 m against a 0.200 m limit. **None would terminate.**
The dominant failure mode in the acceptance suite — line clearance, 20 of 22 Monte
Carlo line failures and 96 of 130 h-sweep line failures — is produced by frame
bookkeeping, not by the plant.

The genuine physical signal underneath is real but small: 2–6 cm of upward cutter
drift at the path endpoint under degraded hydraulics, which is where reduced
`e_flow` and increased seal leakage would be expected to act. That is a legitimate
ROM-B robustness signal, currently buried under an offset five times its size.

> **ROM-B training completed. Whether domain randomization succeeded remains
> unresolved**, because the principal clearance acceptance signal is dominated by a
> frame inconsistency. `model_53`'s acceptance, all nine h-sweep verdicts, and any
> `model_607` comparison must be re-read once the clearance signal is
> frame-consistent.

## 4. Delivered this session

| item | status |
|---|---|
| `runtime/acceptance.py` — `max(pooled CP, seed-group t)`, per shape / per (h,shape), worst-shape, `phi` reported never gating | done, tested |
| Replay on existing audit: MC **FAIL** (line 0.05647), h-sweep **FAIL** 13/45 conditions | done |
| `evaluate.py` reports the new statistic alongside the legacy gate | done |
| Exact integer `unsuccessful_environments` / `replicas` from the probe | done |
| `_PreTerminationRecorder` + `_summarize_path_profile` + full frame decomposition fields | done, exercised live |
| `scripts/diagnose.py` — replays audit seeds with diagnostics | done, exercised live |
| `scripts/check_cutter_frame.py` — datum and cutter-frame query | done, with the USD caveat in §0 |
| `train.py` — `--remediation`, `--curriculum_eval_seed_groups`, `--curriculum_eval_seed_offset`, `--minimum_full_scale_rounds`, `--train_safety_terminations` | implemented, **not smoke-tested** |
| `--hard_reset` | implemented, **measured insufficient**, default off, documented |
| 29 offline tests | pass |

Not started: remediation training, `model_607` control arm, `--train_safety_terminations`
and `--remediation` smoke tests. `model_53` untouched.

## 5. Open items, in order

1. **Decide A or B** (§2), then repair the clearance signal in the shared parent.
2. **Probe position-dependence** — same seed gives 7–13 failures depending on
   sequence position (`rom_b_probe_reproducibility.md`). Only verified fix today is
   a fresh process per probe.
3. Re-run acceptance for `model_53` and `model_607` on the corrected signal.
4. Then, and only then, remediation — with `--train_safety_terminations` as the
   single behavioural change, and reward shaping left alone.
