# Pre-fix contact-loss diagnostic — PPO3 (ROM-A) and PPO4 (ID/FD)

Written 2026-08-02, before any reward, observation or plant change was made.
Every number is read from a stored artifact in this directory; the file is
named next to it.

**Reproducer**

```bash
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
$PY PPO3_SOLE_PhysX/scripts/probe_contact_loss.py \
  --checkpoint logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt \
  --actuator rom --headless --device cuda:0 \
  --num_envs 32 --seed 17 --steps 960 --stage 5 \
  --eval_terrain --path_family s_curve --terrain_height_scale 1.0 \
  --report PPO3_SOLE_PhysX/reports/contact_safe_v1/prefix_ppo3_s_curve_seed17.json
```

`--actuator idfd` with `logs/rsl_rl/ppo4_sole_idfd/shared_v2_idfd/model_validated.pt`
produces the PPO4 row. Both runs have `sole_teacher_mode = "off"` and
`enable_contact_tracking_demo = False`, recorded in the report header.

---

## 1. The failure is real, large, and invisible to every existing gate

Held-out mesh, stage 5, path family `s_curve`, terrain height scale 1.0,
960 steps × 32 environments (30 720 scored samples per actuator):

| | PPO3 / ROM-A | PPO4 / ID-FD |
|---|---:|---:|
| `ground_contact_fraction` (gate ≥ 0.90) | **0.9486 PASS** | 0.8945 FAIL |
| post-acquisition contact fraction | 0.9527 | 0.9007 |
| contact-loss events | 428 | 458 |
| visible lifts (geometric gap > 5 mm) | 334 | 359 |
| impact/ejection events | 201 | 212 |
| hard impacts (> 30 kN) | 286 | 278 |
| takeoff events | 314 | 337 |
| **max consecutive airborne steps** | **45** (0.75 s) | **79** (1.32 s) |
| **max airborne gap** | **0.297 m** | **0.675 m** |
| p95 airborne gap | 0.178 m | 0.262 m |
| upward separations | 73.4 % | 73.6 % |
| max upward cutter velocity | 4.41 m/s | 7.93 m/s |
| mean in-contact force | 19.7 kN | 20.1 kN |
| p95 in-contact force | 35.4 kN | 37.7 kN |
| **max contact force** | **119 kN** | **170 kN** |
| p95 \|dF/dt\| | 1.18e6 N/s | 1.12e6 N/s |
| p95 / max cylinder target lead | 0.0262 / 0.0300 m | 0.0265 / 0.0406 m |
| min scan validity | 0.889 | 0.889 |
| limit-shield active fraction | 0.0000 | 0.0005 |
| min cylinder headroom | 0.0150 m | 0.0153 m |

**The first row is the point.** PPO3 passes the shipped `ground_contact` gate
at 0.9486 while spending three quarters of a second up to 30 cm off the verge.
`ground_contact_fraction` is a duty cycle: 428 short flights and continuous
light support are the same number to it. No previous campaign could have found
this by tuning against the existing gate set, because the existing gate set
says this policy is fine.

## 2. Mechanism attribution

Every post-acquisition `contact → no contact` transition, classified in the
order impact/ejection → invalid sensing → saturation → actuator lag →
path geometry (summed over the loaded reports; see `PREFIX_SUMMARY.md`):

| mechanism | PPO3 / ROM-A | PPO4 / ID-FD |
|---|---:|---:|
| impact / ejection | 804 | 848 |
| actuator lag | 776 | 820 |
| path geometry | 132 | 164 |
| **invalid sensing** | **0** | **0** |
| **cylinder / workspace saturation** | **0** | **0** |

Two of the five candidate mechanisms are ruled out outright, and not by
argument: minimum scan validity is 0.889 across the whole run, the limit
shield is active on 0.00–0.05 % of samples, and minimum cylinder headroom
never falls below 15 mm. Neither sensing nor saturation is involved in a
single event.

The remaining mass is split almost evenly between impact/ejection and
actuator lag, which is consistent with them being **one mechanism seen from
two sides** — see §4.

## 3. A representative event trace

`prefix_ppo3_s_curve_seed17.json`, environment 0, separation at step 95 and
again at 106. Every stored event carries a ±12-step window of all 20 scalar
and 6 per-cylinder channels, so this can be re-derived without re-simulating.

| step | contact | force (N) | geom. gap (m) | v_z (m/s) | target lead (m) | headroom (m) | scan | head align |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 89 | 1 | 17 753 | 0.0328 | +0.064 | 0.0254 | 0.0592 | 1.00 | 0.9997 |
| 90 | 1 | 17 753 | 0.0360 | +0.162 | 0.0261 | 0.0597 | 1.00 | 0.9997 |
| 91 | 1 | 5 565 | 0.0404 | +0.234 | 0.0260 | 0.0602 | 1.00 | 0.9999 |
| 92 | 1 | 9 354 | 0.0419 | +0.057 | 0.0261 | 0.0605 | 1.00 | 0.9998 |
| 93 | 1 | 20 956 | 0.0401 | −0.138 | 0.0251 | 0.0608 | 1.00 | 0.9996 |
| 94 | 1 | 20 956 | 0.0502 | +0.572 | 0.0249 | 0.0614 | 1.00 | 0.9998 |
| **95** | **0** | **0** | **0.0595** | **+0.525** | 0.0246 | 0.0617 | 1.00 | 0.9999 |
| 96 | 0 | 0 | 0.0599 | −0.004 | 0.0236 | 0.0621 | 1.00 | 0.9997 |
| 97 | 1 | **54 057** | 0.0503 | −0.594 | 0.0237 | 0.0614 | 1.00 | 0.9996 |
| 98 | 1 | 40 977 | 0.0549 | +0.292 | 0.0236 | 0.0614 | 1.00 | 0.9996 |
| … | | | | | | | | |
| 104 | 1 | 54 384 | 0.0583 | −0.134 | 0.0248 | 0.0615 | 1.00 | 0.9997 |
| 105 | 1 | 54 384 | 0.0677 | +0.595 | 0.0244 | 0.0620 | 1.00 | 0.9998 |
| **106** | **0** | **0** | **0.0854** | **+1.141** | 0.0257 | 0.0624 | 1.00 | 1.0000 |
| 107 | 0 | 0 | 0.1011 | +1.021 | 0.0272 | 0.0627 | 1.00 | 1.0000 |

Four things are visible at once:

1. **Force chatters between 5 kN and 54 kN with a period of a few control
   steps.** This is not a tool riding a verge; it is a tool hammering it.
2. **Separation follows the force peak by one step and leaves upward** —
   +0.525 m/s at 95, +1.141 m/s at 106, both immediately after 54 kN samples.
3. **The cylinder target lead is pinned at 0.023–0.027 m for the entire
   window.** It never relaxes. It is saturated against its 25 mm bound.
4. **Nothing else is wrong.** Scan validity 1.00, headroom 0.06 m, head
   alignment 0.9996–1.0000 throughout. The head-alignment problem that
   dominates `HANDOVER_PPO3_PPO4.md` is simply not present in this window.

The worst single event in the run (`env 1, step 508`) reaches a 0.242 m peak
gap after a 57.5 kN sample, separating at +1.32 m/s, with scan validity 1.0,
55 mm of headroom and head alignment 0.9993.

## 4. Measured root cause

**`cylinder_target_max_lead_m` is saturated at its inherited 25 mm on BOTH
actuators, and it is not an ID/FD field.**

It is read once, unconditionally, at
`PPO2_ROM_PhysX/runtime/mulag_upper_level_env.py:2454`, inside
`_run_cylinder_id_pipeline`, on the code path both ROM-A and ID/FD take. The
value comes from `PPO2_ROM_PhysX/tasks/env_cfg.py:123`, whose own comment says
"25 mm is needed only during the transient through a mechanically coupled
configuration" — a transient allowance that the terrain task runs at
continuously.

The arithmetic that makes this the mechanism is already in this repository's
notes. With the cylinders at ~5e6 N/m, contact force *is* commanded position:

| commanded lead | implied force |
|---|---|
| 0.40 mm | 2 kN |
| 3.9 mm | 19.7 kN (the measured mean) |
| 23.6 mm | 118 kN (the measured PPO3 maximum) |
| 25.0 mm | **125 kN — the bound** |

The measured peaks (119 kN on ROM-A, 170 kN on ID/FD — the latter exceeding
the bound because ID/FD's lead itself overshoots to 0.0406 m) are the order
the saturated integrator can deliver. The reference integrator winds up
against terrain the cylinders cannot follow, stores up to 25 mm of commanded
position error, and releases it into the ground as an impulse. The tool is
ejected, the integrator re-winds during the flight, and the cycle repeats —
which is why "impact/ejection" and "actuator lag" split the event mass almost
evenly. **They are the same loop measured before and after the impact.**

Two secondary findings:

* **The binary contact reward pays for this.** `terrain_contact_reward_weight`
  was raised 2.0 → 4.0 on 2026-07-31 to reduce skipping. A binary bonus on
  measured contact pays a 119 kN impact exactly what it pays 6 kN of stable
  support, and the impact is what causes the separation one step later. The
  measured stable-contact force is 18.9 kN mean, against a 2 kN design target.
* **The force observation cannot see any of it.** The policy's force channel
  is `clamp(F / 1000, 0, 10)`. Every force above 10 kN is the same input, and
  the in-contact distribution has a 19.7 kN mean. There is no force-rate
  channel at all, so an approaching impact is not merely hard to see — it is
  not represented.

## 5. A methodological finding that invalidates past multi-seed claims

`--seed` **cannot vary a held-out evaluation.** The held-out mesh is pinned at
seed 1337 (`terrain_env.py::_mow_strip_cfg`, deliberately, so a stage-5 score
means the same thing across runs), `randomize_start_phase` is False, and the
actor is evaluated at its mean. The rollout is therefore fully deterministic.

Measured, not inferred: the first pass of this diagnostic ran seeds 17, 909,
1337 and 4242, and all four returned **bit-identical** event counts, gaps,
forces and target leads. Those four runs are preserved in
`prefix_identical_seed_artifact/` as the evidence.

`terrain_eval_seed_override` (`--eval_terrain_seed`) is new and is the only
way to move the held-out mesh. With it, seed 909 gives PPO3 408 events /
289 lifts / 0.292 m max gap / 36 airborne steps against seed 17's
428 / 334 / 0.297 / 45 — genuinely different, and the same conclusion.

## 6. What this rules in and out for the fix

Ruled **out** by measurement, before any tuning:

* invalid sensing (0 events; min scan validity 0.889)
* cylinder/workspace saturation (0 events; shield ~never active, ≥15 mm headroom)
* head alignment as a cause (0.9996–1.0000 through every trace examined)

Ruled **in**:

* stored command energy released as impact (saturated 25 mm target lead)
* an objective that pays for impacts as much as for support
* an observation that cannot represent either the force or its rate
* a gate set in which none of the above is measurable

The fix therefore has to change four things together — the bound on stored
command error, the reward, the observation, and the gates. Changing only the
contact reward weight, which is what the shipped configuration tried, moves
the policy further into the failure.
