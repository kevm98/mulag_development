# Reference consistency check (offline)

Question: are the two references the policy is judged against mutually consistent?

* **Reward target** — `_original_reference_tool` = `cutter_truck` from the
  trajectory library, compared in the truck-chassis frame against
  `_compute_ee_pos_local()`.
* **Clearance gate target** — `_original_reference_world_z` = `cutter_world_z`
  from the same library rows, compared in world coordinates against
  `_compute_cutter_world_z()`.

Both are read from the same sample row (`policy/env.py:241-243`), so any
disagreement is authored into the library, not introduced at runtime.

**Answer: they are not consistent, and the worst inconsistency sits exactly where
the failures are.**

---

## 1. The two cutter definitions are not the problem

There are two cutter computations in `tasks/cylinder_env.py` and they differ:

| | definition | orientation-aware |
|---|---|---|
| `_compute_cutter_pos_w` (feeds the reward) | `ee_pos_w + quat_apply(ee_q, [0,0,-0.79])` | yes |
| `_compute_cutter_world_z` (feeds the gate) | `ee_origin_z + (-0.79)` | **no** |

They agree only when the head is exactly vertical, differing by
`0.79·(1 − cos θ)`. At the observed failure alignment of 0.9996 that is **0.3 mm**.
Not the explanation. The docstring already flags the split as deliberate.

## 2. The plant-projection gap is not the problem either

The library carries `cutter_truck` ("exact requested geometry"),
`desired_cutter_truck`, and `plant_projected_cutter_truck` (what the plant can
actually reach). If the reward chased an unreachable ideal while the gate measured
the reachable one, that would explain a standing offset. It does not:

| shape | mean z-gap | max z-gap | max 3-D gap |
|---|---:|---:|---:|
| line | +0.0001 | +0.0122 | 0.0134 |
| arc | −0.0000 | −0.0035 | 0.0065 |
| sine | −0.0003 | −0.0361 | 0.1060 |
| circle | +0.0000 | +0.0126 | 0.0238 |
| square | +0.0005 | +0.0195 | 0.0223 |

For `line` the ideal and the reachable point agree to 13 mm. Ruled out.

## 3. The world-Z reference is inconsistent with the truck-frame reference

For a rigid, level, parked truck, `cutter_world_z` and `cutter_truck[2]` must
differ by a **constant** — the truck base height. Across all 3615 library samples
that difference has sd 0.0439 m; it is not constant for any shape.

`line` is the cleanest demonstration, because its truck-frame reference is exactly
flat:

```
cutter_truck[2]  : one unique value across all 241 samples -> 1.550000
cutter_world_z   : min 0.0333  max 0.1056  range 0.0722 m  sd 0.0125
```

The commanded height is dead constant while the world-Z reference swings 7.2 cm.

## 4. The worst of it is in the failure band

First twelve `line` samples, ordered by path position:

| path_s (m) | world_z ref | step |
|---:|---:|---:|
| 0.0000 | 0.0746 | |
| 0.0104 | **0.0333** | **−0.0413** |
| 0.0208 | 0.0665 | **+0.0332** |
| 0.0312 | 0.0574 | −0.0091 |
| 0.0417 | 0.0589 | +0.0014 |
| 0.0521 | 0.0607 | +0.0018 |
| 0.0625 … 0.1146 | 0.0618 → 0.0654 | +0.0006 … +0.0026 |

| segment | sd | range |
|---|---:|---:|
| `path_s <= 0.06` | 0.0127 | **0.0413 m** |
| `0.5 <= path_s <= 2.0` | 0.0035 | 0.0169 m |

There is a **4.1 cm single-sample notch at `path_s = 0.0104`**, after which the
reference settles into millimetre-scale steps. `_interpolate_trajectory` linearly
interpolates this column, so the runtime reference inherits the notch.

**15 of the 18 captured clearance failures occurred at `path_s` = 0.011–0.048 m,
with clearance positive (cutter too high) in every case.** A reference that dips
4 cm in that band raises measured clearance by 4 cm — 20 % of the 0.20 m limit —
at precisely the path position where the arm decelerates into the endpoint and
therefore dwells for many timesteps.

That is not the whole story, but it is a real defect and it is concentrated
exactly on the failure mode.

## 5. What this does and does not settle

**Settled:**

* The reward target and the clearance target are not mutually consistent.
* The inconsistency is authored into `cutter_world_z`, is largest at the line path
  origin, and coincides with the failure location and sign.
* Neither the dual cutter definition (0.3 mm) nor the plant-projection gap (13 mm)
  explains it.

**Not settled:** the ~0.17 m *baseline* clearance offset. The notch is 4 cm; the
observed clearance sits at 0.16–0.19 before the drift even begins. Explaining the
remaining ~0.13 m needs the runtime truck base height, which was not being logged.
It is one of two things:

1. a static frame offset between the authoring configuration and this environment's
   spawn — i.e. a specification bug, in which case no shaping, randomization or
   retraining fixes it; or
2. genuine standing tracking error in world Z that the truck-frame reward does not
   see.

These have opposite implications, so the distinction matters more than anything
else currently on the plan.

## 6. Instrumentation added to close it

`_PreTerminationRecorder` now also captures, per step per environment:
`reference_world_z`, `cutter_world_z`, `ee_origin_world_z`,
`truck_base_world_z`, `reference_cutter_truck` (3), `measured_cutter_truck` (3).

That is exactly the decomposition:

```
clearance = cutter_world_z - reference_world_z
          = (truck_base_world_z + measured_cutter_truck[2] - reference_cutter_truck[2])   <- tracking
          + (reference_cutter_truck[2] + truck_base_world_z - reference_world_z)          <- reference disagreement
```

One short probe on a known failing seed resolves which term carries the 0.17 m.
27 offline tests pass, including a new assertion that all six fields are captured.

## 7. Consequence for the plan

If term 2 dominates, the ordering changes completely: regenerate or repair
`cutter_world_z` first, and every prior clearance measurement — including
`model_53`'s acceptance and the `model_607` control arm — has to be re-read against
the corrected reference. Remediation training on the current reference would be
fitting a policy to a defective target.

Recommend running the decomposition probe before the control arm. It is ~2 minutes
of compute:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/diagnose.py \
  --headless --device cuda:0 \
  --num_envs 128 --eval_steps 120 --seed 17 \
  --suite monte_carlo --rounds 1 --trajectory_shapes line
```
