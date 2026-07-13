# Lower-level ID→FD tracker — measured results

Branch: `fix/lower-level-idfd-tracker`  ·  Bench: `ID_FD_pipline/ppo1/tests/lower_level_bench.py`
Env: `Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0` (4-D action == cylinder DLS action; no state
machine, no EE IK, no truck). `step_dt = 0.016667 s` (sim dt 1/120, decimation 2).
Raw traces + JSON summaries: `outputs/lower_level_tracker/`.

Status legend: RUNTIME TESTED = executed in Isaac Sim and logged.

---

## 0. Headline

**The lower-level controller cannot track because the PhysX cylinder drives have
essentially no position-target authority at the mowing pose.** All four cylinder
actuators are permanently effort-saturated. This is upstream of ID, FD, the guards, the
Jacobian, the trajectory profile and the models. No controller — learned, PID or
DirectCylinderDriver — can track a cylinder reference on this plant as configured.

A second, independent finding: **endpoint A and the whole ground-band map are *sag*
poses, not commandable poses.** They record where the arm fell to while its drives were
saturated, not where it was told to go.

---

## 1. Actuator configuration is internally inconsistent  [RUNTIME TESTED]

From `env.robot.actuators` and `robot.data`:

| cylinder | joint | stiffness | effort_limit | saturates beyond |
|---|---|---|---|---|
| 1 | `Drehzapfen_joint` | 7.5e7 | 5.0e4 | **0.67 mrad** |
| 2 | `Ausleger_I_joint` | 3.0e7 | 1.0e5 | **3.3 mrad** |
| 3 | `Ausleger_II_joint` | 5.0e6 | 5.0e4 | **10 mrad** |
| 4 | `Messerkopf_Schwenk_joint` | 5.0e6 | 2.0e4 | **4.0 mrad** |

`saturates beyond = effort_limit / stiffness` — the tracking error at which the implicit
PD drive clips to maximum force. Every drive is a bang-bang actuator for any error larger
than a few milliradians.

Measured at the **reset pose, merely holding still at its own measured position**:

```
gravity tau  : [   0.2,      2.3,     0.3,    -2.2]   N·m
computed tau : [-16520.9, -114754.4, -12619.2, -924.9]
applied  tau : [-16520.9, -100000.0, -12619.2, -924.9]   <- cyl2 CLIPPED
effort limit : [ 50000.0,  100000.0,  50000.0, 20000.0]
```

The static gravity load is ~2 N·m, but the PD demands 1.1e5 N·m to hold a 3 mrad error.
Cylinder 2 is already at its effort limit while doing nothing.

Measured at **endpoint A** (after DirectCylinderDriver pre-positioning):

```
computed tau : [14653522.0, 2214866.0, 104524.6, 390686.2]
applied  tau : [   50000.0,  100000.0,  50000.0,  20000.0]   <- ALL FOUR CLIPPED
```

Cylinder 1 demands **293× its effort limit**. All four drives are pinned at maximum force.

## 2. Plant authority at endpoint A is zero  [RUNTIME TESTED]

`--mode plant`: DirectCylinderDriver commands `target = A + δ`, settles 4 s, measures.
This bypasses ID, FD, the DLS and the guards entirely — it is the plant's own
position-drive response.

```
cyl1  δ=-50mm -> +0.021mm   δ=-10mm -> +0.007mm   δ=-5mm -> +0.009mm
      δ=+ 5mm -> +0.009mm   δ=+10mm -> +0.019mm   δ=+50mm -> +0.020mm
cyl2  δ=-50mm -> +0.041mm   δ=+50mm -> -0.946mm
cyl3  δ=±any  ->  0.000mm   (at its soft lower limit, q = -0.160 = q_soft_lo)
cyl4  δ=-50mm -> -0.313mm   δ=+50mm -> -1.803mm
```

**d(measured)/d(target) ≈ 0.00 for every cylinder, in both directions, out to ±50 mm.**
The arm is frozen in a saturated equilibrium. Note several gains are *negative*: the
residual motion is coupling/settling noise, not command response.

This alone explains every downstream symptom. It is not a controller bug.

## 3. Endpoint A is a sag pose, not a commandable pose  [RUNTIME TESTED]

Effort-limit sweep at the ground-band endpoint-A cylinder command
(`x = [0.1353, 0.4147, 0.6780, 0.6014]`):

| effort × | reached `|q_err|` (rad) | saturated? | resulting cutter (truck frame) |
|---|---|---|---|
| **1 (authored)** | [0.195, 0.071, 0.000, 0.060] | all 4 | (5.456, −4.311, 1.312) |
| 4 | [0.040, 0.000, 0.000, 0.005] | 1,2 | (2.893, −2.783, 0.495) |
| **16** | [0.001, 0.025, 0.000, 0.002] | none | **(2.283, −2.240, 0.779)** |
| 1024 | diverges | none | (7.254, 2.605, 5.762) — unstable |

At the authored effort limits the arm **never reaches the commanded configuration** — it
stalls 0.195 rad away and settles at cutter (5.46, −4.31, 1.31). Give it enough force to
actually execute the command and the cutter goes to **(2.28, −2.24, 0.78)** — over 3 m
away.

`ground_band_map.json` records `target_x` (commanded) and `cylinder_x` (settled) with
gaps up to **0.25 m**. Those entries are sag equilibria. Consequences:

* The arc geometry (centre 4.327, −3.224; r = 1.286; θ ∈ [−26.7°, 9.3°]) was circle-fitted
  to sag poses.
* The 0.88 m "max safe arc" is a property of where the arm falls, not where it can go.
* `ee_jacobian_ground_truck3x4.json` was measured by perturbing targets at a pose where
  targets do nothing — so it is a Jacobian of noise.
* The cutter-at-ground band is achieved by the arm *drooping* onto the ground, which is
  also why it looked like it "reaches" the ground so conveniently.

## 4. Where the plant *does* have authority, the ID→FD loop is unstable  [RUNTIME TESTED]

At the **reset pose** the drives are near-linear (`target − measured` offset ≤ 3.6 mm) and
the plant moves 235–1275 µm per control step. Cylinder step/sine matrix through the full
ID→FD chain (`--mode cyl --pose reset`, PPO residual = 0, truck frozen):

| probe | p95 err | final err | goal | verdict |
|---|---|---|---|---|
| cyl1 +5 mm step | 41.3 mm | **−34.7 mm** | 5 mm | overshoots ~8× and reverses |
| cyl2 −5 mm step | 46.3 mm | −33.1 mm | −5 mm | overshoot |
| cyl2 sine 0.10 Hz ±10 mm | 107.2 mm | −87.1 mm | ±10 mm | diverges |
| cyl3 +10 mm step | 23.8 mm | −24.7 mm | 10 mm | overshoot |
| cyl4 +10 mm step | 56.6 mm | −55.6 mm | 10 mm | overshoot |

Every cylinder overshoots and limit-cycles. **Cylinder-level acceptance (p95 < 10 mm) fails
on all four.** Mechanism: the DLS asks for `clamp(0.1·err, ±10 mm)` ≈ 0.5 mm/step of FD
delta for a 5 mm error, but the plant actually moves 0.6–1.3 mm/step — the true
action→motion gain is ~2.6× what the controller's inverted FD Jacobian believes. The DLS
inverts *FD's predicted delta*, not the plant's response to the integrated target, and an
open target integrator in series with a lagging saturating drive closes an over-gained
loop. Valve sits at 0.62–0.87 (near saturation) and the direction guard zeroes 12–51% of
FD outputs, adding a hard nonlinearity.

## 5. Static hold at endpoint A — a false pass  [RUNTIME TESTED]

| handoff | EE mean | EE p95 | EE max | cyl p95 | drift/10 s |
|---|---|---|---|---|---|
| target-synced | 1.7 mm | 2.5 mm | 2.6 mm | 1.4 mm | 1.5 mm |
| measured-synced (production) | 19.1 mm | **61.3 mm** | 76.5 mm | 5.8 mm | 6.6 mm |

The target-synced hold appears to pass the static gate — but §2 shows it passes because
the arm is *stuck*, not because it is controlled. This is a vacuous gate; do not count it.

The measured-synced result is nonetheless a real production bug:
`ID_FD_pipline/ppo1/env.py:438` sets `_cylinder_pos_target = measured joint pos` after
pre-positioning. The plant was in equilibrium under the *commanded* target, which differs
from measured by up to 0.19 m, so the drive command drops by that much at handoff and the
arm sags ~7 cm. The sampler anchor and IK seed anchor are captured *before* that sag
(`env.py:417-435`), so the ~7 cm offset is then baked into the arc goal permanently. That
matches the reported 0.10–0.13 m residual static EE error.

## 6. Trajectory time-profile bug  [CONFIRMED by inspection]

`trajectories/arc_mowing.py` applies quintic scaling to `tau = stroke_dist / L_arc`, where
`stroke_dist` is already advanced at constant path speed by `TrajectorySampler.advance()`.
The path is therefore **time-scaled twice**: the geometric EE speed is
`path_speed · h_v(tau)`, i.e. 0 at both endpoints and 1.875 × path_speed mid-stroke.

At the reported `norm_prog = 0.0511`: `h_p = 10τ³ − 15τ⁴ + 6τ⁵ = 0.00125`, i.e. **1.1 mm of
geometric displacement on an 0.88 m arc.** The 10 s run really did contain ≈ 0 mowing —
that part of the diagnosis holds.

Also: `linear_velocity_b` / `linear_acceleration_b` are derivatives with respect to **path
distance**, not time (see the `dp/ds` derivation at `arc_mowing.py:98-128`), so they are
dimensionless/`1/m`, not m/s and m/s². Anything consuming them as velocity feedforward
would be wrong by a factor of the path speed.

## 7. Truck

`enable_scripted_vehicle_motion = False` and `ParkingBrake.engage()` every step: truck root
displacement measured at **0.94 mm over the 10 s hold** (well inside tolerance), no wheel
velocity targets issued. Truck stationarity is not a contributor. A full 60 s zero-action
proof is implemented (`--mode truck`) but was not run — it is not on the critical path
while §2 stands.

---

## 8. Bottleneck ranking (runtime evidence)

1. **Cylinder actuator effort limits vs. stiffness (BLOCKING).** All four drives saturated;
   zero position-target authority at the mowing pose. Fix before anything else.
2. **Ground-band map / arc geometry / ground Jacobian are all derived from sag poses
   (BLOCKING).** They must be re-derived from poses the arm can actually command and hold,
   once (1) is fixed.
3. **DLS loop is over-gained and unstable** where the plant does respond (reset pose).
   Real, fixable, but not measurable until (1) and (2) are fixed.
4. **Handoff syncs the PhysX target to *measured* instead of the commanded target**
   (`ppo1/env.py:438`), and the anchors are captured before the resulting sag.
5. **Double time-scaling in the arc trajectory** + velocity fields carrying `d/ds`, not
   `d/dt`.

## 9. Answers to the 12 required questions (as far as evidence currently supports)

1. *Apparent freeze mainly caused by short run + time profile?* **Partly.** The profile bug is
   real (1.1 mm at the observed progress), but even a correctly sized run cannot mow: the
   plant has no authority (§2).
2. *Can ID produce both valve directions?* **Yes** — both signs observed, valve 0.6–0.9, no
   sign lock.
3. *Can FD produce the required cylinder deltas?* FD emits deltas of the right order
   (~1e-4 m/step) in both signs; the direction guard zeroes 12–51% of them.
4. *Is the composite IDFD map full-rank along the arc?* **Not yet answerable, and not yet
   meaningful** — a map measured at a pose where the plant cannot move is a map of noise.
5. *Can the controller hold endpoint A?* Only vacuously (the arm is stuck). **No** as a
   control result.
6. *Track the teacher cylinder trajectory?* **No** (§4: p95 15–107 mm on ±5 mm steps).
7. *Track the task-space arc?* **No.**
8. *Which guards bind?* Direction guard (12–51% of steps), valve near saturation
   (0.62–0.87), FD ratio cap rarely (<8%). Neutral hold never fired.
9. *Are the checkpoints adequate?* **Unproven either way** — cannot be judged on an
   uncontrollable plant.
10. *Is fine-tuning required?* **No evidence yet that it is.** Do not fine-tune against this
    plant: any data collected now encodes the saturation.
11. *Max deterministic speed / arc length that passes?* **None.** No gate passes.
12. *Is the truck provably stationary?* Yes for the arm bench (0.94 mm / 10 s, no wheel
    commands); full 60 s proof implemented, not yet run.
