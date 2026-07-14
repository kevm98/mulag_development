# ID/FD wiring audit

Scope: ordering, signs, units, normalization, timing, FD output semantics — **before** any
tuning or training. Scripts:

* `ppo1/tests/wiring_audit_offline.py` — source-data + learned-FD matrices (no Isaac)
* `ppo1/tests/wiring_audit_runtime.py` — runtime-composite + direct-PhysX matrices, guard ablation

---

## 1. Ordering and layout — CLEAN

| | layout | matches runtime? |
|---|---|---|
| ID input (28) | `Extension(4), ExtensionVelocity(4), dP(4), Force(4), TargetDeltaExtension(4), TargetDeltaExtensionVelocity(4), PreviousValveCmd(4)` | ✅ `build_id_input` emits exactly this order |
| ID output (4) | `ValveCmd_1..4`, `target_mode = normalized_delta_valve_cmd` | ✅ runtime adds `prev_valve_cmd` (delta semantics honoured) |
| FD input (20) | `ValveCmd(4), Extension(4), ExtensionVelocity(4), dP(4), Force(4)` | ✅ `CylinderFDRuntime.step` concatenates in this order |
| FD output (12) | `DeltaExtension_next(4), DeltaExtensionVelocity_next(4), DeltaDP_next(4)` | ✅ sliced `[0:4] / [4:8] / [8:12]` |

The residualisation `zero_input[:, 16:24] = 0` correctly zeroes exactly the two target blocks
for both the 24- and 28-wide layouts.

Cylinder↔USD scaling is a clean affine map (`q = scale·x + offset`, `scale = −1` on all four).
Normalizers in both checkpoints match the source-data statistics **exactly** (max diff 0.0).

**No ordering, sign-convention or normalization defect found.**

## 2. Timing / units — TWO DEFECTS

Source data: `sample_time = 0.0500 s`, **exactly constant** across all 411 840 samples.
Runtime: `_learned_model_dt = _control_dt = 1/60 = 0.01667 s`.

### 2a. `TargetDeltaExtensionVelocity` is the wrong physical quantity

The dataset's own `target_definition` states:

> "TargetDeltaExtension/Velocity is the **ACHIEVED transition** Extension(k+1)−Extension(k) /
> **ExtensionVelocity(k+1)−ExtensionVelocity(k)**"

So feature block 20:24 is a **velocity increment** `Δẋ = ẋ(k+1) − ẋ(k)`.

The runtime instead computes (`PPOActionToCylinderTargetAdapter.convert`, mode `cmd_minus_xdot`):

```
desired_next_xdot = target_delta_x / learned_model_dt
target_delta_xdot = desired_next_xdot - xdot_current        # a velocity ERROR
```

These are different quantities. The data proves it:

* `corr(TDX, TDXV)` in training data = **[0.004, 0.004, 0.001, −0.045]** — the two features are
  *independent*. At runtime they are deterministically linked (`TDXV = 60·TDX − ẋ`).
* The runtime formula fails to reproduce the data's TDXV by **407–729 %** of its std at
  `dt = 1/60`, and still by 24–37 % at `dt = 0.05`.

### 2b. `TargetDeltaExtension` is built over the wrong interval

`TDX` at runtime is the delta wanted over one control step (1/60 s); in training it is the
delta over 0.05 s. A 3× magnitude mismatch on a feature whose training std is only
`[0.0066, 0.0077, 0.0245, 0.0177]`.

FD's output *is* correctly rescaled (`cylinder_fd_scale_delta_by_sample_time = True` →
`× step_dt/0.05 = 1/3`), so the **FD output semantics are handled; the ID input semantics are
not.**

## 3. FD valve gain is 9–19× too large — MODEL DEFECT

`∂(DeltaExtension)/∂(ValveCmd)`, identical perturbation `ε = 0.05`, averaged over 4000 real
training states (not the data mean — that is the valve deadband and gives a misleading local
slope):

```
SOURCE DATA (OLS, R^2 = 0.95-0.98, VIF ~ 1.0 so the partials are trustworthy)
   -0.000094  -0.000002  -0.000017  -0.000008
   +0.000001  -0.000052  -0.000009  -0.000007
   -0.000027  -0.000003  -0.000449  +0.000039
   +0.000005  -0.000017  +0.000084  -0.000306
   diag = [-9.4e-05, -5.2e-05, -4.49e-04, -3.06e-04]   cond 9.2

LEARNED FD (mean over 4000 real data states)
   -0.001687  +0.000008  -0.000061  +0.000037
   -0.000068  -0.000977  +0.000006  -0.000118
   +0.000060  +0.000196  -0.004881  -0.000462
   +0.000064  -0.000032  +0.000483  -0.002833
   diag = [-1.69e-03, -9.77e-04, -4.88e-03, -2.83e-03]  cond 5.0

cosine similarity  +0.97      diagonal sign agreement  4/4
diag ratio FD/data  [18.0, 18.8, 10.9, 9.3]
```

Structure, signs and conditioning are **right**. The **gain is 9–19× too large**.

FD still scores **R² = 0.993–0.997** on its own training data — because `ExtensionVelocity`
dominates the variance of `ΔExtension` (`Δx ≈ ẋ·dt`), so the network can predict `Δx` almost
perfectly while getting the *valve sensitivity* badly wrong. That sensitivity is a small
correction term in the fit — and it is **precisely the term the controller depends on**.

This is why the previous "FD is rank-2 → rank-promoting fine-tune" work did not help: it
treated a symptom (conditioning) and never checked the gain against the source data.

> **Consequence for training:** an FD trained only on one-step `ΔExtension` MSE is not
> constrained where it matters. The fine-tune must include an explicit Jacobian /
> valve-sensitivity term, which is what the identity-composition loss provides.

## 4. The four response matrices (identical perturbations)

`outputs/lower_level_tracker/wiring_audit_runtime.json`.

### 4a. direct-PhysX — the teacher is perfect

`∂(measured x)/∂(target x)`, ε = 5 mm, ID/FD/DLS bypassed:

```
   +1.004645   -0.002918   -0.075224   -0.007179
   +0.021583   +0.985098   -0.045085   -0.010979
   +0.013310   -0.012326   +0.982159   -0.010604
   -0.005656   +0.001931   +0.012875   +0.995320
   diag = [1.005, 0.985, 0.982, 0.995]      cond 1.1   rank 4
```

The repaired plant is **essentially the identity**. This is the teacher the composite must
reproduce.

### 4b. runtime-FD matches offline learned-FD — inputs are fed correctly

`∂(FD delta)/∂(valve)` at the live runtime state: `diag = [-0.0035, -0.0025, -0.0066, -0.0013]`,
cond 6.7 — same order, same signs as the offline learned-FD matrix. `dP` and `force` are set to
the training means exactly. ID input z-scores are healthy: `max|z| = 2.67`, nothing beyond 3σ.

**No FD input-wiring defect, no out-of-distribution defect.**

### 4c. runtime-composite — the identity contract fails by 2–250×

`∂(applied FD delta)/∂(target_delta_x)` — this **must be `I`**:

```
UNGUARDED  (the map the DLS builds and inverts)
   +0.011930   -0.000664   -0.011332   -0.038767
   -0.001281   +0.001071   -0.000354   +0.044937
   -0.042522   +0.029071   +0.127548   +0.352957
   -0.019801   -0.008784   -0.018880   +0.459913
   diag = [0.012, 0.001, 0.128, 0.460]      cond 482    max|J - I| = 0.999

GUARDED    (the delta the PLANT actually receives)
   diag = [0.012, 0.004, 0.153, 0.460]      cond 127    max|J - I| = 0.996
   -- every off-diagonal is exactly ZERO (see 4d)
```

The composite delivers **0.4 %–46 %** of the commanded cylinder delta. It is also strongly
state-dependent: at a second pose the diagonals of cylinders 1 and 3 came out **negative**
(`diag = [-0.018, +0.156, -0.021, +0.208]`) — commanding +Δx produces −Δx.

That is the complete explanation of the tracking failure: the DLS closes a loop on a map that
delivers a few percent of what it asks for, in a sign that flips with pose.

### 4d. Guard ablation — the ratio cap never binds; the direction rule zeroes everything

The guard is `same = delta*cmd >= 0; limit = ratio*|cmd|; delta = same ? min(|delta|,limit) : 0`.

For a **single-cylinder** command, every off-diagonal has `cmd_i = 0`, hence
`limit = ratio × 0 = 0` — so the guard forces **all off-diagonal response to exactly zero**,
for any ratio. And any diagonal whose sign disagrees with the command is zeroed outright,
which is why cylinders 1 and 3 were completely dead in the step tests.

| ratio | guarded diag | max\|J − I\| |
|---|---|---|
| 1.5 | [0.012, 0.004, 0.153, 0.460] | 0.996 |
| 2.0 | [0.012, 0.004, 0.153, 0.460] | 0.996 |
| 3.0 | [0.012, 0.004, 0.153, 0.460] | 0.996 |

**Identical.** The ratio cap never binds — the responses (1–46 % of commanded) sit far below
the 150 % cap. Raising the ratio, as previously proposed, does nothing at all.

### 4e. The DLS inverts a different map from the one the plant receives

`CylinderDLSController.predict_fd_delta()` — used by `local_model()` to build the Jacobian the
DLS inverts — does **not** apply the direction guard. `_run_cylinder_id_pipeline()` does.
The controller therefore inverts the *unguarded* map (cond 482, full off-diagonals) while the
plant is handed the *guarded* one (strictly diagonal). These are different matrices.

## 5. Verdict

| item | status |
|---|---|
| feature ordering | ✅ clean |
| sign conventions | ✅ clean |
| normalization | ✅ exact match to source data (max diff 0.0) |
| FD output semantics + 0.05→1/60 s rescaling | ✅ correct |
| FD runtime inputs (dP/force/state) | ✅ correct, in-distribution (max\|z\| = 2.67) |
| direct-PhysX teacher | ✅ identity, cond 1.1 |
| **ID target-velocity feature** | ❌ **wrong quantity** — velocity *error* vs velocity *increment* |
| **ID target-delta timing** | ❌ **wrong interval** — 1/60 s vs the data's 0.05 s |
| **FD valve gain** | ❌ **9–19× too large** vs source data |
| **direction guard** | ❌ zeroes ALL off-diagonal response; ratio cap never binds |
| **DLS inverts the unguarded map** | ❌ plant receives the guarded one |
| **composite ≈ identity** | ❌ **delivers 0.4–46 % of command, sign flips with pose** |

## 6. What this means for the fix

* **Gain tuning cannot work** and must not be attempted: the composite is off by 2–250× with a
  pose-dependent sign. No Kp/Kd repairs that. (Confirmed empirically: 12 Kp×Kd combinations
  all overshot 7–11×.)
* **Raising the guard ratio cannot work**: it never binds. The guard must be *removed* from the
  main experiment; it is a per-axis rule applied to an inherently cross-coupled map.
* **One-step MSE training cannot fix FD's gain**: FD already scores R² = 0.993–0.997 while its
  valve gain is 18× wrong. The loss must constrain the *Jacobian* (identity-composition), not
  just the prediction.

Fix order: (1) remove the direction guard, (2) fix the ID target-feature wiring
(quantity + interval), (3) collect excitation data over the repaired commandable workspace,
(4) fine-tune FD with an explicit valve-Jacobian term, (5) train ID through the **frozen** FD
with identity-composition + multi-step rollout losses, (6) gate all four cylinders against the
direct-PhysX teacher (hold / step / sine / coupled / rollout) before any tool-space work.

### Constraint on "excitation data from the plant"

The simulator's cylinders have **no valve input** — the only actuator interface is the PhysX
joint position target. Valve→Δx labels therefore cannot be collected from the plant, and
fabricating them from `DirectCylinderDriver` is forbidden (and circular: `set_debug_valve_cmd`
routes valve→Δx *through FD itself*). So:

* FD's valve labels stay the **Simscape hydraulic data** (411 840 samples, physically valid),
  fine-tuned with a valve-Jacobian loss against the source-data partials.
* The **repaired plant** supplies the *state distribution* and the achievable Δx per step, and
  is the teacher for the identity-composition and rollout losses.
