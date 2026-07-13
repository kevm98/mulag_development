# Historical investigation: rank-deficient FD model

> This is a preserved diagnostic snapshot, not the current runbook. The FD,
> cylinder controller, EE trajectory controller, and residual-RL gates now pass.
> Use the repository-level `ID_FD_pipline/README.md` for current commands and
> architecture.

## TL;DR of the task
Fine-tune the existing **FD cylinder model** on the **ORIGINAL** training data with a
**rank-promoting objective**, so that `d(fd_delta)/d(valve)` becomes **full rank (4/4)**
instead of the current rank ~2. This single defect is the root cause of every
downstream failure (RL and classical control alike). Do not touch RL until the gate passes.

---

## 1. Background: what was diagnosed (do not re-derive)

The ID_FD pipeline is: `PPO action -> target_delta_x -> ID(valve) -> FD(delta) -> PhysX target`.

Stage-wise Jacobian measurement (`ID_FD_pipline/tests/rank_gate.py`) showed:

| stage | rank | cond | verdict |
|---|---|---|---|
| action -> target_delta_x (adapter) | 4/4 | 1 | healthy |
| action -> valve (**ID model**) | 4/4 | 53 | healthy |
| action -> fd_delta (**FD model**) | **~2/4** | **243** | **THE BUG** |
| action -> cylinder_x | ~2/4 | 301 | inherits FD defect |

**The FD model collapses a full-rank 4-D valve command into ~2-D cylinder motion.**
Consequence: arbitrary 4-D cylinder goals (and 3-D EE goals) are unreachable, so RL
always collapsed to a "hold still" policy and every classical controller (P, DLS in
cylinder space and EE task space) failed to track. It was never an RL/reward/tuning
problem.

### Why the FD is rank-2: it UNDER-FIT the valve (NOT a data deficiency)

Critically, the **original training data already contains full-rank valve information**.
Measuring the valve's *partial* effect (regress the state out first, since the plant's
gravity/coupling dynamics dominate the raw signal):

```
ORIGINAL data, valve -> DeltaExtension, state controlled:
  partial-effect singular values: [0.0103, 0.0099, 0.0061, 0.0049]
  cond = 2.1        <-- WELL-CONDITIONED: all 4 valve directions are identifiable
```

So all four valve directions ARE learnable from the existing data. The valve's effect is
**real and full-rank but small** relative to the state-driven dynamics (partial corr only
~0.04-0.08). An MSE-trained model fits the big state-driven term and lets the valve
pathway collapse — exactly what happened. The plant is fine: the user confirmed in
Simscape that **cylinders 3 and 4 do have hydraulic authority** (a step on valve 3/4 moves
them). Raw `corr(valve, motion)` reads ~0 only because the dynamics swamp it.

### DEAD ENDS — do not repeat
- The later excitation CSV/parquet was proven response-identical across valve
  channels and was removed during the final cleanup. Do not recreate or train
  from that dataset.
  is **BROKEN**. The plant response is bit-identical across all excitation channels
  (including all-valves-zero), i.e. the valve never metered flow. **DO NOT TRAIN ON IT.**
  Training on it makes the model strictly worse (it has *less* valve information than the
  original: 1/4 usable directions vs 2/4).
- Do NOT re-tune RL (leak, exploration, entropy, net size, reward penalties, action_scale)
  — all were exhausted and all fail while the FD is rank-deficient.
- Do NOT redesign around a "2-DOF arm" — the arm is genuinely 4-DOF controllable.

---

## 2. Exact artifacts and paths

### The FD model to fix
```python
from ID_FD_pipline.models import CYLINDER_MODELS_ROOT, OLD_FD_CHECKPOINT
# OLD_FD_CHECKPOINT = "fd_cylinder_space /runs_fd_cylinder/fd_cylinder_rollout_v2/best_fd_cylinder.pt"
ckpt_path = Path(CYLINDER_MODELS_ROOT) / OLD_FD_CHECKPOINT
```
Note: `CYLINDER_MODELS_ROOT` currently resolves to
`/home/rrlab/rrlab_mulag_newobs_pruned_20260701/kevin_integration/new_observations_pipeline/Models_cylinder /`
(note the **trailing spaces** in the dir names — they are real).

### The model loader (REUSE IT, do not re-implement the architecture)
`<CYLINDER_MODELS_ROOT>/fd_cylinder_space /fd_cylinder_common.py` provides:
```python
load_model_from_checkpoint(path, device=...) -> (model: FDCylinderMLP, checkpoint: dict, normalizers: dict)
# normalizers = {"x_mean"(20), "x_std"(20), "y_mean"(12), "y_std"(12)}  as torch tensors
build_model(config) -> FDCylinderMLP     # class at line ~340, builder at ~407
normalize_x / normalize_y / denormalize_y
load_fd_data(path) -> FDData
```
Import it the same way the runtime does (see
`kevin_integration/new_observations_pipeline/ppo1/cylinder_models_runtime.py:280` →
`CylinderFDRuntime`, which uses `_import_from_dir(fd_dir, "fd_cylinder_common")`).

### The ORIGINAL training data (USE THIS)
`/home/rrlab/rrlab_mulag/ID_FD_pipline/models/fd/FD_cylinder_data.mat`  (HDF5 / v7.3)
```
X_train_fd : (20, 411840)   <-- NOTE: stored TRANSPOSED, do .T -> (N, 20)
Y_train_fd : (12, 411840)
X_val_fd   : (20, 45760)
Y_val_fd   : (12, 45760)
x_mean_fd, x_std_fd (20) ; y_mean_fd, y_std_fd (12)
TrainRunID, TrainClass, TrainSampleTime, TrainTimeIndex
```
Read with `h5py` (scipy.io.loadmat will fail — it's v7.3).

### FD I/O contract (must be preserved exactly)
```
model_config: FDCylinderMLP, input_dim=20, output_dim=12, hidden_dim=512,
              num_blocks=4, activation="silu", dropout=0.02, layer_norm=True
input_layout  (20): [ValveCmd(4), Extension(4), ExtensionVelocity(4), dP(4), Force(4)]
output_layout (12): [DeltaExtension(4), DeltaExtensionVelocity(4), DeltaDP(4)]
target_mode: "normalized_delta_cylinder_state"   # model operates in NORMALIZED space
sample_time_s: 0.05
```

---

## 3. What to implement

Create **`/home/rrlab/rrlab_mulag/ID_FD_pipline/scripts/finetune_fd_rank.py`**.

### Step 1 — load + normalize
Load `FD_cylinder_data.mat`, transpose to `(N,20)` / `(N,12)`. Normalize with the
**checkpoint's** `x_mean/x_std/y_mean/y_std` (do NOT recompute — the pretrained weights
depend on this scaling).

Split the normalized input: `valve = X[:, 0:4]`, `state = X[:, 4:20]`.

### Step 2 — compute the empirical valve partial-effect matrix `B` (12x4), in NORMALIZED space
This is the *target* for the model's valve-Jacobian. It is full-rank (cond ~2), which is
what makes the fix possible.
```python
# regress the state out of both the target and the valve, then fit valve -> residual target
bs, *_ = np.linalg.lstsq(state, Y, rcond=None);   R  = Y   - state @ bs     # (N,12)
bu, *_ = np.linalg.lstsq(state, valve, rcond=None); Ur = valve - state @ bu  # (N,4)
B,  *_ = np.linalg.lstsq(Ur, R, rcond=None)       # (4,12) -> use B.T as (12,4)
```
Sanity-check: SVD of the `ΔExtension` rows (the 4x4 block) should give cond ≈ 2 and
4 comparable singular values. If it doesn't, stop — the premise is broken.

### Step 3 — fine-tune with a rank-promoting objective
```
loss = mse(model(x), y)  +  lambda_jac * || J_u(model) - B ||^2
```
where `J_u(model) = d(model_output)/d(valve_inputs)` is the model's (12x4) valve-Jacobian,
computed **cheaply** by central finite differences on the 4 valve input dims:
```python
for j in range(4):
    xp = x.clone(); xp[:, j] += eps
    xm = x.clone(); xm[:, j] -= eps
    J[:, :, j] = (model(xp) - model(xm)) / (2*eps)     # (batch, 12)
```
(`eps` ~ 1e-2 in normalized units. 8 extra forward passes per batch — acceptable.)

This term **directly forces the model's valve sensitivity to match the data's full-rank
empirical effect**, which is precisely what MSE alone failed to learn.

Training config:
- Adam, **low LR 1e-4** (fine-tune, not retrain), batch 8192, **10–30 epochs**
- `lambda_jac`: start at **1.0**; sweep {0.1, 1, 10} if needed. Increase it if the rank
  gate fails; decrease it if val MSE regresses.
- Keep dropout/weight-decay LOW (the original wd=1e-4 helped suppress the valve pathway).
- **Monitor every epoch**: (a) val MSE (must not regress vs the pretrained baseline),
  (b) `rank` and `cond` of the model's mean valve-Jacobian (this is the thing we're fixing).

### Step 4 — save a pipeline-compatible checkpoint
Save to **`/home/rrlab/rrlab_mulag/ID_FD_pipline/models/fd/best_fd_cylinder_rank4.pt`**
with **all** of the original keys so `load_model_from_checkpoint` accepts it:
```
model_state_dict, model_config, x_mean, x_std, y_mean, y_std, sample_time_s,
target_mode, input_layout, output_layout, input_feature_names, output_feature_names,
checkpoint_version
```
(Copy them straight from the loaded checkpoint dict; only `model_state_dict` changes.)

### Step 5 — point the pipeline at the new FD
Override in `ID_FD_pipline/tasks/env_cfg.py` → `IDFDCylinderSpaceEnvCfg`:
```python
fd_cylinder_checkpoint = "<path to best_fd_cylinder_rank4.pt>"
```
(or update `OLD_FD_CHECKPOINT` in `ID_FD_pipline/models/__init__.py`). Keep the old
checkpoint intact so we can A/B it.

---

## 4. THE GATE — must pass before anything else

```bash
./rrlab.sh -p ID_FD_pipline/tests/rank_gate.py --num_envs 1 --headless
```
Requires the isaac env: `conda activate env_isaaclab`.

**PASS = `action -> fd_delta` is rank 4/4 with cond < ~30.**
(Baseline before the fix: rank ~2, cond 243.) The script prints an explicit
`[GATE] PASS/FAIL` line.

Also confirm the FD didn't forget: **val MSE must not be worse than the pretrained model.**

If the gate fails after sweeping `lambda_jac`, the fallback conclusion is that the valve
signal is too weak per-step to learn, and a **re-collection with larger valve amplitude**
is required (the user's new collection is broken and must be fixed first — see Dead Ends).

---

## 5. After the gate passes (the roadmap — do these in order, each with its own gate)

1. **Fine-tune the ID model too** (same idea, same data). The ID is already rank-4 but was
   trained on the same data, so its inverse for independent motions is unvalidated
   extrapolation. Gate: `FD(ID(independent_desired)) ≈ independent_desired`.
2. **Low-level controller (no RL).** A cylinder-space DLS controller should now track
   cylinder setpoints. Gate: tracks arbitrary setpoints **both directions to < 0.01 m**.
3. **Trajectory execution (still no RL).** Feed a time-varying setpoint (line/curve/sweep);
   truck provides forward motion. Gate: tool tracks a mowing sweep.
4. **RL as the upper layer.** The policy outputs *setpoints* (not raw cylinder actions);
   the controller executes them. This is why RL will now converge — the action space is
   finally reliable and correctly signed.

---

## 6. Working config already fixed in `ID_FD_pipline/tasks/env_cfg.py` (keep)
- `robot_cfg.init_state.pos = (-25.4, 4.3, 1.26)` — grounding fix (the parent's z=0.05
  buried the truck ~1 m; MULAG's root origin is ~1.11 m above wheel contact).
- `cylinder_target_integrates_from_previous_target = True` — reference-space target
  integration (the parent default re-anchors to the sagged measured pos = "ratchet",
  which destroys action authority).
- `cylinder_target_leak = 0.01` — light anti-runaway on that integrator.
- `action_scale = 0.02` — 0.003 + a strong leak starved cylinder authority.

**Note:** the leak was a band-aid for a symptom of the rank-deficient FD. Once the FD is
full-rank, re-evaluate whether the leak is needed at all (try `cylinder_target_leak = 0`).

## 7. Other notes
- The **sign is inverted**: `+action` *decreases* cylinder extension (consistent in both
  the data and IsaacLab). It's consistent, so it's learnable, but keep it in mind when
  writing controllers — a naive `action = +Kp*error` P-controller **diverges**.
- Cylinders are heavily cross-coupled (moving one moves the others) — a controller must
  use the full 4x4 Jacobian (DLS), not 4 independent SISO loops.
