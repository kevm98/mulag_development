# Stop-and-Mow Rebuild Plan

**Date:** 2026-07-13
**Git commit:** fd1f674
**IsaacLab:** v2.3.2

## Summary of root causes

1. **PPO trajectory is a one-way graph-diameter arc, not back-and-forth mowing** — `planner.arc()` gives 7 hops, 3.8 m forward arc, no return, no cosine turnaround.
2. **Memory.md explicitly warns this arc path is invalid for runtime IK.**
3. **Cylinder 4 not reserved for head-down** — full 4×4 Jacobian used; head orientation uncontrolled.
4. **No stop-and-mow architecture** — truck moves continuously (policy-controlled), not in discrete advance/brake phases.
5. **No ground-height feedback** — path assumes flat ground from map samples.
6. **PPO reward encourages driving during mowing** — exponential truck credit active even when truck should be parked.
7. **Path speed 0.045 m/s is imperceptibly slow** — 1.27 m travel over 28 s looks like the arm stopped.
8. **Vehicle transition is not repeatably reliable** (per `memory.md` and `stop_and_mow_report.md`) — multi-station must remain arm-only until fixed.

## Target architecture

```
Mission state machine (per env)
        ↓
Shared trajectory generator (line / bowed_curve / sine_sweep)
        ↓
Deterministic ground-height correction
        ↓
Bounded PPO residual (±0.03 m lateral, ±0.03 m height, ±20% speed)
        ↓
Cached ground cutter Jacobian DLS (cylinders 1-3, cylinder 4 reserved)
        ↓
Measured-cylinder DLS
        ↓
Learned ID → Valve → Learned FD → PhysX
```

## Implementation phases

### Phase A — Audit and plan (CURRENT)

**Status:** IN PROGRESS

Files produced:
- `docs/current_ppo_trajectory_audit.md`
- `docs/stop_and_mow_references.md`
- `docs/stop_and_mow_rebuild_plan.md` (this file)

No functional changes.

### Phase B — Trajectory primitives

**Goal:** Create shared, testable trajectory generators used by both deterministic gates and PPO env.

**Files to create:**
```
ID_FD_pipline/trajectories/__init__.py
ID_FD_pipline/trajectories/types.py          # CutterTrajectorySample dataclass
ID_FD_pipline/trajectories/time_scaling.py   # cosine_scaling(), quintic_scaling()
ID_FD_pipline/trajectories/line.py           # centered lateral line
ID_FD_pipline/trajectories/bowed_curve.py    # bowed lateral curve
ID_FD_pipline/trajectories/sine_sweep.py     # three-lobe sine sweep
ID_FD_pipline/trajectories/sampler.py        # batched continuous sampler
ID_FD_pipline/trajectories/tests/__init__.py
ID_FD_pipline/trajectories/tests/test_time_scaling.py
ID_FD_pipline/trajectories/tests/test_shapes.py
```

**Requirements:**
- All generators take `(progress_m, shape_params)` and return `CutterTrajectorySample`
- Batch-compatible (accept `(N,)` tensors)
- Deterministic, side-effect free, no env reads
- Position and velocity continuity (cosine turnaround for reversals)
- Correct accumulated path length vs unique spatial span
- Explicit frame: all positions in truck `base_link` frame

**Verified envelope per `memory.md`:**
- Line: centred 1.00 m lateral span
- Bowed curve: 0.80 m lateral, 0.10-0.18 m forward bow, exact return
- Sine sweep: 0.80 m lateral, 0.06 m forward amplitude, 3 lobes, exact return

**Time scaling:**
- Cosine: `h(τ) = 0.5 × (1 − cos(π × τ))` where τ ∈ [0,1] per half-stroke
- Quintic (optional): `h(τ) = 10τ³ − 15τ⁴ + 6τ⁵`

**Unit tests must verify:**
- Endpoint positions (start == end for closed shapes)
- Velocity zero at reversals
- Accumulated length ≥ unique span
- Batch consistency
- Frame naming

### Phase C — Deterministic shared controller

**Goal:** Move trajectory generation out of `stop_and_mow_gate.py` into a shared module. Make existing gates use the shared generator. Prove parity.

**Files to modify:**
- `ID_FD_pipline/dls_solver/tests/stop_and_mow_gate.py` — replace inline `work_reference()` with shared trajectory generators
- `ID_FD_pipline/dls_solver/tests/long_trajectory_gate.py` — use shared generators where applicable

**Files to create:**
- `ID_FD_pipline/stop_mow/__init__.py`
- `ID_FD_pipline/stop_mow/reference_manager.py` — wraps trajectory generators + ground-height correction + cylinder seed computation

**Acceptance:** Existing `stop_and_mow_gate.py` tests still PASS with identical metrics.

### Phase D — Arm-only state machine

**Goal:** Implement deterministic per-environment state machine with PARK → DEPLOY → MOW → STOW → COMPLETE phases. Truck remains disabled.

**Files to create:**
```
ID_FD_pipline/stop_mow/__init__.py
ID_FD_pipline/stop_mow/phases.py          # StopMowPhase enum
ID_FD_pipline/stop_mow/state_machine.py   # per-env phase transitions
ID_FD_pipline/stop_mow/parking.py         # set/release parking brake
ID_FD_pipline/stop_mow/reference_manager.py
ID_FD_pipline/stop_mow/ground_following.py
ID_FD_pipline/stop_mow/tests/__init__.py
ID_FD_pipline/stop_mow/tests/test_state_machine.py
```

**Phases:**
```python
class StopMowPhase(IntEnum):
    RESET_SETTLE = 0   # brake, wait for truck stop, settle cylinders
    DEPLOY = 1         # lower cutter to start pose, verify height + head-down
    MOW_OUT = 2        # outward stroke
    MOW_RETURN = 3     # return stroke (cosine turnaround)
    STOW = 4           # lift cutter to travel height
    COMPLETE = 5       # cycle done
```

**Per-env state:** phase, phase_step, station_index, stroke_index, path_progress, stroke_direction.

**Transition conditions:**
- RESET_SETTLE → DEPLOY: truck speed < 0.02 m/s AND cylinder speeds below threshold AND min settle duration
- DEPLOY → MOW_OUT: cutter at start pose within tolerance AND head-down within tolerance AND held for N steps
- MOW_OUT → MOW_RETURN: path progress ≥ 1.0 (stroke complete)
- MOW_RETURN → (MOW_OUT or STOW): path progress ≥ 2.0 if repeating, else STOW
- STOW → COMPLETE: cutter above safe height AND cylinder errors stable

**Multi-stroke accumulation:** Repeat MOW_OUT → MOW_RETURN cycles to reach `long_path_m ≥ 5.0`.

### Phase E — Speed ladder

**Goal:** Determine maximum reliable mowing speed for each shape.

**Method:** Test each shape at [0.03, 0.05, 0.075, 0.10, 0.125, 0.15] m/s. Record at each speed: mean/p95/max tracking error, ground fraction, head-down alignment, per-cylinder travel, cylinder velocity, valve saturation, FD clipping, truck drift.

**Acceptance:** Speed where p95 < 0.05 m, ground ≥ 95%, head-down ≥ 0.95.

### Phase F — Vehicle diagnostics

**Goal:** Make truck advance/brake transition repeatable.

**Files to create:**
- `ID_FD_pipline/stop_mow/truck_advance.py`
- `ID_FD_pipline/stop_mow/tests/vehicle_transition_gate.py`

**Test:** 20 consecutive stow → advance 0.50 m → brake → settle cycles. Log wheel velocity (commanded/actual), wheel effort, chassis velocity, contact forces, forward/lateral displacement, heading error.

**Acceptance:** Forward within ±0.10 m of target, lateral < 0.10 m, heading < 5°, post-brake speed < 0.02 m/s, no impulse.

**BLOCKER:** `memory.md` says this is not currently repeatable. Do not proceed to multi-station PPO until this passes.

### Phase G — PPO zero-action environment

**Goal:** Integrate state machine into `ppo1/env.py`. Force actions to zero. Prove equivalence.

**Files to modify:**
- `ID_FD_pipline/ppo1/env.py` — replace arc() path with state-machine-driven trajectory
- `ID_FD_pipline/ppo1/env_cfg.py` — new config fields for shape, speed, stroke length
- `ID_FD_pipline/ppo1/tests/zero_action_contract.py` — extend for all shapes

**Key invariants:**
- Zero PPO action → exact deterministic baseline for all shapes
- PPO actions masked during RESET_SETTLE, DEPLOY, STOW
- PPO actions only during MOW_OUT and MOW_RETURN
- Truck controlled by state machine, NOT PPO

### Phase H — Residual observations and rewards

**Goal:** Compact phase-aware observations and phase-specific rewards.

**Files to modify:**
- `ID_FD_pipline/ppo1/observations.py` — new observation builder
- `ID_FD_pipline/ppo1/rewards.py` — phase-specific rewards

**Observation components (candidate):**
- Nominal cutter position error XYZ (3)
- Corrected cutter position error XYZ (3)
- Measured cutter velocity XYZ (3)
- Desired cutter velocity XYZ (3)
- Left/right cutter height (2)
- Mean height error (1)
- Head-down alignment (1)
- Cylinder positions (4)
- Cylinder velocities (4)
- Cylinder goals (4)
- Current valve commands (4)
- Corrected FD delta (4)
- Previous PPO action (3)
- Phase one-hot (6 or N)
- Normalized phase progress (1)
- Normalized stroke progress (1)
- Stroke direction (±1) (1)

**Reward structure (mowing phase only):**
- Tracking: `exp(-(pos_error / 0.05)^2)` — bell-shaped, max 1.0
- Height: `exp(-(height_error / 0.03)^2)`
- Head-down: `exp(-(alignment_error / 0.05)^2)`
- Progress: `path_progress × tracking_confidence`
- Action magnitude: small negative weight
- Action rate: small negative weight
- No truck speed reward during parked mowing

### Phase I — Arm-only PPO training

**Goal:** Train residual policy for arm-only mowing.

**Curriculum:**
1. Line only, 0.03 m/s, 50 iterations
2. Line, speed ladder up to verified max
3. Add bowed curve
4. Add sine sweep
5. Mixed shapes

**Acceptance:** PPO must beat zero-action baseline on at least one metric without degrading safety metrics.

### Phase J — Terrain variation

**Goal:** Add ground-height variation and limited domain randomization.

Progressive difficulty: flat → gentle slope → random height offsets.

### Phase K — Multi-station evaluation

**Only after Phase F (vehicle transition) passes.**

PPO remains arm-residual only. Truck advance is deterministic.

## Files deliberately NOT changed

Per mission constraints:
- ID/FD checkpoint architecture and weights
- Cylinder coordinate mapping and learned limits
- Paired ID/FD residual contract
- FD residual gains and direction guards
- Robot USD geometry, collision geometry, joint names
- Physics timestep (1/120), control decimation (2)
- PhysX memory settings
- Terrain implementation
- RSL-RL library internals
- IsaacLab source code
- `teacher_student/` preserved snapshot

## Known blockers

1. **Vehicle transition not repeatable** (Phase F) — must be fixed before multi-station PPO.
2. **Maximum spatial envelopes are 1.00 m (line) / 0.80 m (sweep/semicircle)** — cannot claim longer without deterministic proof.
3. **Cylinder 4 head-down authority not independently verified** — must measure before giving PPO control.
4. **Isaac Sim is required for all runtime tests** — syntax checks are insufficient.

## Next actions

1. Complete Phase A (this document)
2. Begin Phase B: create `trajectories/` module with time_scaling and shape generators
3. Run unit tests on trajectory modules
