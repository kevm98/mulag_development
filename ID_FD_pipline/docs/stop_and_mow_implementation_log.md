# Stop-and-Mow Implementation Log

**Started:** 2026-07-13
**Git commit:** fd1f674
**IsaacLab:** v2.3.2

---

## Phase A — Audit and plan

**Status:** COMPLETED

Documents produced:
- [current_ppo_trajectory_audit.md](current_ppo_trajectory_audit.md)
- [stop_and_mow_references.md](stop_and_mow_references.md)
- [stop_and_mow_rebuild_plan.md](stop_and_mow_rebuild_plan.md)

---

## Phase B — Trajectory primitives

**Status:** IMPLEMENTED, UNIT TESTED

**Files created:**
```
ID_FD_pipline/trajectories/__init__.py
ID_FD_pipline/trajectories/types.py          # CutterTrajectorySample dataclass
ID_FD_pipline/trajectories/time_scaling.py   # cosine_scaling, quintic_scaling
ID_FD_pipline/trajectories/line.py           # centered lateral line (1.00 m span)
ID_FD_pipline/trajectories/bowed_curve.py    # lateral + forward bow (0.80 m + 0.14 m)
ID_FD_pipline/trajectories/sine_sweep.py     # lateral + 3-lobe sweep (0.80 m + 0.06 m)
ID_FD_pipline/trajectories/sampler.py        # TrajectorySampler with batched progress
ID_FD_pipline/trajectories/tests/__init__.py
```

**Tests run:**
- Time scaling: endpoints, monotonicity, symmetry, batching — all PASS
- Line shape: Y span [-0.5, 0.5], closed path, zero-velocity turnarounds — PASS
- Bowed curve: Y span [0, 0.8], forward bow ~0.14 m, closed — PASS
- Sine sweep: Y span [0, 0.8], 3 lobes, closed — PASS
- Sampler: advance, reset, set_speed, per-env state — PASS
- Full integration: trajectory → sampler → ground correction → obs → reward — PASS

**Bugs fixed during development:**
- Bowed curve and sine sweep did not reverse lateral direction on return stroke
- Forward bow/sweep sign needed stroke_direction multiplier for exact path retrace

---

## Phase C — Deterministic shared controller

**Status:** IMPLEMENTED

**Files created:**
```
ID_FD_pipline/stop_mow/reference_manager.py  # trajectory + ground correction + cylinder seeds
ID_FD_pipline/stop_mow/ground_following.py   # height correction from cutter world-Z
```

The reference manager wraps TrajectorySampler + GroundFollower. It is shared between the deterministic gate (future) and the PPO environment (current).

---

## Phase D — Arm-only state machine

**Status:** IMPLEMENTED, UNIT TESTED

**Files created:**
```
ID_FD_pipline/stop_mow/__init__.py
ID_FD_pipline/stop_mow/phases.py             # StopMowPhase enum, StopMowState dataclass
ID_FD_pipline/stop_mow/state_machine.py      # RESET→DEPLOY→MOW_OUT↔MOW_RETURN→STOW→COMPLETE
ID_FD_pipline/stop_mow/parking.py            # ParkingBrake (stiffness/damping toggle)
ID_FD_pipline/stop_mow/tests/__init__.py
```

**Phases:**
- RESET_SETTLE: brake, wait for truck stop + cylinder settle (300 steps or 3 s)
- DEPLOY: hold cutter at start pose (60 steps)
- MOW_OUT: outward stroke, cosine turnaround
- MOW_RETURN: return stroke
- Repeat MOW_OUT/MOW_RETURN until accumulated path ≥ min_accumulated_path_m (5.0 m default)
- STOW: lift cutter (300 steps)
- COMPLETE: episode ends

**Tests:** StopMowState transitions, GroundFollower corrections — PASS

---

## Phase G — PPO zero-action environment

**Status:** IMPLEMENTED, SYNTAX VERIFIED

**Files modified:**
- `ID_FD_pipline/ppo1/env.py` — full rewrite with state machine integration
- `ID_FD_pipline/ppo1/env_cfg.py` — new config fields for stop-and-mow
- `ID_FD_pipline/ppo1/scripts/train.py` — simplified arm-only training

**Key changes in env.py:**
- Replaced arc() path with TrajectorySampler + ReferenceManager
- Replaced `_update_residual_trajectory()` with StopMowStateMachine
- PPO actions masked (zeroed) during RESET_SETTLE, DEPLOY, STOW, COMPLETE
- PPO active only during MOW_OUT and MOW_RETURN
- Action: (dy ±0.03 m, dz ±0.03 m, speed_scale [0.8-1.2])
- Truck reward disabled (arm-only)
- Parking brake engaged during all arm work
- Cylinder 4 reserved for head-down via ground Jacobian J[:,:3]

**Config changes:**
- `episode_length_s`: 35 → 60 (accommodates multi-stroke cycles)
- `init_std`: 0.15 → 0.10 (more conservative for bounded residual)
- `desired_kl`: 0.008 → 0.02 (prevents LR starvation with wider init)
- New fields: `residual_mow_shape`, `residual_lateral_span_m`, `residual_min_accumulated_path_m`, etc.

---

## Phase H — Residual observations and rewards

**Status:** IMPLEMENTED, SYNTAX VERIFIED

**Observations:** 33-D (unchanged width, semantics now phase-aware via state machine masking)
**Rewards:** Exponential-weighted tracking + action smoothness. Truck reward disabled for arm-only.

---

## Training command

```bash
# Line shape, 1.00 m span, 5 m accumulated path
TERM=xterm ./rrlab.sh -p ID_FD_pipline/ppo1/scripts/train.py \
  --num_envs 256 --max_iterations 500 --headless --shape line --speed 0.05

# Bowed curve
TERM=xterm ./rrlab.sh -p ID_FD_pipline/ppo1/scripts/train.py \
  --num_envs 256 --max_iterations 500 --headless --shape bowed_curve --speed 0.05 --span 0.80

# Sine sweep
TERM=xterm ./rrlab.sh -p ID_FD_pipline/ppo1/scripts/train.py \
  --num_envs 256 --max_iterations 500 --headless --shape sine_sweep --speed 0.05 --span 0.80
```

## Open risks

- **Runtime verification requires Isaac Sim** — all tests so far are syntax + unit (no GPU/PhysX)
- **Phase F (vehicle transition) is not addressed** — remains a known blocker per memory.md
- **Zero-action contract test** (`zero_action_contract.py`) needs updating for new env API
- **stop_and_mow_gate.py** still uses inline trajectory code; should be migrated to shared generators
