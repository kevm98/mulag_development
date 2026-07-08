# PPO1 Horizon ID/FD Pipeline — Upper-Level EE Trajectory Tracking

Upper-level PPO controls the tool (Messerkopf) contact point of the MULAG mower arm.
The learned hydraulic dynamics (ID/FD) sit between PPO and the simulator and are
**always in the loop** — PPO never commands cylinders directly.

## Architecture

```
PPO action (ΔEE, base-frame Y/Z, [-1,1])
  → × ee_action_scale_m
  → empirical J_eff (damped least-squares, measured through the full deployment chain)
  → desired cylinder-space motion  dx_des  (per-cylinder + global clips)
  → ID model   (PhysX-finetuned VelocityToValveMLP: [dx, v, x, xd] → valve)
  → FD model   (sequence GRU, Simscape hydraulic data: [valve, x, xd] → Δextension)
  → closed-loop horizon rollout (H steps, zero-baseline subtracted)
  → cylinder position targets (smooth-ramped over H env steps, joint-limit guarded)
  → PhysX implicit position drives (hydraulic-lock stiffness, hydraulic-scale force caps)
```

- **ID** answers "what valve command produces this motion" (finetuned on PhysX data).
- **FD** answers "what motion do the hydraulics actually produce for this valve command"
  (trained on Simscape hydraulic data — this is where hydraulic realism lives).
- The **implicit position drives** only realize FD's predicted motion; a closed
  hydraulic valve is nearly rigid, which is what high drive stiffness models.

## Key files

| File | Role |
|---|---|
| `horizon_idfd_env_wrapper.py` | Horizon wrapper: J_eff, ID/FD rollout, smooth ramp, rewards, dones, ground trajectories |
| `mulag_upper_level_env.py` | Base DirectRLEnv: robot, actuation backends, manual cylinder targets |
| `mulag_upper_level_env_cfg.py` | All config: actuators (drive gains/force caps), guards, rewards, stages |
| `ground_workspace_monte_carlo.py` | Workspace sampling (kinematic + dynamic hold), PID sweeps, observe/validation gates |
| `../low_level_controller/high_level_ee_controller.py` | J_eff measurement + damped-LS EE→cylinder mapping |
| `../../scripts/train_upper_level_ppo_horizon_idfd.py` | PPO training (use `--trajectory_mode ground_*`; the stage-0 path is unvalidated) |
| `../../scripts/play_upper_level_ppo_horizon_idfd.py` | Roll out a trained checkpoint (pass the same trajectory flags as training) |

Key artifacts (under `outputs/ppo1_horizon_idfd/`):

- `dynamic_hold_workspace.json` — **the** trajectory workspace: 1400 poses the plant
  can physically hold (≤1 cm cylinder error, ≤5 cm drift over 3 s). Covers 100% of the
  kinematically reachable base-YZ points within 5 cm.
- `reachable_workspace_base_yz.json` — legacy kinematic workspace (state-writing only);
  do not use for trajectories.
- `observe_*_summary.json` — machine-readable gate results of every validation run.

## Validation gates (run these before/after any plant or pipeline change)

### 1. Zero-hold plant stability gate (must pass before any training)

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/ppo1/ground_workspace_monte_carlo.py \
  --mode observe \
  --workspace_json outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json \
  --trajectory_mode ground_straight_line --trajectory_plane base_yz \
  --hold_target --observe_policy zero --observe_steps 60 \
  --actuator_backend implicit_position \
  --summary_json outputs/ppo1_horizon_idfd/observe_zero_hold_summary.json \
  --enforce_pass --headless --device cuda:0 --fast_exit
```

Gate (`gate_mode: zero_hold_drift`): tool drift from its own settled pose ≤ 5 cm,
cylinder error ≤ 1 cm, positive tool clearance, no NaNs/huge transforms/hidden resets.
Exit code 0 = pass, 3 = fail. Current status: **passes** (drift 2.9 cm, cyl err 5 mm).

### 2. Greedy tracking validation (no learning, pure pipeline)

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/ppo1/ground_workspace_monte_carlo.py \
  --mode observe \
  --workspace_json outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json \
  --trajectory_mode ground_straight_line --trajectory_plane base_yz \
  --observe_policy greedy --greedy_gain 1.0 --observe_steps 200 \
  --actuator_backend implicit_position \
  --summary_json outputs/ppo1_horizon_idfd/observe_greedy_summary.json \
  --save_plots_dir outputs/ppo1_horizon_idfd/observe_greedy_plots \
  --headless --device cuda:0 --fast_exit
```

Watch `err` descending in the log and `root_motion` staying at cm scale (truck must
not be pushed by the arm). Current status: error 0.72 → 0.60 m and descending over
200 steps, root motion 3 cm. The log's `fd=` vs `meas=` vectors show ID/FD fidelity.

To **watch in the GUI**, drop `--headless` and add `--live_plots` (use `--num_envs 1`).

### 3. Dynamic hold workspace regeneration (after any plant change)

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/ppo1/ground_workspace_monte_carlo.py \
  --mode sample_dynamic --samples 8192 --num_envs 32 \
  --workspace_json outputs/ppo1_horizon_idfd/reachable_workspace_base_yz.json \
  --dynamic_workspace_json outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json \
  --dynamic_records_json outputs/ppo1_horizon_idfd/dynamic_hold_records.json \
  --coverage_reference_points_json outputs/ppo1_horizon_idfd/reachable_workspace_base_yz_points.json \
  --coverage_tolerance_m 0.05 \
  --trajectory_plane base_yz --actuator_backend implicit_position \
  --headless --device cuda:0 --fast_exit
```

Per candidate: write pose → settle → hold 3 s → accept only if stable. Prints
`[dynamic_sampler/coverage]` (target: ≥90% of reference points within 5 cm — currently
100%). Merge additional runs with `--merge_dynamic_json <previous.json>` and a new
`--random_seed`.

### 4. PID / drive-gain sweeps (plant tuning)

```bash
# cfg-attribute sweep (any config key, plus drive gains via
# drive_stiffness_per_cylinder / drive_damping_per_cylinder trial keys)
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/ppo1/ground_workspace_monte_carlo.py \
  --mode sweep --sweep_json <trials.json> \
  --sweep_output_json outputs/ppo1_horizon_idfd/sweep_results.json \
  --actuator_backend implicit_position --headless --device cuda:0 --fast_exit
```

Results are ranked safety-first (no limit hits/resets/NaNs before error magnitude).
Reference sweeps: `drive_stiffness_bracket.json` (found the ×5 stiffness optimum),
`pin_root_ab.json` (proved root pinning pumped energy).

## PPO1 training

**Every command in this section has been run and verified on the current plant.**
The only validated training path is ground trajectory tracking
(`--trajectory_mode ground_*`). Do NOT use `--ppo_stage local_reach_stage0` — stage0
is broken on the new plant (see Known limitations).

### Ground trajectory tracking — straight line (validated)

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 16 --max_iterations 2000 \
  --trajectory_mode ground_straight_line --trajectory_plane base_yz \
  --workspace_json outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json \
  --ground_max_episode_steps 300 \
  --headless --device cuda:0
```

Validated as a 16-env / 40-iteration smoke: mean reward −330 → −268, tracking error
0.92 → 0.64 m, episodes ran to the 300-step timeout with **zero** failure resets
(run `logs/rsl_rl/horizon_idfd_ppo/2026-07-07_23-10-53`).

**Env count vs wall time (measured 2026-07-08, after the per-step CPU optimization,
RTX laptop GPU):**

| envs | per wrapper step | per iteration (24 steps) | samples/s |
|---|---|---|---|
| 16 | 204 ms | ~4.6 s | ~83 |
| 64 | 247 ms | ~6 s | ~260 |
| 256 | 509 ms | ~12 s | ~510 |
| 1024 | 1550 ms | ~37 s | ~660 |

Scaling is sub-linear (the profile is PhysX-dominated; the wrapper's old per-step
USD stage traversals — 88% of step time — are replaced by cached GPU tensor math,
change 14). **Recommended: 256 envs** — best throughput per cost on this hardware.
1024 envs runs (healthy J_eff, 44/1024 degenerate probes median-replaced) but
construction takes ~25 min, host RAM sits at 25/30 GB, and samples/s only improves
~30% over 256. The ceiling is scene construction time/host RAM, not the GPU
(`replicate_physics=False` means full USD clones: 256 envs ≈ 8 min build; GPU memory
stays ≤ ~3.2 GB even at 1024). Note the PPO batch size grows with envs
(`num_envs × 24`), so scale `--max_iterations` down accordingly — e.g. 256 envs
× 300–500 iterations ≈ the sample count of 2000 iterations at 64 envs.

The trajectory target advances along the path each wrapper step and wraps around
(`ground_path_loop`); episodes truncate after `--ground_max_episode_steps` wrapper
steps. Reward: tracking-error improvement + proximity + success bonus − action-rate
− guard/clip penalties − hydraulic-consistency penalty (all in the wrapper).

Healthy training looks like: `Diagnostics/timeout_rate ≈ 1`, `failure_rate ≈ 0`,
`Trajectory/tracking_error` ↓, mean reward ↑, `Trajectory/root_motion_norm` at cm
scale. If `failure_rate` is ~1.0 and episodes last only a few steps **from iteration
0**, the env is killing episodes before the policy can learn — stop and diagnose;
more iterations will not help.

### Semicircle and mower sweep (same code path, smoke-test before long runs)

`--trajectory_mode ground_semicircle` and `--trajectory_mode ground_mower_sweep` use
the exact same env construction and were verified to generate paths inside
`dynamic_hold_workspace.json` bounds, but only the straight line has had a training
smoke run. Before committing to a long run on either, do a cheap smoke first:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 16 --max_iterations 40 \
  --trajectory_mode ground_semicircle --trajectory_plane base_yz \
  --workspace_json outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json \
  --ground_max_episode_steps 300 \
  --headless --device cuda:0
```

and check the healthy-training signals above before scaling up.

### Evaluate a trained checkpoint (validated)

The play script must be given the **same trajectory flags as training** — it rebuilds
the env, and the checkpoint's observation/action dims only match in that mode:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/scripts/play_upper_level_ppo_horizon_idfd.py \
  --checkpoint logs/rsl_rl/horizon_idfd_ppo/<run>/model_<iter>.pt \
  --trajectory_mode ground_straight_line --trajectory_plane base_yz \
  --num_envs 1 --episodes 5 --headless --device cuda:0
```

Prints per-episode steps, total reward, and mean/final tracking error. Validated
against the 40-iter smoke checkpoint (300-step episodes, mean tracking 0.65–0.73 m —
consistent with that checkpoint's training curve). Drop `--headless` to watch in the
GUI.

## Changes implemented (2026-07-07) and why

1. **Low-level backend switched to `implicit_position`** (cfg default; wrapper no
   longer forces `explicit_effort_pid_bias`). The explicit force PID at ~120 Hz with
   implicit-scale gains was structurally unstable (limit cycles; a 22-trial gain sweep
   had zero passes). PhysX implicit drives realize the ID/FD cylinder targets;
   hydraulic realism stays in ID/FD.
2. **Drive stiffness set by measurement** (`drive_stiffness_bracket_results.json`):
   K=[7.5e7, 3e7, 5e6, 5e6] for [Drehzapfen, Ausleger_I, Ausleger_II, Messerkopf_Schwenk]
   — the ×5 ladder minimizes hold drift; softer sags under gravity (~0.9 m tool sag at
   the old values), stiffer excites the solver.
3. **Root pinning removed** (`horizon_pin_root_pose` default off; `--pin_root` to
   re-enable). Teleporting the truck to its spawn pose every step fought the tire
   contacts and continuously pumped energy into the boom — proven by overdamped probes
   (10× damping changed nothing) and a pin on/off A/B (0.72 m vs 0.004 m drift at the
   same gains). The truck now stands physically on braked wheels.
4. **Physical force hierarchy** (fixes "arm pushes the truck"): arm effort caps at
   hydraulic scale (5e4/1e5/5e4/2e4 N·m — measured masses: truck 6.8 t, arm 542 kg, so
   the caps keep arm forces below the truck's friction budget), arm velocity limits
   1.0 (hydraulic flow analog), wheels as parking brakes (stiffness 6e8/damping 2e6),
   steering damped locked (1e9) — wheel/steering/velocity-limit values adopted from the
   teacher-student project's tuned MULAG asset. Result: truck motion during greedy
   tracking dropped from 0.93 m to 0.03 m.
5. **Guard-done misattribution fixed**: transient workspace-guard freezes were labeled
   `cylinder_limit` and hard-reset the env every ~10 steps during diagnostics;
   `cylinder_limit` now requires actual joint-limit proximity.
6. **Hold-target ratchet fixed**: the neutral-action retarget used to fire every step,
   making the hold target follow the sagging plant; it now anchors once per hold phase.
7. **ID/FD input scaling** (from a checkpoint audit): authority clips re-opened ~20×
   (`ee_per_cyl_clip_m` 0.15 mm → 3 mm etc.) so a max command drives ID at ~1σ of its
   training inputs instead of 0.05σ; the horizon rollout now propagates FD-predicted
   velocity (self-consistent closed loop).
8. **J_eff measurement hardened**: waits (bounded) for the free-standing truck to be
   still, judged by root *displacement* — the root velocity buffer reads a constant
   phantom ~0.11 m/s and must not be trusted for stillness.
9. **Dynamic hold workspace machinery** (`--mode sample_dynamic`): per-candidate
   settle-and-hold acceptance, run merging, coverage reporting vs the kinematic cloud,
   full accepted cloud stored in the JSON. The old kinematic JSON only proves poses can
   be *written*, not *held*.
10. **Training plumbing**: `--trajectory_mode/--trajectory_plane/--workspace_json/`
    `--ground_max_episode_steps` on the training script, ground-mode episode timeouts,
    and looping trajectory paths.
11. **Per-env ground targets**: path points are stored in canonical (env-0) world
    coordinates and are now shifted by each env's grid origin when broadcast — before
    this, every training env except env 0 tracked a target at another env's location
    (~30 m tracking error, no learning signal).
12. **Play script fixed and validated (2026-07-08)**: it used to rebuild the env with
    the default (stage0) config, so trajectory-trained checkpoints could not be
    evaluated (obs/action dim mismatch), and its step-unpacking crashed against the
    RSL-RL wrapper (4 return values, not 5). It now takes the same
    `--trajectory_mode/--trajectory_plane/--workspace_json/--id_checkpoint` flags as
    training, mirrors the env construction, and reports per-episode tracking error.
13. **Ground plate sized to the env grid (2026-07-08)**: the ground was a fixed
    200×200 m cuboid (edges at ±100 m). At 64 envs (8×8 grid, spacing 20 m, truck
    spawn ~20 m in −x of its origin) the far-corner trucks (envs 62/63, x ≈ −90…−103)
    straddled the edge, slid off, and free-fell forever — the J_eff settle wait timed
    out at 62 m/window and the Jacobian came out rank-1 (cond 3e11), which presented
    as "training stuck before iteration 1". The plate now scales with
    `env_spacing × √num_envs / 2 + 60 m` half-extent (minimum the old 200 m, so all
    ≤16-env results are unchanged). Debug aid: `MULAG_SETTLE_DEBUG=1` prints per-window
    max root displacement + worst env during the J_eff settle wait.
14. **Per-step CPU bottleneck removed (2026-07-08)**: 88% of training step time was
    `lowest_non_tool_arm_geometry_z` doing a full USD stage traversal + BBoxCache
    query per arm body per env, twice per step (once for dones, once for logging).
    Now: prim-path lookups are cached; each arm body's local AABB corners are
    captured once from USD (AABB and prim transform from the same snapshot, so the
    local corners are pose-independent) and re-posed per step from PhysX link states
    entirely on GPU; the duplicate `ground_safety_metrics` call is reused. Measured:
    16 envs 1796 → 204 ms/step, 64 envs 555 → ~6 s/iteration (94×). Parity proven:
    zero-hold gate passes with identical numbers, greedy tracking matches the
    pre-optimization run to 4 decimals, and a 40-iter smoke reproduces the baseline
    learning curve. Bonus: arm clearance now tracks live link poses (the old USD
    read could go stale in headless GPU sim). Fallback to the slow path (with a
    printed warning) if the cache cannot be built.
15. **J_eff robust to noisy envs at high env counts (2026-07-08)**: with hundreds of
    envs, a few trucks' residual ~1 cm rocking swamps their own Jacobian probe and
    collapses that env's J_eff to rank 1 (observed: rank-1/cond 2e6 at 256 envs).
    Since all envs are identical plants at the same pose, envs whose measured J is
    degenerate (rank-deficient, ill-conditioned, or non-finite) now inherit the
    median J of the healthy envs (9/256 replaced; rank 2 / cond ≈ 1100 after).
    Healthy envs keep their own measurement — an A/B smoke showed wholesale median
    replacement shifts early learning, so ≤64-env behavior is untouched (0 envs
    replaced there). `MULAG_JEFF_MEDIAN=0` disables.

## Known limitations / next steps

- **ID re-finetune across the workspace (DONE — result: parity, not a win)**: a
  workspace-wide ID was collected (28,874 samples from 354 accepted hold poses,
  extension std 24–47 mm vs the old 4–17 mm) and finetuned to
  `outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune_ws/id_physx_finetuned_best.pt`
  (sign accuracy ~0.81 per cylinder). A/B results vs the old single-pose ID:
  greedy tracking near the home pose is slightly worse (mean 0.91 vs 0.72 m,
  endpoints comparable); a 40-iteration PPO smoke on `ground_straight_line` is
  statistically identical (reward −338.7→−272.2 vs −330.4→−268.1, final tracking
  0.62 vs 0.64 m, zero resets in both). The J_eff feedback loop absorbs ID model
  error near operating points, so ID quality is not the training bottleneck at
  smoke scale. Keep the default (old) checkpoint for now; the workspace-wide one
  is available via `--id_checkpoint .../model_finetune_ws/id_physx_finetuned_best.pt`
  and may matter for mower sweeps that visit workspace edges. If a v2 is ever
  needed: per-cylinder masked-loss collection (current teacher-signal yield is
  only ~13–15% of rows with all four signs correct).
- **Stage0 local reach is unvalidated (do not use)**: a 297-iteration 64-env run
  (2026-07-08) showed 98.7% early termination at iteration 0 with a random policy,
  zero successes, and a collapsed policy (action std 0.11, reward −19 → −42). Part of
  that was the ground-edge bug (change 13): envs 62/63 were in free fall and cycling
  through resets every ~6 steps, flooding the completed-episode stats. That bug is
  fixed, but stage0 has not been re-run or revalidated since the plant overhaul
  (backend switch, new gains, pinning removal, effort caps). Before using it:
  instrument per-cause done rates, re-run a short smoke, and re-tune. Until then,
  train only with `--trajectory_mode ground_*`.
- **Scene construction time at high env counts**: `replicate_physics=False` clones
  the full USD scene per env, so startup grows with env count (256 envs ≈ 8 min).
  Per-step cost is no longer the issue (change 14); if very large env counts are
  needed routinely, investigating `replicate_physics=True` + explicit collision
  filtering would cut startup, but that changes scene construction and must be
  re-gated.
- **Shared trajectory index**: all envs track the same advancing path point
  (`_ground_path_index` is a scalar). Fine for training on one geometry; per-env
  indices/start offsets would decorrelate envs further.
- **Scripted driving stage**: wheels are configured as parking brakes. Before enabling
  `enable_scripted_vehicle_motion` (moving-truck mowing), switch wheel drive gains to
  velocity mode (stiffness 0, damping ~2e6) — wheel velocity targets cannot act
  against a 6e8 position drive.
- **Timescale factor 3**: the ID/FD rollout models H×0.05 s of hydraulic motion but the
  ramp executes in H×0.0167 s. Absorbed by the J_eff feedback for now; reconcile if
  `fd` vs `meas` shows a consistent scale factor after the ID re-finetune.
