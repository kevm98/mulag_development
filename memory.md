# Project memory

## Environment
- Main Python env: `/home/rrlab/miniconda3/envs/env_isaaclab/bin/python`
- IsaacLab path: `~/IsaacLab`
- RRLAB project path: `~/rrlab_mulag`
- Codex model: DeepSeek V4 Pro (with Moon Bridge routing)

## Known constraints
- Laptop GPU: RTX 3060 Laptop, 6 GB VRAM
- Large IsaacLab runs can OOM; use small num_envs and disable cameras/terrain for debugging
- DeepSeek V4 Pro needs Moon Bridge/LiteLLM for routing (Responses API → DeepSeek bridge)

## Working commands
- Python import check: `/home/rrlab/miniconda3/envs/env_isaaclab/bin/python test_import_mulag.py`
- VS Code Copilot CLI: `gh copilot`
- Activate conda: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab`

## DeepSeek integration status
- Routing: VS Code extension injects `modelProvider = "moonbridge"` for deepseek-v4-pro
- Moon Bridge: Handles Codex Responses API → DeepSeek translation
- Config: `~/.config/codex-deepseek/deepseek-v4-pro.config.toml` (workspace-write sandbox)
- Writable paths: `/home/rrlab/rrlab_mulag`, `/home/rrlab/rrlab_mulag.worktrees`, `/home/rrlab/IsaacLab`, `/tmp`

## Verified working patterns
- AGENTS.md loaded by Codex/VS Code before task execution
- Agent autonomy: No asking before read-only operations, grep, Python checks
- Terminal execution: All commands (python/bash/grep/git) run without approval gate
- Memory system: `<<` reads memory.md before task; `>> <text>` saves to memory.md
- Explicit task framing: "Do not ask permission" → agent behaves as executor, not advisor

## ID/FD mowing trajectory lessons (2026-07-12)
- Do not use `CylinderDLSConfig(invert_learned_pipeline=True)` online. The deployed
  paired-residual pipeline includes clipping/direction guards, so the finite-difference
  inverse is not the runtime map; a long-line test drove the cutter more than 1 m upward.
- Do not use `low_level_servo_mode=id_ff_plus_cylinder_velocity_pid` for the current
  locked checkpoints. It cross-coupled a cylinder-3 line command into cylinder-1 slew
  and produced about 1.42 m mean EE error.
- `idfd_step_gate.py --model_mode absolute` fails every 0.08 m cylinder step. The paired
  contract with `cylinder_fd_residual_output_gain=[20,10,2,2]` passes all four axes at
  0.0033-0.0047 m error. Keep the paired contract and calibration.
- The replicated `ground_band_map.json` natural slew arc is valid in the standalone
  plant and passes a 1.574 m trajectory, but its PCA line/serpentine graph branches are
  not transferable to the standalone closed-loop linkage. Never reuse those raw graph
  branches as runtime IK seeds.
- A one-environment reverse line remap stayed mostly on ground but produced only 1.228 m
  accumulated motion with 0.694 m spatial span; it does not meet the meaningful >=1.5 m
  line requirement. The forward remap contained linkage branch jumps and is invalid.
- Do not run different long shapes sequentially in one Isaac process. `env.reset()` does
  not fully restore the closed-loop linkage branch: line and serpentine pass separately
  but fail when invoked as `--shapes line,serpentine`. The long gate intentionally accepts
  exactly one shape per process; arc may still test stationary+moving in one process.
- Fixed truck speed exposes a nonlinear breakaway: 3.0 km/h stalls, while 4.5 km/h moves
  reliably. At 4.5 km/h the truck-composed line/serpentine are stable in mean/final error
  but have p95 transients (0.0646/0.1159 m). These are deliberately recorded as the future
  residual-PPO targets; do not retune the lower controller endlessly or claim strict p95.
- Verified architecture: cached stepped-PhysX cutter Jacobian + CPU DLS for EE IK;
  measured cylinder feedback through paired ID/FD for actuation. No MPC and no OSC.

## Stop-and-mow phase lessons (2026-07-12)
- Correct operational architecture is `PARK -> ARM WORK -> RETURN/LIFT -> STRAIGHT
  TRUCK ADVANCE -> BRAKE -> REPEAT`. The truck must be stationary during every scored
  cutter path. Continuous truck-composed mowing is not an acceptable substitute.
- Verified parked sweep is a real 0.35 m lateral out-and-back stroke with constant
  truck-frame X: mean 0.0282 m, p95 0.0344 m, ground 100%, truck drift 0.013 m.
- Verified parked curve uses a 0.35 m lateral stroke with a smooth 0.10 m forward bow
  and exact return. It stays entirely in front of the truck: mean 0.0273 m,
  p95 0.0354 m, ground 100%, truck drift 0.014 m.
- A 0.40 m forward semicircle tracked with the arm but left a bad truck load/traction
  state. Do not reuse it; the smaller bowed lateral curve is the stable arm geometry.
- The multi-station wheel/brake transition is not repeatably green. One sweep cycle
  passed (0.505 m forward, 0.029 m lateral, both work stations green), but later
  deterministic reruns overshot or moved laterally after brake release. Treat this as
  an unresolved vehicle-phase blocker, not an arm or ID/FD failure.
- Do not pre-align steering under a loaded brake: it produced a large lateral release
  impulse. Do not use equal/common steering feedback from the failed trials: it either
  amplified lateral motion or bound the axle.
- Do not hide brake-release drift in an unscored settle. One such trial moved 0.162 m
  laterally before measurement. Do not zero the articulation root velocity as a parking
  brake; it injected constraint impulses and worsened lateral motion.
- Repeatedly churning several sweep cycles at one station accumulates hydraulic
  hysteresis/cutter lift. One closed arm pass per station is the robust pattern.
- Before any PPO work, instrument commanded/actual wheel velocities, wheel effort,
  chassis velocity, and contact force across the drive-to-brake transition and make the
  deterministic multi-station gate repeatable. RL must not compensate for this bug.
- Preferred PPO baseline is now truck-disabled `stop_and_mow_gate.py --arm_only`: five
  cosine-turnaround 0.35 m out-and-back strokes. Verified 4.856 m goal travel,
  5.139 m cutter travel, p95 0.0407 m, ground 100%, truck drift 0.003 m.
- Reserve cylinder 4 for mower-head pitch. The cached ground position Jacobian using
  cylinders 1-3 is rank 3 with condition 3.85, so position remains controllable. Holding
  cylinder 4 at the calibrated ground reference gave only 0.3 mm travel and minimum
  mower-face/world-down alignment 0.973. The Messerkopf USD mowing-face direction is
  authored local +Z (not -Z); do not repeat the wrong-axis orientation check.
- Arm-only default now provides `line`, `semicircle`, and three-lobe `sweep`, each with
  smooth cosine turnarounds and 7 m minimum accumulated travel. Full sine gate passed:
  goal path 12.485 m, cutter path 14.507 m, p95 0.0428 m, ground 100%, truck drift
  0.002 m, cylinder travel [0.0080, 0.0160, 0.0552, 0.0003] m, head-down 0.973.
- Do not confuse accumulated length with unique spatial span. Maximum accepted spatial
  envelopes are: centred line 1.00 m (p95 0.0456, cylinder3 travel 0.1661 m), and
  semicircle/sine lateral span 0.80 m (p95 0.0371/0.0363). Centred line 1.10 m failed
  p95 at 0.0519; 1.20 m failed at 0.0621; 1.40 m diverged. The coarse 3.78 x 2.75 m map
  extent is disconnected and is not a valid continuous trajectory.
- Active cylinder-4 orientation calibration at +/-15 mm and +/-3 mm silently terminated
  Isaac. Three learned-pipeline head recovery strategies also failed to retract x4 from
  about 0.68 m after a 1.20 m full-redundant-DLS stroke. Keep cylinder 4 on the fixed
  down-facing branch for the accepted controller; do not repeat online x4 probing until
  the actuator issue is diagnosed separately.

## DeepSeek PPO residual policy overhaul (2026-07-13) — identifier: `deepseek-stages-a-d`

### What was changed and why

**Stage A — PPO Hyperparameters** (`PPO1_ID_FD_PhysX/ppo1/env_cfg.py`):
- `hidden_dims`: [64,64] → [128,64] (more capacity for residual mapping over 4-cylinder system)
- `init_std`: 0.05 → 0.15 (wider initial exploration; zero-entropy policy died at 0.05)
- `entropy_coef`: 0.0 → 0.005 (gentle exploration pressure; zero entropy let action std collapse)
- `learning_rate`: 1e-4 → 5e-4 (5× faster; original was ~10× below IsaacLab recommended)
- `num_learning_epochs`: 3 → 5 (more PPO update steps per batch)
- `clip_param`: 0.1 → 0.2 (IsaacLab default)
- `desired_kl`: 0.01 → 0.008 (tighter KL constraint compensates higher LR)

**Stage B — Reward Restructure** (`PPO1_ID_FD_PhysX/ppo1/rewards.py`):
- Replaced hard-gated truck progress (`if error < 0.05: reward += truck_bonus`) with exponential-weighted: `progress_weight * (kmh/max) * exp(-error/decay_m)`. Now credit is always nonzero, giving smooth gradient.
- Added action smoothness penalty: `-0.01 * ||a_t - a_{t-1}||` to discourage jitter.
- Returns 3-tuple `(reward, error, extras_dict)` for richer logging.

**Stage C — Enriched Observations** (`PPO1_ID_FD_PhysX/ppo1/observations.py`, `env.py`):
- Added 12 dimensions: `cylinder_goal_x[4]`, `current_valve_cmd[4]`, `fd_delta_corrected[4]`.
- Total observation: 21 → 33 dims.
- Policy can now see what the base controller is doing internally.

**Stage D — Truck-Speed Curriculum** (`PPO1_ID_FD_PhysX/ppo1/scripts/train.py`, `env.py`):
- Three-phase curriculum managed by `residual_truck_curriculum_phase`:
  - Phase 0 (iter 0-99): truck disabled, arm-only tracking
  - Phase 1 (iter 100-299): slow truck, span=2.0 km/h
  - Phase 2 (iter 300-499): full truck, span=4.5 km/h
- `train.py` calls `runner.learn()` in three segments, toggling `env_cfg.residual_truck_curriculum_phase` between them.
- `env.py::_effective_truck_span()` maps phase → effective speed span.

### Verification
- All 4 files pass Python syntax check.
- Observation shape (33) matches constant.
- Reward function: zero-error gives full truck credit (1.125); large-error gives near-zero credit.
- Action smoothness penalty scales correctly (0.01 × action delta norm).

### Files modified
- `PPO1_ID_FD_PhysX/ppo1/env_cfg.py` — full rewrite (config changes)
- `PPO1_ID_FD_PhysX/ppo1/rewards.py` — full rewrite (new reward logic)
- `PPO1_ID_FD_PhysX/ppo1/observations.py` — full rewrite (enriched obs)
- `PPO1_ID_FD_PhysX/ppo1/env.py` — targeted edits (obs wiring, reward wiring, truck span)
- `PPO1_ID_FD_PhysX/ppo1/scripts/train.py` — full rewrite (curriculum loop)

### Training command
```bash
TERM=xterm ./rrlab.sh -p PPO1_ID_FD_PhysX/ppo1/scripts/train.py --num_envs 256 --max_iterations 500 --headless
```

For GPU (RTX 3060 6GB), 256 envs is safer than 1000.
Add `--phase1_iters 100 --phase2_iters 200` to override curriculum boundaries.

## Canonical PPO2-ROM-PhysX setup (2026-07-19)

- The maintained hydraulic backend is the single command-conditioned ROM at
  `ROM-cylinders_pipeline/checkpoints/plant_reachable_v1_finetune_blend25/best_rom.pt`.
  PPO2 does not use ID or FD.
- The only deployment/reference library is
  `PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v1.json`. Its exact shapes
  are line, arc, sine, circle, and a genuinely sharp square at fixed, distinct heights.
- Exact zero-policy full-cycle results at requested 0.18 m/s are: line 0.0221/0.0430 m,
  arc 0.0189/0.0391 m, sine 0.0255/0.0422 m, circle 0.0215/0.0389 m, and square
  0.0200/0.0398 m mean/p95 tool error.
- The truck and PPO are hard-interlocked. PPO acts only during `ARM_TRACK`, corrects
  cylinders 1-3 by at most 2 mm, and cannot control wheels, truck root, or cylinder 4.
- Canonical runtime smoothing is ROM direction guard enabled, hydraulic target filter
  beta 0.5, EE correction gain 0.25, and 0.25 s seed/reference lead.
- The new PPO training profile is 500 iterations from a fresh actor: 350 exact-only,
  then 150 at a 70/30 exact/procedural mix, with 64 environments, seed 17, and requested
  speed 0.18 m/s. Do not resume the old stage checkpoints for this run.
- The observation remains 41-D but uses a 0.25 s local XY reference preview instead of
  template phase. There is no shape identifier. Ten conservative size/offset/height
  variants are plant-solved and gated for the mixed phase.
- Retain only decision metrics: mean and p95 tool error, p95 measured tool acceleration,
  minimum head alignment, maximum post-lock truck drift, and mean absolute PPO residual.
- A new path is not automatically safe: solve it inside the measured head-down workspace,
  reject it if the plant projection fails, and keep speed/curvature inside the trained
  envelope before asking the policy to track it.
- PPO2 training reset latches the settled truck pose after ROM warm-up and applies the
  planar fixture both before and after each physics step. The preflight uses stage-4
  reference-relative clearance, limits raw suspension reaction to 5 cm, and limits
  retained post-constraint drift to 2 cm (the verified seed-17 maximum was 0.999 mm).
- The first 2026-07-19 PPO2 run was intentionally stopped after `model_200.pt`. A
  vectorized identical-state five-shape screen selected `model_100.pt`: zero/model100/
  model200 aggregate mean error was 0.02708/0.02511/0.02630 m, p95 was
  0.04893/0.04572/0.04805 m, and p95 acceleration was 28.97/28.09/27.95 m/s².
  Model 200 regressed sine and circle and used 0.492 mm mean residual versus 0.236 mm
  for model 100. Fine-tune model 100 with a fresh optimizer, fixed 1e-4 learning rate,
  zero entropy, std 0.03, action-rate -0.20, bounded acceleration penalty -0.20, and
  save interval 10; do not resume model 200.
- The fine-tune completed at
  `logs/rsl_rl/ppo2_rom_residual/2026-07-19_18-55-23/model_249.pt` and is accepted/frozen
  for now. Its identical-state screen improves zero-policy aggregate mean error
  0.02649->0.02401 m, p95 0.04820->0.04726 m, and acceleration 27.73->26.09 m/s².
  Use `./PPO2_ROM_PhysX/visualize.sh` for the canonical quiet five-station GUI run.
- The canonical GUI run draws the current station reference directly in the Isaac
  viewport in cyan and appends the measured mower-tool trail in orange. The reference
  is transformed from the truck frame after every relocation; headless runs skip it.
- A separate four-cylinder residual experiment now lives under
  `PPO2_ROM_PhysX/residual_policy_4cyl/`. It transfers the accepted model_249 trunk and
  first three outputs exactly, inserts a zero observation column, zero-initializes the
  fourth output, and unlocks cylinder 4 through 0/0.25/0.50 mm stages. Never overwrite
  or reinterpret the accepted 41-observation/3-action checkpoint as a four-action model.
