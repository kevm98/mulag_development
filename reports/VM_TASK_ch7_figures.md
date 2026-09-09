# VM task brief — Chapter 7 figures

Purpose: produce the two figures Chapter 7 currently carries as placeholders,
and refresh the cross-stack figures so they describe the **rebuilt** AF stack.

Seeds are deliberately excluded. Nothing here trains anything; all runs are
evaluation only.

Why this is needed: every figure artifact in the repo dates from **2026-08-18**
(`outputs/stage31_openloop`, `experiments/E1_tracking_and_control_effort/results`, `experiments/E2_transport_delay_and_identifiability/results`,
`outputs/e1_traces`). The AF command path was rebuilt after that, so the stored
adaptive traces describe the retired architecture and cannot be replotted into
anything truthful.

---

## 0. Blocking code change (must land before Run 1)

The trace recorder never captured pressure. Confirmed on all three stored
traces: recorded fields are `step, cutter_x, cutter_y, cutter_z, reference_y,
lateral_error, cut_height_error, z_hat, z_hat_raw, valve_cmd, valve_applied`
(plus `theta_*` on the perturbed stacks). There is no `dP`.

Add to the per-step `TraceRecorder.record(...)` call:

* `dP` — the four differential-pressure entries the actuator model carries
  (`self._dP`), i.e. the ROM's internal state on R and the frozen FD's
  recursive state on IF and AF
* `mass_scale_now` — so the flip instant is recoverable from the trace itself
  rather than inferred

Everything else already exists.

---

## 1. Pressure response and AF tracking, all three stacks (the priority)

One run per stack, one mesh, tracing on, through a mid-episode change of load.
This single run produces **both** placeholder figures.

Plant condition, identical for all three stacks:

```
MULAG_PAYLOAD_RANGE=1.0,1.0
MULAG_MASS_STEP_TIME_S=20.0
MULAG_MASS_STEP_SCALE=3.0
MULAG_MASS_STEP_JITTER_S=8.0
MULAG_MASS_BIMODAL=1.0,3.0
```

Arms:

| arm | checkpoint | extra |
|---|---|---|
| base (frozen) | `logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt` | no residual |
| control | same base | `--residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/rebal17_nohistory/model_399.pt --residual_no_history` |
| adaptive | same base | `--residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/rebal17_history/model_399.pt` |

Common flags: `--residual_alpha 0.5 --residual_history 20 --num_envs 176
--steps 900 --meshes 1337 --seed 17`, and `MULAG_DIRECT_VALVE=1`,
`MULAG_RMA_CONTEXT=1` as in `experiment_scripts/seeds_score_only.sh`.

For R and IF, run the same mesh/seed/steps through their own tasks so the
pressure panels are comparable:

* R: `logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt`
* IF: `logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt`,
  task `RRLAB-Mulag-Verge-IDFD-v0`

**Deliver:** `outputs/ch7_pressure/{rom,idfd,af_base,af_control,af_adaptive}_m1337.npz`

Each `.npz` must contain `step, dP, mass_scale_now, lateral_error,
cut_height_error, valve_cmd, valve_applied`.

That gives Figure `fig:eval:pressure-traces` (dP per cylinder across the flip,
three stacks) and Figure `fig:eval:arms` (lateral and cut-height traces for the
three AF arms, flip instant marked).

---

## 2. Cross-stack refresh (only if time allows)

The tracking, gate, chain, step and effort figures are currently two-stack (R
and IF) because the stored AF numbers describe the retired path. To restore AF
to them, re-score it on the E1 protocol:

* `experiments/E1_tracking_and_control_effort/code/run_e1_tracking.sh`, replacing the third row's checkpoint
  and task with the rebuilt stack (base + `rebal17_history` residual, direct
  valve), leaving R and IF rows untouched
* **Deliver:** `outputs/e1_tracking/` refreshed, and
  `experiments/E1_tracking_and_control_effort/results/control_effort.json`

Note this is a genuinely different comparison from before: AF no longer shares
the action contract with R and IF, so it should be reported as its own condition
rather than ranked against them.

---

## 3. Do NOT run

* Seed replication (23, 42, 101, 7) — long, and you are handling it
* The sham control (`e3_e5_close.sh`) — long
* Any training

---

## Pull back to the Mac

```
rsync -rvz --checksum --partial --exclude='__pycache__/' \
  -e "ssh -i <key>" <vm>:isaaclab-cloud/projects/rrlab_mulag/outputs/ \
  /Volumes/DATA/rrlab_mulag/outputs/
```

`--checksum` is required: `/Volumes/DATA` is FAT32 and cannot store mtimes, so
timestamp-based sync re-transfers everything.

Disk note: the volume is at ~97 %, about 1 GB free.
