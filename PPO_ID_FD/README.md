# PPO4_1 — the PPO3_1 verge task over ID→FD

`PPO4_1 : PPO3_1 :: PPO4 : PPO3`. Same task, same terrain, same reward, same
gates, same observation and action contract. The **only** difference is what
converts a cylinder command into motion:

    PPO3_1:  policy -> command ROM (ROM-A)                      -> PhysX
    PPO4_1:  policy -> ID (target -> valve) -> FD (valve -> dx) -> PhysX

Everything is inherited by subclassing `Ppo31SoleEnvCfg` / `Ppo31SoleEnv`, and
the gates and reward are **imported** from PPO3_1 rather than copied. A copy
would drift, and the moment it did this comparison would be measuring the drift
instead of the actuator.

---

## Result: ID→FD tracks better, but does NOT pass the contract

Measured on **six** held-out meshes (1337, 17, 909, 4242, 555, 2024), worst-mesh
worst-family, 480 steps x 32 envs, identical distillation settings.

| metric | PPO3_1 (ROM-A) | PPO4_1 (ID→FD) | better |
|---|---|---|---|
| **worst-mesh lateral** | 0.0817 m | **0.0628 m** | **ID→FD by 23%** |
| **worst-mesh ground contact** | **0.7854** | 0.6882 | **ROM-A** |
| contact margin vs 0.700 gate | **+0.0854** | **−0.0118 (FAIL)** | ROM-A |
| meshes failing contact | **0 of 6** | 1 of 6 (mesh 555) | ROM-A |
| contact mean ± σ across meshes | 0.8424 ± 0.030 | 0.7435 ± 0.042 | ROM-A |

**ROM-A passes all eight gates on six meshes. ID→FD does not** — it fails ground
contact on mesh 555.

### The three-mesh result was mesh luck

An earlier run on three meshes (1337/17/909) reported ID→FD passing contact with
a margin of **+0.0069**, and this README previously claimed both pipelines
passed. That was wrong. Those three meshes happen to exclude 555, the one ID→FD
struggles on.

Contact varies **±0.042 across meshes** — six times the margin that run was
arguing about. `gates.py` requires a minimum of three meshes, which is enough to
detect gross failure but **not enough to certify a thin pass**. Treat any contact
margin under ~0.05 as unresolved until measured on more meshes.

### The failure is actuator-specific, not terrain-specific

On mesh 555, ROM-A scores **0.8715** — its second-best result of the six — while
ID→FD scores **0.6882**, its worst. The mesh is not hard in general; it is hard
for ID→FD. So this is a property of the actuator, not of the terrain.

### What ID→FD is genuinely better at

The lateral advantage is robust and holds on every mesh (0.048–0.063 against
ROM-A's 0.060–0.082), and it survives outside the trained amplitude band:

| amplitude | ROM-A lateral | ID→FD lateral | band |
|---|---|---|---|
| 0.20 | 0.0687 | **0.0403** | outside |
| 0.45 | 0.0522 | **0.0341** | inside |
| 0.75 | 0.0777 | **0.0758** | inside |
| 0.85 | 0.0984 | **0.0771** | outside |
| 1.00 | 0.0675 | **0.0548** | outside |

Both degrade gracefully to 1.33x the trained maximum (ROM-A 1.29x, ID→FD 1.61x)
while staying 3–4x inside the 0.300 lateral gate. Neither is *adaptive* — there
is no online context inference; see "Open questions".

### Reading it

The axis of difference is **lateral precision versus vertical ground following**.
ID→FD's learned valve map gives finer lateral authority; ROM-A holds the cutter
on the ground more reliably. ID→FD also runs slightly gentler (hard impacts
0.0515 vs 0.0609, peak tool acceleration 63.1 vs 65.2 m/s²).

**For this task, where tracking is the objective and contact is a gate, ROM-A is
the shippable pipeline** — it is the only one that meets the contract.

## What is deliberately NOT different

For the comparison to mean anything, only the actuator may vary. Verified by
building both configs and diffing them — **14 fields identical, 2 differing, and
both differences are the actuator swap**:

| identical | value |
|---|---|
| terrain amplitude / sweep gain / height scales | 0.7 / 0.5 / (0.50, 0.70, 0.85, 1.00) |
| randomization band | amplitudes (0.30, 0.45, 0.60, 0.75), gains (0.25, 0.40, 0.55) |
| relief profile | `flat_patches` |
| observation / action space | 66 / 4 |
| c4 authority (per-axis leads, scales) | (0.018, 0.018, 0.018, 0.30) / (0.004, 0.004, 0.004, 0.012) |
| arm stiffness scale, truck speed, episode length | 0.35, 4.0 km/h, 16 s |
| PPO regime (lr, schedule, entropy, init_std, nets) | inherited from `Ppo31PPORunnerCfg` |

| differing | PPO3_1 | PPO4_1 |
|---|---|---|
| `cylinder_dynamics_model` | `command_rom` | `fd` |
| `command_rom_checkpoint` | ROM-A path | `disabled` |

**The c4 authority is deliberately PPO3_1's, not PPO4's.** `PPO4_ID_FD_PhysX`
uses a scalar `cylinder_target_max_lead_m` of 0.025; adopting it here would mean
the two pipelines differed in both the actuator *and* the command authority, and
any measured gap could be attributed to either.

`PPO_ID_FD/scripts/preflight.py` runs PPO3_1's checks against the ID/FD
plant — 10/10, with P4 reporting the same eleven families and the same 0.125 m/s
worst rate demand, and S5 reporting identical reward gradients.

---

## Commands

Everything goes through PPO3_1's scripts with `--task` / `TASK=` selecting the
pipeline. There is no separate copy to keep in sync.

```bash
# preflight (10 checks, on the ID/FD plant)
PPO_ID_FD/scripts/preflight.py --num_envs 32 --steps 120

# distil a student from the DLS teacher
verge_task/scripts/distill_terrain.py \
    --task RRLAB-Mulag-Verge-IDFD-v0 --actuator idfd \
    --stage 6 --num_envs 128 --seed 17 --iterations 120 \
    --out logs/rsl_rl/ppo4_1_sole/student_distilled_v2.pt \
    --report PPO_ID_FD/reports/distillation_v2.json --headless

# score on the same 8 gates, one process per mesh
TASK=RRLAB-Mulag-Verge-IDFD-v0 \
  PPO_ROM/scripts/score_per_mesh.sh ppo4_1_student \
  logs/rsl_rl/ppo4_1_sole/student_distilled_v2.pt 480 32

# watch it in the viewer
TASK=RRLAB-Mulag-Verge-IDFD-v0 \
  PPO_ROM/visualize.sh logs/rsl_rl/ppo4_1_sole/student_distilled_v2.pt

# PPO fine-tune (measured to LOSE on PPO3_1; expect the same here)
NUM_ENVS=256 STAGE_ITERATIONS=600 FREEZE_ITERATIONS=150 \
  STUDENT=logs/rsl_rl/ppo4_1_sole/student_distilled_v2.pt \
  LOG_DIR=logs/rsl_rl/ppo4_1_sole/polish \
  PPO_ROM/scripts/supervise_training.sh
```

Deliverable: `logs/rsl_rl/ppo4_1_sole/student_distilled_v2.pt`
Report: `PPO_ROM/reports/frontier/ppo4_1_student_gates.json`

---

## Requirements

The validated ID/FD pair from PPO1, both asserted present at config
construction — falling back to ROM-A would make every number here a PPO3_1
number wearing a PPO4_1 label:

    PPO1_ID_FD_PhysX/models/id/best_id_cylinder_control.pt
    PPO1_ID_FD_PhysX/models/fd/best_fd_cylinder_strict.pt   <- STRICT, not control

The strict FD is the rank-promoted retrain; the earlier FD was rank ~2 and
under-fit the valve.

---

## Open questions

1. ~~Re-measure ID→FD ground contact.~~ **Done** — six meshes, fails at 0.6882
   on mesh 555. The three-mesh pass was mesh luck.
2. **Why is ID→FD worse vertically and better laterally?** The split is clean
   enough to be mechanistic rather than incidental. Worth probing the FD model's
   response on the vertical axis specifically.
3. **Does PPO fine-tuning behave the same here?** On PPO3_1 it reliably bought
   contact by spending lateral across six runs, which is why it lost there. Here
   that trade points the RIGHT way: contact is the failing axis (−0.0118) and
   lateral has 4–5x of margin to spend. This is the one configuration where the
   measured-to-lose result may not transfer. **Running.**
4. **Adaptive policy (RMA).** Neither pipeline infers the terrain it is on. A
   privileged policy over the true terrain parameters plus a context-inference
   module trained on recent history would extend the envelope measured above.
   Scheduled as a later experiment; `_terrain_amplitude_scales` and
   `_terrain_sweep_gains` are already stored per environment as the ground-truth
   context vector it would regress against.

## Layout

    env_cfg.py   Ppo41IDFDEnvCfg — subclasses Ppo31SoleEnvCfg, swaps the actuator
    env.py       Ppo41IDFDEnv — asserts the ID/FD plant is really running
    __init__.py  gym registration; re-exports PPO3_1's gates and reward
    scripts/preflight.py  thin wrapper that runs PPO3_1's checks on this task
