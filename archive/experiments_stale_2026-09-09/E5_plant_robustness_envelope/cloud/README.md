# E5 on the GCP L4 VM — environment and cloud harness

Written 2026-08-28 on `isaaclab-l4`. Covers what it took to make the ROM (R)
stack actually run off the Legion box, and what is still open.

## The environment question, settled

The repo does not pin an Isaac Lab version anywhere, and the first three
candidate images all failed. The combination that runs is:

| Component     | Version   |
| ------------- | --------- |
| Isaac Lab     | **2.3.2** |
| `isaaclab_rl` | **0.5.2** |
| `rsl-rl-lib`  | **5.0.1** |
| Isaac Sim     | **5.1.0** |
| Python        | **3.11.13** |

The catch: the **published `nvcr.io/nvidia/isaac-lab:2.3.2` image is not
sufficient**. It is built from the `v2.3.2` *tag*, which carries `isaaclab_rl`
0.4.7 — and 0.4.7 has no `RslRlMLPModelCfg`, which five config modules in this
repo import. Isaac Lab's **`main` branch** sits at the same `VERSION` 2.3.2 but
carries `isaaclab_rl` 0.5.2, which does have it. So the Legion box was running a
main-branch checkout, not a release tag.

`Dockerfile.isaaclab-2.3.2-rl052` reconstructs that: stock `:2.3.2`, with the
`isaaclab_rl` extension replaced by main's 0.5.2 and `rsl-rl-lib` moved
3.1.2 -> 5.0.1. Nothing else is touched — same Isaac Sim, same PhysX.

```bash
docker build -t isaac-lab:2.3.2-rl052 .   # needs source/isaaclab_rl from IsaacLab main
IMAGE=isaac-lab:2.3.2-rl052 ~/bin/isaaclab-run <cmd>
```

Three independent facts pick this combination, and all three must hold:

1. `model_598.pt` stores **separate** `actor_state_dict` / `critic_state_dict`,
   each with its own `obs_normalizer.*`, plus `distribution.std_param`. That is
   the rsl-rl >= 5.0 layout, so `isaaclab_rl` >= 0.5.0 is required. There is no
   new -> old checkpoint converter, so the older API cannot be substituted.
2. The repo reads `cfg.sim.physx`. Isaac Lab 3.0 renamed that field to
   `.physics`, so 3.0 is out. PhysX is still the backend on both sides of the
   rename (Newton is opt-in), but the attribute is not.
3. Container Python is 3.11, matching the documented thesis environment.

### Why not the other images

* `:2.0.0` — `isaaclab_rl` too old, no `RslRlMLPModelCfg`.
* `:2.3.2` — same problem (0.4.7, built from the tag).
* `:3.0.0-beta2-post1` — has `RslRlMLPModelCfg`, but renamed
  `SimulationCfg.physx` -> `.physics` and changed
  `root_physx_view.get_masses()` to return a non-torch array. The latter is a
  data-path change on the mass/COM channel that `mass_scale` and `com_z` ride,
  so it is not a safe substitution for this experiment.

## Repo changes made for the cloud run

All are written to be **no-ops on the Legion box**, so nothing regresses there.

* `PPO_ROM/scripts/{score_per_mesh,supervise_training,select_checkpoint}.sh` —
  `ROOT`/`PY` made `${VAR:-default}` overridable instead of hardcoded.
* `PPO_ROM/scripts/{preflight,training_status,train,evaluate_gates}.py` —
  shebang `~/miniconda3/...` -> `/usr/bin/env python3`. Inert either
  way, since these are invoked through `$PY`.
* `experiments/E5_plant_robustness_envelope/code/make_e5_figure.py` — output path had a stray space
  (`"thesis/thesis/..."`); now derived from `$ROOT`.
* `mulag_plant/runtime/physx_compat.py` (new) + an import in
  `mulag_upper_level_env_cfg.py` — restores `cfg.sim.physx` on builds that
  renamed it. Self-disabling where the field already exists.

That shim also fixes a latent bug worth knowing about independently of the
version question: `sim_memory.py:37` reads
`getattr(sim_cfg, "physx", sim_cfg)`, so on a renamed build it would have
silently applied **none** of the GPU memory settings rather than raising.

## Open: the ROM path is not reproducible

**This blocks the sweep and must be resolved before any curve is read.**

Two runs, identical config, same seed (17), same mesh (1337), 8 envs x 240
steps:

| metric                     | run A  | run B  | delta      |
| -------------------------- | ------ | ------ | ---------- |
| `mean_abs_lateral_error_m` | 0.0722 | 0.0805 | **+11.5%** |
| `ground_contact_fraction`  | 0.7579 | 0.7873 | **+2.9 pp** |

224 fields differ. Against the predeclared thresholds — onset at lateral error
>= 1.5x nominal, or ground contact >= 5 pp below nominal — run-to-run noise is
already **59% of the ground-contact threshold**. A reported "degradation onset"
could be noise.

This also contradicts the project's own stated expectation: `gates.py` says
*"varying `--seed` alone returns bit-identical numbers."*

### Confirmed: true nondeterminism, not small-sample noise

Repeated at the sweep's own production settings (`ENVS=16 STEPS=2808`, mesh
1337, seed 17) — 12x the data. Variance did **not** shrink:

| metric                     | smoke (8x240) | production (16x2808) | vs. threshold        |
| -------------------------- | ------------- | -------------------- | -------------------- |
| lateral error              | 11.5%         | **10.8%**            | 22% of the 1.5x onset |
| ground contact             | 2.9 pp        | **3.68 pp**          | **74% of the 5 pp onset** |
| cut-height error           | —             | 0.04%                | negligible           |
| `safety_termination_rate`  | —             | **38.3%** (0.262 vs 0.162) | —              |

The mechanism is visible in the per-distance-band breakdown. Sample counts are
**identical** run to run (3337, 3345, 3345, 3347, ...), as are
`scan_valid_fraction`, `episode_termination_rate` and `max_truck_travel_m` — so
this is not an episode-count or sampling difference. Instead the divergence
**grows monotonically with distance travelled**:

| band (m) | lateral rel. diff |
| -------- | ----------------- |
| 0–3      | 0.95%             |
| 6–10     | 4.16%             |
| 13–16    | 11.15%            |
| 23–26    | 13.70%            |

That is the signature of chaotic amplification of floating-point differences
(GPU PhysX solver / atomics ordering) over a long closed-loop rollout, not a
harness fault. Cut-height stays deterministic (0.04%) because it is a local
regulation channel; the lateral/path channel integrates error over the run.

**Consequence for the design.** A single run per stress level cannot resolve
the predeclared thresholds — run-to-run noise on ground contact (3.68 pp) is
74% of the 5 pp onset criterion. Each stress point needs repeats, with the
noise band measured at nominal and an effect required to exceed it. Raising
`ENVS` also averages the per-env chaos down (~1/sqrt(N)); at 16 envs one
evaluator held well under the L4's 23 GB, so 64–128 envs is affordable and
would roughly halve to quarter the spread.

## Open: nominal may already breach the absolute breakdown thresholds

Both nominal runs, measured against the predeclared *absolute* functional
breakdown criteria (ground contact < 0.700 **or** safety termination > 0.080):

| run | worst-family ground contact | safety termination rate |
| --- | --------------------------- | ----------------------- |
| A   | 0.2658  (breach)            | 0.2619  (breach)        |
| B   | 0.4099  (breach)            | 0.1617  (breach)        |

If this holds up, the envelope sweep is vacuous in the handover's own sense —
degradation cannot be measured from a baseline that already reads as broken.

**Not yet confirmed, and it must be before anyone acts on it.** These are
single-mesh (1337) numbers with `merge_gates` refusing to score (it requires
>= 3 distinct meshes), and they read `worst_family_ground_contact_fraction`,
which may not be the aggregate the thresholds were written against. Re-check
with a proper >= 3-mesh gate evaluation before drawing any conclusion.
