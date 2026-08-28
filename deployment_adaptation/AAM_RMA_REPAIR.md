# AAM repair plan — rebuild in RMA order

Written 2026-08-28. **Do not start until the E5 ROM degradation curve is
complete, committed and pushed** (`experiments/E5/HANDOVER_GCP_SWEEP.md`). E5
and this plan both want the L4 and the budget; running them concurrently
doubles the burn rate and mixes two sets of results on one disk.

This supersedes the "filter the latent on the policy path" recommendation in
Chapter 8, which E6 withdrew. It does not supersede
`deployment_adaptation/README.md` stages 0-31, which remain the build record.

---

## The finding this plan responds to

The AAM was built in the wrong order. RMA is: **train a policy that is
addicted to a plant-identity code, then learn to infer that code.** What was
built instead was a code trained for `SD_z`, an adaptation site behind a frozen
inverse model, and a policy lightly fine-tuned from an initialisation that
already worked without the code.

Verified against the repo, not inferred:

- **No z-conditional expert exists.** Every run in
  `logs/rsl_rl/ppo_id_aam_fd/` carries
  `initialization_source: "PPO3 terrain PPO"`, and `headroom_priced_v1`
  resumes from `idfd_init_speed_2_7/model_init.pt` — an ID-FD policy with no
  latent. RMA Stage 1 was never run.
- **The deployed fine-tune was small.** `headroom_priced_v1/run_manifest.json`:
  `learning_rate 3e-06`, `entropy_coef 0.001`, `freeze_actor_iterations 0`,
  `obs_normalizer "freeze"`, `online_aam_residual_scale 0.25`. (The 1e-5 /
  entropy 0 / 100-frozen-iteration figures quoted in some notes come from the
  README's example command and from
  `privileged_true_l20_s5_combined_train_20260816`, not from the deployed run.
  Do not cite them for `headroom_priced_v1`.)
- **E4's oracle arm is a substitution, not an expert.** `aam_true` swaps the
  privileged latent into `headroom_priced_v1`'s latent slot. It therefore
  measures whether *this* policy can use plant identity, and answers no. It is
  **not** evidence that no headroom exists on the task. Any thesis sentence
  claiming the latter must be corrected.
- **`mu` cannot encode terrain or truck speed.**
  `sd_model/stage23_frozen/dz2_nodp_holdq/protocol.json` gives
  `factor_keys = [flow, compliance, seal, pressure, mass_scale, com_z]` and
  `z = mu(e)` is a 6->2 bottleneck on that vector alone. The real
  representation problem is *within* those six: `SD_z`'s objective keeps
  whatever most moves 50 ms ROM dynamics — the hydraulic channels — and
  squeezes out `com_z`, whose train std is 0.036 over a [-0.10, 0.05] range.
- **`d_z=2` has no axis semantics.** It is an unstructured bottleneck. There is
  no "leak axis" and no "CoM axis" today. Gate 1 below *creates* that
  semantics; it does not preserve it.
- **The 89% preclip saturation figure is not a measurement of the deployed
  stack.** `AAM_module/stage28_id_transfer.md` evaluated the frozen historical
  ID on **Stage-22 Simscape-era** runs under the legacy `training_mean`
  feature contract with force absent. It is the leading hypothesis for why the
  adaptation site is dead. It has never been measured in the Stage-31 loop.
  **Gate 0 measures it.** Do not write it into the thesis as a current fact
  until Gate 0 does.

## Gate order and the stopping rule

Gates run in order. **Each gate can stop the plan**, and a stop is a result,
not a failure to report. Do not train another AAM until Gate 1 passes.

```
Gate 0  measure the adaptation site            ~1 GPU-hour
Gate 1  privileged z-conditional expert        the decision point
Gate 2  excite com_z in the deployment loop    only if Gate 1 says CoM is dead
Gate 3  on-policy AAM (RMA Stage 2)            only after Gate 1 passes
Gate 4  A-RMA fine-tune + acceptance panel
```

---

## Gate 0 — measure the adaptation site before rebuilding it

Two cheap measurements that decide whether the expensive gates are justified.
Neither trains anything.

### 0a. In-loop clip fraction of `u_base`

Instrument the Stage-31 deployed stack and record, per control tick, the
fraction of the four valve elements where `u_base = ID_frozen(...)` is already
at its preclip limit before the adapter residual is added, plus the fraction
where `u_final` differs from `u_base` by less than 1% of the residual bound.

Run on the same 33-cell panel E4 used: `FACTOR_DOMAIN=combined_test`, meshes
1337/17/909, `headroom_priced_v1/model_299`.

**Predeclare the reading, now, before the number is seen:**

| clip fraction | reading |
|---|---|
| > 0.50 | the adaptation site is saturated. Gate 0c is mandatory. |
| 0.15 - 0.50 | partially saturated; Gate 0c is optional, Gate 1 proceeds either way. |
| < 0.15 | the ID is **not** the blocker. Stage 28's 89% does not transfer. Skip Gate 0c, go to Gate 1, and correct the thesis. |

### 0b. Fail-safe fallback rate

`online_aam_fail_safe` falls back to the frozen ID on invalid or slow AAM
inference (`online_aam_max_inference_latency_ms: 50.0`). Record the per-episode
fraction of ticks that took the fallback. If this is non-trivial the loop has
been running partly without `z_hat` and every E4/E6 arm is diluted by an
unmeasured amount. **This must be reported beside every latent result from now
on**, whatever it turns out to be.

### 0c. Unsaturate the command path — only if 0a says > 0.50

Two acceptable interfaces. Pick one and record why:

- **A (classic RMA).** The policy emits the cylinder command directly,
  `pi(x, z) -> u`. The frozen ID becomes a warm start or an auxiliary loss, not
  a clip in the command path.
- **B (current stack, repaired).** Rebuild the ID as z-conditional and
  unsaturated in the operating region:
  `u_ff = ID_theta(q, qdot, qddot_des, z)`, trained in PhysX over the factor
  design. Pass criterion is physical, not a loss value: at long versus short
  reach the hold command must scale with the CoM channel, and at fixed pose the
  sag must scale with the seal/leak channel.

B keeps the PPO-ID-FD comparison intact and is the smaller change to the
thesis narrative. A is faster to get working. **This is a design decision for
the user, not for the runner** — bring the 0a number and ask.

---

## Gate 1 — RMA Stage 1: a privileged z-conditional expert

**This is the decision point for the whole plan.** Everything before it is
measurement; everything after it is wasted unless it passes.

Train a policy **from scratch** — not resumed from any `PPO3 terrain PPO`
descendant — conditioned on the **true** privileged latent.

Configuration, each item a deliberate reversal of what the deployed run did:

| setting | deployed run | Gate 1 |
|---|---|---|
| init | resume from no-latent ID-FD | **from scratch** |
| learning rate | 3e-06 | normal PPO schedule |
| entropy coef | 0.001 | normal, non-zero |
| `freeze_actor_iterations` | 0 (n/a, tiny lr) | 0, and the lr is real |
| `obs_normalizer` | `freeze` (fitted without z) | `adapt`, refit including the z columns |
| latent in critic | inherited | explicit, plus LayerNorm on the z columns |
| latent mode | `estimated` | `true` |

`online_aam_latent_mode` already accepts `true`, and
`online_runtime.py:198` requires an adapter whose `latent_source` matches. Both
adapters exist and were verified 2026-08-28 by reading the checkpoint payloads:
`id_adapter_cnn_l20_s5/zhat_adapter.pt` declares `latent_source="zhat"` and
`id_adapter_cnn_l20_s5_true/true_adapter.pt` declares `"true"`. No adapter
retraining is needed to start this gate.

**Gate 1 must override the adapter path.** `env_cfg.py:33` defaults
`ADAPTER_MODEL` to the *zhat* adapter, so `latent_mode='true'` with the default
raises `latent_mode='true' requires an adapter trained with
latent_source='true'`. Set it explicitly:

```bash
export MULAG_AAM_ADAPTER_MODEL=outputs/stage31_rom/id_adapter_cnn_l20_s5_true/true_adapter.pt
```

Note that `TRUE_LATENT_MODEL` (`sd_model/stage31_frozen/z2_rom`) is a different
artifact — the frozen encoder that computes `z` from `e`, needing `encoder.pt`
and `metrics.json` in that directory. Both are present on the VM. The adapter
and the encoder are both required for `latent_mode='true'`; supplying one is
not enough.

Add `||dec(z) - e||^2` as an auxiliary loss so the two axes acquire physical
meaning. Without it `d_z=2` stays unstructured and Gate 3's per-axis checks
have nothing to check.

Evaluate `zero` / `shuffled` / `mean` against `true` **every N iterations
during training**, not only at the end. That is the instrument; if it never
separates, the run is telling you so while it is cheap to stop.

### Stop condition — predeclare it and do not move it

Score `true` vs `mean` vs `zero` on the **same 33-cell panel and the same
metrics** as E4, so the numbers are directly comparable to
`experiments/E4/results/e4_summary.json`.

- **Pass:** `true` beats `mean` on mean absolute lateral error with a bootstrap
  95% CI **excluding zero**, and beats it in a clear majority of the 33 cells.
  E4's oracle-vs-mean was -3.4% with the CI through zero and 18/33 cells; a
  result of that shape is a **fail**, not a marginal pass.
- **Fail:** the gap does not appear. Then either `com_z` is still unexcited
  (-> Gate 2 once, then retry Gate 1) or the command path still eats the
  credit (-> Gate 0c if it was skipped). **Do not train an AAM.** If both
  retries fail, the honest result is that this task is not adaptation-limited,
  and that is the chapter.

---

## Gate 2 — excite `com_z` in the deployment loop

Only if Gate 1 fails and the per-axis diagnostic shows the CoM channel is
unidentifiable.

Keep the 1.0 s AAM window and keep `d_z=2` — **do not add latent dimensions**.
Change the episode content so the CoM channel is observable: every episode must
contain a hold at short reach, a slow extension, and a hold at long reach. The
hydraulic channels are already excited by the existing motion; this is a CoM
fix only.

The Stage 12-13 collection design (`AAM_module/stage12_*`, `stage13_*`) has the
posture and PRBS structure to copy from, but this must land in the **PPO
episode**, not in a separate identification collection. That distinction is the
whole point of Gate 3.

Then return to Gate 1. One retry.

---

## Gate 3 — RMA Stage 2: on-policy AAM

Only after Gate 1 passes.

Freeze `pi`, `mu` and the ID. Train `phi` on **closed-loop rollouts of the
policy that will deploy** — not on a second DLS/PRBS dump. The existing
`AAM_module/datasets/rom_deployment_stage31_v1_frozen` (2.3 GB) is an
identification collection at classical-controller excitation; it is not the
2-7 km/h stage-6 truck loop, and training `phi` on it is the distribution
mismatch that Stage 26's offline numbers hid.

Loss:

```
L_phi = ||z_hat - z||^2  +  ||dec(z_hat) - e||^2  +  lambda * ||FD(q, qdot, u, z_hat) - qdot_next||^2
```

Two changes from Stage 26:

- **Condition FD on the latent.** It was deliberately left unconditioned, which
  removed the only term that would have anchored `z_hat` to plant identity
  rather than to the time-varying residual. This is the fix for the 13.6% /
  0.43% decomposition.
- **Drop or heavily downweight the adjacent-window temporal-consistency term**
  (currently weight 1.0) until per-axis plant-identity R^2 against `e` is
  measurably real. Consistency on a code that carries 0.43% identity just
  produces a smoother wrong signal.

Gate: per-axis R^2 of `dec(z_hat)` against the true `e` channels, on held-out
factor configurations, reported per channel. A latent that cannot recover
`seal` and `com_z` above chance does not proceed.

---

## Gate 4 — A-RMA Stage 3 and the acceptance panel

Unfreeze `pi` (and a thin ID residual if on interface B), condition on
`z_hat`, normal learning rate and entropy, obs-normaliser refit with the z
columns included.

`online_aam_residual_scale = 0.25` may be reintroduced **only after** the
oracle gap exists. It cannot create the gap; at Gate 4 it is a safety bound,
not a mechanism.

Rerun the full 33-cell panel. **Predeclared acceptance ordering**, on mean
absolute lateral error:

```
true z  <  z_hat  <  mean z  <  zeros / shuffled
```

with the `true` vs `mean` contrast resolved (CI excluding zero).

- If `true z` is still tied with `mean z`: you are back in Gate 1, not in
  Gate 4. Do not tune.
- If `z_hat` lands **above** `mean z` — the E4 outcome — the estimator is the
  problem and Gate 3 is where to return. This is the first point in the plan
  where "improve the AAM" is a legitimate response to a bad number.

Report the median and the per-cell win count beside every mean. E6's whole
finding was that the means were tail-driven while the medians were flat; a
mean-only table would have read as a success.

---

## Traps carried forward

- **A flat curve has two readings** — the controller absorbed it, or it never
  arrived. Every gate here needs its positive control. Gate 1's is the
  during-training `zero`/`shuffle` eval; Gate 3's is the per-axis R^2.
- **`FACTOR_DOMAIN=combined_test` is mandatory** for every scoring run. Under
  `nominal` the plant does not vary, the latent is near-constant, and every
  arm collapses to the same thing.
- **Validate the instrument before believing a null.** E6 raises rather than
  reports when the low-pass never reached the config. Gate 1 needs the
  equivalent: assert that the `true`/`mean`/`zero` arms actually produce
  different latent tensors, or a plumbing bug scores as a clean negative
  result.
- **Sequential audits in one process have fabricated failures in this repo.**
  Reproduce any gate failure in a fresh process before acting on it.
- **Thresholds are predeclared above and must not move after a curve is seen.**
  With the cross-stack comparison already dropped from E5, these are the only
  thing standing between this plan and a cherry-picked result.

## VM prerequisites beyond the E5 set

The E5 handover pushed the ROM and ID/AIF checkpoints, `id_fd_model/`,
`sd_model/` and the USD assets. This plan additionally needs, and `.gitignore`
excludes all of them (`outputs/`, `logs/`, `datasets/`, `*.pt`):

```
outputs/stage31_rom/aam_cnn/L20_d64/aam.pt          + metrics.json   AAM
outputs/stage31_rom/id_adapter_cnn_l20_s5/zhat_adapter.pt
                                                   + manifest.json  estimated adapter
outputs/stage31_rom/id_adapter_cnn_l20_s5_true/true_adapter.pt
                                                   + manifest.json  true adapter (Gate 1)
AAM_module/datasets/rom_deployment_stage31_v1_frozen/raw/manifest.csv
                                                   7.9 KB, factor draws
sd_model/stage31_frozen/z2_rom/                                     frozen encoder
logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt           Gate 0 subject
```

**Pushed and sha256-verified on the VM 2026-08-28**, and the container
was then confirmed to register `RRLAB-Mulag-Verge-ID-AAM-FD-v0` and resolve
`DEFAULT_MANIFEST`. `rrlab_assets` needs no work: `~/bin/isaaclab-run` already
puts `extensions/rrlab_assets` on `PYTHONPATH`. The working image is
`isaac-lab:2.3.2-rl052`, not the wrapper's `:2.0.0` default, so every Gate must
be invoked with `IMAGE=isaac-lab:2.3.2-rl052`. The last two were already
there from the E5 push; the rest were not, and Gate 0 cannot start without
them: `env_cfg.py:45` and `:35` name the AAM and adapter as defaults, so the
env does not build, and with no adapter there is no residual and no
`fallback_active`, which makes 0a's residual half and the whole of 0b
unmeasurable.

The `.pt` files each need their sidecar `metrics.json` / `manifest.json` — the
loaders read the normalisation statistics and the `latent_source` field from
them, and `online_runtime.py:198` refuses a mismatched source.

**`manifest.csv` is the correction to an error in the first version of this
plan**, which said the Stage-31 dataset was not needed. The 2.3 GB of runs are
not, but `raw/manifest.csv` is: it is 7.9 KB, it is the file
`verge_task/deployment_factors.py:30` and `terrain_env_cfg.py:269` read, and it
carries the only copy of the factor draws — 16 combined/test rows, 48
combined/train, 40 hydraulic, 12 mechanical, 2 nominal. Without it, passing
`--factor_domain` at all sets `deployment_factor_enabled=True` and the run
crashes. `FACTOR_DOMAIN=combined_test` is mandatory for every Gate, and under
`nominal` the plant does not vary and every arm collapses to the same thing.

Still blocking from the E5 handover and equally blocking here: the hardcoded
`ROOT` / `PY` Legion paths in `score_per_mesh.sh` and the shebangs of
`evaluate_gates.py` / `preflight.py` / `train.py`, and `rrlab_assets` needing
to be importable in the container. Unlike the ROM path, this stack also needs
`RRLAB-Mulag-Verge-ID-AAM-FD-v0` to register — `PPO_ID_AAM_FD/__init__.py:11`
registers it itself, so no `rrlab_tasks` dependency, but confirm it.

## Cost

Gate 0 is about an hour. Gate 1 is a from-scratch PPO run and is the expensive
item — `headroom_priced_v1` was a 300-iteration fine-tune at 256 envs, and
from-scratch is a different order of magnitude on an L4. **Estimate Gate 1's
wall-clock from a 20-iteration timing run and bring the number to the user
before launching the full run.** At $0.88/hour against a 237 EUR budget, a
multi-day from-scratch run is a budget decision, not a runner's decision.

Stop the VM between gates: `cloud/bin/isaac-stop`. Copy results back and commit
after every gate — git is the only backup and `/Volumes/DATA` is FAT32 and has
already lost files once.
