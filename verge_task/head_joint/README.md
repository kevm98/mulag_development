# head_joint — isolated q5 experiment

Everything here is an **experiment**. The production PPO3 line outside this
folder stays on the verified 4-action contract and is not changed by any work
in here.

## Scope

The original request was narrow: **stop the tool head bouncing while tracking,
and stop the truck swaying.** That is already delivered on the main line and did
not require touching the arm:

| metric | before | after | how |
|---|---|---|---|
| p95 tool acceleration | 111.6 m/s² | **91.7** | `terrain_constrain_truck_per_substep = True` |
| p95 chassis sway | 54.77 mm | **17.72** | same (zero-action floor is 16.40 mm) |
| p95 attitude sway | 0.629° | **0.206** | same (floor 0.202°) |
| p95 cut error | — | 0.0631 m | unchanged contract |

The truck was feeding motion into the boom between control-step corrections.
Pinning the chassis every physics substep removed it at the source. **No arm or
actuator change was involved.**

## Verified production baseline (do not regress)

- checkpoint `logs/rsl_rl/ppo3_sole_terrain/lawn_rough_stage5_v5/model_best.pt`
- `sole_head_joint_action = False` (4-action interface)
- **16/17 held-out gates**, failing only `wheel_road_support` (0.1947 vs 0.20)
- 16/17 and 14/17 on two varied terrain meshes
- artifacts: `../reports/FINAL_16of17.json`, `../reports/FINAL_REPORT.md`

Reproduce with:

```bash
~/miniconda3/envs/env_isaaclab/bin/python verge_task/scripts/evaluate.py \
  --device cuda:0 --stage 5 --num_envs 128 --steps 960 --eval_terrain \
  --checkpoint logs/rsl_rl/ppo3_sole_terrain/lawn_rough_stage5_v5/model_best.pt
```

## What the q5 experiment is for

A separate, unresolved problem: worst-case head alignment. `min_head_alignment`
fails on varied terrain in every configuration measured (0.9500 held-out is a
zero-margin pass; 0.9396 and 0.9185 on varied meshes), while `p01` stays healthy
at 0.976–0.988 — rare, mesh-dependent excursions.

`Messerkopf_joint` carries no cylinder, is not hydraulic, and is **not commanded
by ROM-A**. It was slaved to c4 by a fixed `0.76·q4` feedforward with no
feedback, and misses that reference by 0.2096 rad at p99. Every PPO fine-tuning
campaign collapsed on head alignment — the one axis the policy could not command.

The code for the 5-action interface is committed and gated behind
`sole_head_joint_action`, default **False**. Nothing runs it unless a config in
this folder turns it on.

## Open question that gates everything else

Arithmetic on the committed actuator values (`stiffness 5e6`, `damping 1.12e6`,
`effort_limit 20000`):

| quantity | value |
|---|---|
| position error saturating effort alone | 0.004 rad (0.23°) |
| velocity saturating effort alone | 0.0179 rad/s (1.02°/s) |
| reference slew we command | 0.6 rad/s |
| ratio | **33.6×** |
| measured p99 reference miss | 0.2096 rad = **52×** the 0.004 rad threshold |

If the effort log confirms this, the joint is a saturated torque source during
essentially all motion, which would explain why every feedback law written for
it lost to plain feedforward. **Confirm by logging applied vs computed effort
before changing anything.** If it is not confirmed, the actuator must not be
touched.

## Measurements already taken (treat as provisional)

Several comparisons from the 2026-07-31 session were confounded and need
independent re-derivation before being trusted:

- the "static head" baseline was **not** static — the parent rewrote the
  reference to the coupling every substep
- the "coupling only" case had the levelling correction silently disabled
- two edits were bit-identical no-ops that looked like fixes

Head reference variants, 128 envs × 480 steps, stage 5 (provisional):

| variant | min align | p01 | contact | p95 cut | p95 accel |
|---|---|---|---|---|---|
| coupling + levelling | 0.9296 | 0.9858 | 0.8937 | 0.0574 | 67.8 |
| frozen (no boom tracking) | 0.8369 | 0.9569 | 0.9056 | 0.0496 | 67.2 |
| coupling, levelling off | 0.8662 | 0.9567 | 0.9065 | 0.0484 | 67.3 |
| + tilt P+D, sign − | 0.9161 | 0.9790 | 0.8838 | 0.0732 | 105.0 |
| + tilt P+D, sign + | 0.8475 | 0.8884 | 0.8924 | 0.0553 | 121.1 |

Both signs of a tilt-feedback law lose to no feedback at all, and acceleration
rises sharply — consistent with two controllers (c4's tilt assist and the head
loop) fighting one error, into a saturated drive.

q5 → tool tilt was measured at ≈0.36 rad/rad from a **single** probe point that
assumed a zero baseline tilt, which was never measured. Re-measure 3-point
(−1 / 0 / +1, boom fixed) before relying on it.

## Rule for this folder

Measurement first, plant changes only on confirmed evidence, and one variable at
a time. The production line already satisfies the original request; nothing here
is worth regressing it for.
