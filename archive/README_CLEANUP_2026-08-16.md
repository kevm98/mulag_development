# Archive cleanup — 2026-08-16

Tree cleanup after the AAM / stage31 pipeline session. **Nothing was deleted.**
Everything below was *moved*, and every tracked file removed from the working tree
was *recovered from git* before its deletion was committed.

## What was kept live (the working set)

Determined by (a) files modified in the last 24 h and (b) references from live code.

**`logs/rsl_rl/` — 6.4 GB → 1.7 GB**

| run | size | why kept |
|---|---|---|
| `ppo_id_aam_fd` | 456M | active session; 53 sub-runs, latest 2026-08-16 17:26 |
| `ppo3_1_sole` | 856M | referenced by 19 live files; current branch topic |
| `ppo4_1_sole` | 306M | referenced by 58 live files; current branch topic |
| `ppo2_a_trajectory_2d` | 51M | referenced by live trajectory-2D code |
| `ppo2_a_policy_v3` | 4.0M | referenced by live code |
| `ppo3_1_sole_c4_authority` | 4.0K | referenced by live code |

**`outputs/` — 703 MB → 460 MB.** Kept `stage31_rom` (live) and the
`aam_stage25` → `stage30` lineage still cited by the `AAM_module/stage*.md` docs.

## What was retired

### `checkpoints_2026-08-16/` — 4.8 GB, 84 run dirs + 14 stray logs
Every `logs/rsl_rl` run with **no reference from live code** and untouched since
2026-08-10. Mostly the PPO1 `stage0/stage1` lineage, the PPO2 ROM-B/ROM-C
residual family, the `id_fd_*` pre-residual runs, and the ~30 `ppo3_sole_rom_direct_q5_*`
ablation sweep. Per-run sizes and timestamps in `MANIFEST.tsv`.

### `outputs_retired_2026-08-16/` — 243 MB, 26 dirs
Two groups, tagged in `MANIFEST.tsv`:
- **bug-produced / quarantined** — `aam_stage25_buggy_zhat`, `aam_stage26_buggy_zhat`,
  `aam_stage25_corrected_interrupted`, `stage22_v6_quarantine_20260809`,
  `stage29_latent_adapter_retry`, `stage30_pressure_adapter_aligned_cpu`,
  `aam_stage25_smoothed`.
- **pre-August legacy** — PPO1 / PPO2 / early `id_fd` outputs, all superseded.

### `ppo_rom_retired_2026-08-16/` — 812 KB, 96 files
The PPO_ROM / PPO_ID_FD files deleted from the working tree during this session.
They existed **only in git history**, so each was restored with `git show HEAD:<path>`
into this directory before the deletion was committed. Contents: the 82-file
`PPO_ROM/reports/frontier/` sweep, 7 `PPO_ROM/scripts/` probes
(`evaluate.py`, `probe_c4_authority.py`, `probe_path_feasibility.py`,
`probe_target_anchoring.py`, `run_warmup_finetune.sh`, `score_latest.sh`, `smoke.py`),
5 `PPO_ROM/reports/` JSONs, `PPO_ROM/CHALLENGES_FOR_REVIEW.md`, and
`PPO_ID_FD/reports/distillation_v2.json`. All 96 verified non-empty.

### `stale_runtime_state_2026-08-16/`
`.speed_randomization_campaign.lock` from `experiments/` — held pid 751838, dead
since 2026-08-13, so the lock was blocking nothing.

## Not archived, but labeled

`outputs/stage31_rom/promotion/STALE_VERDICTS.md` records that 4 of 5
`promotion_verdict.json` files carry a `promotion_contract` PPO field that
contradicts their own `ppo.pass=false`. The verdicts were left byte-for-byte
intact for provenance. `promoted=false` in all five, so the system-level
conclusion was never wrong — **the system remains correctly unpromoted.**

## Restoring anything

Everything is a plain move:

```bash
mv archive/checkpoints_2026-08-16/<run>       logs/rsl_rl/
mv archive/outputs_retired_2026-08-16/<dir>   outputs/
cp -r archive/ppo_rom_retired_2026-08-16/PPO_ROM/<path> PPO_ROM/<path>
```

The retired PPO_ROM files are also still recoverable from git history at the
commit preceding this cleanup.
