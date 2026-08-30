# Step 1 naming proposal — 2026-08-10

This is a proposal only. No folders, task IDs, checkpoints, or historical
report paths were renamed.

## Canonical experiment labels

| Human label | Existing folder | Proposed experiment key | Runtime actuator metadata |
|---|---|---|---|
| PPO-ROM | `PPO_ROM/` | `ppo_rom` | `rom` |
| PPO-ID-FD | `PPO_ID_FD/` | `ppo_id_fd` | `idfd` |
| PPO-ID-FD + integrator | `PPO_ID_FD/` | `ppo_id_fd_integrator` | `idfd_integrator` |
| PPO-ID(AAM)-FD | `PPO_AAM_FD/` | `ppo_id_aam_fd` | `id_aam_fd` |

The integrator is a runtime correction arm over the frozen ID-FD pipeline, not
a fourth implementation folder. Its manifest should carry an explicit
`integrator_enabled: true` field and a separate experiment key.

The current registered task IDs that must remain resolvable are
`RRLAB-Mulag-Verge-ROM-v0`, `RRLAB-Mulag-Verge-IDFD-v0`, and
`RRLAB-Mulag-Verge-ID-AAM-FD-v0`, together with their `-PLAY-v0` viewer
variants. The proposal changes display labels and manifest fields, not these
registry strings.

## Proposed metadata fields

Every final paired-evaluation manifest should record:

```json
{
  "arm": "PPO-ID(AAM)-FD",
  "experiment_key": "ppo_id_aam_fd",
  "folder": "PPO_AAM_FD",
  "actuator": "id_aam_fd",
  "integrator_enabled": false,
  "policy_checkpoint_sha256": "...",
  "id_checkpoint_sha256": "...",
  "fd_checkpoint_sha256": "...",
  "aam_checkpoint_sha256": "...",
  "factor_manifest_sha256": "..."
}
```

Display labels may be improved immediately in reports and manifests without
moving any files. New documentation should use the four labels above and avoid
older package-era labels.

## Migration cost and recommendation

Renaming is not recommended now. The cost is material and the benefit is only
cosmetic:

- Task IDs are registered strings used by launchers, viewers, evaluation
  scripts, checkpoint metadata, and historical reports. A rename requires
  compatibility aliases and a complete import/preflight sweep.
- Checkpoints carry actuator metadata. A folder or task-ID migration would
  require revalidating every checkpoint's actuator tag, observation width, and
  associated manifest.
- Stored report paths and shell visualizers currently resolve through the
  existing folders. Moving them would make historical reports and rerun
  commands non-resolvable unless redirect files were added.
- Script references, experiment names, log directories, and task-registration
  strings would all need a synchronized migration; a partial rename would be
  worse than the current naming.

Keep the existing filesystem and task IDs, add the canonical display labels and
manifest fields above, and use the current folders as compatibility roots. A
future migration is worthwhile only if a clean repository-wide alias plan is
prepared and all historical paths remain resolvable.
