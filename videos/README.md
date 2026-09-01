# Rollout videos

`adaptive_rollout_mesh1337.mp4` — the adaptive arm (PPO-AAM-FD, seed 17,
`model_399`, `--residual_alpha 0.5`) driving the verge task on held-out mesh
1337 under the mid-episode change of load. Recorded on the IsaacLab VM,
1 September 2026, from `outputs/videos/demo_adaptive/mesh1337/`.

This is the arm the mechanism claim of Section 7.5 rests on: the one whose
capacity-matched control, identical but for a zeroed history window, does not
reproduce the gain.

`aam_fd/` — earlier rollouts, kept with their own note.
