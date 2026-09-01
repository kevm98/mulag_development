# Rollout videos

Three views of the adaptive stack (PPO-AAM-FD, seed 17, `model_399`,
`--residual_alpha 0.5`) driving the verge task on held-out mesh 1337. Recorded
on the IsaacLab VM, 1-2 September 2026.

| file | view | trajectory | source |
|---|---|---|---|
| `adaptive_rollout_mesh1337_side_view.mp4` | side | - | `outputs/videos/demo_adaptive/mesh1337/` |
| `adaptive_rollout_mesh1337_front_view.mp4` | front | s-curve | `outputs/videos/demo_front_scurve/mesh1337/` |
| `adaptive_rollout_mesh1337_back_view.mp4` | back | meander | `outputs/videos/demo_back_meander/mesh1337/` |

These are separate runs of the same arm, not three cameras on one drive: the
source directory names carry the trajectory each was recorded on. Say so if any
of them is used where a single drive is implied.

This is the arm the mechanism claim of Section 7.5 rests on: the one whose
capacity-matched control, identical but for a zeroed history window, does not
reproduce the gain.

`aam_fd/` - earlier rollouts, kept with their own note.
