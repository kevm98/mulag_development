# Learned Model Checkpoints

| Model | Path | Sign Accuracy | Trained On |
|-------|------|---------------|------------|
| Fine-tuned ID | `outputs/.../id_physx_finetuned_best.pt` | 97.2% | Valve-space PID teacher |
| Original FD (sequence) | `Models_cylinder /fd_cylinder_space /.../best_fd_cylinder_sequence_motion_v1.pt` | 88% | Dynamic cylinder data |
| Original ID (sequence) | `Models_cylinder /id_cylinder_space /.../best_id_cylinder_sequence_delta_clean_v1.pt` | N/A (positional) | Training dataset |
