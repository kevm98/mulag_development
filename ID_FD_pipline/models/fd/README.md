# Forward-dynamics model

Active checkpoint: `best_fd_cylinder_rank4.pt`.

It was fine-tuned from the original HDF5 MATLAB data in
`FD_cylinder_data.mat` with MSE plus a valve-Jacobian rank objective. The
resulting valve Jacobian is rank 4/4 with condition 2.29; validation MSE also
improved by 7.6% over the original checkpoint.

This model is locked. `scripts/finetune_fd_rank.py` is retained only for
provenance and reproducibility.
