# id_fd_model — ID -> FD cylinder models

The actuator model for the **PPO_ID_FD** pipeline: two learned maps in series,
cylinder target -> valve (ID), then valve -> delta extension (FD).

    vendor_models/                  model DEFINITIONS (imported)
    best_id_cylinder_control.pt     ID: target -> valve
    best_fd_cylinder_strict.pt      FD: valve -> delta extension

**Use the STRICT FD, not the control FD.** `best_fd_cylinder_control` was rank ~2
and under-fit the valve; the strict file is the rank-promoted retrain and is the
validated partner for the control ID.

`vendor_models/` is not optional and is not a checkpoint. The runtime imports
definitions from `<root>/id_cylinder_space ` and `<root>/fd_cylinder_space ` --
**the trailing space in those two directory names is real**. Pointing
`cylinder_models_root` at a checkpoint directory instead gives
`ModuleNotFoundError: id_cylinder_common`.

`PPO_ID_FD` asserts both checkpoints exist at config construction: a missing file
would fall back to ROM-A and every ID/FD number would silently be a ROM number
wearing an ID/FD label.

Provenance: copied from `PPO1_ID_FD_PhysX/models/`, now under
`archive/model_training_trees/`.
