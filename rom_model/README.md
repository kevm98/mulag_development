# rom_model — ROM-A command ROM

The actuator model for the **PPO_ROM** pipeline: cylinder command -> internal
valve -> hydraulics, learned from Simscape data and executed at 50 ms.

    ROM_pipeline/                                     model DEFINITIONS (imported)
    plant_reachable_v1_finetune_blend25/best_rom.pt    the deployed command ROM
    command_run_001/best_rom.pt                        PPO2 upper-level default
    base_run_001/best_base_rom.pt                      base ROM (PPO2 tasks)

`ROM_pipeline/` is not optional and is not a checkpoint. The runtime puts this
folder on `sys.path` and does `from ROM_pipeline.models import load_checkpoint`,
so the definitions must sit beside the weights -- exactly like `vendor_models/`
in `id_fd_model/`. Loaded by `mulag_plant/runtime/command_rom_runtime.py`
and `base_rom_runtime.py`.

Provenance: copied from `ROM-cylinders_pipeline/`, now under
`archive/model_training_trees/`. Only the validated artefacts were taken.
