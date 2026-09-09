# rom_model — ROM-A command ROM

The actuator model for the **PPO_ROM** pipeline: cylinder command -> internal
valve -> hydraulics, executed at 50 ms.

**Provenance is not "Simscape only".** The grey-box model was trained on the
Simscape command-ROM export, and the deployed checkpoint then received a
closed-loop calibration *in IsaacLab/PhysX*: 4,084 transitions rolled out on the
five plant-reachable shapes, with ONLY the bounded (+-3 mm) command-gated
`command_extension_correction_head` trainable and the Simscape dataset replayed
at weight 0.25 -- hence `blend25`. `command_run_001` is the pre-calibration
checkpoint and has no correction head at all. The correction is command-gated,
so it is exactly zero at hold and the stationary-hold contract survives.

This is also why ROM + PhysX is the **deployment plant** for the AAM/SD work
(`deployment_adaptation/README.md`): the ROM is the only actuator here that has
been fitted to the plant it runs in. The ID/FD pair is Simscape-identified end
to end and has had no equivalent calibration.

    ROM_pipeline/                                     model DEFINITIONS (imported)
    plant_reachable_v1_finetune_blend25/best_rom.pt    the deployed command ROM
    command_run_001/best_rom.pt                        PPO2 upper-level default
    base_run_001/best_base_rom.pt                      base ROM (optional; not present in this checkout)

`ROM_pipeline/` is not optional and is not a checkpoint. The runtime puts this
folder on `sys.path` and does `from ROM_pipeline.models import load_checkpoint`,
so the definitions must sit beside the weights -- exactly like `vendor_models/`
in `learned_models/id_fd_model/`. The deployed Stage 31 plant uses
`mulag_plant/runtime/command_rom_runtime.py`: its input is the 8-D desired
cylinder command `[delta_extension(4), delta_velocity(4)]` and its output
includes the internal valve command. It does not accept a 4-D valve vector as
its plant input. The later ID-FD pipeline has the opposite contract: ID emits a
4-D valve command and FD consumes that valve. `base_rom_runtime.py` is a
separate valve-input interface, but no valid `base_run_001/best_base_rom.pt`
checkpoint is available here, so it must not be selected for the ROM
collection.

For Stage 31, collect through the command-ROM interface and record
`rom_predicted_valve_cmd`. That recorded command is the supervised target for
the later ID/AAM fine-tuning stage; it is not fed back as a direct command to
the ROM during collection.

Provenance: copied from `ROM-cylinders_pipeline/`, now under
`archive/model_training_trees/`. Only the validated artefacts were taken.
