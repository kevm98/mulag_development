# Stage 1 — ROM deployment interface

Status: **complete and verified against code**.

The deployment plant is the command-conditioned hydraulic ROM followed by
PhysX. The selected checkpoint is
`rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt`.

```text
desired cylinder extension change + velocity
  -> ROM internal valve prediction/dead-zone
  -> pressure, velocity and extension dynamics at 20 Hz
  -> three held increments
  -> 60 Hz PhysX cylinder position target
```

The external command is desired cylinder motion, not a direct valve command.
This interface is the basis for every later collection and SD model.
