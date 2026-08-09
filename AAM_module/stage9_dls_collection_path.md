# Stage 9 — True DLS collection path

Status: **complete and verified**.

```text
task reference
  -> cylinder-space DLS teacher
  -> normalized 4-D action
  -> desired cylinder motion
  -> command ROM
  -> PhysX cylinder target
```

DLS closes the loop on measured cylinder state. It does not bypass the deployed
plant with direct joint targets. Identification segments place PRBS on the same
4-D action port and record it separately from DLS output.
