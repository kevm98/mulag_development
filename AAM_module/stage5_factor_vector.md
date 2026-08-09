# Stage 5 — Final physical factor vector

Status: **complete**.

```text
e = [flow, compliance, seal, pressure, mass_scale, com_z]
```

Final Stage-22 ranges:

| Factor | Range |
|---|---:|
| flow | `[-0.25, 0.25]` |
| compliance | `[-0.25, 0.10]` |
| seal | `[0, 1]` |
| pressure | `[-0.15, 0.15]` |
| mass scale | `[0.8, 1.5]` |
| COM-z | `[-0.10, 0.05] m` |

Hydraulic endpoints are one-factor anchors. Combined draws are constrained to
avoid excessive compounded response gain.
