# Switching-oracle matrix summary

Diagnostic only; this is not ROM-B certification evidence.

| target h | ROM-B RMSE mm | oracle RMSE mm | ROM-B acc95 | oracle acc95 | controlled acceleration interaction | positive rows | unique outcomes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 22.17 | 22.03 | 8.42 | 4.11 | +2.39 m/s² | 3/3 | 1 |
| 0.7 | 22.58 | 22.53 | 7.04 | 7.82 | -2.70 m/s² | 0/3 | 1 |
| 1.0 | 22.80 | 22.57 | 8.14 | 6.21 | +0.00 m/s² | 0/3 | 1 |
| 1.2 | 22.71 | 22.23 | 6.64 | 4.25 | +0.47 m/s² | 3/3 | 1 |

**Seed warning:** metrics were bit-identical across the requested seeds. The rows are deterministic repeats and do not establish repeated-seed evidence.
