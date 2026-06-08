# Locked ROM Split Report

- Dataset: `/home/rrlab/rrlab_mulag/kevin_integration/rom_cylinders/pytorch_hydraulic_rom_dataset.mat`
- Seed: `42`
- Original validation used as locked test: `True`
- Complete-run splitting verified: `True`

| Split | Source Split | Runs | Samples |
|---|---|---:|---:|
| train | train | 842 | 362902 |
| val | train | 94 | 40514 |
| test | val | 104 | 44824 |

## Overlap Checks

- Run overlap count: `0`

## Sequence Boundary Checks

- `train` violations: `0`
- `val` violations: `0`
- `test` violations: `0`

The locked `test` split is the original validation set and should not be used for iterative model development.
