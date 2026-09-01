# E3 — method R2 (lr 1e-4): abandoned screen

Pre-registration `reports/PREREG_R2_2026-09-01.md`; abandoned under its
2026-09-01 06:45 amendment, which fixed the go/no-go screen in advance.

**This is an abandoned screen, not a five-seed table.** One mesh (1337), n=176
per arm, scored against `base_R1`. It is reported here so the R2 runs are on the
record; it is never presented as a result table.

## What R2 changed

One thing: `--lr 3e-4 -> 1e-4`. Aimed at the R1 failure mode — seed 7 peaked at
14729 mean reward and collapsed to 6957 by the final iteration, which is the
checkpoint the protocol scores.

## Screen result

| seed | cut | lateral | R1 cut (n=528) |
|---|---|---|---|
| 17  | **−15.0%*** | +59.1%* | +63.7% |
| 101 | −2.5% | **−63.7%*** | +12.5% |
| 23  | +53.6%* | +58.7%* | +39.6% |
| 42  | **−40.2%*** | **−10.9%*** | −23.1% |
| 7   | +14.0% | +8.1%* | −86.9% |

`*` = bootstrap CI on the paired difference excludes zero.

**1 of 5 improves both (seed 23), against 2 of 5 for R1 at full n.** R2 is worse
than the method it was meant to repair, and is abandoned.

## What it taught, which is worth keeping

**The learning rate trades seeds against each other rather than stabilising
them.** The change rescued exactly the seed it was designed for — seed 7,
−86.9% → +14.0% — and destroyed the best seed in the campaign, 17, from +63.7%
to −15.0% with the CI excluding zero.

So the R1 diagnosis was wrong. The seed variance is not an optimisation-stability
problem that a smaller step fixes; a smaller step just moves which seeds fail.
That makes learning rate a dead lever in both directions and rules it out for
any later method version — R2 is evidence against its own hypothesis, which is
worth more than another inconclusive sweep.

Training-curve shape agreed in advance: end/peak reward fell for every completed
seed under R2 (17: 1.00 → 0.63, 101: 0.80 → 0.03, 23: 0.77 → 0.43), and the
declared risk in the R2 pre-registration — that a 3x lower lr at unchanged
budget would under-train — is what happened.

## Cost

5 training runs (~5 h wall) + 5 evals (~26 min). The screen saved the other 10
evals (~52 min) that a full R2 table would have cost.
