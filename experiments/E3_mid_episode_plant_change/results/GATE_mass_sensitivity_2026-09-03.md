# Gate: is the frozen base sensitive to cutter mass?

Run 2026-09-03 on `isaaclab-l4`. Pre-registration:
`reports/PREREG_STARVED_2026-09-03.md`, "Gate before paying for five seeds".

## Why this had to be measured first

The starved-encoder campaign only makes sense if the plant variation it asks the
encoder to identify is one a fixed policy cannot already absorb. If the frozen
base is insensitive to cutter mass, there is nothing for live history to
recover and the campaign is not worth its GPU time.

## Design

Frozen base only (`phase1_init_70.pt`, no residual). Constant cutter mass held
for the whole episode, one value per run, `MULAG_PAYLOAD_RANGE=x,x`. No
mid-episode flip. Mesh 1337, 176 environments, 900 steps, evaluation seed 17.
Artefacts in `outputs/precheck_mass/`.

## Result: the base degrades strongly and monotonically

| cutter mass | lateral median (mm) | vs 1.0 | cut-height median (mm) | vs 1.0 |
|---|---|---|---|---|
| 1.0 | 46.31 | — | 5.027 | — |
| 1.5 | 52.47 | +13.3% | 6.861 | +36.5% |
| 2.0 | 60.09 | +29.8% | 8.023 | +59.6% |
| 2.5 | 76.50 | +65.2% | 14.275 | +184.0% |
| 3.0 | 113.88 | +145.9% | 29.247 | +481.8% |
| 4.0 | 396.37 | +755.8% | 277.061 | +5411.8% |
| 5.0 | 550.78 | +1089.2% | 444.725 | +8747.3% |

Both channels rise at every step of the ladder. The gate passes: this plant
axis is one the frozen policy demonstrably cannot absorb.

## Range chosen for the campaign: 1.0 to 3.0

Over 1.0-3.0 the base loses a factor of 2.5 in lateral error and 5.8 in
cut height, which is a wide, clean signal for an encoder to identify, while the
policy still tracks the task and a residual has something to recover.

At 4.0 and 5.0 the base has effectively failed: 396 mm and 551 mm of lateral
error against a 46 mm nominal, with cut-height error larger than the errors the
task is scored on. A range extending that far would spend most of its mass draws
in a regime where no bounded residual could recover the task, which would test
saturation rather than adaptation.

## What this does and does not establish

It establishes that cutter mass is a control-relevant, observable-in-principle
plant axis over 1.0-3.0. It says nothing about whether live history recovers it:
that is the campaign this gate authorises, and its primary comparison is the
capacity-matched control rather than this frozen base.
