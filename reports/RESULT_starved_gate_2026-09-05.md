# Gate result: base sensitivity to constant cutter mass

Run 2026-09-05 on the GCP L4 as Phase 0 of the campaign pre-registered in
`PREREG_STARVED_2026-09-03.md`. The gate asks whether the frozen base actually
degrades with cutter mass. If a fixed policy is insensitive to the draw, the
encoder has nothing to identify and the campaign is not worth paying for.

## Design

Frozen base `logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt`, no
residual, scored on held-out mesh 2718 over 176 environments and 900 steps.
Cutter mass held constant per run at three levels via
`MULAG_PAYLOAD_RANGE=x,x`. No mid-episode flip.

## Result

| payload | cut-height median | lateral median |
|---|---:|---:|
| 1.0x | 3.353 mm | 46.122 mm |
| 3.0x | 19.387 mm | 101.699 mm |
| 5.0x | 320.314 mm | 410.447 mm |

The gate passes, and by a wide margin. Cut-height error grows 95x from 1.0 to
5.0 and lateral error 8.9x. At 5.0 the frozen policy is effectively failing the
task rather than tracking it imperfectly.

## What this changes

The 2026-09-02 campaign ran on the manifest's own 0.8-1.5 mass range, which
`PPO_AAM_FD/env_cfg.py` already records as costing a fixed policy nothing
measurable. Section 7.4.3 of the thesis reads the resulting null partly as a
power problem: the robust baseline left only a small deficit for adaptation to
recover, so extra residual capacity and live plant information could produce
similar gains. That explanation does not apply here. Over 1.0-5.0 the deficit is
large and grows monotonically, so this campaign tests the history question on a
plant where there is something substantial to identify.

The campaign uses `MULAG_PAYLOAD_RANGE=1.0,5.0` with
`MULAG_PAYLOAD_RESAMPLE=1`, a fresh constant draw per episode.

## Caveat

The 5.0 end may be beyond what any bounded residual at alpha = 0.5 can repair.
A failure of the adaptive arm at that end is not by itself evidence about
history, and the per-dose breakdown should be read before the pooled result.
