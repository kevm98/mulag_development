# Time-varying payload, confirmed on held-out meshes

Payload steps 1.0 -> 5.0x mid-episode at 6 s +- 2 s of jitter, 176 environments,
900 steps. Meshes **3141 and 8191**, which were never used to design, diagnose or
read this campaign.

## Plant gate, frozen policy only, run before any adaptive number was read

| mesh | constant 1.0x | STEPPING | constant 5.0x | verdict |
|---|---:|---:|---:|---|
| 3141 | 5.05 mm | 84.14 mm | 399.17 mm | pass |
| 8191 | 4.41 mm | 127.98 mm | 509.76 mm | pass |

The frozen policy on the stepping plant sits far above its constant-1.0x value
and well below constant-5.0x, which is what a step landing at 4-8 s of a 15 s
episode should produce.

## Result

| mesh | arm | cut | lateral | contact | safety |
|---|---|---:|---:|---:|---:|
| 3141 | ctxzero | 84.14 mm | 160.16 mm | 0.766 | 0.133 |
| 3141 | **ctxest** | **9.38 mm** | **99.57 mm** | **0.881** | **0.036** |
| 8191 | ctxzero | 127.98 mm | 185.77 mm | 0.715 | 0.175 |
| 8191 | **ctxest** | **10.23 mm** | **99.67 mm** | **0.881** | **0.027** |

Paired `ctxzero -> ctxest`:

| mesh | cut | lateral | contact | outcome |
|---|---:|---:|---:|---|
| 3141 | +452.1 %* | +31.3 %* | +0.111* | **joint improvement** |
| 8191 | +758.4 %* | +53.5 %* | +0.161* | **joint improvement** |

Both channels improve with intervals excluding zero, on both reserved meshes.

## What this closes

The time-varying claim was previously selection-mesh only, which was listed as
an open gap. It is now confirmed where the campaign was not designed, and the
held-out effects are again LARGER than the selection mesh's (+636 % cut there
against +452 % and +758 % here on a comparable dose).

Combined with the constant-payload held-out result, the evidence is:

    constant heavy payload, held out     4/4 pass, 3 of them joint
    time-varying, held out               2/2 joint improvements
    safety and contact                   improved in 10/10 configurations

## What it does NOT change

The nominal-payload limitation stands: at constant 1.0x the arm loses lateral,
and four fixes have failed. The governing rule is unchanged -- the arm helps
where the frozen controller is losing ground contact and costs lateral where it
is not -- and the stepping plant qualifies because the frozen policy's contact
falls to 0.766 and 0.715 on it.

It also does not establish TRACKING. The freeze ablation and the periodic test
both found a stale estimate performs as well as a live one, so this is load
identification driving a conditioned policy, not a parameter tracker.
