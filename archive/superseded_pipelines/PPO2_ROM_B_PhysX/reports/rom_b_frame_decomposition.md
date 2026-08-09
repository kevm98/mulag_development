# Frame decomposition probe — result

Probe: `diagnose.py --num_envs 128 --eval_steps 120 --seed 17 --suite monte_carlo
--rounds 1 --trajectory_shapes line`, checkpoint `model_53`.
Result `term = 0.05469`, 7 failures — **reproduces the audit row exactly**, so this
is the audit's own round-1 line condition, not a different experiment.

Decomposition, as specified:

```
clearance_error = cutter_world_z - reference_world_z
                = (measured_cutter_truck_z - reference_cutter_truck_z)          <- tracking
                + (truck_base_world_z + reference_cutter_truck_z - reference_world_z)  <- reference/frame
                + (cutter_world_z - cutter_pos_w_z)                             <- cutter definition
                + closure                                                       <- truck tilt / residual
```

`cutter_pos_w_z` is the orientation-aware cutter world Z
(`_compute_cutter_pos_w`), logged explicitly rather than assumed negligible.

---

## 1. Per-failed-environment table (at termination)

| env | cause | path_s | clearance | tracking | reference/frame | cutter-def | tilt | truck_base_z |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 107 | clearance | 0.1461 | +0.2084 | +0.0570 | +1.7374 | −1.5796 | −0.0063 | 0.2562 |
| 20 | truck_raw_reaction | 0.9461 | +0.0810 | −0.0223 | +1.7042 | −1.5795 | −0.0214 | 0.2583 |
| 21 | clearance | 0.0137 | +0.2005 | +0.0227 | +1.7579 | −1.5800 | −0.0001 | 0.2517 |
| 2 | clearance | 0.0141 | +0.2038 | +0.0273 | +1.7566 | −1.5800 | −0.0001 | 0.2517 |
| 84 | clearance | 0.0130 | +0.2037 | +0.0237 | +1.7601 | −1.5800 | −0.0001 | 0.2517 |
| 30 | clearance | 0.0171 | +0.2013 | +0.0201 | +1.7477 | −1.5799 | +0.0134 | 0.2523 |
| 111 | clearance | 0.0480 | +0.2155 | +0.0589 | +1.7433 | −1.5800 | −0.0068 | 0.2533 |
| **mean** | | | **+0.1877** | **+0.0268** | **+1.7439** | **−1.5799** | **−0.0031** | 0.2533 |

Over all 84 pre-termination samples (12-step window x 7 replicas):

| term | mean | sd | min | max |
|---|---:|---:|---:|---:|
| clearance | +0.1527 | 0.0375 | +0.0719 | +0.2155 |
| tracking | −0.0015 | 0.0285 | −0.0831 | +0.0589 |
| reference / frame | +1.7366 | 0.0158 | +1.6977 | +1.7601 |
| cutter definition | −1.5798 | 0.0003 | −1.5800 | −1.5789 |
| tilt / closure | −0.0025 | 0.0079 | −0.0282 | +0.0194 |

## 2. The level of the clearance reading is a frame artifact

Two terms of order 1.6–1.7 m nearly cancel:

```
reference / frame   +1.7439
cutter definition   -1.5799
                    -------
residual            +0.1640   <- 87% of the 0.1877 m the gate reads
tracking            +0.0268   <- the only physically meaningful part
tilt / closure      -0.0031
```

**The gate's zero point is wrong by about 0.164 m.** That consumes 82 % of the
0.200 m limit before the arm has done anything, leaving roughly **0.036 m of
budget** for real tracking error.

### 2a. Where each 1.6 m term comes from

**Cutter definition, −1.5799 m, sd 0.0003.** The gate uses
`_compute_cutter_world_z = ee_origin_z − 0.79`, ignoring orientation. The
orientation-aware `_compute_cutter_pos_w` places the cutter 1.58 m higher. Working
backwards: `quat_apply(q, [0,0,−0.79])_z = +0.79`, i.e. in the head-down pose the
EE body Z axis points **downward in world**, so an offset authored as −0.79 in body
coordinates resolves to +0.79 in world. One of the two is wrong about which way the
cutter hangs. Numerically the orientation-aware value is the self-consistent one:
`truck_base_world_z + measured_cutter_truck_z = 0.2533 + 1.577 = 1.830 =
cutter_pos_w_z` to within the closure term. The gate's value, 0.25 m, is not
reachable from the truck-frame pipeline.

**Reference / frame, +1.7439 m.** `truck_base_world_z + cutter_truck[2] −
cutter_world_z = 0.2533 + 1.55 − 0.06`. The library's own two reference columns
describe points 1.74 m apart under this environment's spawn. Consistent with the
per-shape library deltas measured offline (line mean −1.4536, i.e. authored as if
the truck base sat near world −1.48 rather than +0.25).

I cannot determine offline which of the two cutter computations is physically
correct — that needs the actual cutter prim's world position or the ground plane
queried in sim. **The conclusion does not depend on resolving it:** whichever is
right, the gate's numerator and its reference are expressed in frames that differ
by ~0.164 m after cancellation, and that is what the gate reports as "clearance".

## 3. The rise that triggers termination is mostly real

Change over the 12-step approach, per replica:

| env | Δclearance | Δtracking | Δreference/frame | Δtruck_base_z | Δlibrary ref |
|---:|---:|---:|---:|---:|---:|
| 107 | +0.0472 | +0.0453 | −0.0163 | −0.0129 | +0.0034 |
| 20 | −0.0366 | −0.0245 | +0.0065 | +0.0062 | −0.0004 |
| 21 | +0.0378 | +0.0205 | +0.0173 | +0.0000 | −0.0173 |
| 2 | +0.0415 | +0.0274 | +0.0141 | +0.0000 | −0.0141 |
| 84 | +0.0385 | +0.0189 | +0.0196 | −0.0000 | −0.0196 |
| 30 | +0.0506 | +0.0190 | +0.0002 | −0.0035 | −0.0037 |
| 111 | +0.0571 | +0.0538 | +0.0034 | +0.0000 | −0.0033 |
| **mean** | **+0.0337** | **+0.0229** | **+0.0064** | −0.0015 | −0.0079 |

**68 % of the rise is genuine truck-frame tracking error** — the cutter really does
drift up 2.3 cm in the truck frame while its commanded height is constant at
1.55 m. 19 % is the reference/frame term, of which the library reference falling
away as `path_s → 0` (−0.0079) is the larger part and suspension motion
(−0.0015) the smaller.

So your third branch is the one that holds: **both contribute, to different
things.** The reference/frame term sets the offset; the tracking term supplies the
motion that crosses the limit.

### 3a. What this implies

With the 0.164 m frame offset removed, the clearance reading in these replicas
would be roughly 0.027 m rising to 0.05 m against a 0.200 m limit. **None of the
seven would terminate.** The dominant failure mode in the ROM-B acceptance suite —
line clearance, 96 of 130 h-sweep line failures and 20 of 22 Monte Carlo line
failures — is being produced by a frame bookkeeping error, not by the plant.

The underlying physical signal is real but small: 2–6 cm of upward drift at the
path endpoint under degraded hydraulics, which is where reduced `e_flow` and
increased seal leakage would be expected to bite. That is a legitimate ROM-B
robustness signal and it is currently invisible, buried under an offset five times
its size.

## 4. Corrections to my earlier statements

* **"They must differ by a constant" was too strong.** You were right. Measured:
  `truck_base_world_z` mean 0.2533, sd 0.00286, range **0.0178 m** — the truck does
  move vertically under suspension loading. It is a real contributor, just a small
  one (−0.0015 m of the +0.0337 m rise).
* **The 4.1 cm library notch is now second-order.** It is real and it sits in the
  failure band, but it is 4 cm against a 16 cm frame offset. Per your caution: do
  not smooth it. It should be re-derived once the frame question is settled, at
  which point it may not even be anomalous.
* **The cutter-definition residual is not negligible** — logging it rather than
  assuming it was the right call. At −1.58 m it is the single largest term in the
  decomposition.

## 5. Recommended conclusion

> ROM-B training completed. Whether domain randomization succeeded is
> **unresolved**, because the principal clearance acceptance signal is dominated by
> a ~0.164 m frame inconsistency between the gate's cutter computation and the
> trajectory library's world-Z reference. The genuine physical signal underneath it
> is 2–6 cm of endpoint drift, well inside the 0.200 m limit. `model_53`'s
> acceptance, the h-sweep verdicts, and any `model_607` comparison must all be
> re-read against a corrected reference.

## 6. Ordering from here

1. **Establish ground truth for the cutter's world Z** — query the cutter prim's
   world position and the ground plane in sim, and compare against both
   computations. Do not repair either until this is measured.
2. **Reconcile the two library reference columns** against that ground truth and
   this environment's spawn configuration. Re-derive `cutter_world_z`; do not
   patch the notch in isolation.
3. **Re-run the acceptance suite** for `model_53` and `model_607` on the corrected
   reference. Only then is the control arm interpretable.
4. **Then** decide on remediation. On current evidence the line clearance mode
   largely disappears, which would change the entire remediation target.
5. Independently, quantify the probe's position dependence
   (`rom_b_probe_reproducibility.md`) — same seed, 7 to 13 failures depending on
   sequence position. That affects both the seed-group statistics and the
   reproduction of individual failures.

No training was started. `model_53` untouched. 27 offline tests pass.
