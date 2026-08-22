# E2 — Plant characterisation: transport delay and identifiability

**Defends:** *Hydraulic Manipulator* · **Covers:** all three stacks (plant property)
**Status:** delay complete; nonlinearity outstanding; thesis §7.6, §7.2

## Run

    python experiments/E2/code/transport_delay.py \
        --raw-root AAM_module/datasets/rom_openloop_v1 \
        --out outputs/stage31_openloop/transport_delay.json
    python experiments/E2/code/excitation_drift.py --raw-root ... --out ...
    python experiments/E2/code/online_refinement_probe.py --model-dir ... --root ... --raw-root ...

## Result — transport delay

Command to differential pressure is **117 ms (7 control steps at 60 Hz)**, stable
across three of four amplitude tiers. That is 2.3 model ticks at the 20 Hz
estimator rate and roughly a tenth of the AAM's 1.0 s window.

**Positive control passes exactly**: cylinder extension drives joint angle
through a static linkage, so the command-to-extension and command-to-joint
onsets must coincide — they agree to **0.0 steps** at every amplitude.

## Method notes

- Onset detection, **not** cross-correlation. A correlation peak conflates dead
  time with rise time and produced a "delay" falling 317 -> 117 ms as amplitude
  grew, which is a detection-threshold artefact.
- The figure annotates the **median per-edge onset**, not the onset of the mean
  trace; the latter reads earlier because averaging edges of differing delay
  smears the leading edge forward. Figure and table therefore agree.
- **Separate two claims**: the valve's dynamic *model* is not identifiable in
  either dataset; its transport *latency* is. Do not use the first to dismiss
  the second.

## Result — identifiability envelope

Plant identity is decodable only at low excitation: seal R^2 = 0.703 at
amplitude 0.05, 0.505 at 0.15, and gone (<= 0.05) at 0.35 and 0.70.

Cause is **operating-point drift, not saturation** — command saturation reaches
only 10% even at 0.70. Runs desynchronise in position from 1.5 mm to 54.2 mm
across the episode, half a probe stroke, so a one-second window cannot separate
"different because leaky" from "different because sitting 5 cm lower".

**TRAP:** amplitude is confounded with episode time in `rom_openloop_v1` — the
plan ramps 0.05/0.15/0.35/0.70, one tier per quarter. **No within-episode trend
may be read from this collection.** A successor collection must interleave the
tiers and re-home between them.
