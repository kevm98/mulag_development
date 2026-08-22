# E5 — Robustness degradation curve

**Defends:** *Hydraulic Manipulator* · **Covers:** all three stacks
**Status:** not started

## Design

Sweep randomisation scale at 0x, 0.5x, 1x, 1.5x, 2x the nominal factor ranges
and report a **degradation curve**, not a pass/fail verdict. A breakdown point is
a result; a green tick is not. Separate interpolation from extrapolation.

## Traps

- This project has repeatedly produced **vacuous gates** — thresholds no run
  could violate, yielding passing verdicts that carry no information. Every gate
  here needs a **measured floor** and a **positive control** before any failure
  is believed.
- Earlier ROM-B "robustness failures" were 4 vacuous gates plus worst-of-N
  binomial noise, not real degradation. Check gate-versus-termination-threshold
  coincidence first.
- Sequential audits in one process have fabricated failures; reproduce any
  failure in a fresh process before believing it.
