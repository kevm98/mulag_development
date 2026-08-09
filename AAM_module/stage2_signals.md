# Stage 2 — Observable and privileged signals

Status: **complete**.

- Observable: cylinder extension/velocity, joint position/velocity, task pose,
  controller command and recent command/state history.
- Recorded diagnostics: pressure difference, predicted valve, ROM increments,
  saturation flags and timing.
- Privileged: the six physical factors and their encoded latent `z`.
- Final deployment mirror: `dP[4]` is withheld from the SD input, while causal
  history and joint context remain available.

Do not confuse the physical factor vector, the privileged latent, the ROM
hidden state and the online AAM estimate.
