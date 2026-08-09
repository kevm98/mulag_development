# Stage 17 — Learning timebase

Status: **complete**.

- Physics: 120 Hz.
- Environment control and recording: 60 Hz.
- Hydraulic ROM and learning transition: 20 Hz (`0.05 s`).

Samples use exact control rows `0,3,6,...`: state at row `t` and command at
`t` predict state at `t+3`. No interpolation is used. Grid validation occurs
before derived data is built.
