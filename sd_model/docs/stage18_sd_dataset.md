# Stage 18 — System Dynamics dataset

Status: **complete**.

At each 50 ms ROM tick:

- state: `[extension(4), velocity(4), dP(4)]`;
- target: next-tick 12-D state delta;
- context: joint position/velocity and desired cylinder command;
- history: six causal ROM ticks, contained within one run.

Joint context is never predicted. The final derived samples live under
`datasets/master_v3_stage22_wide_frozen/sd/`.
