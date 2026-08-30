# The per-env arm instrument was aliased against terrain difficulty

**Status: bug found, fixed, revalidation running.**
**Consequence: every `--arms_per_env` number produced before this fix is void.**

## The bug

Arm assignment was

```python
modes = [arms[i % len(arms)] for i in range(raw.num_envs)]
```

while the terrain assigns, per environment index,

- height scale: `values[env_index % len(values)]`, `terrain_height_scales = (0.50, 0.70, 0.85, 1.00)` -> period **4**
- path family: `families[(env_index + offset) % 11]` -> period **11**

With **4 arms** the arm index and the height scale have the **same period**, so each
arm was locked to exactly one relief scale for the whole run. The arms were not
compared on the same task.

## How it showed up

A/A control, `--arms_per_env privileged,true,estimated,privileged`. On the
`privileged_e` route these three modes return the *same tensor* by construction,
so all groups must score identically.

| arm | locked relief | n | median lateral |
|---|---|---|---|
| privileged | 0.50 and 1.00 | 32 | 34.406 mm |
| true | 0.70 | 16 | 32.879 mm |
| estimated | 0.85 | 16 | 36.488 mm |

Ordered by relief, not by context. Repeating with `MULAG_SHARED_TERRAIN=1`
changed nothing (30.43 / 27.91 / 29.42 mm, spreads 1.0-2.5 mm) **because the
height scale never depended on the terrain seed** -- sharing the seed removes
chunk geometry, not difficulty.

This is the same magnitude as the sequential-instrument bias it was built to
replace (1.1-2.4 mm), and larger than the ~1 mm effects under test.

Earlier I read the wide unpaired CI ([-7.8, +4.7] mm) as the instrument passing
its A/A control. It was not passing; it was underpowered, and the bias was
sitting inside the interval.

## The fix

11 families and 4 scales are coprime, so a block of **44** covers every
(family, scale) pair exactly once:

```python
CYCLE = 44
modes = [arms[(i // CYCLE) % len(arms)] for i in range(raw.num_envs)]
```

`num_envs` must be a multiple of `44 * n_arms` (enforced, raises otherwise).
Verified offline: every arm receives all 44 (family, height) pairs exactly once.
With `MULAG_SHARED_TERRAIN=1` the seed is shared as well, so env `j` and env
`j+44` differ **only** by arm -- an exactly paired comparison.

## Gate

`outputs/rif_delay/p44_aa.json` (176 envs, 900 steps). The A/A must come back at
~0 with a tight paired CI. Until it does, no arm result may be read.
