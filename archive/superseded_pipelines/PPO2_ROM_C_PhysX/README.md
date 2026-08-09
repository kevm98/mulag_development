# PPO2-ROM_C — five-factor online adaptation of locked ROM-B

ROM-C is the adaptive continuation of ROM-B. It keeps the same five-factor
hydraulic plant family, inner DLS/cascade controller, deterministic parked-truck
contract, trajectories, four residual actions, rewards, and safety thresholds.
The only policy-interface change is five causal AAM estimates appended to the
locked 42-dimensional ROM-B observation.

```text
reachable arm reference
        ↓
unchanged DLS/cascade inner loop + PPO residual
        ↓
five-factor ROM-B hydraulic plant → PhysX
        ↑
e_comp and d_seal_1..4 sampled privately at reset

valve, extension, velocity, pressure, force history (10 × 20)
        ↓
online AAM → five estimates → widened PPO observation (42 + 5)
```

Flow and relief-pressure factors from the older seven-output AAM bundle are
discarded. They do not perturb ROM-B and cannot enter the ROM-C policy.

## Evidence gate before training

An estimator that predicts only the population mean is not online adaptation.
Before PPO fine-tuning, run:

```bash
PYTHONPATH=/home/rrlab/rrlab_mulag:/home/rrlab/rrlab_mulag/ROM-cylinders_pipeline \
  /home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_C_PhysX/scripts/audit_aam_rom_b_compatibility.py
```

For every supported factor, the audit requires:

- lower RMSE than a constant-mean predictor;
- positive held-out-style R²;
- positive correlation above 0.1 across individual ROM-B plants.

The current AAM bundle fails all five checks. Therefore `scripts/train.py`
blocks adaptive PPO training unless a passing compatibility report exists.
This prevents a long run from converting a cross-domain estimator bias into a
policy feature and calling it adaptation.

## Policy transfer and fine-tuning contract

- Source policy: locked factor-blind ROM-B checkpoint, not PPO2_A.
- Observations: 47 = unchanged ROM-B 42 + five normalized AAM estimates.
- Initialization: exact 42→47 zero-column expansion; the initial deterministic
  actor is identical when AAM inputs are neutral.
- Trainable actor parameters: only the five new input columns in the teacher and
  pitch branches.
- Frozen: accepted ROM-B mapping, actor normalizer, exploration standard
  deviation, downstream actor layers, inner controller, truck supervisor.
- Critic: trainable.
- Hidden plant: the exact same independent, reset-fixed five-factor ROM-B
  distribution used by the robust baseline.

This is an A-RMA-style final fine-tuning stage for estimator error. It is only
meaningful after (1) the true-factor oracle beats ROM-B and (2) the AAM passes
the ROM-B compatibility audit.

## Transfer-only smoke test

This checks construction and exact checkpoint widening without starting PPO:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_C_PhysX/scripts/train.py \
  --headless --device cuda:0 --num_envs 16 --transfer_only
```

Do not launch the full ROM-C training command with the current AAM checkpoint.
The compatibility report is intentionally failing.

## Required comparison after a future compatible AAM exists

Use identical held-out plants, shapes, and seeds for:

1. locked ROM-B;
2. ROM-C with AAM estimates;
3. the same ROM-C policy with estimates zeroed;
4. the same estimates shuffled across environments;
5. the true-factor oracle upper bound.

Claim online adaptation only if AAM-on beats zero and shuffle, improves the
predeclared arm metrics without weakening any safety gate, and stays below the
oracle. Truck transport remains deterministic and outside PPO in every arm
comparison.
