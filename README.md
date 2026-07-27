# RRLAB MULAG hydraulic policy workflows

The repository is organized around two chosen nominal V3 policies and their future
robust/adaptive variants.

## Workflow order and naming

| Order | Workflow | Hydraulic backend | Meaning | Status |
|---:|---|---|---|---|
| 1 | **PPO1_A** | ID → FD → PhysX | Nominal models | Current V3; selected `model_575.pt` |
| 2 | **PPO1_B** | Randomized ID → FD → PhysX | Randomized models | Future |
| 3 | **PPO2_A** | Command ROM → PhysX | Nominal ROM | Current V3; selected workflow |
| 4 | **PPO2_B** | Randomized command ROM → PhysX | Factor-blind robust PPO | Implemented; termination-certified, circle acceleration limitation documented |
| 5 | **PPO2_C** | Command ROM + online AAM | Nominal policy fine-tuned with AAM latent variables | Implemented; promotion pending |
| 6 | **PPO2_D** | Command ROM + diagnostic AAM | Upper-bound diagnostic | Future; not deployment |

“A” always means nominal model parameters. “B” means randomized model parameters.
PPO2_C is the nominal PPO2 policy fine-tuned for online adaptation with AAM latents.
PPO2_D is a privileged/upper-bound diagnostic and must not be reported as the deployed
controller.

The physical source directories retain their established names for import and checkpoint
compatibility:

```text
PPO1_A -> PPO1_ID_FD_PhysX/
PPO2_A -> PPO2_ROM_PhysX/
```

Both implementations now use the public package name `policy/`; the completed PPO1_A
training was allowed to finish before this package migration.

## 1. PPO1_A — nominal ID-FD V3

Train:

```bash
cd /home/rrlab/rrlab_mulag
./PPO1_ID_FD_PhysX/train_v3.sh
```

Visualize a selected checkpoint after training:

```bash
./PPO1_ID_FD_PhysX/visualize_v3.sh
```

See [PPO1_A README](PPO1_ID_FD_PhysX/README.md).

## 2. PPO2_A — nominal ROM V3

Train:

```bash
cd /home/rrlab/rrlab_mulag
./PPO2_ROM_PhysX/train_large_v3.sh
```

Visualize the selected smooth checkpoint:

```bash
./PPO2_ROM_PhysX/visualize.sh
```

See [PPO2_A README](PPO2_ROM_PhysX/README.md) and the
[hydraulic ROM README](ROM-cylinders_pipeline/README.md).

## 3. PPO2_B — robust PPO over randomized ROM plants

ROM-B keeps the PPO observation factor-blind and samples five hidden hydraulic
factors independently at reset. The factors remain fixed within each episode.
The truck motion is outside PPO; the GUI player uses a fixed-chassis fixture so
only the arm tracks the requested trajectory.

See [PPO2_B README](PPO2_ROM_B_PhysX/README.md) for training, isolated
evaluation, current evidence, and visualization commands.

## 4. PPO2_C — ROM V3 with online AAM

Fine-tune PPO2_A using only causal AAM estimates (no privileged factors):

```bash
cd /home/rrlab/rrlab_mulag
./PPO2_ROM_C_PhysX/train.sh
```

See [PPO2_C README](PPO2_ROM_C_PhysX/README.md) for the 49-observation
contract, exact checkpoint provenance, visualization command, and promotion
criteria.

## Shared references

- [Universal Policy](Universal_Policy/README.md): exact observations, actions,
  rewards, plant loop, and A/B/C/D taxonomy.
- [DLS solver](DLS_solver/README.md): outer tool and inner cylinder controller.
- `extensions/`: shared Isaac Lab robot assets and task support.
- `teacher_student/`: retained research branch for later comparison.

## What `mdp/` means

MDP means **Markov Decision Process**. The `mdp/` directories contain internal Isaac
Lab task functions such as action sanitization and development-task observation/reward
helpers. They are not the deployed PPO Policy. The maintained implementations are under
`PPO1_ID_FD_PhysX/policy/` and `PPO2_ROM_PhysX/policy/`.

TensorBoard logs are stored under `logs/rsl_rl/`.
