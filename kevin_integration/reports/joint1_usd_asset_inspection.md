# Joint 1 USD / Articulation Inspection

Status: inspection scaffolding is complete, but the final USD/PXR and Isaac runtime answers are pending until the scripts are run through `./rrlab.sh` in an environment with Isaac Lab and PXR available. This report records what can be proven statically from the repository and what the runtime scripts now print.

## Static Findings

1. Exact USD file path used:
   - Configured in `extensions/rrlab_assets/rrlab_assets/mulag.py` as:
     `${RRLAB_ASSETS_DATA_DIR}/Robots/Unimog_Mulag_FME500/unimog_mulag_fme500_cylinder_as_actuator.usd`
   - The repository contains a binary USD crate at:
     `extensions/rrlab_assets/data/Robots/unimog_mulag_fme500_cylinder_as_actuator.usd`
   - Runtime inspection must confirm how `RRLAB_ASSETS_DATA_DIR` resolves, because the configured path includes `Robots/Unimog_Mulag_FME500/`.
2. Robot cfg file/class used:
   - `extensions/rrlab_assets/rrlab_assets/mulag.py`
   - `MULAG_CFG = ArticulationCfg(...)`
   - Kevin env uses `robot_cfg: ArticulationCfg = MULAG_CFG.replace(prim_path="/World/envs/env_.*/Robot")`.
3. Active actuator config:
   - `actuators={"all_joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None, effort_limit=None, velocity_limit=None)}`
   - Per-arm actuator configs with explicit large effort limits are present only as commented code and are not active.
4. Kevin expected joint order:
   - State/debug joints: `Drehzapfen_joint`, `Ausleger_I_joint`, `Ausleger_II_joint`, `Messerkopf_Schwenk_joint`, `Messerkopf_joint`
   - Commanded joints: `Drehzapfen_joint`, `Ausleger_I_joint`, `Ausleger_II_joint`, `Messerkopf_Schwenk_joint`
5. Kevin normal effort path:
   - `MulagAdaptiveRLEnv._apply_action()` calls:
     `self.robot.set_joint_effort_target(self._current_torque, joint_ids=self._command_joint_ids)`
   - `_command_joint_ids` are resolved from `command_joint_names` at env creation.
6. Direct effort test behavior:
   - `test_direct_joint_effort_authority.py` now resolves effort targets by joint name.
   - It prints resolved IDs/names, effort command shape, effort limits, velocity limits, and whether `Drehzapfen_joint` is included.
   - It replaces the Kevin env `_pre_physics_step` and `_apply_action` methods during the diagnostic, then uses `env.step()` with zero actions. This proves the normal learned action pipeline cannot overwrite the direct effort command inside that test.

## Required Runtime Answers

| Question | Current answer | How to confirm |
| --- | --- | --- |
| Does `Drehzapfen_joint` exist in the USD? | Pending runtime USD inspection. | `inspect_mulag_usd_asset.py` |
| What is its USD joint type? | Pending. | `inspect_mulag_usd_asset.py` |
| Is it revolute or fixed? | Pending. | `inspect_mulag_usd_asset.py` |
| Lower/upper limits? | Pending. | `inspect_mulag_usd_asset.py` |
| Axis? | Pending. | `inspect_mulag_usd_asset.py` |
| Parent and child bodies? | Pending. | `inspect_mulag_usd_asset.py` |
| Drive stiffness/damping/maxForce? | Pending. Active Isaac actuator cfg has `None` for these values, so USD-authored values matter. | `inspect_mulag_usd_asset.py` and `inspect_mulag_joint_mapping.py` |
| Is maxForce zero or very small? | Pending. | `inspect_mulag_usd_asset.py` |
| Mimic/coupling constraints involving `Drehzapfen_joint`? | Pending. | `inspect_mulag_usd_asset.py` |
| Similar names that could confuse mapping? | Pending. | `inspect_mulag_usd_asset.py` |
| Runtime Isaac index of `Drehzapfen_joint`? | Pending. | `inspect_mulag_joint_mapping.py` |
| Kevin action index mapped to `Drehzapfen_joint`? | Expected action index `0`; runtime Isaac ID pending. | `inspect_mulag_joint_mapping.py` |
| Is direct effort definitely applied to `Drehzapfen_joint`? | The test now prints the proof path, but runtime confirmation is pending. | `test_direct_joint_effort_authority.py --single_joint_name Drehzapfen_joint` |
| Can direct effort be overwritten by env control? | In the normal env, yes, `_apply_action()` sets effort every step. In the hardened direct test, no: the test replaces `_pre_physics_step` and `_apply_action` before stepping. | Direct test prints override call counts. |

## Current Classification

Classification: Case 5 pending runtime USD/Isaac evidence.

Reason: the repository confirms that direct effort must be applied by name-resolved joint IDs and that the normal env action path would overwrite manual effort unless bypassed. The hardened direct effort test now bypasses that path. However, this shell cannot yet prove whether the USD joint is locked, whether its drive/maxForce prevents motion, or whether runtime Isaac resolves a different joint index.

Most likely next classifications after runtime:

- Case 1 if `Drehzapfen_joint` is fixed, locked, constrained, or excluded from the articulation.
- Case 2 if `Drehzapfen_joint` is free but resolves to an unexpected action/index mapping.
- Case 3 if a future direct test is run without the override and effort is overwritten by `_apply_action()`.
- Case 4 if effort is applied to the correct revolute joint and not overwritten, but drive/maxForce/actuator settings prevent motion.
- Case 5 if all mapping, limits, drive, and override checks look correct but joint 1 still does not move.

## Recommended Next Fix

Do not apply a fix yet. First run:

```bash
./rrlab.sh -p kevin_integration/scripts/inspect_mulag_usd_asset.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --headless

./rrlab.sh -p kevin_integration/scripts/inspect_mulag_joint_mapping.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 1 \
  --headless

./rrlab.sh -p kevin_integration/scripts/test_direct_joint_effort_authority.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 1 \
  --steps_per_test 100 \
  --settle_steps 20 \
  --joint_names Drehzapfen_joint,Ausleger_I_joint,Ausleger_II_joint,Messerkopf_Schwenk_joint \
  --single_joint_name Drehzapfen_joint \
  --effort_values 5,10,20,50 \
  --headless
```

Then choose the fix from the classification above. Do not train, tune reward, tune torque scale, or change SD/FD/ID until this physical/mapping question is resolved.
