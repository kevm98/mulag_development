"""Actuator selection for the shared terrain task.

PPO3 and PPO4 are the SAME task, the same 66-D observation, the same 5-action
contract, the same terrain, the same compliant plant and the same gates. They
differ in exactly one thing: what converts a cylinder command into motion.

    rom   -> command ROM (ROM-A)                      [PPO3]
    idfd  -> ID(target->valve) -> FD(valve->delta x)  [PPO4]

Training PPO4 therefore does not get its own copy of the training scripts.
Copies drift -- one gets a fix, the other does not, and months later a
comparison between them is quietly measuring the divergence instead of the
actuator. Instead `scripts/train.py`, `scripts/distill_terrain.py`,
`scripts/evaluate.py` and `scripts/train_until_converged.py` all take
``--actuator`` and run the identical code path either way. "PPO4 trains exactly
like PPO3" is then a property of the code, not a promise in a README.

Checkpoints carry their actuator so a ROM-trained policy cannot be silently
evaluated on ID/FD or vice versa. The two are structurally interchangeable --
same observation width, same action width -- so nothing else would catch it.
"""

from __future__ import annotations

ACTUATORS = ("rom", "idfd")

#: Written into checkpoint ``infos`` and every report.
ACTUATOR_LABELS = {
    "rom": "command ROM (ROM-A)",
    "idfd": "ID -> FD",
}


def resolve_actuator(name: str):
    """Return ``(task_id, runner_cfg_cls, label)`` for an actuator name.

    Imports are deferred: PPO4 imports PPO3, so a module-level import here
    would be circular.
    """
    key = str(name).lower()
    if key == "rom":
        # PPO3_1, not PPO3: verge_task remains the shared library (terrain,
        # observations, teacher, training scripts) but the PIPELINE is PPO3_1.
        import PPO_ROM as package
        from PPO_ROM.env_cfg import Ppo31PPORunnerCfg

        return package.TERRAIN_TASK_ID, Ppo31PPORunnerCfg, ACTUATOR_LABELS[key]
    if key == "idfd":
        # PPO4_1, not PPO4: PPO4_1 is the current ID/FD pipeline (PPO3_1's
        # feasibility-derived terrain, rebuilt reward and eight-gate contract).
        # PPO4 is the superseded package and lives under archive/.
        import PPO_ID_FD as package
        from PPO_ID_FD.env_cfg import Ppo41IDFDPPORunnerCfg

        return package.TERRAIN_TASK_ID, Ppo41IDFDPPORunnerCfg, ACTUATOR_LABELS[key]
    raise ValueError(f"--actuator must be one of {ACTUATORS}, got {name!r}")


def assert_checkpoint_actuator(infos, actuator: str, *, context: str) -> None:
    """Refuse a checkpoint trained on a different actuator.

    Checkpoints written before this field existed carry no actuator. Those are
    ROM checkpoints by construction -- PPO4 did not exist -- so they are
    accepted as 'rom' and rejected against 'idfd' rather than waved through,
    which would be the silent-mislabel failure this whole mechanism prevents.
    """
    stored = str((infos or {}).get("actuator", "") or "rom").lower()
    requested = str(actuator).lower()
    if stored != requested:
        raise RuntimeError(
            f"{context}: checkpoint was produced on actuator {stored!r} but the "
            f"environment is running {requested!r}. The observation and action "
            "contracts are identical between them, so nothing else would catch "
            "this -- the numbers would describe the wrong plant."
        )


__all__ = [
    "ACTUATORS",
    "ACTUATOR_LABELS",
    "resolve_actuator",
    "assert_checkpoint_actuator",
]
