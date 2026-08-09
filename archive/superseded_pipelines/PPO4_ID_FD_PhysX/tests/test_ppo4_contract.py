"""Contract: PPO4 is PPO3's plant with ONLY the actuator swapped.

The value of PPO4 is comparability. If anything physical drifts between the two
packages, a PPO4-vs-PPO3 difference stops being attributable to ID/FD and the
experiment is worthless. These checks are structural and import-free so they run
without Isaac Sim, alongside PPO3's own contracts.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PKG = ROOT / "PPO4_ID_FD_PhysX"
_CFG_SOURCE = (PKG / "terrain_env_cfg.py").read_text()
_ENV_SOURCE = (PKG / "terrain_env.py").read_text()


def test_every_ppo4_module_parses():
    """Syntax guard: a broken docstring must not surface hours later in training."""
    sources = sorted(
        list(PKG.glob("*.py"))
        + list((PKG / "scripts").glob("*.py"))
        + list((PKG / "tests").glob("*.py"))
    )
    assert sources, "no PPO4 sources discovered"
    broken = []
    for path in sources:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as error:
            broken.append(f"{path.relative_to(ROOT)}:{error.lineno}: {error.msg}")
    assert not broken, "unparseable PPO4 sources:\n  " + "\n  ".join(broken)


def test_ppo4_inherits_ppo3_rather_than_copying_the_plant():
    """Subclassing is what keeps the plants identical as PPO3 changes.

    A copied config would drift the moment someone tunes PPO3's compliance or
    terrain, and the drift would be invisible until a comparison went wrong.
    """
    assert "from PPO3_SOLE_PhysX.terrain_env_cfg import" in _CFG_SOURCE
    assert "class TerrainIDFDEnvCfg(TerrainSolePolicyEnvCfg)" in _CFG_SOURCE
    assert "from PPO3_SOLE_PhysX.terrain_env import TerrainSolePolicyEnv" in _ENV_SOURCE
    assert "class TerrainIDFDEnv(TerrainSolePolicyEnv)" in _ENV_SOURCE


def test_ppo4_inherits_the_adaptive_trajectory_without_overrides():
    for field in (
        "terrain_tracking_frequency_scale",
        "terrain_tracking_frequency_jitter",
        "terrain_composite_trajectory",
        "terrain_composite_families",
        "terrain_tracking_active_length_m",
    ):
        assert field not in _CFG_SOURCE, f"PPO4 must inherit shared field {field}"


def test_ppo4_corrects_measured_head_polarity_and_restores_in_xyz_nullspace():
    ppo3_cfg = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "terrain_c4_action_to_axis_y_sign: float = -1.0" in ppo3_cfg
    assert "terrain_head_nullspace_restore_gain: float = 1.0" in ppo3_cfg
    assert "terrain_head_nullspace_restore_deadband_rad: float = 0.01" in ppo3_cfg
    assert "terrain_head_nullspace_restore_max_action: float = 0.60" in ppo3_cfg
    assert "terrain_head_restore_preserve_xyz: bool = True" in ppo3_cfg
    assert "terrain_head_nullspace_restore_gain =" not in _CFG_SOURCE


def test_ppo4_does_not_redefine_plant_or_task_physics():
    """Only the actuator may differ. Anything else breaks attribution."""
    forbidden = (
        "terrain_arm_stiffness_scale",
        "terrain_head_joint_stiffness_scale",
        "terrain_compliant_contact_stiffness",
        "truck_spawn_z",
        "mow_strip_resolution_m",
        "target_cut_height_m",
        "terrain_contact_reward_weight",
        "sole_action_filter",
        "max_depenetration_velocity",
        "sole_head_joint_action",
        "sole_head_action_to_c4_gain",
        "terrain_model_error_observation",
        "terrain_head_feedforward_from_measured",
        "terrain_head_penalty_far_weight",
        "terrain_head_shield_soft_alignment",
        "terrain_head_shield_hard_alignment",
        "terrain_c4_action_to_axis_y_sign",
        "terrain_head_nullspace_restore_gain",
        "terrain_head_nullspace_restore_deadband_rad",
        "terrain_head_nullspace_restore_max_action",
        "terrain_head_restore_preserve_xyz",
        "terrain_messerkopf_level_gain",
        # Contact safety is a TASK definition, not an actuator property. If
        # PPO4 set its own acquisition hysteresis, gap tolerance or contact
        # reward weights, "PPO4 lifts more than PPO3" would stop being a
        # statement about ID/FD and become a statement about two different
        # definitions of lifting.
        "terrain_contact_acquire_steps",
        "terrain_contact_loss_grace_steps",
        "terrain_max_airborne_gap_m",
        "terrain_takeoff_velocity_threshold_mps",
        "terrain_hard_impact_force_n",
        "terrain_impact_ejection_force_n",
        "terrain_contact_safety_reward",
        "terrain_contact_support_weight",
        "terrain_contact_loss_weight",
        "terrain_contact_force_target_n",
        "terrain_contact_force_std_n",
        "terrain_contact_force_soft_limit_n",
        "terrain_contact_gap_weight",
        "terrain_contact_takeoff_weight",
        "terrain_contact_observation",
        "terrain_terminate_on_contact_loss",
        "terrain_terminate_on_hard_impact",
        "terrain_lift_terminal_reward",
    )
    # Mentioning a name in prose is fine; assigning it is not.
    tree = ast.parse(_CFG_SOURCE)
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
                elif isinstance(target, ast.Attribute):
                    assigned.add(target.attr)
    leaked = sorted(set(forbidden) & assigned)
    assert not leaked, (
        f"PPO4 re-defines plant/task physics {leaked}; it must inherit these "
        "from PPO3 so an ID/FD-vs-ROM comparison stays attributable"
    )


def test_ppo4_selects_the_idfd_actuator_and_disables_rom():
    assert 'cylinder_dynamics_model = "fd"' in _CFG_SOURCE
    assert "use_cylinder_id_pipeline = True" in _CFG_SOURCE
    assert 'rom_cylinder_checkpoint = "disabled"' in _CFG_SOURCE
    assert "best_id_cylinder_control.pt" in _CFG_SOURCE
    # The STRICT FD is the rank-promoted retrain; the earlier FD was rank ~2 and
    # under-fit the valve, so naming the wrong file silently restores that bug.
    assert "best_fd_cylinder_strict.pt" in _CFG_SOURCE


def test_ppo4_refuses_to_run_over_rom_a():
    """A silent ROM fallback would report PPO3 behaviour under a PPO4 label."""
    assert 'if dynamics != "fd"' in _ENV_SOURCE
    assert "raise ValueError" in _ENV_SOURCE
    # Verified against the constructed runtime, not just the requested config.
    assert '_cylinder_id' in _ENV_SOURCE
    assert "raise RuntimeError" in _ENV_SOURCE


def test_ppo4_fails_loudly_when_checkpoints_are_missing():
    assert "FileNotFoundError" in _CFG_SOURCE, (
        "a missing ID/FD checkpoint must stop the run, not fall back"
    )


def test_ppo4_registers_its_own_task_ids():
    init_source = (PKG / "__init__.py").read_text()
    assert "RRLAB-Mulag-PPO4-Verge-IDFD-v0" in init_source
    assert "PPO4_ID_FD_PhysX.terrain_env:TerrainIDFDEnv" in init_source

    # Must not silently reuse PPO3's task ids or entry points, or reports would
    # be mislabelled. Checked on the string VALUES in the module, not on prose:
    # the docstring legitimately mentions PPO3.
    tree = ast.parse(init_source)
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    routed_to_ppo3 = [
        text
        for text in literals
        if text.startswith("RRLAB-") and "PPO4" not in text
    ] + [
        text
        for text in literals
        if ":" in text and text.startswith("PPO3_SOLE_PhysX")
    ]
    assert not routed_to_ppo3, (
        f"PPO4 registration references PPO3 ids/entry points: {routed_to_ppo3}"
    )


def test_teacher_probe_gates_before_training():
    """The teacher probe must check actuator authority, not just tracking.

    The FD model was once rank ~2: two cylinders barely responded while tracking
    metrics still looked plausible. Authority is therefore measured explicitly
    and first.
    """
    source = (PKG / "scripts" / "probe_teacher.py").read_text()
    assert "bidirectional_authority" in source
    assert "all_cylinders_bidirectional" in source
    assert "authority_ok" in source
    assert '"passed"' in source
    # It must be able to produce the ROM-A reference on identical terrain.
    assert '"ppo3"' in source


def test_ppo2_and_ppo3_remain_rom_only():
    """Widening the actuator guard must not loosen PPO2 or PPO3.

    PPO4 runs the PPO2 task class over ID/FD by widening
    ``_supported_dynamics_models``. If that attribute's default ever stops being
    command_rom, PPO2 and PPO3 would silently accept an actuator they were never
    validated against.
    """
    source = (ROOT / "PPO2_ROM_PhysX" / "tasks" / "cylinder_env.py").read_text()
    assert '_supported_dynamics_models: frozenset = frozenset({"command_rom"})' in source, (
        "PPO2's task must default to command_rom only"
    )
    # The ID/FD residual switches are a misconfiguration under a ROM and the
    # validated contract under ID/FD, so that check must be scoped, not global.
    assert "if dynamics in self._rom_dynamics_models:" in source

    ppo3_cfg = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "_supported_dynamics_models" not in ppo3_cfg
    ppo3_env = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env.py").read_text()
    assert "_supported_dynamics_models" not in ppo3_env


def test_ppo4_widens_the_guard_to_fd_only():
    assert '_supported_dynamics_models = frozenset({"fd"})' in _ENV_SOURCE, (
        "PPO4 must widen the actuator guard to fd, and to nothing else"
    )


# ----------------------------------------------------------------------
# "PPO4 trains exactly like PPO3" must be a property of the code
# ----------------------------------------------------------------------

_SCRIPTS = ROOT / "PPO3_SOLE_PhysX" / "scripts"


def test_ppo4_has_no_training_scripts_of_its_own():
    """Copies drift. PPO4 must reuse PPO3's scripts via --actuator.

    If PPO4 ever grows its own train/distill/evaluate, a fix applied to one
    package silently stops applying to the other, and a later PPO4-vs-PPO3
    comparison measures that divergence instead of the actuator.
    """
    own = {path.name for path in (PKG / "scripts").glob("*.py")}
    forbidden = {"train.py", "distill_terrain.py", "evaluate.py", "train_until_converged.py"}
    assert not (own & forbidden), (
        f"PPO4 must not carry its own copies of {sorted(own & forbidden)}; "
        "run PPO3's scripts with --actuator idfd instead"
    )


def test_training_scripts_take_the_actuator_switch():
    for name in ("train.py", "distill_terrain.py", "evaluate.py", "train_until_converged.py"):
        source = (_SCRIPTS / name).read_text()
        assert '"--actuator"' in source, f"{name} has no --actuator switch"
        assert '"rom"' in source and '"idfd"' in source, f"{name} lacks both actuators"
        assert 'default="rom"' in source, (
            f"{name} must default to rom so PPO3's behaviour is unchanged"
        )


def test_supervisor_forwards_the_actuator_to_both_children():
    """A block and its evaluation must never disagree about the plant."""
    source = (_SCRIPTS / "train_until_converged.py").read_text()
    train_cmd = source[source.index("scripts/train.py"):]
    train_cmd = train_cmd[: train_cmd.index("return ")]
    assert "--actuator" in train_cmd, "training blocks do not receive --actuator"
    eval_cmd = source[source.index("scripts/evaluate.py"):]
    eval_cmd = eval_cmd[: eval_cmd.index("return ")]
    assert "--actuator" in eval_cmd, "held-out evaluation does not receive --actuator"


def test_checkpoints_record_and_enforce_their_actuator():
    """ROM and ID/FD checkpoints are structurally interchangeable.

    Same observation width, same action width -- so without an explicit tag a
    ROM-trained policy would load and evaluate on ID/FD without complaint, and
    the report would describe the wrong plant.
    """
    for name in ("train.py", "distill_terrain.py"):
        assert '"actuator"' in (_SCRIPTS / name).read_text(), (
            f"{name} does not tag checkpoints with their actuator"
        )
    for name in ("train.py", "evaluate.py"):
        assert "assert_checkpoint_actuator" in (_SCRIPTS / name).read_text(), (
            f"{name} does not verify the checkpoint's actuator against the env"
        )
    resolver = (ROOT / "PPO3_SOLE_PhysX" / "actuators.py").read_text()
    # Untagged checkpoints predate PPO4 and are ROM by construction; they must
    # be treated as rom, not waved through against any actuator.
    assert 'or "rom"' in resolver


def test_startup_banner_reports_the_actual_actuator():
    """A banner that always says ROM-A makes an ID/FD log look like a ROM run."""
    source = (ROOT / "PPO3_SOLE_PhysX" / "env.py").read_text()
    assert "plant=ROM-A command ROM" not in source, (
        "the startup banner must not hard-code the actuator"
    )
    assert 'cylinder_dynamics_model' in source
    assert '"ID -> FD"' in source


# ----------------------------------------------------------------------
# No ROM-A anywhere in the ID/FD pipeline
# ----------------------------------------------------------------------


def test_ppo4_config_disables_every_rom_checkpoint():
    """PPO4 must name no ROM model, not merely decline to run one.

    PPO3's config points at ROM-A. Inheriting that path leaves a reference to a
    model this package must never use, which later reads as "PPO4 uses ROM-A
    too". The parent's __post_init__ also re-asserts it, so disabling has to
    happen in both the class body and __post_init__.
    """
    for field in (
        "rom_cylinder_checkpoint",
        "command_rom_checkpoint",
        "base_rom_checkpoint",
    ):
        assert f'{field} = "disabled"' in _CFG_SOURCE, (
            f"PPO4 must disable {field} in the class body"
        )
        assert f'self.{field} = "disabled"' in _CFG_SOURCE, (
            f"PPO4 must re-disable {field} in __post_init__; the parent "
            "re-asserts the ROM-A path after the class body is evaluated"
        )
    assert "rom_reset_warmup_steps = 0" in _CFG_SOURCE
    assert "command_rom_direction_guard = False" in _CFG_SOURCE


def test_ppo4_names_no_rom_model_artifact():
    """No ROM checkpoint file may be referenced from the ID/FD package."""
    for path in sorted(PKG.rglob("*.py")):
        # This file has to spell the artifact names in order to look for them.
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text()
        for artifact in ("best_rom.pt", "best_base_rom.pt", "ROM-cylinders_pipeline"):
            assert artifact not in text, (
                f"{path.relative_to(ROOT)} references the ROM artifact {artifact!r}"
            )


def test_ppo4_refuses_to_construct_any_rom_runtime():
    assert "_command_cylinder_rom" in _ENV_SOURCE
    assert "_base_cylinder_rom" in _ENV_SOURCE
    assert "must contain no ROM model" in _ENV_SOURCE


def test_teacher_preflight_is_calibrated_for_the_compliant_plant():
    """The preflight must not demand student performance from the teacher.

    Measured on the compliant plant, the old thresholds failed BOTH actuators
    (ID/FD contact 0.7989, ROM-A 0.8884, against a 0.90 bar), so they blocked
    distillation for a reason unrelated to either actuator.
    """
    source = (ROOT / "PPO3_SOLE_PhysX" / "scripts" / "distill_terrain.py").read_text()
    assert 'report["ground_contact_fraction"] >= 0.70' in source
    assert 'report["p01_head_alignment"] >= 0.75' in source
    assert 'report["min_head_alignment"] >= 0.65' in source
    # Still catches a genuinely broken teacher.
    assert 'report["all_finite"]' in source
    assert 'report["slew_limit_respected"]' in source
    assert 'not report["direct_target_bypass_enabled"]' in source


def test_readiness_check_validates_the_viewer_path_not_the_training_path():
    """The PLAY/visual config is not exercised by training and can rot silently."""
    source = (PKG / "scripts" / "check_ready.py").read_text()
    assert "TERRAIN_PLAY_TASK_ID" in source
    assert "enable_road_scenery = True" in source
    assert "enable_visual_grid_floor = True" in source
    # Must prove the ID/FD plant, not just that something ran.
    assert "assert_checkpoint_actuator" in source
    assert "_command_cylinder_rom" in source
    # Must prove motion, not merely absence of crashes.
    assert "cutter is moving" in source


def test_every_train_checkpoint_save_is_actuator_tagged():
    """An untagged checkpoint reads as 'rom' and the evaluator refuses it.

    That is not a theoretical risk: it killed an ID/FD run after block 1,
    because one of train.py's three save paths omitted the tag.
    """
    source = (_SCRIPTS / "train.py").read_text()
    infos_blocks = source.count('"schema": "terrain_shared_v2"')
    tagged = source.count('"actuator": str(a.actuator)')
    assert tagged >= infos_blocks, (
        f"{infos_blocks} checkpoint-info blocks but only {tagged} actuator tags"
    )


def test_the_target_lead_bound_is_shared_by_both_actuators():
    """`cylinder_target_max_lead_m` is not an ID/FD-only field.

    It is read once, in ``mulag_upper_level_env._run_cylinder_id_pipeline``,
    on the path BOTH actuators take, and the pre-fix probe measured it
    saturated on both (ROM-A p95 0.0262 / max 0.0300; ID/FD p95 0.0265 /
    max 0.0406 against the inherited 0.025 cap). Treating it as a PPO4 knob
    would leave PPO3 storing the same command error and releasing it into the
    ground.
    """
    runtime = (
        ROOT / "PPO2_ROM_PhysX" / "runtime" / "mulag_upper_level_env.py"
    ).read_text()
    consumption = runtime[runtime.index('getattr(self.cfg, "cylinder_target_max_lead_m"'):]
    consumption = consumption[: consumption.index("self._cylinder_pos_target =")]
    assert "dynamics_model" not in consumption, (
        "the lead clamp is applied unconditionally, so it governs ROM-A too"
    )
    train = (ROOT / "PPO3_SOLE_PhysX" / "scripts" / "train.py").read_text()
    assert '"--cylinder_target_max_lead_m"' in train, (
        "PPO3's training entry point must be able to sweep the shared bound"
    )


def test_ppo4_closes_the_head_feedforward_on_the_measured_boom():
    """A learned actuator does not reach its commanded q4.

    The head is slaved to the boom by joint6_coupling_a * q4. Taking q4 from the
    COMMAND assumes the actuator arrives; ID/FD does not, and the residual
    becomes a standing head-alignment offset applied downstream of the policy
    every substep. That is the signature observed: p01 head alignment sat at
    0.79 across ten PPO blocks while contact, cut error and acceleration all
    improved.
    """
    ppo3_cfg = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "terrain_head_feedforward_from_measured: bool = True" in ppo3_cfg
    assert "terrain_head_feedforward_from_measured =" not in _CFG_SOURCE
