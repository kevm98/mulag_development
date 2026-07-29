"""Contract: no classical controller is in the scored control path.

The whole claim of PPO3 is that the EE Jacobian cascade and the cylinder DLS
controller have been removed from the loop.  Both objects still exist -- the
parent class builds them and reset-time deployment uses them -- so "we deleted
the inner loop" is exactly the kind of statement that quietly stops being true
after a refactor and that no tracking metric would ever catch.

These checks are structural and run without Isaac Sim, so they execute in the
same pytest pass as everything else rather than only inside a GPU session.  A
runtime version that steps the real environment is in ``scripts/evaluate.py``
under ``--control-path-audit``.
"""

from __future__ import annotations


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ast
import torch

from PPO3_SOLE_PhysX.observations import finite_difference_kinematics


_ENV_SOURCE = (ROOT / "PPO3_SOLE_PhysX" / "env.py").read_text()
_TREE = ast.parse(_ENV_SOURCE)
_TERRAIN_ENV_SOURCE = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env.py").read_text()
_TERRAIN_TREE = ast.parse(_TERRAIN_ENV_SOURCE)

_FORBIDDEN_IN_CONTROL_PATH = ("_cascade", "_cylinder_dls")


def _method(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"SolePolicyEnv.{name} not found in PPO3_SOLE_PhysX/env.py")


def _terrain_method(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TERRAIN_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"TerrainSolePolicyEnv.{name} not found in PPO3_SOLE_PhysX/terrain_env.py"
    )


def _attributes_used(node: ast.AST) -> set[str]:
    return {
        child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
    }


_TEACHER_SYMBOLS = ("_teacher_action", "_compute_teacher_action", "_teacher_execute_mask")


def test_pre_physics_step_touches_no_classical_controller():
    used = _attributes_used(_method("_pre_physics_step"))
    leaked = sorted(set(_FORBIDDEN_IN_CONTROL_PATH) & used)
    assert not leaked, (
        f"the scored control path references {leaked}; PPO3 must reach the ROM "
        "without the cascade or the cylinder DLS"
    )


def test_every_teacher_reference_is_behind_the_teacher_mode_guard():
    """The distillation teacher must be unreachable when the mode is 'off'.

    The teacher is allowed to drive during distillation -- that is the whole
    point of the phase -- so ordering alone is not the invariant here. The
    invariant is the guard: with sole_teacher_mode='off' no teacher statement
    executes at all. Without this check the teacher would be exactly the
    loophole that lets a cascade-driven number be reported as a sole-controller
    result.
    """
    node = _method("_pre_physics_step")

    guarded: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.If) and "_teacher_mode" in _attributes_used(child.test):
            for inner in ast.walk(child):
                guarded.add(id(inner))

    unguarded = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Attribute)
            and child.attr in _TEACHER_SYMBOLS
            and id(child) not in guarded
        ):
            unguarded.append((child.attr, child.lineno))

    assert not unguarded, (
        f"teacher symbols {unguarded} are reachable in _pre_physics_step without "
        "a sole_teacher_mode guard"
    )


def test_teacher_cannot_influence_the_command_before_it_is_final():
    """No teacher statement may precede the assignment of _current_action."""
    node = _method("_pre_physics_step")
    final_command_line = max(
        (
            target.lineno
            for stmt in ast.walk(node)
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Attribute) and target.attr == "_current_action"
        ),
        default=None,
    )
    assert final_command_line is not None, "_current_action is never assigned"

    early = [
        (child.attr, child.lineno)
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and child.attr in _TEACHER_SYMBOLS
        and child.lineno < final_command_line
    ]
    assert not early, (
        f"teacher symbols {early} appear before _current_action is assigned at "
        f"line {final_command_line}; the teacher must never shape the student's "
        "own command"
    )


def test_teacher_mode_defaults_to_off_and_evaluation_enforces_it():
    cfg_source = (ROOT / "PPO3_SOLE_PhysX" / "env_cfg.py").read_text()
    assert 'sole_teacher_mode: str = "off"' in cfg_source, (
        "the distillation teacher must be off by default"
    )
    eval_source = (ROOT / "PPO3_SOLE_PhysX" / "scripts" / "evaluate.py").read_text()
    assert "assert_teacher_disabled" in eval_source, (
        "evaluation must refuse to report a number produced with the teacher on"
    )


def test_ppo_and_evaluation_instantiate_the_terrain_task():
    """The A/B task must not silently train against the legacy stored path."""
    for script in ("train.py", "evaluate.py"):
        source = (ROOT / "PPO3_SOLE_PhysX" / "scripts" / script).read_text()
        assert "PPO3_SOLE_PhysX.TERRAIN_TASK_ID" in source, (
            f"{script} does not instantiate the terrain-following gym task"
        )
        assert "verge_observation_dim() + 1" in source, (
            f"{script} does not use the terrain observation contract"
        )


def test_terrain_task_never_randomizes_the_legacy_path_start_phase():
    """Terrain diversity must not change the arm's reset deployment."""
    cfg_source = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "randomize_start_phase: bool = False" in cfg_source
    for script in ("train.py", "evaluate.py"):
        source = (ROOT / "PPO3_SOLE_PhysX" / "scripts" / script).read_text()
        assert "randomize_start_phase = False" in source, (
            f"{script} does not explicitly enforce the fixed terrain deployment"
        )


def test_terrain_deployment_uses_measured_target_and_canonical_rom_a():
    cfg_source = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "terrain_deployment_cylinder_target_m" in cfg_source
    assert "plant_reachable_v1_finetune_blend25/best_rom.pt" in cfg_source
    assert "plant_reachable_trajectory_library_v3_large.json" in cfg_source
    terrain_source = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env.py").read_text()
    assert "_install_terrain_deployment_target()" in terrain_source
    assert "self._seeds[0, :] = target" in terrain_source
    assert "self._seeds[-1, :] = target" in terrain_source


def test_terrain_head_contract_has_margin_lock_and_strict_evaluation():
    cfg_source = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "terrain_head_threshold: float = 0.995" in cfg_source
    assert "terrain_head_penalty_max_deviation: float = 2.0" in cfg_source
    assert "head_down_termination_alignment: float = 0.98" in cfg_source
    assert "sole_action_authority: tuple = (1.0, 1.0, 1.0, 1.0)" in cfg_source
    assert "terrain_head_tilt_shield_soft: float = 0.08" in cfg_source
    assert "terrain_head_tilt_shield_hard: float = 0.16" in cfg_source
    eval_source = (ROOT / "PPO3_SOLE_PhysX" / "scripts" / "evaluate.py").read_text()
    assert "env_cfg.defer_safety_terminations = True" in eval_source
    assert 'GATE_SPECS["head_alignment_hard_min"].threshold' in eval_source
    assert "raw.terrain_safety_violation()" in eval_source


def test_terrain_reward_forwards_configured_smoothing_and_safety_scales():
    cfg_source = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "terrain_acceleration_max_ratio: float = 10.0" in cfg_source

    reward_source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_get_rewards")
    )
    for field in (
        "residual_action_magnitude_weight",
        "terrain_action_magnitude_weight",
        "sole_rate_penalty_weight",
        "sole_limit_penalty_weight",
        "sole_limit_margin_m",
        "sole_acceleration_weight",
        "sole_acceleration_scale_mps2",
        "terrain_acceleration_max_ratio",
    ):
        assert field in reward_source

    for contract in (
        "terrain_clearance_barrier_weight: float = 0.0",
        "terrain_clearance_margin_m: float = 0.0",
        "terrain_contact_reward_weight: float = 2.0",
        "dense_lateral_error_weight: float = 3.0",
        "sole_acceleration_weight: float = -0.50",
        "sole_rate_penalty_weight: float = -0.30",
        "terrain_action_filter: float = 0.70",
        "terrain_action_magnitude_weight: float = -1.0",
        "terrain_cut_progress_weight: float = 1.0",
        "terrain_lateral_progress_weight: float = 0.5",
        "terrain_head_progress_weight: float = 0.25",
        "terrain_arm_damping_scale: float = 1.0",
    ):
        assert contract in cfg_source


def test_terrain_observation_updates_live_kinematics_and_reset_relatches_last():
    obs_source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_get_observations")
    )
    assert "_update_measured_tool_kinematics()" in obs_source
    assert "head_down_axis_xy=head_axis_world[:, :2]" in obs_source

    shield_source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_task_action_shield")
    )
    assert "directional_tilt_shield" in shield_source

    reset_source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_reset_idx")
    )
    sync_at = reset_source.index("_synchronize_deployed_robot_state(env_ids)")
    anchor_at = reset_source.index("_drive_anchor_base_xy")
    relatch_at = reset_source.index("_relatch_terrain_temporal_state(env_ids)")
    assert sync_at < anchor_at < relatch_at

    reward_source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_get_rewards")
    )
    update_at = reward_source.index("_update_measured_tool_kinematics()")
    acceleration_at = reward_source.index(
        "torch.linalg.norm(self._measured_tool_acceleration_truck"
    )
    assert update_at < acceleration_at

    relatch_source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_relatch_terrain_temporal_state")
    )
    assert "_tool_kinematics_common_step" in relatch_source
    assert "common_step_counter" in relatch_source
    assert "_previous_abs_cut_error" in relatch_source
    assert "_previous_abs_lateral_error" in relatch_source

    progress_at = reward_source.index("cut_progress =")
    memory_update_at = reward_source.index("self._previous_abs_cut_error = torch.where")
    assert progress_at < memory_update_at
    assert "valid_active" in reward_source[memory_update_at:]
    assert "self._prev_head_alignment =" not in reward_source


def test_tool_kinematics_refresh_is_idempotent_within_common_step():
    """Reward and observation must not finite-difference the same pose twice."""
    method_node = _method("_update_measured_tool_kinematics")
    module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "finite_difference_kinematics": finite_difference_kinematics,
    }
    exec(compile(module, filename="<sole-kinematics-method>", mode="exec"), namespace)
    refresh = namespace["_update_measured_tool_kinematics"]

    class DummyEnv:
        step_dt = 0.1
        common_step_counter = 7
        _tool_kinematics_common_step = None

        def __init__(self):
            self.position = torch.tensor([[0.1, 0.0, 0.0]])
            self._prev_tool_pos_truck = torch.zeros_like(self.position)
            self._measured_tool_velocity_truck = torch.zeros_like(self.position)
            self._previous_measured_tool_velocity_truck = torch.zeros_like(
                self.position
            )
            self._measured_tool_acceleration_truck = torch.zeros_like(self.position)

        def _compute_ee_pos_local(self):
            return self.position.clone()

    env = DummyEnv()
    first = refresh(env)
    assert torch.equal(first, torch.tensor([[0.1, 0.0, 0.0]]))
    assert torch.allclose(
        env._measured_tool_velocity_truck, torch.tensor([[1.0, 0.0, 0.0]])
    )
    assert torch.allclose(
        env._measured_tool_acceleration_truck, torch.tensor([[10.0, 0.0, 0.0]])
    )

    env.position[:] = torch.tensor([[0.2, 0.0, 0.0]])
    second = refresh(env)
    assert torch.equal(second, torch.tensor([[0.2, 0.0, 0.0]]))
    # Same common step: return the live pose but preserve the one real sample.
    assert torch.allclose(
        env._measured_tool_velocity_truck, torch.tensor([[1.0, 0.0, 0.0]])
    )
    assert torch.allclose(
        env._measured_tool_acceleration_truck, torch.tensor([[10.0, 0.0, 0.0]])
    )
    assert torch.equal(env._prev_tool_pos_truck, first)

    env.common_step_counter += 1
    refresh(env)
    assert torch.allclose(
        env._measured_tool_velocity_truck, torch.tensor([[1.0, 0.0, 0.0]])
    )
    assert torch.equal(
        env._measured_tool_acceleration_truck,
        torch.zeros_like(env._measured_tool_acceleration_truck),
    )


def test_obsolete_stored_path_distillation_entrypoint_is_removed():
    assert not (ROOT / "PPO3_SOLE_PhysX" / "scripts" / "distill.py").exists()
    assert (ROOT / "PPO3_SOLE_PhysX" / "scripts" / "distill_terrain.py").is_file()


def test_terrain_training_accepts_only_versioned_teacher_student_checkpoints():
    source = (ROOT / "PPO3_SOLE_PhysX" / "scripts" / "train.py").read_text()
    assert 'schema != "terrain_teacher_student_v1"' in source
    assert '"PPO3 terrain teacher distillation v1"' in source
    assert "All pre-correction" in source


def test_terrain_distillation_uses_terrain_task_and_exact_raw_labels():
    source = (
        ROOT / "PPO3_SOLE_PhysX" / "scripts" / "distill_terrain.py"
    ).read_text()
    assert "PPO3_SOLE_PhysX.TERRAIN_TASK_ID" in source
    assert "raw.terrain_teacher_raw_action" in source
    assert 'cfg.sole_teacher_mode = "execute"' in source
    assert "ReplayBuffer" in source


def test_every_vectorized_terrain_entrypoint_resizes_gpu_physx_buffers():
    for script in ("distill_terrain.py", "train.py", "evaluate.py"):
        source = (ROOT / "PPO3_SOLE_PhysX" / "scripts" / script).read_text()
        assert "configure_terrain_gpu_physx_memory" in source, (
            f"{script} can silently drop contacts at large environment counts"
        )


def test_terrain_visual_theme_is_dark_asphalt_without_changing_task_geometry():
    cfg_source = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "road_color: tuple = (0.11, 0.12, 0.13)" in cfg_source
    assert "road_roughness: float = 0.92" in cfg_source
    assert "dark_visual_background: bool = True" in cfg_source
    theme = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_configure_dark_visual_theme")
    )
    assert 'GetAttribute("visibleInPrimaryRay")' in theme
    assert "attribute.Set(False)" in theme
    grid = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_spawn_visual_grid_floor")
    )
    assert "collision_enabled=False" in grid
    assert "VISUAL_GRID_ROOT" in grid


def test_evaluator_selects_one_path_family_and_supports_live_tracking_plot():
    source = (ROOT / "PPO3_SOLE_PhysX" / "scripts" / "evaluate.py").read_text()
    assert '"--path_family"' in source
    assert "env_cfg.tracking_path_family = str(a.path_family)" in source
    assert '"--plot_tracking"' in source
    assert "class LiveTrackingPlot" in source
    assert "tracking_plot.sample(raw, step)" in source


def test_terrain_teacher_uses_student_action_interface_without_direct_write():
    source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_compute_teacher_action")
    )
    assert "learned_delta_to_usd_delta" in source
    assert "apply_sole_action" in source
    assert "_task_action_shield" in source
    assert "_write_direct_learned_targets" not in source


def test_pre_physics_step_commands_the_plant_with_the_policy_action():
    """_current_action must be assigned from the sole-action chain, not a goal."""
    source = ast.get_source_segment(_ENV_SOURCE, _method("_pre_physics_step"))
    assert "apply_sole_action" in source
    assert "_run_cylinder_id_pipeline" in source
    # The parent's residual path routes through the DLS to produce action_4d.
    assert "action_4d" not in source, "DLS inversion leaked into the control path"


def test_observation_excludes_the_cascade_goal_unless_hint_is_enabled():
    """_baseline_cylinder_goal may only be read inside the ik_hint branch."""
    node = _method("_get_observations")
    baseline_reads = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute) and child.attr == "_baseline_cylinder_goal"
    ]
    if not baseline_reads:
        return
    # Every read must sit inside a branch whose test consults self._sole_ik_hint.
    guarded = False
    for child in ast.walk(node):
        if isinstance(child, ast.If):
            test_attrs = _attributes_used(child.test)
            if "_sole_ik_hint" in test_attrs:
                if "_baseline_cylinder_goal" in _attributes_used(child):
                    guarded = True
    assert guarded, (
        "_baseline_cylinder_goal is read in the observation outside the "
        "sole_ik_hint guard; the IK solution must not reach the default policy"
    )


def test_default_config_has_the_hint_disabled():
    from PPO3_SOLE_PhysX.observations import SOLE_OBS_DIM, sole_observation_dim

    cfg_source = (ROOT / "PPO3_SOLE_PhysX" / "env_cfg.py").read_text()
    assert "sole_ik_hint: bool = False" in cfg_source, (
        "the IK-hint ablation must be off by default"
    )
    assert sole_observation_dim(ik_hint=False) == SOLE_OBS_DIM


def test_reset_path_may_still_use_the_classical_controller():
    """Deployment and ROM priming legitimately keep the DLS; assert it survives.

    This is the counterpart to the exclusion tests: if a later refactor rips the
    cascade out entirely, reset-time deployment silently loses its verified
    hold, and this test is what says so.
    """
    parent = ast.parse((ROOT / "PPO2_ROM_PhysX" / "policy" / "env.py").read_text())
    for node in ast.walk(parent):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_warm_command_rom_at_target"
        ):
            warm = ast.dump(node)
            break
    else:
        raise AssertionError("ResidualPolicyEnv._warm_command_rom_at_target not found")
    assert "_cylinder_dls" in warm, (
        "reset-time ROM priming no longer uses the verified DLS hold"
    )


def test_sole_path_applies_configured_hydraulic_target_filter_after_rom():
    source = ast.get_source_segment(_ENV_SOURCE, _method("_pre_physics_step"))
    snapshot_at = source.index("previous_target =")
    rom_at = source.index("self._run_cylinder_id_pipeline()")
    filter_at = source.index("hydraulic_target_filter_beta")
    assert snapshot_at < rom_at < filter_at
    assert "target_beta * previous_target" in source
    assert "self._cylinder_adapter.joint_reference_indices" in source
    assert "_joint6_coupling_target_ref_index" in source


def test_terrain_reset_clears_travel_before_parent_deployment():
    """Old episode travel must not displace the new episode's scanner.

    The parent deployment calls the terrain override of ``_constrain_truck_xy``.
    If travel is cleared only after ``super()._reset_idx``, every reset advances
    the robot and scanner by the preceding rollout distance while the terrain
    mesh remains fixed.
    """
    node = _terrain_method("_reset_idx")
    parent_reset_lines = [
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "_reset_idx"
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "super"
    ]
    travel_clear_lines = [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "_truck_travel_m"
            for target in child.targets
        )
    ]
    assert len(parent_reset_lines) == 1, "terrain reset must call exactly one parent reset"
    assert travel_clear_lines, "terrain reset never clears _truck_travel_m"
    assert min(travel_clear_lines) < parent_reset_lines[0], (
        "_truck_travel_m must be cleared before the parent deployment can call "
        "_constrain_truck_xy"
    )


def test_terrain_first_reset_rechecks_drive_state_after_parent_deployment():
    """The first parent reset initializes the kinematic drive mid-reset."""
    source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_reset_idx")
    )
    parent_reset_offset = source.index("super()._reset_idx(env_ids)")
    post_parent = source[parent_reset_offset:]
    assert 'getattr(self, "_kinematic_drive_ready", False)' in post_parent, (
        "terrain reset uses only the stale pre-parent drive state and will skip "
        "first-reset terrain synchronization"
    )


def test_straight_truck_path_constrains_heading_about_base_link():
    """Holding XY while allowing chassis attitude is not a stable road path."""
    source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_constrain_truck_xy")
    )
    for contract in (
        "_drive_anchor_base_heading_xy",
        "_drive_anchor_base_quat",
        "torch.atan2",
        "root_relative_to_base",
        "quat_inv",
        "quat_mul",
    ):
        assert contract in source
    assert "root_pose[:, :3] = base_state[:, :3] + rotated_relative" in source

    reset_source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE, _terrain_method("_reset_idx")
    )
    anchor_at = reset_source.index("_drive_anchor_base_heading_xy")
    relatch_at = reset_source.index("_relatch_terrain_temporal_state(env_ids)")
    assert anchor_at < relatch_at


def test_deployment_synchronization_uses_fixed_capacity_independent_template():
    cfg_source = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "synchronized_deployment_template_env: int = 0" in cfg_source
    source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE,
        _terrain_method("_synchronize_deployed_robot_state"),
    )
    assert "synchronized_deployment_template_env" in source
    assert "torch.argmin" not in source


def test_terrain_arm_damping_tuning_starts_only_after_deployment():
    init_source = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE,
        _terrain_method("__init__"),
    )
    assert "_terrain_nominal_arm_damping" in init_source
    assert "_cylinder_joint_ids" in init_source
    assert "_tool_joint_id" in init_source
    assert "_apply_terrain_arm_drive_tuning()" not in init_source
    tuning = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE,
        _terrain_method("_apply_terrain_arm_drive_tuning"),
    )
    assert "terrain_arm_damping_scale" in tuning
    assert "write_joint_damping_to_sim" in tuning
    assert "write_joint_effort_limit_to_sim" not in tuning

    reset = ast.get_source_segment(
        _TERRAIN_ENV_SOURCE,
        _terrain_method("_reset_idx"),
    )
    restore_at = reset.index("_restore_terrain_arm_deployment_damping()")
    parent_at = reset.index("super()._reset_idx(env_ids)")
    tune_at = reset.index("_apply_terrain_arm_drive_tuning()")
    relatch_at = reset.index("_relatch_terrain_temporal_state(env_ids)")
    assert restore_at < parent_at < tune_at < relatch_at


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
