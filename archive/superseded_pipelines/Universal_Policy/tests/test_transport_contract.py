"""Regression gates that keep transport mutations out of PPO arm tracking."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIPELINES = ("PPO1_ID_FD_PhysX", "PPO2_ROM_PhysX")


def _source(pipeline: str, relative: str) -> str:
    return (ROOT / pipeline / relative).read_text()


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and candidate.name == name
    )
    return ast.get_source_segment(source, node) or ""


def test_parked_fixture_preserves_trained_velocity_contract():
    for pipeline in PIPELINES:
        env = _source(pipeline, "policy/env.py")
        constraint = _function_source(env, "_constrain_truck_xy")
        assert "write_root_pose_to_sim" in constraint
        assert "write_root_velocity_to_sim" not in constraint


def test_reset_fixture_runs_once_per_control_interval():
    for pipeline in PIPELINES:
        env = _source(pipeline, "policy/env.py")
        reset = _function_source(env, "_run_reset_preposition")
        # Both deployment and settle loops retain the checkpoint's original
        # once-per-decimated-control-step planar correction.
        assert reset.count("            self._constrain_truck_xy()") == 2
        assert "                self._constrain_truck_xy()" not in reset


def test_supervisors_do_not_hard_write_arm_or_tracking_render_state():
    for pipeline in PIPELINES:
        supervisor = _source(pipeline, "policy/scripts/stop_and_work.py")
        direct = _function_source(supervisor, "_direct_arm_step")
        latch = _function_source(supervisor, "_latch_station_for_arm_tracking")
        main = _function_source(supervisor, "main")

        assert "def _write_transport_arm_lock" not in supervisor
        assert "_truck_initial_xy" not in supervisor
        assert "_truck_initial_quat" not in supervisor
        assert "route_origin = settled_base[:2]" in main
        if pipeline == "PPO1_ID_FD_PhysX":
            assert "raw.cfg.sim.render_interval = 1_000_000_000" in main
            assert "raw.sim.render()" in main
        else:
            assert "render_interval =" not in main
            assert "raw.sim.render()" not in main
        assert "raw._constrain_truck_xy()" not in main

        assert "write_joint_position_to_sim" not in direct
        assert "write_joint_velocity_to_sim" not in direct
        assert "_write_prescribed_truck_base_xy" not in direct
        assert "raw._constrain_truck_xy()" in direct
        assert "write_root_velocity_to_sim" in latch
        assert "write_joint_position_to_sim" not in latch
        assert main.count("_latch_station_for_arm_tracking(raw)") == 1


def test_prescribed_chassis_motion_is_drive_only_and_quintic():
    for pipeline in PIPELINES:
        supervisor = _source(pipeline, "policy/scripts/stop_and_work.py")
        stow = _function_source(supervisor, "_stow_arm")
        drive = _function_source(supervisor, "_drive_to_next_station")
        assert "_write_prescribed_truck_base_xy" not in stow
        assert "_write_prescribed_truck_base_xy" in drive
        assert "10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5" in drive
        assert "set_joint_position_target(arm_rest_pos, joint_ids=arm_ids)" in drive
        assert "transport_settle_s" in drive
        assert "transport_arm_stiffness_scale" in drive
        assert "write_joint_stiffness_to_sim(arm_stiffness" in drive
        assert "arm_rest_pos = raw.robot.data.joint_pos" in drive
        assert "write_joint_position_to_sim(arm_rest_pos" in drive
        assert "write_joint_velocity_to_sim(" in drive
        assert "write_joint_position_to_sim(arm_rest_pos" not in supervisor.replace(
            drive, ""
        )
        assert "min_tool_clearance" in drive
        assert "max_tool_z_excursion" in drive
        assert "max_tool_relative_excursion" in drive
        assert "transport_max_arm_joint_error" in drive
        raise_index = drive.index('"transport stress gate failed: "')
        gate = drive[drive.rindex("    if (", 0, raise_index) : raise_index]
        if pipeline == "PPO1_ID_FD_PhysX":
            assert "max_tool_z_excursion >" in gate
            assert "max_tool_relative_excursion" in gate
        else:
            assert "max_tool_z_excursion >" not in gate
            assert "max_tool_relative_excursion" not in gate
        assert "transport stress gate failed" in drive

        if pipeline == "PPO1_ID_FD_PhysX":
            assert "transport_chain_ids" in drive
            assert "lock_transport_chain()" in drive
            assert "if joint_id not in moving_joint_ids" in drive


def test_ppo1_prescribed_drive_uses_one_constant_transport_root_reference():
    source = _source("PPO1_ID_FD_PhysX", "policy/scripts/stop_and_work.py")
    prescribed = _function_source(source, "_write_prescribed_truck_base_xy")
    assert "transport_root_pose" in prescribed
    assert "transport_base_xy" in prescribed
    assert "root_pose = transport_root_pose" in prescribed
    assert "target_base_xy - transport_base_xy" in prescribed
    assert "target_heading" not in prescribed
    assert "quat_mul" not in prescribed


def test_ppo1_viewport_overlay_is_reference_only():
    source = _source("PPO1_ID_FD_PhysX", "policy/scripts/stop_and_work.py")
    overlay = _function_source(source, "ViewportTrajectoryLines")
    main = _function_source(source, "main")
    assert "REFERENCE_COLOR = 0xFF00FFFF" in overlay
    assert "raw._goals" in overlay
    assert "_compute_ee_pos_local" not in overlay
    assert "MEASURED_COLOR" not in overlay
    assert "0xFFFF8800" not in overlay
    assert "viewport_lines.begin_station(raw)" in main
    assert main.index("viewport_lines.sample_tool(raw)") < main.index(
        "raw.sim.render()", main.index("obs, _, dones, extras = env.step(action)")
    )


def test_quintic_route_is_monotonic_and_has_zero_endpoint_speed():
    distance = 2.0
    speed_limit = 0.6
    duration = 1.875 * distance / speed_limit
    position = []
    speed = []
    for index in range(1001):
        u = index / 1000.0
        position.append(10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5)
        speed.append(distance * 30.0 * u**2 * (1.0 - u) ** 2 / duration)
    assert position[0] == 0.0
    assert abs(position[-1] - 1.0) < 1.0e-12
    assert all(right >= left for left, right in zip(position, position[1:]))
    assert speed[0] == 0.0
    assert speed[-1] == 0.0
    assert max(speed) <= speed_limit + 1.0e-12
