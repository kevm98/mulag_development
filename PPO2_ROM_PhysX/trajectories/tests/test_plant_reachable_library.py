from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "models/plant_reachable_trajectory_library_v1.json"
EXPANDED_LIBRARY = ROOT / "models/plant_reachable_trajectory_library_v2_candidate_b.json"
LARGE_LIBRARY = ROOT / "models/plant_reachable_trajectory_library_v3_large.json"
CIRCLE_C3_LIBRARY = ROOT / "models/plant_reachable_trajectory_library_v3_circle_c3.json"
CIRCLE_C4_LIBRARY = ROOT / "models/plant_reachable_trajectory_library_v3_circle_c4_retime.json"
CIRCLE_C7_LIBRARY = ROOT / "models/plant_reachable_trajectory_library_v3_circle_c7_retime.json"


def _shape(payload: dict, name: str) -> np.ndarray:
    return np.asarray(
        [row["cutter_truck"] for row in payload["shapes"][name]["samples"]],
        dtype=np.float64,
    )


def test_rebuilt_library_is_planar_and_passes_plant_projection_gate() -> None:
    payload = json.loads(LIBRARY.read_text())
    exact = {"line", "arc", "sine", "circle", "square"}
    assert exact.issubset(payload["shapes"])
    assert set(payload["generation_contract"]["exact_shape_names"]) == exact
    assert len(payload["generation_contract"]["procedural_shape_names"]) == 10
    for name, shape in payload["shapes"].items():
        goals = _shape(payload, name)
        assert np.ptp(goals[:, 2]) < 1.0e-10
        assert shape["length_m"] >= 1.8
        report = shape["plant_reachability"]
        assert report["plant_projection_mean_error_m"] <= 0.015
        assert report["plant_projection_p95_error_m"] <= 0.030
        assert report["plant_projection_max_error_m"] <= 0.040


def test_circle_is_geometrically_circular_and_closed() -> None:
    payload = json.loads(LIBRARY.read_text())
    goals = _shape(payload, "circle")
    assert np.linalg.norm(goals[0] - goals[-1]) < 1.0e-10
    center = np.mean([goals[:, :2].min(axis=0), goals[:, :2].max(axis=0)], axis=0)
    radius = np.linalg.norm(goals[:, :2] - center, axis=1)
    assert np.ptp(radius) < 1.0e-10


def test_square_has_four_exact_sharp_boundaries() -> None:
    payload = json.loads(LIBRARY.read_text())
    goals = _shape(payload, "square")
    assert np.linalg.norm(goals[0] - goals[-1]) < 1.0e-10
    xy = goals[:, :2]
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    assert np.allclose(high - low, [0.6, 0.6], atol=1.0e-10)
    distance_to_boundary = np.min(
        np.column_stack((
            np.abs(xy[:, 0] - low[0]),
            np.abs(xy[:, 0] - high[0]),
            np.abs(xy[:, 1] - low[1]),
            np.abs(xy[:, 1] - high[1]),
        )),
        axis=1,
    )
    assert np.max(distance_to_boundary) < 1.0e-10
    corners = np.asarray([
        [low[0], low[1]],
        [low[0], high[1]],
        [high[0], high[1]],
        [high[0], low[1]],
    ])
    assert all(np.min(np.linalg.norm(xy - corner, axis=1)) < 1.0e-10 for corner in corners)


def test_expanded_v2_geometry_is_larger_and_exact() -> None:
    payload = json.loads(EXPANDED_LIBRARY.read_text())
    sine = _shape(payload, "sine")
    circle = _shape(payload, "circle")
    square = _shape(payload, "square")

    assert np.allclose(np.ptp(sine, axis=0), [2.4, 0.6, 0.0], atol=1.0e-10)
    assert np.allclose(np.ptp(circle, axis=0), [0.9, 0.9, 0.0], atol=1.0e-10)
    assert np.allclose(np.ptp(square, axis=0), [0.7, 0.7, 0.0], atol=1.0e-10)

    circle_center = 0.5 * (
        circle[:, :2].min(axis=0) + circle[:, :2].max(axis=0)
    )
    assert np.ptp(np.linalg.norm(circle[:, :2] - circle_center, axis=1)) < 1.0e-10

    low = square[:, :2].min(axis=0)
    high = square[:, :2].max(axis=0)
    distance_to_boundary = np.min(
        np.column_stack((
            np.abs(square[:, 0] - low[0]),
            np.abs(square[:, 0] - high[0]),
            np.abs(square[:, 1] - low[1]),
            np.abs(square[:, 1] - high[1]),
        )),
        axis=1,
    )
    assert np.max(distance_to_boundary) < 1.0e-10


def test_large_v3_geometry_matches_training_contract() -> None:
    payload = json.loads(LARGE_LIBRARY.read_text())
    expected = {
        "line": [2.5, 0.0, 0.0],
        "arc": [2.2, 0.5, 0.0],
        "sine": [3.2, 0.9, 0.0],
        "circle": [1.1, 1.1, 0.0],
        "square": [0.9, 0.9, 0.0],
    }
    for name, span in expected.items():
        assert np.allclose(np.ptp(_shape(payload, name), axis=0), span, atol=1.0e-10)


def test_circle_c3_changes_only_the_inverse_circle_reference() -> None:
    source = json.loads(LARGE_LIBRARY.read_text())
    candidate = json.loads(CIRCLE_C3_LIBRARY.read_text())
    for name in source["shapes"]:
        if name != "circle":
            assert candidate["shapes"][name] == source["shapes"][name]
    assert np.array_equal(_shape(candidate, "circle"), _shape(source, "circle"))
    contract = candidate["generation_contract"]["circle_continuity_repair"]
    selected = contract["selected"]
    assert contract["tool_geometry_changed"] is False
    assert selected["passed_offline"] is True
    assert selected["roughness_p95_ratio"] < 0.60
    assert selected["roughness_max_ratio"] < 0.40
    assert selected["tool_p95_error_m"] < 0.010
    assert selected["minimum_predicted_head_alignment"] >= 0.98


def test_circle_c3_is_closed_in_cylinder_position() -> None:
    payload = json.loads(CIRCLE_C3_LIBRARY.read_text())
    commands = np.asarray(
        [row["cylinder_cmd"] for row in payload["shapes"]["circle"]["samples"]]
    )
    assert np.array_equal(commands[0], commands[-1])


def test_circle_c4_retimes_without_changing_geometry_or_cylinders() -> None:
    source = json.loads(CIRCLE_C3_LIBRARY.read_text())
    candidate = json.loads(CIRCLE_C4_LIBRARY.read_text())
    for name in source["shapes"]:
        if name != "circle":
            assert candidate["shapes"][name] == source["shapes"][name]
    source_rows = source["shapes"]["circle"]["samples"]
    candidate_rows = candidate["shapes"]["circle"]["samples"]
    for before, after in zip(source_rows, candidate_rows, strict=True):
        timing = after["timing_s_m"]
        assert np.isfinite(timing)
        assert {key: value for key, value in after.items() if key != "timing_s_m"} == before
    timing = np.asarray([row["timing_s_m"] for row in candidate_rows])
    assert timing[0] == 0.0
    assert np.all(np.diff(timing) > 0.0)
    contract = candidate["generation_contract"]["circle_retiming"]
    assert contract["tool_geometry_changed"] is False
    assert contract["cylinder_commands_changed"] is False
    assert contract["geometric_start_phase_distribution_preserved"] is True


def test_circle_c7_revises_only_timing() -> None:
    source = json.loads(CIRCLE_C4_LIBRARY.read_text())
    candidate = json.loads(CIRCLE_C7_LIBRARY.read_text())
    for name in source["shapes"]:
        if name != "circle":
            assert candidate["shapes"][name] == source["shapes"][name]
    for before, after in zip(
        source["shapes"]["circle"]["samples"],
        candidate["shapes"]["circle"]["samples"],
        strict=True,
    ):
        assert before["path_s_m"] == after["path_s_m"]
        assert before["cutter_truck"] == after["cutter_truck"]
        assert before["cylinder_cmd"] == after["cylinder_cmd"]
    timing = np.asarray([
        row["timing_s_m"] for row in candidate["shapes"]["circle"]["samples"]
    ])
    assert timing[0] == 0.0
    assert np.all(np.diff(timing) > 0.0)
    contract = candidate["generation_contract"]["circle_retiming"]
    assert contract["tool_geometry_changed"] is False
    assert contract["cylinder_commands_changed"] is False
    assert contract["minimum_total_speed_scale"] < 0.70
