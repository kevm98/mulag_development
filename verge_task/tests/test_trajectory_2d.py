import numpy as np

from verge_task.trajectory_2d import circle, line, make_trajectory, rounded_square, sine_chain


def test_all_public_shapes_have_valid_arc_length_geometry():
    for name in ("line", "sine_chain", "circle", "rounded_square"):
        path = make_trajectory(name)
        assert path.length_m > 0.5
        assert np.all(np.diff(path.s_m) > 0.0)
        assert np.allclose(np.linalg.norm(path.tangent_xy, axis=1), 1.0, atol=1e-6)
        assert np.allclose(path.normal_xy, np.stack((-path.tangent_xy[:, 1], path.tangent_xy[:, 0]), axis=-1))


def test_closed_shapes_wrap_and_round_trip():
    for path in (circle(), rounded_square()):
        point_a, tangent_a, normal_a = path.sample(0.0)
        point_b, tangent_b, normal_b = path.sample(path.length_m)
        assert np.allclose(point_a, point_b, atol=2e-3)
        assert np.allclose(tangent_a, tangent_b, atol=2e-2)
        assert np.allclose(normal_a, normal_b, atol=2e-2)


def test_nearest_point_returns_signed_normal_for_line():
    path = line(length_m=2.0)
    result = path.nearest_point(np.array([[0.2, 0.15], [-0.3, -0.2]]))
    assert np.allclose(result.point_xy[:, 1], 0.0)
    assert np.allclose(result.signed_normal_m, (0.15, -0.2), atol=1e-6)
    assert np.allclose(result.distance_m, (0.15, 0.2), atol=1e-6)


def test_sine_chain_is_open_and_rounded_square_has_fillets():
    sine = sine_chain(cycles=4.0)
    square = rounded_square(side_m=1.0, corner_radius_m=0.2)
    assert not sine.closed
    assert square.closed
    assert np.max(np.abs(sine.xy[:, 1])) <= 0.18 + 1e-6
    # A filleted square has points strictly inside the sharp-corner box.
    assert np.any(np.abs(square.xy[:, 0]) < 0.5)
    assert np.any(np.abs(square.xy[:, 1]) < 0.5)
