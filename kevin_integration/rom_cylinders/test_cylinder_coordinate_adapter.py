from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import torch

try:
    from cylinder_coordinate_adapter import CYLINDER_ORDER, CylinderCoordinateMapping
except ImportError:
    from kevin_integration.rom_cylinders.cylinder_coordinate_adapter import CYLINDER_ORDER, CylinderCoordinateMapping


ROOT = Path(__file__).resolve().parents[2]
ROM_DIR = ROOT / "kevin_integration" / "rom_cylinders"


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1.0e-9) -> None:
    if not torch.allclose(actual, expected.to(actual.device, actual.dtype), atol=atol, rtol=0.0):
        raise AssertionError(f"Mismatch:\nactual={actual}\nexpected={expected}")


def _mapping() -> CylinderCoordinateMapping:
    return CylinderCoordinateMapping.from_values(
        scale=[2.0, -0.5, 1.25, -3.0],
        offset=[0.1, 0.25, -0.2, 1.5],
        isaac_min=[-1.0, -0.5, -2.0, -4.0],
        isaac_max=[3.0, 0.5, 2.0, 4.0],
    )


def _nominal_mapping() -> CylinderCoordinateMapping:
    return CylinderCoordinateMapping.from_nominal_sign_values(
        sign=[1, -1, 1, -1],
        simscape_nominal_extension=[0.5, 0.6, 0.7, 0.8],
        isaac_nominal_position=[0.01, -0.02, 0.03, -0.04],
        isaac_min=[-1.0, -1.0, -1.0, -1.0],
        isaac_max=[1.0, 1.0, 1.0, 1.0],
    )


def test_position_inverse_velocity_delta_batch_dtype() -> None:
    mapping = _mapping()
    x = torch.tensor([[0.0, 0.5, 1.0, -0.5], [0.2, 0.4, -0.1, 0.3]], dtype=torch.float64)
    q_expected = torch.tensor([[0.1, 0.0, 1.05, 3.0], [0.5, 0.05, -0.325, 0.6]], dtype=torch.float64)
    q = mapping.simscape_extension_to_isaac_position(x)
    _assert_close(q, q_expected)
    _assert_close(mapping.isaac_position_to_simscape_extension(q), x)

    xdot = torch.tensor([[1.0, -2.0, 0.4, -0.25]], dtype=torch.float64)
    qdot = mapping.simscape_velocity_to_isaac_velocity(xdot)
    _assert_close(qdot, torch.tensor([[2.0, 1.0, 0.5, 0.75]], dtype=torch.float64))
    _assert_close(mapping.isaac_velocity_to_simscape_velocity(qdot), xdot)

    delta_x = torch.tensor([[0.01, -0.02, 0.03, -0.04]], dtype=torch.float64)
    delta_q = mapping.simscape_delta_to_isaac_delta(delta_x)
    _assert_close(delta_q, torch.tensor([[0.02, 0.01, 0.0375, 0.12]], dtype=torch.float64))
    _assert_close(mapping.isaac_delta_to_simscape_delta(delta_q), delta_x)
    if q.dtype != x.dtype or q.device != x.device:
        raise AssertionError("Adapter did not preserve dtype/device.")


def test_ordering_and_round_trips() -> None:
    if CYLINDER_ORDER != ("Drehzapfen_joint", "Ausleger_I_joint", "Ausleger_II_joint", "Messerkopf_Schwenk_joint"):
        raise AssertionError(f"Unexpected cylinder order: {CYLINDER_ORDER}")
    mapping = _mapping()
    q = torch.tensor([[-0.1, 0.2, 0.3, -0.4]], dtype=torch.float32)
    x = mapping.isaac_position_to_simscape_extension(q)
    _assert_close(mapping.simscape_extension_to_isaac_position(x), q, atol=1.0e-6)
    delta_q = torch.tensor([[0.02, -0.01, 0.04, -0.03]], dtype=torch.float32)
    delta_x = mapping.isaac_delta_to_simscape_delta(delta_q)
    _assert_close(mapping.simscape_delta_to_isaac_delta(delta_x), delta_q, atol=1.0e-6)


def test_limits_mapped_consistently() -> None:
    mapping = _mapping()
    inside = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    outside = torch.tensor([[4.0, 0.0, -3.0, 5.0]])
    if torch.any(mapping.isaac_limit_violation(inside)):
        raise AssertionError("Inside-limit point was marked as violation.")
    expected = torch.tensor([[True, False, True, True]])
    if not torch.equal(mapping.isaac_limit_violation(outside), expected):
        raise AssertionError("Limit violation flags are inconsistent.")


def test_rejections() -> None:
    try:
        CylinderCoordinateMapping.from_values(scale=[1.0, 0.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Zero scale was not rejected.")

    unresolved = {
        "version": "cylinder_coordinate_mapping_v1",
        "mapping_validated": False,
        "source": "unresolved",
        "cylinders": {
            name: {
                "rom_index": i,
                "scale": None,
                "offset": None,
                "isaac_min": -1.0,
                "isaac_max": 1.0,
                "direction_verified": False,
            }
            for i, name in enumerate(CYLINDER_ORDER)
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mapping.json"
        path.write_text(json.dumps(unresolved), encoding="utf-8")
        try:
            CylinderCoordinateMapping.from_json(path, mode="validated")
        except ValueError:
            pass
        else:
            raise AssertionError("Missing mapping was not rejected.")


def test_synthetic_reference_signs() -> None:
    mapping = CylinderCoordinateMapping.from_values(scale=[-0.5, -0.5, -0.5, -0.5], offset=[0.25, 0.25, 0.25, 0.25])
    x_ref = torch.full((1, 4), 0.5)
    q_ref = mapping.simscape_extension_to_isaac_position(x_ref)
    _assert_close(q_ref, torch.zeros((1, 4)))
    delta_x = torch.full((1, 4), 0.1)
    delta_q = mapping.simscape_delta_to_isaac_delta(delta_x)
    _assert_close(delta_q, torch.full((1, 4), -0.05))


def test_nominal_sign_offset_mapping() -> None:
    mapping = _nominal_mapping()
    q_nom = torch.tensor([[0.01, -0.02, 0.03, -0.04]], dtype=torch.float64)
    x_nom = torch.tensor([[0.5, 0.6, 0.7, 0.8]], dtype=torch.float64)
    _assert_close(mapping.isaac_position_to_simscape_extension(q_nom), x_nom)
    q = torch.tensor([[0.03, 0.01, -0.01, -0.08]], dtype=torch.float64)
    expected_x = torch.tensor([[0.52, 0.57, 0.66, 0.84]], dtype=torch.float64)
    x = mapping.isaac_position_to_simscape_extension(q)
    _assert_close(x, expected_x)
    _assert_close(mapping.simscape_extension_to_isaac_position(x), q)

    qdot = torch.tensor([[0.2, 0.2, -0.4, -0.4]], dtype=torch.float64)
    expected_xdot = torch.tensor([[0.2, -0.2, -0.4, 0.4]], dtype=torch.float64)
    _assert_close(mapping.isaac_velocity_to_simscape_velocity(qdot), expected_xdot)
    _assert_close(mapping.simscape_velocity_to_isaac_velocity(expected_xdot), qdot)

    delta_x = torch.tensor([[0.01, 0.02, -0.03, -0.04]], dtype=torch.float64)
    expected_delta_q = torch.tensor([[0.01, -0.02, -0.03, 0.04]], dtype=torch.float64)
    _assert_close(mapping.simscape_delta_to_isaac_delta(delta_x), expected_delta_q)
    _assert_close(mapping.isaac_delta_to_simscape_delta(expected_delta_q), delta_x)


def test_nominal_sign_batch_dtype_and_no_offset_on_rates() -> None:
    mapping = CylinderCoordinateMapping.from_nominal_sign_values(
        sign=[-1, -1, -1, -1],
        simscape_nominal_extension=[10.0, 20.0, 30.0, 40.0],
        isaac_nominal_position=[100.0, 200.0, 300.0, 400.0],
    )
    qdot = torch.ones((2, 4), dtype=torch.float32)
    delta_x = torch.full((2, 4), 0.5, dtype=torch.float32)
    _assert_close(mapping.isaac_velocity_to_simscape_velocity(qdot), -torch.ones((2, 4), dtype=torch.float32), atol=1.0e-6)
    _assert_close(mapping.simscape_delta_to_isaac_delta(delta_x), -torch.full((2, 4), 0.5, dtype=torch.float32), atol=1.0e-6)


def test_nominal_json_rejections_and_affine_availability() -> None:
    base = {
        "version": "cylinder_coordinate_mapping_v1",
        "mapping_mode": "nominal_sign_offset",
        "mapping_validated": False,
        "nominal_reference_complete": True,
        "source": "test",
        "units": {"simscape_extension": "m", "isaac_position": "m"},
        "cylinders": {
            name: {
                "rom_index": i,
                "sign": -1,
                "sign_verified": True,
                "simscape_nominal_extension": 0.5 + i,
                "isaac_nominal_position": 0.01 * i,
                "isaac_min": -1.0,
                "isaac_max": 1.0,
                "scale": None,
                "offset": None,
            }
            for i, name in enumerate(CYLINDER_ORDER)
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mapping.json"
        path.write_text(json.dumps(base), encoding="utf-8")
        try:
            CylinderCoordinateMapping.from_json(path, mode="nominal_sign_offset")
        except ValueError:
            pass
        else:
            raise AssertionError("Nonvalidated nominal reference loaded without explicit provisional flag.")
        loaded = CylinderCoordinateMapping.from_json(path, mode="nominal_sign_offset", allow_provisional_nominal_reference=True)
        if loaded.mapping_mode != "nominal_sign_offset":
            raise AssertionError("Nominal sign mapping did not load.")

        for field in ("sign", "simscape_nominal_extension", "isaac_nominal_position"):
            broken = json.loads(json.dumps(base))
            first = broken["cylinders"][CYLINDER_ORDER[0]]
            first[field] = None
            path.write_text(json.dumps(broken), encoding="utf-8")
            try:
                CylinderCoordinateMapping.from_json(
                    path,
                    mode="nominal_sign_offset",
                    allow_provisional_nominal_reference=True,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"Missing {field} was not rejected.")

    affine = _mapping()
    if affine.mapping_mode != "affine_validated":
        raise AssertionError("Affine fallback mode is unavailable.")


def test_extracted_provisional_nominal_reference() -> None:
    reference_path = ROM_DIR / "simscape_nominal_cylinder_reference.json"
    mapping_path = ROM_DIR / "cylinder_coordinate_mapping.json"
    if not reference_path.exists() or not mapping_path.exists():
        return
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    mapping_json = json.loads(mapping_path.read_text(encoding="utf-8"))
    signs = reference.get("signs")
    if signs != [-1, -1, -1, -1]:
        raise AssertionError(f"Expected all extracted signs to be -1, got {signs}")
    mapping = CylinderCoordinateMapping.from_nominal_sign_values(
        sign=signs,
        simscape_nominal_extension=reference["simscape_extension"],
        isaac_nominal_position=reference["isaac_nominal_position"],
        sign_verified=(True, True, True, True),
    )
    q_nominal = torch.tensor([reference["isaac_nominal_position"]], dtype=torch.float64)
    x_nominal = torch.tensor([reference["simscape_extension"]], dtype=torch.float64)
    _assert_close(mapping.isaac_position_to_simscape_extension(q_nominal), x_nominal)
    _assert_close(mapping.simscape_extension_to_isaac_position(x_nominal), q_nominal)
    _assert_close(mapping.isaac_velocity_to_simscape_velocity(torch.ones((1, 4))), -torch.ones((1, 4)))
    _assert_close(mapping.simscape_delta_to_isaac_delta(torch.ones((1, 4))), -torch.ones((1, 4)))
    if not mapping_json.get("nominal_reference_complete", False):
        raise AssertionError("Extracted nominal reference should be marked complete for provisional diagnostics.")
    if mapping_json.get("mapping_validated", True):
        raise AssertionError("Provisional WARN reference must not mark mapping_validated=true.")
    for entry in mapping_json.get("cylinders", {}).values():
        if entry.get("sign") != -1 or not entry.get("sign_verified", False):
            raise AssertionError("Every extracted nominal cylinder must preserve verified sign=-1.")
        if entry.get("simple_mapping_validated", True):
            raise AssertionError("Provisional nominal cylinder must not mark simple_mapping_validated=true.")
        if entry.get("nominal_reference_confidence") != "WARN":
            raise AssertionError("Extracted nominal cylinder confidence must remain WARN.")
    try:
        CylinderCoordinateMapping.from_json(mapping_path, mode="nominal_sign_offset")
    except ValueError:
        pass
    else:
        raise AssertionError("WARN/nonvalidated nominal reference did not require explicit provisional flag.")
    loaded = CylinderCoordinateMapping.from_json(
        mapping_path,
        mode="nominal_sign_offset",
        allow_provisional_nominal_reference=True,
    )
    if loaded.mapping_validated:
        raise AssertionError("Provisional mapping must remain mapping_validated=false.")
    _assert_close(loaded.sign_tensor(), torch.full((4,), -1.0, dtype=torch.float64))
    _assert_close(loaded.isaac_position_to_simscape_extension(q_nominal), x_nominal)
    _assert_close(loaded.simscape_extension_to_isaac_position(x_nominal), q_nominal)
    _assert_close(loaded.isaac_velocity_to_simscape_velocity(torch.ones((1, 4))), -torch.ones((1, 4)))
    _assert_close(loaded.simscape_delta_to_isaac_delta(torch.ones((1, 4))), -torch.ones((1, 4)))


def test_calibration_log_validates_current_sign_scale_deltas() -> None:
    mapping_path = ROM_DIR / "cylinder_coordinate_mapping.json"
    calibration_path = ROOT / "logs" / "cylinder_coordinate_calibration" / "20260609_185550" / "cylinder_coordinate_calibration.csv"
    if not calibration_path.exists():
        raise AssertionError(f"Missing coordinate calibration log: {calibration_path}")

    mapping_json = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not mapping_json.get("sign_scale_delta_validated", False):
        raise AssertionError("Current mapping must record sign/scale delta validation.")
    if mapping_json.get("mapping_validated", True):
        raise AssertionError("Absolute mapping must remain unvalidated until paired Simscape second-pose data exists.")
    if mapping_json.get("absolute_offset_validation_complete", True):
        raise AssertionError("Absolute offset validation must remain incomplete without paired Simscape pose data.")
    rows = list(csv.DictReader(calibration_path.read_text(encoding="utf-8").splitlines()))
    if len(rows) < 2 * len(CYLINDER_ORDER):
        raise AssertionError("Calibration log does not contain +/- offset samples for every cylinder.")

    for index, name in enumerate(CYLINDER_ORDER):
        entry = mapping_json["cylinders"][name]
        if entry.get("sign") != -1 or not entry.get("sign_verified", False):
            raise AssertionError(f"{name} does not preserve the calibrated sign=-1 convention.")
        if entry.get("simple_mapping_validated", True):
            raise AssertionError("Absolute nominal mapping must remain provisional without paired Simscape pose data.")

        cylinder_rows = [row for row in rows if int(row["actuator_index"]) == index]
        if len(cylinder_rows) != 2:
            raise AssertionError(f"{name} expected two calibration offsets, got {len(cylinder_rows)}.")
        for row in cylinder_rows:
            delta_q = float(row["delta_q"])
            delta_length_proxy = float(row["delta_length_proxy"])
            predicted_delta_extension = float(entry["sign"]) * delta_q
            error = abs(predicted_delta_extension - delta_length_proxy)
            if error > 2.5e-6:
                raise AssertionError(
                    f"{name} calibration delta mismatch: predicted {predicted_delta_extension:.9f}, "
                    f"length proxy {delta_length_proxy:.9f}, error {error:.9f}"
                )
        slope_values = [
            float(row["delta_length_proxy"]) / float(row["delta_q"])
            for row in cylinder_rows
        ]
        mean_slope = sum(slope_values) / len(slope_values)
        if abs(mean_slope + 1.0) > 3.0e-4:
            raise AssertionError(f"{name} length-proxy slope {mean_slope:.9f} is not consistent with sign=-1.")


def test_training_range_excess_blocks_severe_mismatch() -> None:
    mapped = torch.tensor([[0.335, 0.9, 0.493, 0.586]], dtype=torch.float64)
    training_min = torch.tensor([0.0112, 0.2753, 0.3092, 0.3319], dtype=torch.float64)
    training_max = torch.tensor([0.4933, 0.7977, 0.8569, 0.6891], dtype=torch.float64)
    excess = torch.maximum(training_min - mapped.reshape(-1), mapped.reshape(-1) - training_max)
    excess = torch.clamp(excess, min=0.0)
    if float(torch.max(excess)) <= 0.05:
        raise AssertionError("Severe training-range mismatch should exceed the 0.05 m gate.")


def test_cuda_parity_if_available() -> None:
    if not torch.cuda.is_available():
        return
    mapping = _mapping()
    q_cpu = torch.tensor([[0.1, -0.1, 0.2, -0.2]], dtype=torch.float32)
    x_cpu = mapping.isaac_position_to_simscape_extension(q_cpu)
    x_cuda = mapping.isaac_position_to_simscape_extension(q_cpu.cuda()).cpu()
    _assert_close(x_cuda, x_cpu, atol=1.0e-6)


def main() -> None:
    test_position_inverse_velocity_delta_batch_dtype()
    test_ordering_and_round_trips()
    test_limits_mapped_consistently()
    test_rejections()
    test_synthetic_reference_signs()
    test_nominal_sign_offset_mapping()
    test_nominal_sign_batch_dtype_and_no_offset_on_rates()
    test_nominal_json_rejections_and_affine_availability()
    test_extracted_provisional_nominal_reference()
    test_calibration_log_validates_current_sign_scale_deltas()
    test_training_range_excess_blocks_severe_mismatch()
    test_cuda_parity_if_available()
    print("PASS: cylinder coordinate adapter tests passed.")


if __name__ == "__main__":
    main()
