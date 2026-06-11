from __future__ import annotations

import torch

try:
    from rom_data_contract import CORE_INPUT_FEATURE_NAMES, EXTENSION_SLICE, HISTORY_LENGTH
    from valve_conditioned_causal_rom import EXPORT_DIR, ValveConditionedROMInference
except ImportError:
    from kevin_integration.rom_cylinders.rom_data_contract import CORE_INPUT_FEATURE_NAMES, EXTENSION_SLICE, HISTORY_LENGTH
    from kevin_integration.rom_cylinders.valve_conditioned_causal_rom import EXPORT_DIR, ValveConditionedROMInference


def test_exported_runtime_inference_is_finite() -> None:
    inference = ValveConditionedROMInference(EXPORT_DIR, device="cpu")
    history = torch.zeros((2, HISTORY_LENGTH, len(CORE_INPUT_FEATURE_NAMES)), dtype=torch.float32)
    history[:, :, EXTENSION_SLICE] = inference.model.extension_mean.reshape(1, 1, 4)
    out = inference.predict(history)
    if out["final_delta_extension"].shape != (2, 4):
        raise AssertionError(f"Unexpected ROM output shape: {tuple(out['final_delta_extension'].shape)}")
    if not torch.isfinite(out["final_delta_extension"]).all():
        raise AssertionError("Runtime ROM output contains NaN/Inf.")


def test_exported_runtime_has_valve_authority() -> None:
    inference = ValveConditionedROMInference(EXPORT_DIR, device="cpu")
    inference.assert_valve_authority()


def main() -> None:
    test_exported_runtime_inference_is_finite()
    test_exported_runtime_has_valve_authority()
    print("PASS: valve-conditioned runtime tests passed.")


if __name__ == "__main__":
    main()
