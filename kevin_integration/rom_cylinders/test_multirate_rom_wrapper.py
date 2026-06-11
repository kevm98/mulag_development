from __future__ import annotations

import torch

try:
    from multirate_rom_wrapper import MultirateROMConfig, MultirateROMWrapper
except ImportError:
    from kevin_integration.rom_cylinders.multirate_rom_wrapper import MultirateROMConfig, MultirateROMWrapper


class ConstantROM:
    def __init__(self, delta: torch.Tensor) -> None:
        self.delta = delta.float()
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.history_snapshots: list[torch.Tensor] = []

    def __call__(self, history: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.batch_sizes.append(int(history.shape[0]))
        self.history_snapshots.append(history.detach().cpu().clone())
        return self.delta.to(history.device).reshape(1, 4).repeat(history.shape[0], 1)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, message: str) -> None:
    if not torch.allclose(actual, expected, atol=1.0e-9, rtol=1.0e-6):
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


def test_distribution_and_update_rate() -> None:
    rom = ConstantROM(torch.tensor([0.001, -0.002, 0.003, -0.004]))
    wrapper = MultirateROMWrapper(rom, num_envs=1, config=MultirateROMConfig(history_length=1))
    increments = []
    update_flags = []
    for step in range(10):
        obs = torch.full((1, 26), float(step))
        increments.append(wrapper.step(obs))
        update_flags.append(bool(wrapper.rom_update_flag.item()))

    expected_substep = torch.tensor([[0.0002, -0.0004, 0.0006, -0.0008]])
    for step in range(10):
        assert_close(increments[step], expected_substep, f"substep {step}")
    assert update_flags == [True, False, False, False, False, True, False, False, False, False]
    if rom.calls != 2:
        raise AssertionError(f"ROM inference should run exactly twice in 10 calls, got {rom.calls}.")
    assert_close(sum(increments[:5]), torch.tensor([[0.001, -0.002, 0.003, -0.004]]), "five-step sum")
    if wrapper.rom_step_counter.item() != 0:
        raise AssertionError("Counter should wrap to zero after ten 0.01 s calls.")
    if wrapper.predicted_delta_extension_50ms.shape[-1] != 4:
        raise AssertionError("Wrapper exposed a fifth actuator output.")


def test_history_samples_are_50ms_spaced() -> None:
    rom = ConstantROM(torch.tensor([0.001, 0.001, 0.001, 0.001]))
    wrapper = MultirateROMWrapper(rom, num_envs=1, config=MultirateROMConfig(history_length=2))
    for step in range(6):
        obs = torch.full((1, 26), float(step))
        wrapper.step(obs)
    if rom.calls != 1:
        raise AssertionError(f"ROM should run once when the second 0.05 s sample arrives, got {rom.calls}.")
    history = rom.history_snapshots[-1]
    assert_close(history[0, :, 0], torch.tensor([0.0, 5.0]), "history 0.05 s sample spacing")


def test_reset_and_independent_batch_envs() -> None:
    rom = ConstantROM(torch.tensor([0.001, 0.001, 0.001, 0.001]))
    wrapper = MultirateROMWrapper(rom, num_envs=2, config=MultirateROMConfig(history_length=2))
    wrapper.prime_history(torch.ones((2, 2, 26)))
    wrapper.step(torch.zeros((2, 26)))
    wrapper.reset([1])
    if wrapper.history_ready[0].item() is not True or wrapper.history_ready[1].item() is not False:
        raise AssertionError("Independent env reset did not preserve env 0 while clearing env 1.")
    if wrapper.rom_step_counter[1].item() != 0:
        raise AssertionError("Reset did not clear env 1 counter.")
    if torch.any(wrapper.applied_delta_extension_10ms[1] != 0.0):
        raise AssertionError("Reset did not clear env 1 pending increment.")
    out = wrapper.step(torch.ones((2, 26)))
    if not torch.isfinite(out).all():
        raise AssertionError("Wrapper output contains NaN/Inf.")
    if rom.calls != 1:
        raise AssertionError("Reset env should not trigger a ROM call before history is ready again.")


def test_nonfinite_guard() -> None:
    rom = ConstantROM(torch.tensor([0.001, 0.001, 0.001, 0.001]))
    wrapper = MultirateROMWrapper(rom, num_envs=1, config=MultirateROMConfig(history_length=1))
    bad = torch.zeros((1, 26))
    bad[0, 0] = float("nan")
    try:
        wrapper.step(bad)
    except ValueError:
        return
    raise AssertionError("NaN observation should raise ValueError.")


def main() -> None:
    tests = [
        test_distribution_and_update_rate,
        test_history_samples_are_50ms_spaced,
        test_reset_and_independent_batch_envs,
        test_nonfinite_guard,
    ]
    for test in tests:
        test()
    print(f"PASS test_multirate_rom_wrapper: {len(tests)} tests")


if __name__ == "__main__":
    main()
