"""Deterministic probe-excitation schedules for master-dataset collection.

Schedules are generated in normalized action space (4-D, values inside
[-1, 1]) at the native control dt. They are fully described by their params so
the run metadata (Stage 8, #20) can make every run reproducible without the
generator itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    name: str
    start: int
    end: int  # exclusive

    def as_dict(self) -> dict:
        return {"name": self.name, "start_step": self.start, "end_step": self.end}


@dataclass
class MultisineSchedule:
    """Staggered multisine probe, one cylinder active at a time.

    Each cylinder gets a window of `per_cylinder_steps` with a low-frequency
    multisine sum; idle cylinders keep a small dither so the hydraulic state
    does not freeze. All amplitudes are bounded well inside the [-1, 1] action
    clip so the probe stays in the linear-ish authority band.
    """

    control_dt: float = 1.0 / 60.0
    settle_steps: int = 60
    tail_steps: int = 60
    per_cylinder_steps: int = 360
    frequencies_hz: tuple[float, ...] = (0.5, 1.3, 2.7)
    amplitudes: tuple[float, ...] = (0.12, 0.08, 0.05)
    phases: tuple[float, ...] = (0.0, 0.8, 1.7)
    idle_amplitude: float = 0.02
    idle_frequency_hz: float = 0.2

    @property
    def params(self) -> dict:
        return {
            "name": "multisine",
            "control_dt": self.control_dt,
            "settle_steps": self.settle_steps,
            "tail_steps": self.tail_steps,
            "per_cylinder_steps": self.per_cylinder_steps,
            "frequencies_hz": list(self.frequencies_hz),
            "amplitudes": list(self.amplitudes),
            "phases": list(self.phases),
            "idle_amplitude": self.idle_amplitude,
            "idle_frequency_hz": self.idle_frequency_hz,
        }

    def build(self, total_steps: int | None = None) -> tuple[np.ndarray, list[Segment]]:
        if total_steps is None:
            total_steps = (
                self.settle_steps + 4 * self.per_cylinder_steps + self.tail_steps
            )
        settle, tail = self.settle_steps, self.tail_steps
        if total_steps < settle + tail + 8:
            spare = max(total_steps - 8, 0)
            settle = min(settle, spare // 2)
            tail = min(tail, spare - settle)
        active = max((total_steps - settle - tail) // 4, 2)
        t = np.arange(total_steps, dtype=np.float64) * self.control_dt
        action = np.zeros((total_steps, 4), dtype=np.float64)
        segments: list[Segment] = [Segment("settle", 0, settle)]
        for c in range(4):
            start = settle + c * active
            end = min(start + active, total_steps - tail)
            segments.append(Segment(f"cyl{c}", start, end))
            window = np.arange(start, end, dtype=np.float64) * self.control_dt
            signal = np.zeros(window.shape)
            for freq, amp, phase in zip(
                self.frequencies_hz, self.amplitudes, self.phases
            ):
                signal += amp * np.sin(2 * np.pi * freq * window + phase)
            action[start:end, c] = signal
        segments.append(Segment("tail", total_steps - tail, total_steps))
        idle = self.idle_amplitude * np.sin(
            2 * np.pi * self.idle_frequency_hz * t[:, None]
            + np.arange(4)[None, :] * 0.7
        )
        active_mask = np.zeros((total_steps, 4), dtype=bool)
        for seg in segments[1:-1]:
            active_mask[seg.start : seg.end, int(seg.name[3])] = True
        action = np.where(active_mask, action, idle)
        action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32), segments


@dataclass
class TrapezoidSchedule:
    """Staggered trapezoid probe, matching the Stage 4 excitation shape.

    One cylinder at a time steps +A, holds, returns -A, holds, then the next
    cylinder starts. A is an action-space amplitude kept inside [-1, 1].
    """

    control_dt: float = 1.0 / 60.0
    settle_steps: int = 60
    tail_steps: int = 60
    per_cylinder_steps: int = 200
    amplitude: float = 0.3
    ramp_steps: int = 10

    @property
    def params(self) -> dict:
        return {
            "name": "trapezoid",
            "control_dt": self.control_dt,
            "settle_steps": self.settle_steps,
            "tail_steps": self.tail_steps,
            "per_cylinder_steps": self.per_cylinder_steps,
            "amplitude": self.amplitude,
            "ramp_steps": self.ramp_steps,
        }

    def build(self, total_steps: int | None = None) -> tuple[np.ndarray, list[Segment]]:
        if total_steps is None:
            total_steps = (
                self.settle_steps + 4 * self.per_cylinder_steps + self.tail_steps
            )
        settle, tail = self.settle_steps, self.tail_steps
        if total_steps < settle + tail + 8:
            spare = max(total_steps - 8, 0)
            settle = min(settle, spare // 2)
            tail = min(tail, spare - settle)
        active = max((total_steps - settle - tail) // 4, 2)
        action = np.zeros((total_steps, 4), dtype=np.float64)
        segments: list[Segment] = [Segment("settle", 0, settle)]
        for c in range(4):
            start = settle + c * active
            end = min(start + active, total_steps - tail)
            segments.append(Segment(f"cyl{c}", start, end))
            n = end - start
            up = max(self.ramp_steps, 1)
            half = max(n // 3, up + 1)
            for k in range(n):
                if k < up:
                    value = self.amplitude * k / up
                elif k < half:
                    value = self.amplitude
                elif k < half + up:
                    value = self.amplitude * (1.0 - (k - half) / up)
                elif k < 2 * half:
                    value = -self.amplitude * (k - half - up) / max(half - up, 1)
                elif k < 2 * half + up:
                    value = -self.amplitude * (1.0 - (k - 2 * half) / up)
                else:
                    value = 0.0
                action[start + k, c] = value
        segments.append(Segment("tail", total_steps - tail, total_steps))
        action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32), segments


@dataclass
class PRBSSchedule:
    """Full-MIMO pseudo-random binary sequence probe (item 33).

    All four channels are active simultaneously with independent, deterministic
    binary sequences (+-amplitude), held for `hold_steps` per bit. Because the
    sequences are independent per channel the cross-channel correlation is ~0,
    which is the MIMO persistent-excitation requirement (items 28-30).
    """

    control_dt: float = 1.0 / 60.0
    settle_steps: int = 60
    tail_steps: int = 60
    per_cylinder_steps: int = 360
    amplitude: float = 0.15
    hold_steps: int = 10
    seed: int = 20260807

    @property
    def params(self) -> dict:
        return {
            "name": "prbs",
            "control_dt": self.control_dt,
            "settle_steps": self.settle_steps,
            "tail_steps": self.tail_steps,
            "per_cylinder_steps": self.per_cylinder_steps,
            "amplitude": self.amplitude,
            "hold_steps": self.hold_steps,
            "seed": self.seed,
        }

    def build(self, total_steps: int | None = None) -> tuple[np.ndarray, list[Segment]]:
        if total_steps is None:
            total_steps = (
                self.settle_steps + 4 * self.per_cylinder_steps + self.tail_steps
            )
        settle, tail = self.settle_steps, self.tail_steps
        if total_steps < settle + tail + 8:
            spare = max(total_steps - 8, 0)
            settle = min(settle, spare // 2)
            tail = min(tail, spare - settle)
        active = max(total_steps - settle - tail, 2)
        action = np.zeros((total_steps, 4), dtype=np.float64)
        n_bits = int(np.ceil(active / max(int(self.hold_steps), 1)))
        for c in range(4):
            rng = np.random.RandomState(int(self.seed) + 97 * c)
            bits = rng.randint(0, 2, size=n_bits).astype(np.float64) * 2.0 - 1.0
            seq = np.repeat(bits, max(int(self.hold_steps), 1))[:active]
            action[settle : settle + active, c] = seq * float(self.amplitude)
        segments = [
            Segment("settle", 0, settle),
            Segment("mimo_active", settle, settle + active),
            Segment("tail", total_steps - tail, total_steps),
        ]
        return action.astype(np.float32), segments


@dataclass
class ChirpSchedule:
    """Full-MIMO linear frequency sweep probe (item 33, preliminary bandwidth).

    All channels sweep from `f_start_hz` to `f_end_hz` over the active window;
    per-channel phase offsets decorrelate the channels. Mainly for preliminary
    hydraulic-bandwidth estimation before choosing multisine/PRBS lines.
    """

    control_dt: float = 1.0 / 60.0
    settle_steps: int = 60
    tail_steps: int = 60
    per_cylinder_steps: int = 720
    amplitude: float = 0.15
    f_start_hz: float = 0.05
    f_end_hz: float = 5.0

    @property
    def params(self) -> dict:
        return {
            "name": "chirp",
            "control_dt": self.control_dt,
            "settle_steps": self.settle_steps,
            "tail_steps": self.tail_steps,
            "per_cylinder_steps": self.per_cylinder_steps,
            "amplitude": self.amplitude,
            "f_start_hz": self.f_start_hz,
            "f_end_hz": self.f_end_hz,
        }

    def build(self, total_steps: int | None = None) -> tuple[np.ndarray, list[Segment]]:
        if total_steps is None:
            total_steps = (
                self.settle_steps + 4 * self.per_cylinder_steps + self.tail_steps
            )
        settle, tail = self.settle_steps, self.tail_steps
        if total_steps < settle + tail + 8:
            spare = max(total_steps - 8, 0)
            settle = min(settle, spare // 2)
            tail = min(tail, spare - settle)
        active = max(total_steps - settle - tail, 2)
        t = np.arange(active, dtype=np.float64) * self.control_dt
        T = max(float(active) * self.control_dt, 1.0e-9)
        f0, f1 = float(self.f_start_hz), float(self.f_end_hz)
        phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t**2)
        action = np.zeros((total_steps, 4), dtype=np.float64)
        for c in range(4):
            action[settle : settle + active, c] = float(self.amplitude) * np.sin(
                phase + c * np.pi / 4.0
            )
        segments = [
            Segment("settle", 0, settle),
            Segment("mimo_active", settle, settle + active),
            Segment("tail", total_steps - tail, total_steps),
        ]
        return action.astype(np.float32), segments
