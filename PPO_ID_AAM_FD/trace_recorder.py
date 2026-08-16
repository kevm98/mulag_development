"""Per-timestep trace recording for the AAM online-adaptation experiments.

The existing rollout reports aggregate into distance bins. The finest bin is
3.25 m, while the AAM's 1.0 s warmup spans ~0.29 m of travel -- so every
transition worth showing is averaged away before it reaches disk.

The three adaptation figures all need the same time-resolved signals:

  1. warmup transition   arms coincide until the AAM becomes valid at ~1.0 s
  2. step change         z_hat moves -> error spikes -> error recovers
  3. lag vs window       response lag should scale with the history window

so this records them once, per control step, and writes a single npz.

`theta_*` columns are the TRUE plant parameters. They are recorded for
ANALYSIS ONLY and must never be routed into the policy, the ID model or the
AAM input -- the premise of the experiment is that the controller does not
receive theta.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TraceRecorder:
    """Accumulates per-step rows in memory and writes one npz at the end.

    Rows are appended on every call to :meth:`record`. Storage is
    ``max_steps x num_envs x width`` per field; at 32 envs and 2808 steps a
    scalar field is ~0.36 MB, so a dozen fields stay well inside memory.
    """

    def __init__(self, *, num_envs: int, out_path: str | Path,
                 max_steps: int = 20000) -> None:
        self.num_envs = int(num_envs)
        self.out_path = Path(out_path)
        self.max_steps = int(max_steps)
        self._cols: dict[str, list[np.ndarray]] = {}
        self._steps = 0
        self._truncated = False
        self._saved = False

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def truncated(self) -> bool:
        """True if recording stopped early because max_steps was reached."""
        return self._truncated

    @staticmethod
    def _to_np(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().to("cpu", dtype=torch.float32).numpy()
        arr = np.asarray(value, dtype=np.float32)
        return arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)

    def record(self, **fields) -> None:
        """Append one row. Field set must stay constant across calls."""
        if self._steps >= self.max_steps:
            self._truncated = True
            return
        for name, value in fields.items():
            arr = self._to_np(value)
            if arr.shape[0] != self.num_envs:
                raise ValueError(
                    f"trace field {name!r} has {arr.shape[0]} rows, "
                    f"expected num_envs={self.num_envs}"
                )
            if self._steps and name not in self._cols:
                raise ValueError(
                    f"trace field {name!r} appeared at step {self._steps}; "
                    "the field set must be identical on every record() call"
                )
            self._cols.setdefault(name, []).append(arr)
        if self._steps:
            missing = set(self._cols) - set(fields)
            if missing:
                raise ValueError(
                    f"trace fields {sorted(missing)} missing at step {self._steps}; "
                    "the field set must be identical on every record() call"
                )
        self._steps += 1

    def save(self, **metadata) -> Path | None:
        """Write the npz. Returns the path, or None if nothing was recorded."""
        if not self._steps:
            return None
        payload = {
            name: np.stack(rows, axis=0)      # (steps, num_envs, width)
            for name, rows in self._cols.items()
        }
        payload["_num_envs"] = np.asarray(self.num_envs)
        payload["_steps"] = np.asarray(self._steps)
        payload["_truncated"] = np.asarray(self._truncated)
        for key, value in metadata.items():
            payload[f"meta_{key}"] = np.asarray(value)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.out_path, **payload)
        self._saved = True
        return self.out_path
