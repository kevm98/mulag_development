from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from kevin_integration.controllers import LearnedControlPipeline, PipelineConfig


class ControlPolicy:
    """Learned-model-based control policy for the Kevin integration branch.

    This policy wraps the learned-model pipeline (AAM -> ID -> optional FD prediction).

    Expected `observation` input format:
    - `aam_x_t`: (18,) = [q(5), qdot(5), dP(4), valve_cmd(4)]
    - `id_x_t`:  (19,) = [q(5), qdot(5), qddot(5), dP(4)]
    Optional:
    - `fd_x_t`:  (18,) = [q(5), qdot(5), dP(4), valve_cmd(4)]

    The feature ordering must match whatever you used during model training.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = dict(config or {})

        models_dir = self.config.get("models_dir")
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[1] / "models"
        models_dir = Path(models_dir)

        pipeline_cfg = PipelineConfig(**self.config.get("pipeline", {}))
        self._pipeline = LearnedControlPipeline(models_dir=models_dir, cfg=pipeline_cfg)
        self._return_debug = bool(self.config.get("return_debug", False))

    def reset(self) -> None:
        self._pipeline.reset()

    def compute_action(self, observation: Mapping[str, Any]):
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping with keys: aam_x_t, id_x_t, ...")

        aam_x_t = np.asarray(observation["aam_x_t"], dtype=np.float32).reshape(-1)
        id_x_t = np.asarray(observation["id_x_t"], dtype=np.float32).reshape(-1)

        fd_x_t = observation.get("fd_x_t", None)

        if fd_x_t is not None:
            fd_x_t = np.asarray(fd_x_t, dtype=np.float32).reshape(-1)

        out = self._pipeline.step(aam_x_t=aam_x_t, id_x_t=id_x_t, fd_x_t=fd_x_t)

        return out if self._return_debug else out["valve_cmd"]

    def predict_torque(self, fd_x_t):
        fd_x_t = np.asarray(fd_x_t, dtype=np.float32).reshape(-1)
        return self._pipeline.predict_torque(fd_x_t)
