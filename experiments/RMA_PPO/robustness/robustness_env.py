"""CPU surrogate environment for the frozen-PPO robustness envelope.

Differences from ``CpuSDTrackingEnv``, all on the PLANT side:

* the 66-D observation is the real verge contract (``observation.py``) rather
  than arbitrary constants;
* episodes come from a FIXED, pre-declared list of (run, start) pairs, so the
  reference set is bit-identical at every stress level;
* the actuator stress chain adds deadband and rate limit and allows delays far
  beyond the old eight-tick FIFO;
* surrogate-validity diagnostics are recorded per step.

The frozen PPO is never modified, and neither is the reward.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO = Path("/Volumes/DATA/rrlab_mulag")
for extra in (REPO, REPO / "AAM_module"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime  # noqa: E402
from AAM_module.tools.train_sd_z import BeliefEncoder, SDz  # noqa: E402

from observation import VergeObservationAdapter  # noqa: E402
from stress import Stress, StressedActuation  # noqa: E402

MODEL_DIR = REPO / "outputs/stage31_rom/sd_none"
DATASET_ROOT = REPO / "AAM_module/datasets/rom_deployment_stage31_v1_frozen"
ID_CHECKPOINT = REPO / "id_fd_model/best_id_cylinder_control.pt"

DT_S = 0.05
HISTORY_LAG = 6
MAX_DELAY_STEPS = 40
PRESSURE_LIMIT_PA = 2.0e8


@dataclass(frozen=True)
class Episode:
    """One frozen evaluation episode: a run id and a start row."""

    run_id: int
    start: int


class RobustnessEnv:
    """Deterministic, vectorized CPU cylinder-tracking plant with stress."""

    state_dim = 12
    action_dim = 4
    obs_dim = 66

    def __init__(self, *, episodes: Sequence[Episode], horizon: int = 48,
                 obs_mean: torch.Tensor, run_ids: Sequence[int] | None = None):
        self.horizon = int(horizon)
        self.episodes = list(episodes)
        self.num_envs = len(self.episodes)
        if self.num_envs == 0 or self.horizon <= 1:
            raise ValueError("need at least one episode and a horizon above one")

        metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
        if metrics.get("latent_source") != "none":
            raise ValueError("expected the latent-free sd_none checkpoint")
        if not (metrics.get("use_q") and metrics.get("use_belief")):
            raise ValueError("surrogate contract requires q conditioning and causal belief")
        if int(metrics.get("history_lag", 0)) != HISTORY_LAG:
            raise ValueError("unexpected SD history lag")
        self.metrics = metrics
        for key in ("s", "u", "q", "y"):
            setattr(self, f"_{key}_mean", torch.tensor(metrics[f"{key}_stats"]["mean"], dtype=torch.float32))
            setattr(self, f"_{key}_std", torch.tensor(metrics[f"{key}_stats"]["std"], dtype=torch.float32))

        needed = sorted({e.run_id for e in self.episodes} | set(int(x) for x in (run_ids or ())))
        self.runs = self._load_runs(needed)

        self.belief = BeliefEncoder(self.state_dim + 8, int(metrics["belief_dim"]))
        self.sd = SDz(0, state_dim=self.state_dim, hidden=256, depth=3, dropout=0.05,
                      use_q=True, use_belief=True, belief_dim=int(metrics["belief_dim"]))
        self.belief.load_state_dict(torch.load(MODEL_DIR / "belief.pt", map_location="cpu", weights_only=False))
        self.sd.load_state_dict(torch.load(MODEL_DIR / "sd.pt", map_location="cpu", weights_only=False))
        self.belief.eval()
        self.sd.eval()

        self.id_runtime = CylinderIDRuntime(
            models_root=REPO / "id_fd_model/vendor_models",
            checkpoint=ID_CHECKPOINT, num_envs=self.num_envs, device=torch.device("cpu"),
        )
        self.actuation = StressedActuation(dt_s=DT_S, action_dim=4, max_delay_steps=MAX_DELAY_STEPS)
        self.adapter = VergeObservationAdapter(obs_mean)
        self.stress = Stress()

        n, h = self.num_envs, self.horizon
        self.s = torch.zeros(n, self.state_dim)
        self.q = torch.zeros(n, 10)
        self.prev_valve = torch.zeros(n, 4)
        self.prev_action = torch.zeros(n, 4)
        self.belief_state = torch.zeros(n, int(metrics["belief_dim"]))
        self.ref_x = torch.zeros(n, h + 1, 4)
        self.ref_v = torch.zeros(n, h + 1, 4)
        self.t = torch.zeros(n, dtype=torch.long)
        self.last_delta = torch.zeros(n, 4)
        self.prev_pred_delta = torch.zeros(n, 4)
        self.realized_delta = torch.zeros(n, 4)

    # -- data ---------------------------------------------------------------
    def _load_runs(self, run_ids: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
        sd_root = DATASET_ROOT / "sd" / "dataset"
        row_root = DATASET_ROOT / "aam" / "rom_rows"
        out: dict[int, dict[str, np.ndarray]] = {}
        for rid in run_ids:
            sd_path = sd_root / f"{rid:05d}" / "samples.npz"
            row_path = row_root / f"{rid:05d}" / "rom_rows.npz"
            if not sd_path.is_file() or not row_path.is_file():
                continue
            with np.load(sd_path) as sd, np.load(row_path) as rows:
                n = min(len(sd["s_h"]), len(rows["desired_cylinder_position"]))
                out[int(rid)] = {
                    "s": sd["s_h"][:n].astype(np.float32),
                    "q": sd["q"][:n].astype(np.float32),
                    "u": sd["u"][:n].astype(np.float32),
                    "x_ref": rows["desired_cylinder_position"][:n].astype(np.float32),
                    "v_ref": rows["desired_cylinder_velocity"][:n].astype(np.float32),
                }
        if not out:
            raise FileNotFoundError("no usable frozen SD/ROM reference runs")
        return out

    # -- rollout ------------------------------------------------------------
    def reset(self, stress: Stress = Stress()) -> torch.Tensor:
        """Reset every environment to its FIXED episode under the given stress."""
        self.stress = stress.validate(max_delay_steps=MAX_DELAY_STEPS)
        self.t.zero_()
        self.prev_action.zero_()
        self.last_delta.zero_()
        self.prev_pred_delta.zero_()
        self.realized_delta.zero_()
        for i, ep in enumerate(self.episodes):
            run = self.runs[ep.run_id]
            st = ep.start
            self.s[i] = torch.from_numpy(run["s"][st])
            self.q[i] = torch.from_numpy(run["q"][st])
            self.prev_valve[i] = torch.from_numpy(run["u"][st, :4])
            self.ref_x[i] = torch.from_numpy(run["x_ref"][st:st + self.horizon + 1])
            self.ref_v[i] = torch.from_numpy(run["v_ref"][st:st + self.horizon + 1])
            window = np.concatenate([
                (run["s"][st - HISTORY_LAG + 1:st + 1] - self._s_mean.numpy()) / self._s_std.numpy(),
                (run["u"][st - HISTORY_LAG + 1:st + 1] - self._u_mean.numpy()) / self._u_std.numpy(),
            ], axis=-1)
            with torch.no_grad():
                self.belief_state[i].copy_(self.belief.init_from_window(torch.from_numpy(window[None]).float())[0])
        self.actuation.reset(self.num_envs)
        self.id_runtime.reset()
        return self.observation()

    def _id_valve(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(torch.nan_to_num(action.float()), -1.0, 1.0)
        # Frozen deployed command semantics, unchanged from the earlier
        # surrogate: one PPO unit is a 4 mm cylinder motion increment on the
        # validated negative axis sign.
        dx = action * 0.004 * torch.tensor([-1.0, -1.0, -1.0, -1.0])
        dxv = dx / DT_S - self.s[:, 4:8]
        id_input = torch.cat((self.s[:, :4], self.s[:, 4:8], self.s[:, 8:12],
                              torch.zeros_like(self.s[:, :4]), dx, dxv, self.prev_valve), dim=-1)
        return self.id_runtime.predict(id_input, prev_valve_cmd=self.prev_valve)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        commanded_valve = self._id_valve(action)
        applied_valve = self.actuation.step(commanded_valve, self.stress)

        u = torch.cat((self.prev_valve, applied_valve), dim=-1)
        s_n = (self.s - self._s_mean) / self._s_std
        u_n = (u - self._u_mean) / self._u_std
        q_n = (self.q - self._q_mean) / self._q_std
        self.belief_state = self.belief.step(torch.cat((s_n, u_n), dim=-1), self.belief_state)
        delta = self.sd(torch.cat((s_n, u_n, q_n, self.belief_state), dim=-1)) * self._y_std + self._y_mean

        previous_x = self.s[:, :4].clone()
        self.prev_pred_delta = self.last_delta.clone()
        self.s = torch.nan_to_num(self.s + delta, nan=0.0, posinf=0.0, neginf=0.0)
        self.realized_delta = self.s[:, :4] - previous_x
        self.prev_valve = applied_valve.detach().clone()
        self.prev_action = torch.clamp(action.detach(), -1.0, 1.0)
        self.last_delta = delta[:, :4].detach().clone()

        idx = torch.arange(self.num_envs)
        self.t += 1
        ref = self.ref_x[idx, self.t]
        ref_v = self.ref_v[idx, self.t]
        ext_err = (self.s[:, :4] - ref) / 0.02
        vel_err = (self.s[:, 4:8] - ref_v) / 0.10
        reward = -(ext_err.square().mean(dim=-1) + 0.25 * vel_err.square().mean(dim=-1)
                   + 0.01 * self.prev_action.square().mean(dim=-1))

        nonfinite = ~torch.isfinite(self.s).all(dim=-1)
        over_pressure = self.s[:, 8:12].abs().amax(dim=-1) > PRESSURE_LIMIT_PA
        invalid = nonfinite | over_pressure
        done = (self.t >= self.horizon) | invalid

        info = {
            "extension_rmse": torch.sqrt((self.s[:, :4] - ref).square().mean(dim=-1)),
            "velocity_rmse": torch.sqrt((self.s[:, 4:8] - ref_v).square().mean(dim=-1)),
            "control_effort": self.prev_action.abs().mean(dim=-1),
            "saturated": (self.prev_action.abs() > 0.99).float().mean(dim=-1),
            "nonfinite": nonfinite,
            "over_pressure": over_pressure,
            "invalid": invalid,
            "sd_state_z": ((self.s - self._s_mean) / self._s_std).abs().amax(dim=-1),
            "sd_action_z": ((u - self._u_mean) / self._u_std).abs().amax(dim=-1),
            "extension": self.s[:, :4].clone(),
            "applied_valve": applied_valve.clone(),
            "commanded_valve": commanded_valve.clone(),
        }
        return self.observation(), reward, done, info

    def observation(self) -> torch.Tensor:
        idx = torch.arange(self.num_envs)
        ref = self.ref_x[idx, torch.clamp(self.t, max=self.horizon)]
        return self.adapter.build(
            extension=self.s[:, :4],
            velocity=self.s[:, 4:8],
            prev_action=self.prev_action,
            applied_valve=self.prev_valve,
            rom_delta=self.last_delta,
            target_lead=ref - self.s[:, :4],
            prediction_residual=self.realized_delta - self.prev_pred_delta,
        )
