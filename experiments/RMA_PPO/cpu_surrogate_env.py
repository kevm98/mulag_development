"""Small CPU-only tracking environment backed by the frozen Stage-31 SD model.

This module is intentionally a tracking-only surrogate.  The frozen SD model
predicts the 12-D hydraulic state ``[extension, velocity, dP]`` and accepts
the 10-D mechanical context ``q`` conditionally; it does not predict joints,
end-effector pose, terrain, or contact.  PPO actions retain the deployed
contract: a four-cylinder motion command is converted by the frozen ID model
to a valve command before entering SD.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AAM_ROOT = ROOT / "AAM_module"
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime  # noqa: E402
from PPO_ID_AAM_FD.rma_adaptation.core import ActuationConfig, DeploymentActuation  # noqa: E402
from AAM_module.tools.train_sd_z import BeliefEncoder, SDz, causal_windows_np  # noqa: E402


@dataclass(frozen=True)
class TrackingStress:
    """Reachable stress coordinates for this surrogate.

    ``mass_scale`` is deliberately absent: the SD transition is not
    mass-conditioned, so emulating mass through a gain would be invalid.
    """

    actuator_authority_alpha: float = 1.0
    hydraulic_response_time_tau: float = 0.0
    actuator_delay_steps: int = 0

    def validate(self, max_delay_steps: int = 8) -> "TrackingStress":
        if not 0.0 <= float(self.actuator_authority_alpha) <= 1.0:
            raise ValueError("authority must be in [0, 1]")
        if float(self.hydraulic_response_time_tau) < 0.0:
            raise ValueError("lag must be non-negative")
        if int(self.actuator_delay_steps) != self.actuator_delay_steps:
            raise ValueError("delay must be an integer")
        if not 0 <= int(self.actuator_delay_steps) <= int(max_delay_steps):
            raise ValueError("delay exceeds configured FIFO")
        return self

    def as_context4(self) -> torch.Tensor:
        """Return the internal wrapper context; fourth mass coordinate is nominal."""

        self.validate()
        return torch.tensor(
            [self.actuator_authority_alpha,
             self.hydraulic_response_time_tau,
             float(self.actuator_delay_steps),
             1.0], dtype=torch.float32
        )


@dataclass(frozen=True)
class TrackingContextNormalizer:
    """Fixed, explicit normalization for the three exposed stress factors."""

    mean: tuple[float, float, float] = (1.0, 0.08, 1.0)
    std: tuple[float, float, float] = (0.5, 0.08, 1.0)

    def normalize(self, context: torch.Tensor) -> torch.Tensor:
        raw = torch.as_tensor(context, dtype=torch.float32)
        if raw.shape[-1] != 3:
            raise ValueError(f"expected three stress factors, got {tuple(raw.shape)}")
        mean = torch.as_tensor(self.mean, dtype=raw.dtype, device=raw.device)
        std = torch.as_tensor(self.std, dtype=raw.dtype, device=raw.device)
        return (raw - mean) / std


def context_tensor(stress: TrackingStress | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(stress, TrackingStress):
        return torch.tensor(
            [stress.actuator_authority_alpha,
             stress.hydraulic_response_time_tau,
             float(stress.actuator_delay_steps)],
            dtype=torch.float32, device=device,
        )
    value = torch.as_tensor(stress, dtype=torch.float32, device=device)
    if value.shape[-1] != 3:
        raise ValueError("tracking context must have width three")
    return value


class CpuSDTrackingEnv:
    """Vectorized deterministic CPU environment for cylinder tracking.

    Episodes are slices of frozen 50-ms ROM rows.  A batch shares a chosen
    stress per environment, while references and initial states come directly
    from the frozen data.  The environment is not a claim of equivalence to
    PhysX and contains no contact or full task reward.
    """

    state_dim = 12
    action_dim = 4
    base_obs_dim = 66
    context_dim = 3
    dt_s = 0.05
    history_lag = 6

    def __init__(
        self,
        *,
        num_envs: int = 16,
        run_ids: Sequence[int] | None = None,
        horizon: int = 64,
        seed: int = 0,
        device: str | torch.device = "cpu",
        model_dir: str | Path = ROOT / "outputs/stage31_rom/sd_none",
        dataset_root: str | Path = ROOT / "AAM_module/datasets/rom_deployment_stage31_v1_frozen",
        id_checkpoint: str | Path = ROOT / "id_fd_model/best_id_cylinder_control.pt",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("CpuSDTrackingEnv is CPU-only")
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        if self.num_envs <= 0 or self.horizon <= 1:
            raise ValueError("num_envs must be positive and horizon must exceed one")
        self.rng = np.random.default_rng(int(seed))
        self.normalizer = TrackingContextNormalizer()
        self.model_dir = Path(model_dir)
        self.dataset_root = Path(dataset_root)
        self.sd_root = self.dataset_root / "sd" / "dataset"
        self.row_root = self.dataset_root / "aam" / "rom_rows"
        self.run_ids = [int(x) for x in (run_ids if run_ids is not None else range(118))]
        if not self.run_ids:
            raise ValueError("at least one trajectory run is required")
        self.runs = self._load_runs(self.run_ids)

        metrics = json.loads((self.model_dir / "metrics.json").read_text())
        if metrics.get("latent_source") != "none":
            raise ValueError("CpuSDTrackingEnv expects the latent-free sd_none checkpoint")
        if not bool(metrics.get("use_q", False)) or not bool(metrics.get("use_belief", False)):
            raise ValueError("surrogate contract requires q conditioning and causal belief")
        if int(metrics.get("history_lag", 0)) != self.history_lag:
            raise ValueError("unexpected SD history lag")
        self._s_mean = torch.tensor(metrics["s_stats"]["mean"], dtype=torch.float32)
        self._s_std = torch.tensor(metrics["s_stats"]["std"], dtype=torch.float32)
        self._u_mean = torch.tensor(metrics["u_stats"]["mean"], dtype=torch.float32)
        self._u_std = torch.tensor(metrics["u_stats"]["std"], dtype=torch.float32)
        self._q_mean = torch.tensor(metrics["q_stats"]["mean"], dtype=torch.float32)
        self._q_std = torch.tensor(metrics["q_stats"]["std"], dtype=torch.float32)
        self._y_mean = torch.tensor(metrics["y_stats"]["mean"], dtype=torch.float32)
        self._y_std = torch.tensor(metrics["y_stats"]["std"], dtype=torch.float32)

        self.belief = BeliefEncoder(self.state_dim + 8, int(metrics["belief_dim"])).to(self.device)
        self.sd = SDz(0, state_dim=self.state_dim, hidden=256, depth=3,
                      dropout=0.05, use_q=True, use_belief=True,
                      belief_dim=int(metrics["belief_dim"])).to(self.device)
        self.belief.load_state_dict(torch.load(self.model_dir / "belief.pt", map_location="cpu", weights_only=False))
        self.sd.load_state_dict(torch.load(self.model_dir / "sd.pt", map_location="cpu", weights_only=False))
        self.belief.eval()
        self.sd.eval()
        self.id_runtime = CylinderIDRuntime(
            models_root=ROOT / "id_fd_model/vendor_models",
            checkpoint=Path(id_checkpoint), num_envs=self.num_envs, device=self.device,
        )
        self.actuation = DeploymentActuation(
            ActuationConfig(dt_s=self.dt_s, action_dim=4, max_delay_steps=8),
            device=self.device,
        )

        self.s = torch.zeros(self.num_envs, self.state_dim)
        self.q = torch.zeros(self.num_envs, 10)
        self.prev_valve = torch.zeros(self.num_envs, 4)
        self.prev_action = torch.zeros(self.num_envs, 4)
        self.belief_state = torch.zeros(self.num_envs, int(metrics["belief_dim"]))
        self.ref_x = torch.zeros(self.num_envs, self.horizon + 1, 4)
        self.ref_v = torch.zeros(self.num_envs, self.horizon + 1, 4)
        self.run_index = torch.zeros(self.num_envs, dtype=torch.long)
        self.row_index = torch.zeros(self.num_envs, dtype=torch.long)
        self.t = torch.zeros(self.num_envs, dtype=torch.long)
        self.context = torch.zeros(self.num_envs, 3)
        self.last_predicted_delta = torch.zeros(self.num_envs, 4)

    def _load_run_tables(self, run_ids: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
        out: dict[int, dict[str, np.ndarray]] = {}
        for rid in run_ids:
            sd_path = self.sd_root / f"{rid:05d}" / "samples.npz"
            row_path = self.row_root / f"{rid:05d}" / "rom_rows.npz"
            if not sd_path.is_file() or not row_path.is_file():
                continue
            with np.load(sd_path) as sd, np.load(row_path) as rows:
                n = min(len(sd["s_h"]), len(rows["desired_cylinder_position"]))
                if n <= self.history_lag + self.horizon:
                    continue
                out[rid] = {
                    "s": sd["s_h"][:n].astype(np.float32),
                    "q": sd["q"][:n].astype(np.float32),
                    "u": sd["u"][:n].astype(np.float32),
                    "x_ref": rows["desired_cylinder_position"][:n].astype(np.float32),
                    "v_ref": rows["desired_cylinder_velocity"][:n].astype(np.float32),
                }
        if not out:
            raise FileNotFoundError("no usable frozen SD/ROM reference runs")
        return out

    def _load_runs(self, run_ids: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
        return self._load_run_tables(run_ids)

    def reset(self, *, contexts: torch.Tensor | None = None, run_ids: Sequence[int] | None = None) -> torch.Tensor:
        choices = [int(x) for x in (run_ids if run_ids is not None else self.run_ids) if int(x) in self.runs]
        if not choices:
            choices = sorted(self.runs)
        selected = self.rng.choice(np.asarray(choices), size=self.num_envs, replace=True)
        self.run_index = torch.as_tensor(selected, dtype=torch.long)
        starts = np.asarray([
            int(self.rng.integers(self.history_lag - 1, len(self.runs[int(rid)]["s"]) - self.horizon - 1))
            for rid in selected
        ], dtype=np.int64)
        self.row_index = torch.as_tensor(starts, dtype=torch.long)
        self.t.zero_()
        self.prev_action.zero_()
        self.last_predicted_delta.zero_()
        for i, rid in enumerate(selected):
            run = self.runs[int(rid)]
            st = int(starts[i])
            self.s[i] = torch.from_numpy(run["s"][st])
            self.q[i] = torch.from_numpy(run["q"][st])
            self.prev_valve[i] = torch.from_numpy(run["u"][st, :4])
            self.ref_x[i] = torch.from_numpy(run["x_ref"][st:st + self.horizon + 1])
            self.ref_v[i] = torch.from_numpy(run["v_ref"][st:st + self.horizon + 1])
            # Initialize the causal GRU exactly from the recorded six-row
            # window; subsequent blocks use the surrogate's own state.
            blocks = np.concatenate([
                (run["s"][st - self.history_lag + 1:st + 1] - self._s_mean.numpy()) / self._s_std.numpy(),
                (run["u"][st - self.history_lag + 1:st + 1] - self._u_mean.numpy()) / self._u_std.numpy(),
            ], axis=-1)
            with torch.no_grad():
                h0 = self.belief.init_from_window(torch.from_numpy(blocks[None]).float())[0]
            self.belief_state[i].copy_(h0)
        if contexts is None:
            self.context.fill_(1.0)
            self.context[:, 1] = 0.0
        else:
            value = context_tensor(contexts, device=self.device)
            if value.ndim == 1:
                value = value[None].expand(self.num_envs, -1)
            if value.shape != (self.num_envs, 3):
                raise ValueError(f"contexts must have shape ({self.num_envs}, 3)")
            self.context = value.detach().clone()
        self.actuation.reset(self.num_envs)
        self.id_runtime.reset()
        return self.observation()

    def set_context(self, contexts: torch.Tensor) -> None:
        value = context_tensor(contexts, device=self.device)
        if value.ndim == 1:
            value = value[None].expand(self.num_envs, -1)
        if value.shape != (self.num_envs, 3):
            raise ValueError(f"contexts must have shape ({self.num_envs}, 3)")
        self.context = value.detach().clone()

    def _id_valve(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(torch.nan_to_num(action.float()), -1.0, 1.0)
        # The validated coordinate mapping has unit magnitude and negative
        # sign for all four USD cylinder axes.  Delta commands therefore use
        # the exact learned-space sign without inventing an absolute offset.
        dx = action * 0.004 * torch.tensor([-1.0, -1.0, -1.0, -1.0])
        dxv = dx / self.dt_s - self.s[:, 4:8]
        id_input = torch.cat((self.s[:, :4], self.s[:, 4:8], self.s[:, 8:12],
                              torch.zeros_like(self.s[:, :4]), dx, dxv,
                              self.prev_valve), dim=-1)
        return self.id_runtime.predict(id_input, prev_valve_cmd=self.prev_valve)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if action.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"action must have shape {(self.num_envs, self.action_dim)}")
        commanded_valve = self._id_valve(action)
        applied_valve = self.actuation.step(
            commanded_valve,
            torch.cat((self.context, torch.ones(self.num_envs, 1)), dim=-1),
        )
        u = torch.cat((self.prev_valve, applied_valve), dim=-1)
        s_n = (self.s - self._s_mean) / self._s_std
        u_n = (u - self._u_mean) / self._u_std
        q_n = (self.q - self._q_mean) / self._q_std
        self.belief_state = self.belief.step(torch.cat((s_n, u_n), dim=-1), self.belief_state)
        pred = self.sd(torch.cat((s_n, u_n, q_n, self.belief_state), dim=-1))
        delta = pred * self._y_std + self._y_mean
        self.s = torch.nan_to_num(self.s + delta, nan=0.0, posinf=0.0, neginf=0.0)
        self.prev_valve = applied_valve.detach().clone()
        self.prev_action = torch.clamp(action.detach(), -1.0, 1.0)
        self.last_predicted_delta = delta[:, :4].detach().clone()
        idx = torch.arange(self.num_envs)
        self.t += 1
        current_ref = self.ref_x[idx, self.t]
        current_v = self.ref_v[idx, self.t]
        ext_err = (self.s[:, :4] - current_ref) / 0.02
        vel_err = (self.s[:, 4:8] - current_v) / 0.10
        reward = -(ext_err.square().mean(dim=-1) + 0.25 * vel_err.square().mean(dim=-1)
                   + 0.01 * self.prev_action.square().mean(dim=-1))
        done = self.t >= self.horizon
        # Pressure is stored in Pa (the frozen data clips at 150 MPa); use a
        # physical guard rather than a unitless 1e4 threshold that would mark
        # every valid pressure sample as an episode failure.
        invalid = (~torch.isfinite(self.s).all(dim=-1)) | (self.s[:, 8:12].abs().amax(dim=-1) > 2.0e8)
        done |= invalid
        info = {
            "tracking_extension_rmse": torch.sqrt((self.s[:, :4] - current_ref).square().mean(dim=-1)),
            "tracking_velocity_rmse": torch.sqrt((self.s[:, 4:8] - current_v).square().mean(dim=-1)),
            "control_effort": self.prev_action.abs().mean(dim=-1),
            "invalid": invalid,
        }
        return self.observation(), reward, done, info

    def observation(self) -> torch.Tensor:
        """Build the frozen 66-D PPO observation contract as a tracking proxy."""

        obs = torch.zeros(self.num_envs, self.base_obs_dim)
        idx = torch.arange(self.num_envs)
        ref = self.ref_x[idx, torch.clamp(self.t, max=self.horizon)]
        ref_v = self.ref_v[idx, torch.clamp(self.t, max=self.horizon)]
        obs[:, 12:16] = self.s[:, :4]
        obs[:, 16:20] = self.s[:, 4:8]
        obs[:, 20:24] = 0.10
        obs[:, 24:28] = 0.10
        obs[:, 28:32] = self.prev_action
        obs[:, 32:36] = self.prev_valve
        obs[:, 36:40] = self.last_predicted_delta
        obs[:, 40:44] = ref - self.s[:, :4]
        obs[:, 44] = 1.0
        obs[:, 45] = 0.0
        obs[:, 46] = 1.0
        obs[:, 47] = 1.0
        obs[:, 48] = 0.0
        obs[:, 49] = 1.0
        # The 12 model-error channels are not available before the first
        # surrogate step; leaving them zero is causal and finite.
        del ref_v
        return obs

    def normalized_context(self, contexts: torch.Tensor | None = None) -> torch.Tensor:
        value = self.context if contexts is None else context_tensor(contexts, device=self.device)
        return self.normalizer.normalize(value)

    def nominal_step_difference(self, action: torch.Tensor) -> float:
        """Return a pure-wrapper identity diagnostic for a supplied command."""

        raw = torch.as_tensor(action, dtype=torch.float32)
        nominal = DeploymentActuation(ActuationConfig(dt_s=self.dt_s, action_dim=4, max_delay_steps=8))
        nominal.reset(raw.shape[0])
        out = nominal.step(raw, torch.tensor([1.0, 0.0, 0.0, 1.0]))
        return float((out - raw).abs().max())
