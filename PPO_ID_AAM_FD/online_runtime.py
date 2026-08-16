"""Online causal AAM + frozen-ID residual adapter runtime.

This module contains only deployment-time state.  The historical ID and FD
models are loaded by the existing cylinder runtime; this class maintains the
causal AAM window and transforms ``u_base`` into a bounded ``u_final``.  AAM
failures never replace the base action: they select the frozen-ID fallback.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AAM_ROOT = ROOT / "AAM_module"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from tools.train_stage31_aam import AAM  # noqa: E402
from tools.train_stage29_latent_adapter import ResidualAdapter  # noqa: E402
from tools.train_sd_z import FactorEncoder  # noqa: E402


class OnlineAAMIDFDController:
    """Causal ROM-history AAM and Stage-31 adapter around frozen ID.

    ``observe`` is called once per control step. With ``update_period_ticks=3``
    it performs AAM inference every three 60 Hz control steps, matching the
    20 Hz ROM grid. The input row is the causal 18-D deployment observable:

        [joint_q(5), joint_qdot(5), dP(4), previous_valve(4)]
    The first valid AAM estimate appears after ``history_lag`` 50 ms rows
    (20 rows for the promoted 1 s model). Until then the adapter is bypassed
    and the frozen ID action is returned unchanged.
    """

    FEATURE_DIM = 18

    def __init__(
        self,
        *,
        aam_checkpoint: str | Path,
        adapter_checkpoint: str | Path,
        true_latent_checkpoint: str | Path | None = None,
        num_envs: int,
        device: str | torch.device,
        history_lag: int = 20,
        update_period_ticks: int = 3,
        control_rate_hz: float = 60.0,
        max_inference_latency_ms: float = 50.0,
        residual_scale: float = 1.0,
        smoother_window: int = 5,
        enabled: bool = True,
        latent_mode: str = "estimated",
        latent_freeze_tick: int = 0,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        self.num_envs = int(num_envs)
        self.history_lag = int(history_lag)
        self.update_period_ticks = max(int(update_period_ticks), 1)
        self.control_rate_hz = float(control_rate_hz)
        self.max_inference_latency_ms = float(max_inference_latency_ms)
        self.residual_scale = float(residual_scale)
        self.smoother_window = max(int(smoother_window), 1)
        if self.residual_scale < 0.0 or self.residual_scale > 1.0:
            raise ValueError("residual_scale must be in [0, 1]")
        self.enabled = bool(enabled)
        self.latent_mode = str(latent_mode).strip().lower()
        _LATENT_MODES = (
            'estimated', 'true', 'zero', 'mean', 'shuffled', 'frozen'
        )
        if self.latent_mode not in _LATENT_MODES:
            raise ValueError(
                f"latent_mode must be one of {list(_LATENT_MODES)}, "
                f"got {self.latent_mode!r}"
            )
        # `frozen` latches the estimate once, at the configured control tick,
        # and holds it for the rest of the episode.  It is the control for an
        # ONLINE adaptation claim: it supplies a latent of the same magnitude
        # and content as `estimated`, differing only in that it stops
        # updating.  If `estimated` recovers from a mid-episode plant change
        # and `frozen` does not, the recovery came from online estimation
        # rather than from merely having the extra input dimensions.
        self.latent_freeze_tick = int(latent_freeze_tick)
        if self.latent_mode == "frozen" and self.latent_freeze_tick < 1:
            raise ValueError(
                "latent_mode='frozen' requires latent_freeze_tick >= 1 "
                "(the control tick at which the estimate is latched)"
            )
        # The history length is no longer pinned to 5. It is still validated
        # against the checkpoint below (lag_ticks and in_dim must agree), so a
        # mismatched pair fails loudly rather than silently reading the wrong
        # window. L5 is 0.25 s, which is short against the literature this
        # design follows: RMA uses 0.5 s and A-RMA 2.3 s. Measured on the frozen
        # ROM collection, L20 (1.0 s) improves held-out estimator error over
        # L5 and clears the Stage-31 SD/AAM dynamics gate.
        if self.history_lag < 1:
            raise ValueError(f"history_lag must be >= 1, got {self.history_lag}")

        aam_path = Path(aam_checkpoint).expanduser().resolve()
        self.aam_checkpoint_path = aam_path / "aam.pt" if aam_path.is_dir() else aam_path
        self.aam_model_dir = self.aam_checkpoint_path.parent
        self.adapter_checkpoint_path = Path(adapter_checkpoint).expanduser().resolve()
        if not self.aam_checkpoint_path.is_file():
            raise FileNotFoundError(self.aam_checkpoint_path)
        if not self.adapter_checkpoint_path.is_file():
            raise FileNotFoundError(self.adapter_checkpoint_path)

        self.true_latent_model_dir: Path | None = None
        self.true_latent_encoder: FactorEncoder | None = None
        self.true_factor_mean: torch.Tensor | None = None
        self.true_factor_std: torch.Tensor | None = None
        if self.latent_mode == "true" and true_latent_checkpoint is None:
            raise ValueError("latent_mode='true' requires true_latent_checkpoint")
        # The frozen encoder may also be loaded in estimated mode to provide
        # training-only teacher labels.  Loading it does not publish true z to
        # the controller or policy; only ``set_privileged_factors`` does that,
        # and that method remains restricted to latent_mode='true'.
        if true_latent_checkpoint is not None:
            true_path = Path(true_latent_checkpoint).expanduser().resolve()
            self.true_latent_model_dir = true_path.parent if true_path.is_file() else true_path
            encoder_path = self.true_latent_model_dir / "encoder.pt"
            metrics_path = self.true_latent_model_dir / "metrics.json"
            if not encoder_path.is_file() or not metrics_path.is_file():
                raise FileNotFoundError(
                    f"incomplete privileged latent model: {self.true_latent_model_dir}"
                )
            true_metrics = json.loads(metrics_path.read_text())
            if true_metrics.get("latent_source") != "encoder" or int(true_metrics.get("latent_dim", -1)) != 2:
                raise ValueError("privileged latent model must be the frozen 6-D factor -> 2-D encoder")
            normalization = true_metrics["e_normalization"]
            self.true_factor_mean = torch.as_tensor(
                normalization["mean"], dtype=torch.float32, device=self.device
            ).reshape(1, 6)
            self.true_factor_std = torch.as_tensor(
                normalization["std"], dtype=torch.float32, device=self.device
            ).clamp_min(1.0e-8).reshape(1, 6)
            self.true_latent_encoder = FactorEncoder(2).to(self.device)
            self.true_latent_encoder.load_state_dict(
                torch.load(encoder_path, map_location=self.device, weights_only=False)
            )
            self.true_latent_encoder.eval().requires_grad_(False)

        metrics = json.loads((self.aam_model_dir / "metrics.json").read_text())
        if int(metrics["lag_ticks"]) != self.history_lag or int(metrics["in_dim"]) != self.FEATURE_DIM * self.history_lag:
            raise ValueError(
                "AAM checkpoint does not match the causal deployment contract: "
                f"configured history_lag={self.history_lag} expects in_dim="
                f"{self.FEATURE_DIM * self.history_lag}, checkpoint reports "
                f"lag_ticks={metrics['lag_ticks']} in_dim={metrics['in_dim']}"
            )
        if int(metrics.get("feature_dim", self.FEATURE_DIM)) != self.FEATURE_DIM:
            raise ValueError("AAM checkpoint has the wrong Stage 31 observable feature dimension")
        if int(metrics.get("latent_dim", 2)) != 2:
            raise ValueError("Stage 31 online ID adapter requires a 2-D AAM latent")
        norm = metrics["input_normalization"]
        self.aam_mean = torch.as_tensor(norm["per_position_mean"], dtype=torch.float32, device=self.device).reshape(1, -1)
        self.aam_std = torch.as_tensor(norm["per_position_std"], dtype=torch.float32, device=self.device).clamp_min(1.0e-8).reshape(1, -1)
        self.aam_architecture = str(metrics.get("architecture", "mlp"))
        self.aam = AAM(
            int(metrics["in_dim"]), int(metrics["latent_dim"]),
            hidden=int(metrics.get("hidden", 256)),
            depth=int(metrics.get("depth", 3)),
            dropout=float(metrics.get("dropout", 0.05)),
            architecture=self.aam_architecture,
            feature_dim=int(metrics.get("feature_dim", self.FEATURE_DIM)),
            lag=int(metrics["lag_ticks"]),
        ).to(self.device)
        self.aam.load_state_dict(torch.load(self.aam_checkpoint_path, map_location=self.device, weights_only=False))
        self.aam.eval().requires_grad_(False)
        # Pay the CUDA kernel/setup cost before the first deployment tick.  A
        # cold first inference otherwise trips the 50 ms fail-safe during the
        # the long-history warm-up and hides the first valid causal estimate behind a
        # needless fallback.
        with torch.no_grad():
            warmup_input = torch.zeros(
                (self.num_envs, self.FEATURE_DIM * self.history_lag),
                dtype=torch.float32,
                device=self.device,
            )
            _ = self.aam((warmup_input - self.aam_mean) / self.aam_std)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        adapter_payload: dict[str, Any] = torch.load(self.adapter_checkpoint_path, map_location=self.device, weights_only=False)
        if adapter_payload.get("format") != "stage29_latent_conditioned_residual_adapter_v1":
            raise ValueError("ID adapter checkpoint has an unsupported format")
        expected_latent_source = "true" if self.latent_mode == "true" else "zhat"
        if adapter_payload.get("latent_source") != expected_latent_source:
            raise ValueError(
                f"latent_mode={self.latent_mode!r} requires an adapter trained "
                f"with latent_source={expected_latent_source!r}; checkpoint reports "
                f"{adapter_payload.get('latent_source')!r}"
            )
        adapter_lag = adapter_payload.get("latent_lag_ticks")
        if adapter_lag is None:
            raise ValueError(
                "ID adapter checkpoint does not declare latent_lag_ticks; "
                "legacy adapters cannot satisfy the online causal contract"
            )
        if int(adapter_lag) != self.history_lag:
            raise ValueError(
                "ID adapter does not match the AAM history contract: "
                f"configured history_lag={self.history_lag}, "
                f"adapter reports latent_lag_ticks={int(adapter_lag)}"
            )
        adapter_smoothing = adapter_payload.get("smoothing_window_predictions")
        if adapter_smoothing is None:
            raise ValueError(
                "ID adapter checkpoint does not declare "
                "smoothing_window_predictions; legacy adapters cannot satisfy "
                "the online causal contract"
            )
        if int(adapter_smoothing) != self.smoother_window:
            raise ValueError(
                "ID adapter does not match the causal latent smoothing contract: "
                f"configured smoother_window={self.smoother_window}, "
                f"adapter reports smoothing_window_predictions={int(adapter_smoothing)}"
            )
        config = adapter_payload["model_config"]
        if int(config["in_dim"]) != 30 or int(config["out_dim"]) != 4:
            raise ValueError("Stage-31 adapter checkpoint has an incompatible feature contract")
        self.adapter = ResidualAdapter(int(config["in_dim"]), int(config["hidden"]), int(config["depth"]), int(config["out_dim"])).to(self.device)
        self.adapter.load_state_dict(adapter_payload["model_state"])
        self.adapter.eval().requires_grad_(False)
        self.adapter_feature_mean = adapter_payload["feature_mean"].to(self.device, dtype=torch.float32)
        self.adapter_feature_std = adapter_payload["feature_std"].to(self.device, dtype=torch.float32).clamp_min(1.0e-8)
        self.residual_bound = adapter_payload["residual_bound"].to(self.device, dtype=torch.float32).reshape(1, 4) * self.residual_scale
        self.command_limits = adapter_payload["command_limits"].to(self.device, dtype=torch.float32).reshape(1, 4)

        self.history = torch.zeros((self.num_envs, self.history_lag, self.FEATURE_DIM), device=self.device)
        self.valid_rows = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.control_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # `z_hat` is the RAW smoothed estimate; `z_published` is what the
        # controller consumes after the ablation mode is applied. They were one
        # tensor until the routing ablation needed them apart: with a single
        # tensor, setting the CONTROLLER arm to `zero` also zeroed what the
        # POLICY could see, so "PPO adaptation only" was inexpressible and
        # collapsed silently onto the baseline arm.
        self.z_hat = torch.zeros((self.num_envs, 2), device=self.device)
        self.z_published = torch.zeros((self.num_envs, 2), device=self.device)
        self.true_z = torch.zeros((self.num_envs, 2), device=self.device)
        # latch buffers for latent_mode='frozen'
        self.z_frozen = torch.zeros((self.num_envs, 2), device=self.device)
        self.z_frozen_valid = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.true_z_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.z_prediction_history = torch.zeros((self.num_envs, self.smoother_window, 2), device=self.device)
        self.z_prediction_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.aam_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fallback_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_update_tick = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.last_latency_ms = 0.0
        self.max_latency_ms_observed = 0.0
        self._pending_latency_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        self._latency_fault = False
        self._aam_update_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._fallback_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._invalid_input_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.reset_count = 0

    @property
    def aam_update_count(self) -> int:
        return int(self._aam_update_count.detach().cpu())

    @property
    def fallback_count(self) -> int:
        return int(self._fallback_count.detach().cpu())

    @property
    def invalid_input_count(self) -> int:
        return int(self._invalid_input_count.detach().cpu())

    def _consume_cuda_latency(self, *, block: bool = False) -> None:
        """Consume completed CUDA timing events without synchronizing the hot path."""
        pending = self._pending_latency_events
        if pending is None:
            return
        start, end = pending
        if block:
            end.synchronize()
        elif not end.query():
            return
        latency_ms = float(start.elapsed_time(end))
        self.last_latency_ms = latency_ms
        self.max_latency_ms_observed = max(self.max_latency_ms_observed, latency_ms)
        self._latency_fault = (
            self.max_inference_latency_ms > 0.0
            and latency_ms > self.max_inference_latency_ms
        )
        self._pending_latency_events = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        self.history[ids] = 0.0
        self.valid_rows[ids] = 0
        self.control_ticks[ids] = 0
        if self.latent_mode == "true":
            self.z_hat[ids] = self.true_z[ids]
        else:
            self.z_hat[ids] = 0.0
        self.z_published[ids] = self.z_hat[ids]
        self.z_prediction_history[ids] = 0.0
        self.z_prediction_count[ids] = 0
        # a reset starts a new episode, so the latch must be re-armed
        self.z_frozen[ids] = 0.0
        self.z_frozen_valid[ids] = False
        if self.latent_mode == "true":
            self.aam_ready[ids] = self.enabled & self.true_z_valid[ids]
        else:
            self.aam_ready[ids] = False
        self.fallback_active[ids] = False
        self.last_update_tick[ids] = -1
        self.reset_count += int(ids.numel())

    @torch.no_grad()
    def encode_privileged_factors(self, factors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode factor labels without changing the deployed latent route."""
        if self.true_latent_encoder is None or self.true_factor_mean is None or self.true_factor_std is None:
            raise RuntimeError("privileged latent encoder is not initialized")
        values = factors.to(device=self.device, dtype=torch.float32)
        if values.shape != (self.num_envs, 6):
            raise ValueError(
                f"privileged factors must have shape {(self.num_envs, 6)}, got {tuple(values.shape)}"
            )
        finite = torch.isfinite(values).all(dim=-1)
        safe = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        encoded = self.true_latent_encoder(
            (safe - self.true_factor_mean) / self.true_factor_std
        )
        finite &= torch.isfinite(encoded).all(dim=-1)
        encoded = torch.where(
            finite.reshape(-1, 1), encoded, torch.zeros_like(encoded)
        )
        return encoded, finite

    @torch.no_grad()
    def set_privileged_factors(self, factors: torch.Tensor) -> torch.Tensor:
        """Publish frozen-SD ``z(e)`` for privileged PPO warm-start only."""
        if self.latent_mode != "true":
            raise RuntimeError("privileged factors are accepted only in latent_mode='true'")
        encoded, finite = self.encode_privileged_factors(factors)
        self.true_z = torch.where(finite.reshape(-1, 1), encoded, self.true_z)
        self.true_z_valid = finite
        self.z_hat = torch.where(finite.reshape(-1, 1), self.true_z, torch.zeros_like(self.true_z))
        # In latent_mode='true' the AAM is not run, so there is no live
        # estimate to hold apart from what is published.
        self.z_published = self.z_hat
        self.aam_ready = self.enabled & finite
        self.fallback_active = ~finite
        return self.z_published

    @torch.no_grad()
    def observe(
        self,
        feature_row: torch.Tensor,
        *,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append an observable row on the model tick and return ``z_hat``.

        The AAM dataset is built on the 50 ms ROM grid, not the 60 Hz PhysX
        grid. ``sample_mask`` lets the live environment append only the
        control steps on which the ROM/ID-FD model is sampled. The old
        no-mask behavior is retained for standalone 60 Hz protocol tests.
        """
        row = torch.nan_to_num(feature_row.to(self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if row.shape != (self.num_envs, self.FEATURE_DIM):
            raise ValueError(f"AAM feature row must be {(self.num_envs, self.FEATURE_DIM)}, got {tuple(row.shape)}")
        sampled_stream = sample_mask is not None
        if sample_mask is None:
            sample_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            sample_mask = sample_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
            if sample_mask.shape != (self.num_envs,):
                raise ValueError(
                    f"sample_mask must have shape {(self.num_envs,)}, got {tuple(sample_mask.shape)}"
                )
        shifted_history = torch.cat((row.unsqueeze(1), self.history[:, :-1]), dim=1)
        self.history = torch.where(
            sample_mask.reshape(-1, 1, 1), shifted_history, self.history
        )
        self.valid_rows = torch.where(
            sample_mask,
            (self.valid_rows + 1).clamp_max(self.history_lag),
            self.valid_rows,
        )
        self.control_ticks = self.control_ticks + sample_mask.to(dtype=torch.long)
        if sampled_stream:
            # One appended row is one 50 ms model tick; the live ROM and the
            # AAM therefore update together. ``update_period_ticks`` remains
            # a 60 Hz compatibility setting for the no-mask protocol path.
            update_mask = sample_mask & (self.valid_rows >= self.history_lag)
        else:
            update_mask = (self.control_ticks % self.update_period_ticks == 0) & (self.valid_rows >= self.history_lag)
        if not self.enabled:
            self.aam_ready.zero_()
            return self.z_published
        if self.latent_mode == "true":
            self.z_hat = self.true_z
            self.z_published = self.true_z
            self.aam_ready = self.true_z_valid
            self.fallback_active = ~self.true_z_valid
            return self.z_published
        if self.device.type == "cuda":
            self._consume_cuda_latency()
            record_latency = self._pending_latency_events is None
            if record_latency:
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
        else:
            cpu_start = time.perf_counter()

        window = self.history.reshape(self.num_envs, -1)
        pred = self.aam((window - self.aam_mean) / self.aam_std)

        if self.device.type == "cuda":
            if record_latency:
                event_end.record()
                self._pending_latency_events = (event_start, event_end)
        else:
            latency_ms = (time.perf_counter() - cpu_start) * 1000.0
            self.last_latency_ms = float(latency_ms)
            self.max_latency_ms_observed = max(
                self.max_latency_ms_observed, float(latency_ms)
            )
            self._latency_fault = (
                self.max_inference_latency_ms > 0.0
                and latency_ms > self.max_inference_latency_ms
            )

        invalid_prediction = ~torch.isfinite(pred).all(dim=-1)
        bad = invalid_prediction | self._latency_fault
        good = update_mask & ~bad
        bad_mask = update_mask & bad

        shifted_predictions = torch.cat(
            (pred.unsqueeze(1), self.z_prediction_history[:, :-1]), dim=1
        )
        self.z_prediction_history = torch.where(
            good.reshape(-1, 1, 1),
            shifted_predictions,
            self.z_prediction_history,
        )
        next_counts = (self.z_prediction_count + 1).clamp_max(
            self.smoother_window
        )
        self.z_prediction_count = torch.where(
            good, next_counts, self.z_prediction_count
        )
        counts = self.z_prediction_count.clamp_min(1).to(
            dtype=torch.float32
        ).reshape(-1, 1)
        next_z_hat = self.z_prediction_history.sum(dim=1) / counts
        self.z_hat = torch.where(good.reshape(-1, 1), next_z_hat, self.z_hat)
        self.z_published = self._apply_latent_mode(self.z_hat)
        self.aam_ready = (self.aam_ready | good) & ~bad_mask
        self.fallback_active = (self.fallback_active & ~good) | bad_mask
        self.last_update_tick = torch.where(
            good, self.control_ticks, self.last_update_tick
        )
        self._aam_update_count += good.sum(dtype=torch.long)
        self._fallback_count += bad_mask.sum(dtype=torch.long)
        self._invalid_input_count += (bad_mask & invalid_prediction).sum(
            dtype=torch.long
        )
        return self.z_published

    @torch.no_grad()
    def raw_latent(self) -> torch.Tensor:
        """The live estimate BEFORE any controller ablation mode is applied.

        The policy arm of the routing ablation reads this, so that degrading
        the controller latent does not also degrade what the policy sees.
        """
        return self.z_hat

    @torch.no_grad()
    def condition_valve(
        self,
        *,
        u_base: torch.Tensor,
        extension: torch.Tensor,
        velocity: torch.Tensor,
        dP: torch.Tensor,
        force: torch.Tensor,
        target_dx: torch.Tensor,
        target_dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(u_final, residual)``; fallback returns base and zero residual."""
        base = torch.nan_to_num(u_base.to(self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        active = self.enabled & self.aam_ready & ~self.fallback_active
        features = torch.cat((extension, velocity, dP, force, target_dx, target_dv, base, self.z_published), dim=-1)
        finite = torch.isfinite(features).all(dim=-1)
        try:
            normalized = (features - self.adapter_feature_mean) / self.adapter_feature_std
            prediction = self.residual_bound * torch.tanh(self.adapter(normalized))
            finite &= torch.isfinite(prediction).all(dim=-1)
        except Exception:
            finite = torch.zeros_like(active)
            prediction = torch.zeros_like(base)
        usable = active & finite
        fallback = active & ~finite
        residual = torch.where(usable.reshape(-1, 1), prediction, torch.zeros_like(base))
        self.fallback_active |= fallback
        self.aam_ready &= ~fallback
        self._fallback_count += fallback.sum(dtype=torch.long)
        final = torch.clamp(base + residual, -self.command_limits, self.command_limits)
        return final, residual


    @torch.no_grad()
    def _apply_latent_mode(self, z: torch.Tensor) -> torch.Tensor:
        """Degrade the published latent for ablation. `estimated` is a no-op."""
        mode = self.latent_mode
        if mode == "estimated":
            return z
        if mode == "true":
            return self.true_z
        if mode == "zero":
            return torch.zeros_like(z)
        if mode == "mean":
            return z.mean(dim=0, keepdim=True).expand_as(z).contiguous()
        if mode == "shuffled":
            if z.shape[0] < 2:
                return z
            # Derangement where possible: a fixed roll guarantees no environment
            # keeps its own latent, which a random permutation does not.
            return torch.roll(z, shifts=1, dims=0).contiguous()
        if mode == "frozen":
            # Latch the live estimate the first time an environment reaches the
            # freeze tick, then republish that latched value forever.  Before
            # the freeze tick this is identical to `estimated`, so the two arms
            # are matched up to the latch and diverge only afterwards.
            due = (self.control_ticks >= self.latent_freeze_tick) & ~self.z_frozen_valid
            if bool(due.any()):
                self.z_frozen = torch.where(due.reshape(-1, 1), z, self.z_frozen)
                self.z_frozen_valid = self.z_frozen_valid | due
            return torch.where(self.z_frozen_valid.reshape(-1, 1), self.z_frozen, z)
        raise ValueError(f"unhandled latent_mode {mode!r}")

    def diagnostics(self) -> dict[str, Any]:
        if self.device.type == "cuda":
            self._consume_cuda_latency(block=True)
        return {
            "enabled": self.enabled,
            "history_lag": self.history_lag,
            "aam_architecture": self.aam_architecture,
            "update_period_control_ticks": self.update_period_ticks,
            "nominal_update_rate_hz": self.control_rate_hz / self.update_period_ticks,
            "input_grid": "50 ms model ticks when sample_mask is supplied",
            "warmup_rows": self.history_lag,
            "smoother": "causal_rolling_mean",
            "smoother_window_predictions": self.smoother_window,
            "ready_envs": int(self.aam_ready.sum().detach().cpu()),
            "fallback_envs": int(self.fallback_active.sum().detach().cpu()),
            "aam_update_count": self.aam_update_count,
            "fallback_count": self.fallback_count,
            "invalid_input_count": self.invalid_input_count,
            "reset_count": self.reset_count,
            "last_latency_ms": self.last_latency_ms,
            "max_latency_ms_observed": self.max_latency_ms_observed,
            "residual_bound": self.residual_bound.detach().cpu().tolist(),
            "command_limits": self.command_limits.detach().cpu().tolist(),
            "residual_scale": self.residual_scale,
            "latent_mode": self.latent_mode,
            "latent_freeze_tick": self.latent_freeze_tick,
            "latent_frozen_envs": int(self.z_frozen_valid.sum().detach().cpu()),
            "aam_checkpoint": str(self.aam_checkpoint_path),
            "adapter_checkpoint": str(self.adapter_checkpoint_path),
            "true_latent_checkpoint": (
                str(self.true_latent_model_dir) if self.true_latent_model_dir else None
            ),
        }
