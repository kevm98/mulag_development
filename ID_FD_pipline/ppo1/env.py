"""Stop-and-mow residual PPO environment.

The deterministic state machine drives arm-only mowing cycles (line, bowed
curve, or sine sweep).  PPO acts as a bounded residual during mowing phases
only — it corrects lateral position, height, and mowing speed.  Zero action
exactly reproduces the deterministic baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from ID_FD_pipline.dls_solver import CylinderDLSController, EECascadeController
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig
from ID_FD_pipline.stop_mow.parking import ParkingBrake
from ID_FD_pipline.stop_mow.phases import StopMowPhase
from ID_FD_pipline.stop_mow.state_machine import StopMowStateMachine
from ID_FD_pipline.stop_mow.reference_manager import ReferenceManager
from ID_FD_pipline.stop_mow.ground_following import GroundFollower
from ID_FD_pipline.trajectories import (
    arc_mowing_reference,
    line_reference,
    bowed_curve_reference,
    sine_sweep_reference,
    TrajectorySampler,
)
from ID_FD_pipline.tasks.cylinder_env import IDFDCylinderSpaceEnv
from ID_FD_pipline.ppo1.observations import build_residual_observation
from ID_FD_pipline.ppo1.rewards import compute_residual_reward
from plant_study.direct_drive import DirectCylinderDriver

_JACOBIAN_CACHE = Path(__file__).resolve().parents[1] / "models/ee_jacobian_ground_truck3x4.json"


class IDFDEEResidualEnv(IDFDCylinderSpaceEnv):
    """Bounded residual PPO over the deterministic stop-and-mow controller.

    Action (3):
      - dy: lateral correction on nominal goal [m]  (±residual_action_scale_m)
      - dz: height correction [m]                     (±residual_action_scale_m)
      - speed_scale: mowing speed multiplier           [0.8, 1.2]

    PPO is active ONLY during MOW_OUT and MOW_RETURN phases.  Actions are
    zeroed during RESET_SETTLE, DEPLOY, STOW, and COMPLETE.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # --- Residual action ---
        self._residual_scale_m = float(getattr(cfg, "residual_action_scale_m", 0.03))
        self._residual_action = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_residual_action = torch.zeros_like(self._residual_action)
        self._corrected_ee_goal_local = torch.zeros((self.num_envs, 3), device=self.device)

        # --- Parking brake ---
        self._brake = ParkingBrake(self)
        self._brake.engage()

        # --- Shape selection ---
        shape = str(getattr(cfg, "residual_mow_shape", "line")).lower()
        shape_fns = {
            "arc": arc_mowing_reference,
            "line": line_reference,
            "bowed_curve": bowed_curve_reference,
            "sine_sweep": sine_sweep_reference,
        }
        if shape not in shape_fns:
            raise ValueError(f"Unknown mow shape {shape!r}; choose from {list(shape_fns)}")
        self._shape_fn = shape_fns[shape]
        self._shape_kwargs = self._build_shape_kwargs(shape, cfg)

        # --- Trajectory sampler ---
        path_speed = float(getattr(cfg, "residual_path_speed_mps", 0.05))
        lateral_span = float(getattr(cfg, "residual_lateral_span_m",
                                     self._shape_kwargs.get("span", self._shape_kwargs.get("lateral_span", 1.0))))
        self._sampler = TrajectorySampler(
            self.num_envs, self.device, self._shape_fn,
            path_speed_mps=path_speed, step_dt=float(self.step_dt),
            **self._shape_kwargs,
        )

        # --- Ground follower ---
        self._ground_follower = GroundFollower(
            target_clearance_m=float(getattr(cfg, "residual_target_clearance_m", 0.05)),
            max_correction_m=float(getattr(cfg, "residual_max_height_correction_m", 0.12)),
            correction_gain=float(getattr(cfg, "residual_height_correction_gain", 0.5)),
        )

        # --- Reference manager ---
        self._ref_manager = ReferenceManager(
            self._sampler, self._ground_follower,
            ground_pinv=self._load_ground_pinv(),
            head_down_cylinder_seed=float(getattr(cfg, "residual_head_down_cylinder_seed", 0.0)),
        )

        # --- State machine ---
        min_path = float(getattr(cfg, "residual_min_accumulated_path_m", 5.0))
        self._state_machine = StopMowStateMachine(
            self.num_envs, self.device, self._brake,
            reset_settle_steps=300,  # pre-positioned by DirectCylinderDriver
            deploy_hold_steps=60,  # 1s deploy hold
            min_accumulated_path_m=min_path,
            stow_steps=int(getattr(cfg, "residual_stow_steps", 300)),
            step_dt=float(self.step_dt),
        )

        # --- Cascade controller ---
        controller = CylinderDLSController(
            self,
            CylinderDLSConfig(
                invert_learned_pipeline=False,
                error_gain=0.10,
                max_desired_delta_m=0.01,
                max_action=1.0,
                action_smoothing=1.0,
            ),
        )
        self._ee_cascade = EECascadeController(self, cylinder_controller=controller)

        # --- Cylinder seed at anchor (computed at reset) ---
        self._anchor_cylinder_seed = torch.zeros((self.num_envs, 4), device=self.device)


    def _build_shape_kwargs(self, shape: str, cfg) -> dict:
        """Return shape-specific kwargs for the trajectory function."""
        if shape == "arc":
            return {
                "arc_center_xy": (
                    float(getattr(cfg, "residual_arc_center_x", 4.327)),
                    float(getattr(cfg, "residual_arc_center_y", -3.224)),
                ),
                "arc_radius_m": float(getattr(cfg, "residual_arc_radius_m", 1.286)),
                "theta_a_deg": float(getattr(cfg, "residual_arc_theta_a_deg", -26.7)),
                "theta_b_deg": float(getattr(cfg, "residual_arc_theta_b_deg", 9.3)),
            }
        elif shape == "line":
            return {"span": float(getattr(cfg, "residual_lateral_span_m", 1.00))}
        elif shape == "bowed_curve":
            return {
                "lateral_span": float(getattr(cfg, "residual_lateral_span_m", 0.80)),
                "forward_bow": float(getattr(cfg, "residual_forward_bow_m", 0.14)),
            }
        elif shape == "sine_sweep":
            return {
                "lateral_span": float(getattr(cfg, "residual_lateral_span_m", 0.80)),
                "forward_amplitude": float(getattr(cfg, "residual_forward_amplitude_m", 0.06)),
                "num_lobes": int(getattr(cfg, "residual_sweep_lobes", 3)),
            }
        return {}

    def _load_ground_pinv(self) -> torch.Tensor | None:
        """Load the cached ground-Jacobian pseudoinverse if available."""
        import json
        cache_path = Path(str(getattr(self.cfg, "residual_jacobian_cache", str(_JACOBIAN_CACHE))))
        if not cache_path.exists():
            return None
        payload = json.loads(cache_path.read_text())
        J = torch.tensor(payload["jacobian"], device=self.device, dtype=torch.float32)
        J123 = J[:, :3]  # cylinders 1-3 only; reserve cylinder 4 for head-down
        # DLS pseudoinverse on CPU
        u, s, vh = torch.linalg.svd(J123.cpu(), full_matrices=False)
        damping = 0.03 * s[0]
        pinv = vh.T @ torch.diag(s / (s.square() + damping.square())) @ u.T
        return pinv.to(self.device, torch.float32)

    @torch.no_grad()
    def _get_nominal_goal(self) -> torch.Tensor:
        """Return the current nominal cutter goal from the reference manager."""
        sample = self._ref_manager.sample()
        return sample.position_b

    @torch.no_grad()
    def _compute_corrected_goal(
        self, residual: torch.Tensor
    ) -> torch.Tensor:
        """Apply PPO residual + ground correction to the nominal goal.

        Args:
            residual: (N, 3) clamped PPO action.

        Returns:
            (N, 3) corrected EE goal in truck frame.
        """
        cutter_z = self._compute_cutter_world_z()
        sample = self._ref_manager.corrected_sample(cutter_world_z=cutter_z)
        goal = sample.position_b.clone()
        # PPO lateral + height residual
        goal[:, 1] += residual[:, 0]  # dy
        goal[:, 2] += residual[:, 1]  # dz
        return goal

    @torch.no_grad()
    def controller_targets(self, residual_action: torch.Tensor):
        """Pure mapping used by the zero-residual equivalence contract."""
        residual = torch.clamp(residual_action[:, :2], -1.0, 1.0) * self._residual_scale_m
        nominal = self._get_nominal_goal()
        corrected = nominal.clone()
        corrected[:, 1:3] += residual
        cylinder_goal = self._ee_cascade.cylinder_goal(corrected)
        return corrected, cylinder_goal

    def _pre_physics_step(self, actions: torch.Tensor):
        action = torch.nan_to_num(
            actions.to(self.device, torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        if action.shape != (self.num_envs, 3):
            raise RuntimeError(f"Residual action must be ({self.num_envs}, 3), got {tuple(action.shape)}")

        # --- State machine step ---
        cutter_z = self._compute_cutter_world_z()
        _, cyl_vel = self._refresh_cylinder_state()
        self._state_machine.step(
            cutter_height=cutter_z,
            cylinder_velocity=cyl_vel,
            path_progress_delta=torch.zeros(self.num_envs, device=self.device),
        )

        # --- Manage parking brake ---
        mowing = self._state_machine.is_mowing()
        settling = self._state_machine.is_settling()
        if mowing.any():
            self._brake.engage()

        # --- PPO action masking ---
        self._previous_residual_action.copy_(self._residual_action)
        self._residual_action = torch.clamp(action, -1.0, 1.0)
        if (~mowing).any():
            self._residual_action[~mowing] = 0.0  # zero residual outside mowing

        # --- Advance reference only during mowing ---
        # During settle/deploy, the sampler stays at the arc start.
        if mowing.any():
            ids = torch.where(mowing)[0]
            base_speed = float(getattr(self.cfg, "residual_path_speed_mps", 0.05))
            self._sampler.set_speed(float(base_speed))
            self._ref_manager.step()
            delta = base_speed * float(self.step_dt)
            self._state_machine.step(
                cutter_height=cutter_z,
                cylinder_velocity=cyl_vel,
                path_progress_delta=torch.full_like(cutter_z, delta),
            )

        # --- Compute corrected goal and cylinder targets ---
        # Always track the nominal goal, even during settle/deploy.
        residual_m = torch.clamp(self._residual_action[:, :2], -1.0, 1.0) * self._residual_scale_m
        corrected_goal = self._compute_corrected_goal(residual_m)
        self._corrected_ee_goal_local = corrected_goal
        # During RESET_SETTLE/DEPLOY, track the nominal arc goal without advancing
        # the sampler. The cascade was reset in _reset_idx and tracks from there.
        if settling.any():
            cylinder_goal = self._ee_cascade.cylinder_goal(corrected_goal)
            self._cylinder_goal_x[:] = cylinder_goal
            cylinder_action, _ = self._ee_cascade.cylinder.action(cylinder_goal)
            self._current_action[:] = cylinder_action
            self._ppo_raw_action[:] = 0.0
        else:
            cylinder_goal = self._ee_cascade.cylinder_goal(corrected_goal)
            self._cylinder_goal_x[:] = cylinder_goal
            cylinder_action, _ = self._ee_cascade.cylinder.action(cylinder_goal)
            self._ppo_raw_action = cylinder_action
            self._current_action = cylinder_action

        # --- Run ID/FD pipeline ---
        self._run_cylinder_id_pipeline()

    def _get_observations(self) -> dict:
        ee = self._compute_ee_pos_local()
        x, xdot = self._refresh_cylinder_state()
        yaw_rate = self.robot.data.root_ang_vel_w[:, 2]

        nominal = self._get_nominal_goal()
        observation = build_residual_observation(
            ee_pos_local=ee,
            nominal_goal_local=nominal,
            ee_velocity_local=torch.zeros_like(ee),  # placeholder: compute if needed
            cylinder_x=x,
            cylinder_xdot=xdot,
            vehicle_speed_kmh=torch.zeros(self.num_envs, device=self.device),
            vehicle_yaw_rate=yaw_rate,
            previous_residual_action=self._previous_residual_action,
            cylinder_goal_x=self._cylinder_goal_x,
            current_valve_cmd=self._current_valve_cmd,
            fd_delta_corrected=self._fd_delta_corrected,
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        ee = self._compute_ee_pos_local()
        nominal = self._get_nominal_goal()
        reward, error, reward_extras = compute_residual_reward(
            ee,
            nominal,
            error_scale=float(getattr(self.cfg, "residual_error_reward_scale", 20.0)),
            truck_kmh=None,  # no truck speed reward during arm-only mowing
            progress_weight=float(getattr(self.cfg, "residual_progress_weight", 1.5)),
            progress_error_gate_m=float(getattr(self.cfg, "residual_progress_gate_m", 0.05)),
            truck_max_kmh=6.0,
            progress_decay_m=float(getattr(self.cfg, "residual_progress_decay_m", 0.10)),
            action_smooth_weight=float(getattr(self.cfg, "residual_action_smooth_weight", 0.01)),
            current_residual_action=self._residual_action,
            previous_residual_action=self._previous_residual_action,
        )
        self.extras["log"] = {
            "residual/yz_error_mean": error.mean(),
            "residual/action_abs_mean": self._residual_action[:, :2].abs().mean(),
            "residual/truck_kmh_mean": torch.zeros((), device=self.device),
            "residual/accumulated_path": self._state_machine.state.accumulated_path.mean(),
            **reward_extras,
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids]
        nonfinite = ~(
            torch.isfinite(cylinder_q).all(dim=-1)
            & torch.isfinite(self._current_valve_cmd).all(dim=-1)
            & torch.isfinite(self._fd_delta_corrected).all(dim=-1)
        )
        limit = ((cylinder_q < limits[..., 0]) | (cylinder_q > limits[..., 1])).any(dim=-1)
        completed = self._state_machine.is_complete()
        return limit | nonfinite | completed, timeout

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device, torch.long)

        # Engage brake and reset state machine
        self._brake.engage()
        self._state_machine.reset(env_ids)

        # Reset trajectory sampler to the arc start (distance=0)
        self._ref_manager.reset(env_ids)
        ee = self._compute_ee_pos_local()
        x, _ = self._refresh_cylinder_state()
        self._anchor_cylinder_seed[env_ids] = x[env_ids]

        # Set sampler anchor Z to ground-level from map (not spawn EE).
        # The arc goals need to be at mowing height (~1.3m truck Z), not
        # the spawn height (~1.8m truck Z).
        import json, numpy as np
        map_path = str(Path(__file__).resolve().parents[1] / "models/ground_band_map.json")
        try:
            with open(map_path) as fh:
                bd = json.load(fh)
            band = bd.get("ground_band", [])
            if band:
                cutters = np.array([s["cutter_truck"] for s in band])
                nominal_start = self._get_nominal_goal()
                start_xy = nominal_start[0, :2].detach().cpu().numpy()
                dists = np.linalg.norm(cutters[:, :2] - start_xy, axis=1)
                ground_z = float(cutters[int(np.argmin(dists)), 2])
                self._sampler.anchor[env_ids, 2] = ground_z
        except Exception:
            self._sampler.anchor[env_ids, 2] = ee[env_ids, 2]  # fallback

        self._residual_action[env_ids] = 0.0
        self._previous_residual_action[env_ids] = 0.0

        # Pre-position: use DirectCylinderDriver to drive the arm to
        # the arc start position (same pattern as verified gates).
        nominal_start = self._get_nominal_goal()
        import json, numpy as np
        map_path = str(Path(__file__).resolve().parents[1] / "models/ground_band_map.json")
        try:
            with open(map_path) as fh:
                bd = json.load(fh)
            band = bd.get("ground_band", [])
            if band:
                targets = np.array([s["target_x"] for s in band])
                cutters = np.array([s["cutter_truck"] for s in band])
                start_xy = nominal_start[0, :2].detach().cpu().numpy()
                dists = np.linalg.norm(cutters[:, :2] - start_xy, axis=1)
                closest = int(np.argmin(dists))
                seed = torch.as_tensor(
                    targets[closest], device=self.device, dtype=torch.float32
                ).unsqueeze(0).expand(len(env_ids), 4)
                driver = DirectCylinderDriver(self)
                for _ in range(40):
                    driver.step(seed, substeps=2)
        except Exception:
            pass

        # Reset the cascade at the pre-positioned pose
        self._ee_cascade.reset(env_ids)
        self._corrected_ee_goal_local[env_ids] = nominal_start[env_ids]
