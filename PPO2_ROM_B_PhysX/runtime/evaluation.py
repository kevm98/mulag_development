"""Deterministic rollout metrics for ROM-B curriculum promotion."""

from __future__ import annotations

from collections.abc import Callable

import torch


_FACTOR_NAMES = ("e_comp", "d_seal_1", "d_seal_2", "d_seal_3", "d_seal_4")


# Per-environment signals captured in a rolling window so a terminating replica
# can be explained after the fact. Purely observational: nothing here feeds the
# gate metrics computed by `summarize_deterministic_probe`.
_TRACE_SIGNALS: tuple[tuple[str, str, int], ...] = (
    # (record name, environment attribute, expected trailing width)
    ("clearance_signed_m", "", 1),
    ("head_alignment", "", 1),
    ("acceleration_mps2", "", 1),
    ("truck_raw_drift_m", "_truck_drift_xy", 1),
    ("truck_post_constraint_drift_m", "_truck_post_constraint_drift_xy", 1),
    ("path_s_m", "_reference_path_s", 1),
    ("reference_phase", "_traj_phase", 1),
    ("scored_elapsed_s", "_scored_elapsed", 1),
    ("tracking_error_m", "", 1),
    ("residual_raw", "_residual_raw", 4),
    ("residual_filtered", "_residual_filtered", 4),
    ("physical_residual_m", "", 4),
    ("cylinder_pos_m", "", 4),
    ("cylinder_vel_mps", "", 4),
    ("baseline_cylinder_goal_m", "_baseline_cylinder_goal", 4),
    ("corrected_cylinder_goal_m", "_corrected_cylinder_goal", 4),
    ("valve_command", "_current_valve_cmd", 8),
    ("rom_increment_m", "_fd_delta_corrected", 4),
    # Frame decomposition. The reward tracks `cutter_truck` (truck frame) while
    # the clearance gate compares `cutter_world_z` against `cutter_world_z` from
    # the trajectory library. For a rigid level parked truck those two references
    # must differ by a constant; in the library they do not. These fields let the
    # runtime offset be decomposed into reference disagreement versus genuine
    # tracking error, which cannot be done from the library alone.
    ("reference_world_z", "_original_reference_world_z", 1),
    ("cutter_world_z", "", 1),
    # Orientation-aware cutter world Z. Differs from `cutter_world_z` by the
    # cutter-definition residual, which is logged rather than assumed negligible.
    ("cutter_pos_w_z", "", 1),
    ("ee_origin_world_z", "", 1),
    ("truck_base_world_z", "", 1),
    ("reference_cutter_truck", "_original_reference_tool", 3),
    ("measured_cutter_truck", "", 3),
)


class _PreTerminationRecorder:
    """Rolling per-environment trace plus a snapshot at first termination.

    The recorder never influences `counts`, `alive`, or any tensor handed to
    `summarize_deterministic_probe`; it only observes environment state.
    """

    def __init__(self, raw, *, window: int) -> None:
        self.raw = raw
        self.window = max(int(window), 1)
        self.num_envs = int(raw.num_envs)
        self.device = raw.device
        self.cursor = 0
        self.filled = 0
        self.buffers: dict[str, torch.Tensor] = {}
        self.records: list[dict[str, object]] = []

    def _gather(self, tool: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.raw
        clearance = raw._clearance_signal()
        scales = raw._residual_scale_vector().reshape(1, -1)
        values: dict[str, torch.Tensor] = {
            "clearance_signed_m": clearance,
            "head_alignment": raw._measure_head_down_alignment(),
            "acceleration_mps2": torch.linalg.norm(
                raw._measured_tool_acceleration_truck, dim=-1
            ),
            "tracking_error_m": torch.linalg.norm(
                raw._original_reference_tool - tool, dim=-1
            ),
            "physical_residual_m": raw._residual_filtered * scales,
            "cylinder_pos_m": raw.robot.data.joint_pos[:, raw._cylinder_joint_ids],
            "cylinder_vel_mps": raw.robot.data.joint_vel[:, raw._cylinder_joint_ids],
            "cutter_world_z": raw._compute_cutter_world_z(),
            "cutter_pos_w_z": raw._compute_cutter_pos_w()[:, 2],
            "ee_origin_world_z": raw.robot.data.body_link_state_w[
                :, raw._ee_body_id, 2
            ],
            "truck_base_world_z": raw.robot.data.body_link_state_w[
                :, raw._truck_base_body_id(), 2
            ],
            "measured_cutter_truck": tool,
        }
        for name, attribute, _width in _TRACE_SIGNALS:
            if name in values or not attribute:
                continue
            tensor = getattr(raw, attribute, None)
            if isinstance(tensor, torch.Tensor):
                values[name] = tensor
        return {
            name: tensor.detach().float().reshape(self.num_envs, -1)
            for name, tensor in values.items()
        }

    def observe(self, tool: torch.Tensor) -> None:
        values = self._gather(tool)
        for name, tensor in values.items():
            buffer = self.buffers.get(name)
            if buffer is None:
                buffer = torch.full(
                    (self.window, self.num_envs, tensor.shape[-1]),
                    torch.nan,
                    device=self.device,
                )
                self.buffers[name] = buffer
            buffer[self.cursor] = tensor
        self.cursor = (self.cursor + 1) % self.window
        self.filled = min(self.filled + 1, self.window)

    def _chronological(self, name: str, env_id: int) -> list[list[float]]:
        buffer = self.buffers[name]
        order = [
            (self.cursor - offset) % self.window
            for offset in range(self.filled, 0, -1)
        ]
        return [buffer[index, env_id].tolist() for index in order]

    def capture(
        self,
        *,
        env_ids: torch.Tensor,
        causes: dict[str, torch.Tensor],
        simulator_step: int,
        scored_step: torch.Tensor,
        factors: torch.Tensor | None,
    ) -> None:
        """Snapshot every environment that terminated on this simulator step."""

        if not self.buffers:
            return
        for env_id in env_ids.tolist():
            matched = [name for name, mask in causes.items() if bool(mask[env_id])]
            trace = {
                name: self._chronological(name, env_id) for name in self.buffers
            }
            terminal = {name: values[-1] for name, values in trace.items()}
            self.records.append({
                "env_id": int(env_id),
                "simulator_step": int(simulator_step),
                "scored_step": int(scored_step[env_id]),
                "termination_causes": matched or ["unclassified"],
                "factors": (
                    factors[env_id].detach().cpu().tolist()
                    if isinstance(factors, torch.Tensor) else None
                ),
                "at_termination": {
                    name: (values[0] if len(values) == 1 else values)
                    for name, values in terminal.items()
                },
                "trace": trace,
            })


def _expand_fixed_factors(
    fixed_factors: torch.Tensor, template: torch.Tensor
) -> torch.Tensor:
    requested = torch.as_tensor(
        fixed_factors, dtype=template.dtype, device=template.device
    )
    if requested.ndim == 1:
        requested = requested.reshape(1, -1).expand_as(template)
    if requested.shape != template.shape:
        raise ValueError(
            f"fixed_factors must have shape ({template.shape[1]},) or "
            f"{tuple(template.shape)}, got {tuple(requested.shape)}"
        )
    return requested


def held_out_evaluation_seed(
    base_seed: int, shape_index: int, *, offset: int = 100000
) -> int:
    """Return a fixed per-shape seed shared by every curriculum round."""

    if int(shape_index) < 0:
        raise ValueError("shape_index must be non-negative")
    return int(base_seed) + int(offset) + int(shape_index)


def summarize_deterministic_probe(
    *,
    shape: str,
    error: torch.Tensor,
    acceleration: torch.Tensor,
    alignment: torch.Tensor,
    clearance: torch.Tensor,
    drift: torch.Tensor,
    completed: torch.Tensor,
    safety_failed: torch.Tensor,
    timed_out: torch.Tensor,
) -> dict[str, float | int | str]:
    """Reduce per-step, per-environment probe tensors into gate metrics."""

    def _values(name: str, tensor: torch.Tensor) -> torch.Tensor:
        selected = tensor[:, completed] if tensor.ndim == 2 else tensor
        values = selected[torch.isfinite(selected)]
        if values.numel() == 0:
            raise RuntimeError(
                f"deterministic {shape} probe has no completed {name} samples"
            )
        return values

    error_values = _values("tracking-error", error)
    acceleration_values = _values("acceleration", acceleration)
    alignment_values = _values("head-alignment", alignment)
    clearance_values = _values("clearance", clearance)
    drift_values = _values("truck-drift", drift)
    unsuccessful = safety_failed | timed_out | ~completed
    return {
        "shape": str(shape),
        "evaluated_environments": int(completed.numel()),
        "completed_environments": int(torch.count_nonzero(completed)),
        "completion_fraction": float(completed.float().mean()),
        # Exact integer counts, so the acceptance statistic never has to
        # reconstruct them by rounding a reported fraction.
        "unsuccessful_environments": int(torch.count_nonzero(unsuccessful)),
        "replicas": int(completed.numel()),
        "mean_error_m": float(error_values.mean()),
        "p95_error_m": float(torch.quantile(error_values, 0.95)),
        "p95_acceleration_mps2": float(
            torch.quantile(acceleration_values, 0.95)
        ),
        "min_head_alignment": float(alignment_values.min()),
        "max_clearance_error_m": float(clearance_values.max()),
        "max_post_constraint_truck_drift_m": float(drift_values.max()),
        "max_termination_rate": float(unsuccessful.float().mean()),
        "safety_termination_rate": float(safety_failed.float().mean()),
        "timeout_rate": float(timed_out.float().mean()),
    }


def _summarize_path_profile(
    samples: dict[str, torch.Tensor], *, bins: int = 10
) -> list[dict[str, float | int]]:
    """Bin scored samples by normalized path position.

    Answers where along a trajectory acceleration spikes and clearance excursions
    happen. Uses every scored sample, including those from replicas that later
    terminated, so the pre-failure approach is visible.
    """

    path = samples["path_s"].detach().cpu()
    finite = torch.isfinite(path)
    if not bool(finite.any()):
        return []
    low = float(path[finite].min())
    high = float(path[finite].max())
    width = max(high - low, 1.0e-9)
    index = torch.clamp(
        ((path - low) / width * bins).floor().long(), 0, bins - 1
    )
    acceleration = samples["acceleration"].detach().cpu()
    clearance = samples["clearance"].detach().cpu()
    alignment = samples["alignment"].detach().cpu()
    error = samples["error"].detach().cpu()
    profile: list[dict[str, float | int]] = []
    for edge in range(bins):
        mask = finite & (index == edge)
        if not bool(mask.any()):
            continue
        def _stat(tensor: torch.Tensor, reducer) -> float:
            values = tensor[mask]
            values = values[torch.isfinite(values)]
            return float(reducer(values)) if values.numel() else float("nan")
        profile.append({
            "bin": edge,
            "path_s_low_m": low + width * edge / bins,
            "path_s_high_m": low + width * (edge + 1) / bins,
            "samples": int(mask.sum()),
            "mean_error_m": _stat(error, torch.mean),
            "p95_acceleration_mps2": _stat(
                acceleration, lambda v: torch.quantile(v, 0.95)
            ),
            "max_acceleration_mps2": _stat(acceleration, torch.max),
            "mean_clearance_error_m": _stat(clearance, torch.mean),
            "p95_clearance_error_m": _stat(
                clearance, lambda v: torch.quantile(v, 0.95)
            ),
            "max_clearance_error_m": _stat(clearance, torch.max),
            "mean_head_alignment": _stat(alignment, torch.mean),
            "min_head_alignment": _stat(alignment, torch.min),
        })
    return profile


def _summarize_factor_association(
    acceleration: torch.Tensor, factors: torch.Tensor
) -> list[dict[str, float | str]]:
    """Observational association between hidden factors and per-env accel tails."""

    values = acceleration.detach().cpu()
    factors = factors.detach().cpu()
    per_env = torch.nanquantile(values, 0.95, dim=0)
    finite_acceleration = torch.isfinite(per_env)
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(_FACTOR_NAMES):
        factor = factors[:, index]
        finite = finite_acceleration & torch.isfinite(factor)
        x = factor[finite]
        y = per_env[finite]
        if x.numel() < 4:
            continue
        centered_x = x - x.mean()
        centered_y = y - y.mean()
        denominator = torch.sqrt(
            torch.sum(centered_x.square()) * torch.sum(centered_y.square())
        )
        correlation = (
            float(torch.sum(centered_x * centered_y) / denominator)
            if float(denominator) > 0.0 else float("nan")
        )
        low = x <= torch.quantile(x, 0.25)
        high = x >= torch.quantile(x, 0.75)
        rows.append({
            "factor": name,
            "pearson_with_per_env_p95_acceleration": correlation,
            "low_quartile_mean_factor": float(x[low].mean()),
            "low_quartile_mean_p95_acceleration_mps2": float(y[low].mean()),
            "high_quartile_mean_factor": float(x[high].mean()),
            "high_quartile_mean_p95_acceleration_mps2": float(y[high].mean()),
        })
    return rows


def _crosscheck_acceleration_from_positions(
    tool_positions: torch.Tensor,
    online_acceleration: torch.Tensor,
    *,
    dt: float,
) -> dict[str, float | int]:
    """Recompute acceleration independently from the recorded position trace."""

    positions = tool_positions.detach().cpu()
    online = online_acceleration.detach().cpu()[2:]
    offline_vector = (
        positions[2:] - 2.0 * positions[1:-1] + positions[:-2]
    ) / max(float(dt) ** 2, 1.0e-12)
    offline = torch.linalg.norm(offline_vector, dim=-1)
    finite = torch.isfinite(offline) & torch.isfinite(online)
    if not bool(finite.any()):
        return {"matched_samples": 0}
    online_values = online[finite]
    offline_values = offline[finite]
    difference = torch.abs(online_values - offline_values)
    return {
        "matched_samples": int(finite.sum()),
        "dt_s": float(dt),
        "online_p95_acceleration_mps2": float(torch.quantile(online_values, 0.95)),
        "offline_position_difference_p95_acceleration_mps2": float(
            torch.quantile(offline_values, 0.95)
        ),
        "mean_absolute_difference_mps2": float(difference.mean()),
        "p95_absolute_difference_mps2": float(torch.quantile(difference, 0.95)),
    }


def _crosscheck_path_weighting(
    samples: dict[str, torch.Tensor],
    completed: torch.Tensor,
    *,
    bins: int = 10,
) -> dict[str, float | int]:
    """Compare formal time-sample p95 with equal-geometric-bin weighting."""

    keep = completed.detach().cpu().bool()
    path = samples["path_s"].detach().cpu()[:, keep]
    acceleration = samples["acceleration"].detach().cpu()[:, keep]
    finite = torch.isfinite(path) & torch.isfinite(acceleration)
    if not bool(finite.any()):
        return {"matched_samples": 0}
    path_values = path[finite]
    acceleration_values = acceleration[finite]
    low = path_values.min()
    width = torch.clamp(path_values.max() - low, min=1.0e-9)
    index = torch.clamp(
        ((path_values - low) / width * bins).floor().long(), 0, bins - 1
    )
    counts = torch.bincount(index, minlength=bins).float()
    populated = counts > 0
    weights = 1.0 / counts[index]
    order = torch.argsort(acceleration_values)
    sorted_values = acceleration_values[order]
    sorted_weights = weights[order]
    cumulative = torch.cumsum(sorted_weights, dim=0)
    cutoff = 0.95 * cumulative[-1]
    weighted_index = int(torch.searchsorted(cumulative, cutoff).item())
    weighted_index = min(weighted_index, sorted_values.numel() - 1)
    return {
        "matched_samples": int(finite.sum()),
        "geometric_bins": int(bins),
        "populated_bins": int(populated.sum()),
        "formal_time_uniform_p95_acceleration_mps2": float(
            torch.quantile(acceleration_values, 0.95)
        ),
        "equal_geometric_bin_weighted_p95_acceleration_mps2": float(
            sorted_values[weighted_index]
        ),
    }


@torch.no_grad()
def restore_canonical_physical_state(raw) -> None:
    """Write the articulation's spawn state to sim before a scored probe.

    Motivation: `_reset_idx` is a *soft* reset. It redeploys from wherever the
    arm was left (`_preposition_start_x = measured_x`, `policy/env.py:1373`) and
    re-takes the truck anchor from the current chassis pose (`:1385-1390`).
    Because the probe deliberately does not reset failed replicas, each probe
    inherits the terminal physical state of its predecessor, so an identical seed
    yields different results depending on its position in the sequence.

    WARNING - this is NOT sufficient, measured 2026-07-24. Restoring the root
    pose, root velocity, joint state and truck anchor still leaves probes
    position-dependent:

        without hard reset, seed 300032: 11, 11, 10 (positions 1, 2, 3)
        with    hard reset, seed 300017:  5,  4      (positions 1, 3)
        with    hard reset, seed 300032: 10, 12      (positions 2, 4)

    It also shifts the position-1 result away from the value a fresh process
    produces, so it is not a drop-in restoration of canonical state. State must
    still be leaking through something this does not touch - candidates are the
    command-ROM recurrent history, the DLS/cascade controllers, and PhysX solver
    warm-start or contact caches.

    Kept, default off, because the diagnosis it encodes is correct and the fix is
    worth finishing. Do not rely on it for certification. The only currently
    verified way to make a probe a function of its seed alone is a fresh process
    per probe: two separate processes running this seed first agreed to five
    decimal places.
    """

    robot = raw.robot
    root = robot.data.default_root_state.clone()
    root[:, :3] += raw.scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    )
    robot.reset()
    # Force the parked-truck anchor to be re-derived from the spawn pose rather
    # than from wherever the chassis drifted to.
    raw._truck_stationary_initialized = False
    raw._truck_anchor_xy = root[:, :2].clone()
    raw._truck_drift_xy.zero_()
    raw._max_truck_drift.zero_()
    raw._truck_post_constraint_drift_xy.zero_()
    raw._max_truck_post_constraint_drift.zero_()
    raw._prev_tool_pos_truck = None


@torch.inference_mode()
def run_deterministic_probe(
    raw,
    env,
    policy: Callable[[torch.Tensor], torch.Tensor],
    *,
    shape: str,
    seed: int,
    scored_steps: int,
    force_zero_policy: bool = False,
    fixed_factors: torch.Tensor | None = None,
    collect_diagnostics: bool = False,
    diagnostic_window: int = 12,
    hard_reset: bool = False,
) -> dict[str, object]:
    """Evaluate deterministic policy actions without resetting failed replicas.

    `collect_diagnostics` adds observational per-environment failure records and
    per-sample path coordinates. It cannot change any gate metric: the tensors
    passed to `summarize_deterministic_probe` are unaffected by the flag.
    """

    target = int(scored_steps)
    if target <= 0:
        raise ValueError("scored_steps must be positive")
    raw.set_trajectory_shape(shape)
    if hard_reset:
        restore_canonical_physical_state(raw)
    raw.reset(seed=int(seed))
    if fixed_factors is not None:
        runtime = getattr(raw, "_command_cylinder_rom", None)
        factors = getattr(runtime, "true_factors", None)
        if not isinstance(factors, torch.Tensor):
            raise RuntimeError("environment does not expose ROM-B factors")
        factors.copy_(_expand_fixed_factors(fixed_factors, factors))
    obs = env.get_observations()
    num_envs = int(raw.num_envs)
    device = raw.device
    counts = torch.zeros(num_envs, dtype=torch.long, device=device)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    safety_failed = torch.zeros_like(alive)
    timed_out = torch.zeros_like(alive)
    sample_names = [
        "error", "acceleration", "alignment", "clearance", "drift",
        "mean_abs_residual", "cylinder4_abs_residual",
    ]
    if collect_diagnostics:
        # Observational only; never read by `summarize_deterministic_probe`.
        sample_names += ["path_s", "reference_phase", "signed_clearance"]
    samples = {
        name: torch.full((target, num_envs), torch.nan, device=device)
        for name in sample_names
    }
    recorder = (
        _PreTerminationRecorder(raw, window=diagnostic_window)
        if collect_diagnostics else None
    )
    tool_positions = (
        torch.full((target, num_envs, 3), torch.nan, device=device)
        if collect_diagnostics else None
    )
    cause_failed = {
        name: torch.zeros_like(alive)
        for name in (
            "nonfinite", "cylinder_limit", "head_alignment", "clearance",
            "truck_post_constraint", "truck_raw_reaction", "unclassified",
        )
    }

    synchronized_resets = bool(getattr(raw.cfg, "synchronize_resets", True))
    deferred_safety = bool(getattr(raw.cfg, "defer_safety_terminations", False))
    original_force_zero = bool(getattr(raw.cfg, "force_zero_policy", False))
    original_get_dones = raw._get_dones
    captured_terminated = torch.zeros_like(alive)
    captured_truncated = torch.zeros_like(alive)

    def _capture_without_reset():
        nonlocal captured_terminated, captured_truncated
        terminated, truncated = original_get_dones()
        captured_terminated = terminated.bool()
        captured_truncated = truncated.bool()
        return torch.zeros_like(terminated), torch.zeros_like(truncated)

    raw.cfg.synchronize_resets = False
    raw.cfg.defer_safety_terminations = False
    raw.cfg.force_zero_policy = bool(force_zero_policy)
    raw._get_dones = _capture_without_reset
    total_steps = 0
    try:
        for _ in range(target + 4000):
            obs, _, _, _ = env.step(policy(obs))
            total_steps += 1
            active = (
                alive
                & (counts < target)
                & (raw._preposition_counter <= 0)
                & (raw._settle_counter <= 0)
            )
            if recorder is not None:
                recorder.observe(raw._compute_ee_pos_local())
            ids = torch.nonzero(active, as_tuple=False).flatten()
            if ids.numel() > 0:
                rows = counts[ids]
                tool = raw._compute_ee_pos_local()
                if tool_positions is not None:
                    tool_positions[rows, ids] = tool[ids]
                samples["error"][rows, ids] = torch.linalg.norm(
                    raw._original_reference_tool[ids] - tool[ids], dim=-1
                )
                samples["acceleration"][rows, ids] = torch.linalg.norm(
                    raw._measured_tool_acceleration_truck[ids], dim=-1
                )
                samples["alignment"][rows, ids] = (
                    raw._measure_head_down_alignment()[ids]
                )
                clearance = raw._clearance_signal()
                samples["clearance"][rows, ids] = torch.abs(clearance[ids])
                samples["drift"][rows, ids] = (
                    raw._truck_post_constraint_drift_xy[ids]
                )
                scales = raw._residual_scale_vector().reshape(1, -1)
                physical_residual = torch.abs(raw._residual_filtered) * scales
                samples["mean_abs_residual"][rows, ids] = physical_residual[
                    ids
                ].mean(dim=-1)
                samples["cylinder4_abs_residual"][rows, ids] = physical_residual[
                    ids, 3
                ]
                if collect_diagnostics:
                    samples["path_s"][rows, ids] = raw._reference_path_s[ids]
                    samples["reference_phase"][rows, ids] = raw._traj_phase[
                        ids
                    ].float()
                    samples["signed_clearance"][rows, ids] = clearance[ids]
                counts[ids] += 1
            if torch.any(captured_terminated & alive):
                cylinder_q = raw.robot.data.joint_pos[:, raw._cylinder_joint_ids]
                cylinder_qdot = raw.robot.data.joint_vel[:, raw._cylinder_joint_ids]
                limits = raw.robot.data.soft_joint_pos_limits[
                    :, raw._cylinder_joint_ids
                ]
                limit_margin = float(
                    getattr(raw.cfg, "cylinder_limit_margin_m", 0.002)
                )
                current_clearance = raw._clearance_signal()
                cause_masks = {
                    "nonfinite": ~(
                        torch.isfinite(cylinder_q).all(dim=-1)
                        & torch.isfinite(cylinder_qdot).all(dim=-1)
                        & torch.isfinite(raw._current_valve_cmd).all(dim=-1)
                        & torch.isfinite(raw._fd_delta_corrected).all(dim=-1)
                        & torch.isfinite(raw._corrected_cylinder_goal).all(dim=-1)
                    ),
                    "cylinder_limit": (
                        (cylinder_q < limits[..., 0] - limit_margin)
                        | (cylinder_q > limits[..., 1] + limit_margin)
                    ).any(dim=-1),
                    "head_alignment": raw._measure_head_down_alignment()
                    < float(getattr(raw.cfg, "head_down_termination_alignment", 0.95)),
                    "clearance": torch.abs(current_clearance)
                    > float(getattr(raw.cfg, "cutter_clearance_hard_band_m", 0.15)),
                    "truck_post_constraint": raw._truck_post_constraint_drift_xy
                    > float(getattr(raw.cfg, "truck_drift_termination_m", 0.02)),
                    "truck_raw_reaction": raw._truck_drift_xy
                    > float(
                        getattr(raw.cfg, "truck_raw_reaction_termination_m", 0.05)
                    ),
                }
                classified = torch.zeros_like(alive)
                for name, mask in cause_masks.items():
                    matched = alive & captured_terminated & mask
                    cause_failed[name] |= matched
                    classified |= matched
                cause_failed["unclassified"] |= (
                    alive & captured_terminated & ~classified
                )
                if recorder is not None:
                    runtime = getattr(raw, "_command_cylinder_rom", None)
                    recorder.capture(
                        env_ids=torch.nonzero(
                            alive & captured_terminated, as_tuple=False
                        ).flatten(),
                        causes=cause_masks,
                        simulator_step=total_steps,
                        scored_step=counts,
                        factors=getattr(runtime, "true_factors", None),
                    )
            safety_failed |= captured_terminated
            timed_out |= captured_truncated
            alive &= ~(captured_terminated | captured_truncated)
            if bool(torch.all((counts >= target) | ~alive)):
                break
    finally:
        raw._get_dones = original_get_dones
        raw.cfg.synchronize_resets = synchronized_resets
        raw.cfg.defer_safety_terminations = deferred_safety
        raw.cfg.force_zero_policy = original_force_zero

    completed = counts >= target
    summary = summarize_deterministic_probe(
        shape=shape,
        completed=completed.detach().cpu(),
        safety_failed=safety_failed.detach().cpu(),
        timed_out=timed_out.detach().cpu(),
        **{
            name: samples[name].detach().cpu()
            for name in ("error", "acceleration", "alignment", "clearance", "drift")
        },
    )
    summary.update({
        "seed": int(seed),
        "scored_steps": target,
        "simulator_steps": total_steps,
        "force_zero_policy": bool(force_zero_policy),
        "mean_abs_residual_m": float(torch.nanmean(samples["mean_abs_residual"])),
        "max_cylinder4_abs_residual_m": float(
            torch.nan_to_num(samples["cylinder4_abs_residual"], nan=0.0).max()
        ),
        "termination_cause_rates": {
            name: float(mask.float().mean())
            for name, mask in cause_failed.items()
        },
    })
    if recorder is not None:
        summary["diagnostics"] = {
            "window_steps": recorder.window,
            "failed_environments": recorder.records,
            "path_profile": _summarize_path_profile(samples),
            "acceleration_crosscheck": _crosscheck_acceleration_from_positions(
                tool_positions,
                samples["acceleration"],
                dt=float(raw.step_dt),
            ),
            "path_weighting_crosscheck": _crosscheck_path_weighting(
                samples, completed
            ),
        }
    runtime = getattr(raw, "_command_cylinder_rom", None)
    factors = getattr(runtime, "true_factors", None)
    if isinstance(factors, torch.Tensor):
        factors = factors.detach().cpu()
        summary.update({
            "factor_min": factors.min(dim=0).values.tolist(),
            "factor_mean": factors.mean(dim=0).tolist(),
            "factor_max": factors.max(dim=0).values.tolist(),
        })
        if recorder is not None:
            summary["diagnostics"]["factor_association"] = (
                _summarize_factor_association(samples["acceleration"], factors)
            )
    return summary
