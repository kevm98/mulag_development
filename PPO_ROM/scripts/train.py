#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Train the PPO3-1 c4-authority sole controller over ROM-A.

The task, observation, rewards, terrain, path families and gates are inherited
from PPO3's shared verge task. The plant change is the c4 authority
(``cylinder_target_max_lead_per_axis_m`` and
``cylinder_action_scale_per_axis_m``), applied through
``PPO_ROM.terrain_env_cfg.apply_c4_authority_stage`` after every
curriculum transition.

The four measured training fixes from the PPO3/PPO4 handover are available as
flags and recommended for a fresh c4-authority run:

  * ``--reset_critic_head_on_resume``  stale critic after a plant change
  * ``--resume_c4_std``                cap c4 exploration separately
  * keep ``lateral_tracking_std_m >= 0.30`` (the config default)
  * ``terrain_action_magnitude_weight = 0.0`` (the config default)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stages", default="4,5")
p.add_argument("--stage_iterations", type=int, default=400)
p.add_argument("--save_interval", type=int, default=50)
p.add_argument("--learning_rate", type=float, default=None)
p.add_argument("--entropy_coef", type=float, default=None)
p.add_argument(
    "--desired_kl",
    type=float,
    default=None,
    help="Adaptive-schedule KL setpoint (inherited default 0.02; the "
    "teacher_student fine-tune regime used 0.008).",
)
p.add_argument(
    "--std_clamp",
    default=None,
    help="'min,max' bounds clamped onto the per-axis exploration std after "
    "every PPO update (e.g. '0.03,0.15'), mirroring skrl's clip_log_std. "
    "Without it the entropy bonus can inflate std without limit.",
)
p.add_argument("--init_std", type=float, default=None)
p.add_argument("--resume", default=None, help="66x4 checkpoint to resume from.")
p.add_argument(
    "--from_scratch",
    action="store_true",
    help="Explicit ablation: initialize a random terrain actor.",
)
p.add_argument(
    "--reset_optimizer_on_resume",
    action="store_true",
    help="Load actor/critic weights but discard stale PPO optimizer moments.",
)
p.add_argument(
    "--reset_critic_head_on_resume",
    action="store_true",
    help="Zero the resumed critic output head after a reward/plant change.",
)
p.add_argument(
    "--resume_std",
    type=float,
    default=None,
    help="Exploration std forced on ALL actions after resume.",
)
p.add_argument(
    "--resume_c4_std",
    type=float,
    default=None,
    help="Exploration std forced on the head-pitch (c4) action after resume.",
)
p.add_argument(
    "--freeze_actor_iterations",
    type=int,
    default=0,
    help=(
        "Critic warm-up: hold the actor (mean net AND exploration std) exactly "
        "at its resumed weights for the first N iterations while the critic "
        "learns against the actor's rollouts (requires_grad=False, so the "
        "shared Adam skips actor params). The adaptive KL scheduler is "
        "suspended for the window and the learning rate reset at unfreeze. "
        "Measured context: with a reset critic head the value loss stays at "
        "6k-26k and the policy descends its own reward; this is the standard "
        "remedy."
    ),
)
p.add_argument(
    "--task",
    default="",
    help=(
        "Gym task to train. Empty = PPO3_1 (ROM-A). Set "
        "RRLAB-Mulag-Verge-IDFD-v0 for the ID/FD pipeline. WITHOUT this "
        "the script builds PPO3_1's ROM task regardless of which checkpoint is "
        "resumed -- measured 2026-08-06, an ID/FD-distilled student was "
        "silently fine-tuned on the ROM plant and the run manifest read "
        "actuator=rom. The actuator assertion below now refuses that."
    ),
)
p.add_argument("--checkpoint_out", default="")
p.add_argument("--log_dir", default="")
p.add_argument(
    "--cylinder_target_lead_c4_m",
    type=float,
    default=None,
    help="Override the c4 lead cap (metres).",
)
p.add_argument(
    "--c4_action_scale_gain",
    type=float,
    default=None,
    help="Override the c4 action-scale multiplier vs c1-c3.",
)
p.add_argument(
    "--until_converged",
    action="store_true",
    help="Repeat the final curriculum stage until all shared gates pass.",
)
p.add_argument("--convergence_check_interval", type=int, default=50)
p.add_argument("--convergence_steps", type=int, default=300)
p.add_argument("--vehicle_speed_kmh", type=float, default=4.0)
p.add_argument(
    "--vehicle_speed_min_kmh",
    type=float,
    default=None,
    help="With --vehicle_speed_max_kmh: per-episode uniform truck-speed "
    "randomization, the teacher_student repo's regime (constant within an "
    "episode, resampled at reset) adapted to this asset's kinematic drive. "
    "The periodic mid-episode resampler inherited from the PPO2 obstacle "
    "task is disabled so the speed really is per-episode. Evaluation "
    "scripts stay pinned at their own speed and are unaffected.",
)
p.add_argument("--vehicle_speed_max_kmh", type=float, default=None)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

stages = tuple(int(part) for part in a.stages.split(",") if part.strip())
if not stages or any(stage not in (1, 2, 3, 4, 5, 6) for stage in stages):
    p.error("--stages must be a comma-separated subset of 1,2,3,4,5,6")
if a.resume and a.from_scratch:
    p.error("--resume and --from_scratch are mutually exclusive")
if (a.vehicle_speed_min_kmh is None) != (a.vehicle_speed_max_kmh is None):
    p.error(
        "--vehicle_speed_min_kmh and --vehicle_speed_max_kmh must be given "
        "together; one alone would silently pin the speed"
    )
if a.vehicle_speed_min_kmh is not None:
    if not 0.0 < a.vehicle_speed_min_kmh <= a.vehicle_speed_max_kmh:
        p.error("truck speed range must satisfy 0 < min <= max")
if a.until_converged:
    # `convergence_rollout` scores verge_task.terrain_convergence, the
    # LEGACY 26-gate contract -- not PPO_ROM.gates. Three of its
    # thresholds are above anything this plant has ever produced, the
    # privileged DLS teacher included: ground_contact 0.90 and worst-family
    # 0.85 against a teacher that measures 0.75-0.81, and hard_impacts as a
    # raw EVENT COUNT bounded at 0. Re-deriving those three from the
    # achievable band is exactly what gates.py exists to record. So this stop
    # condition cannot fire, `model_converged.pt` is never written, and the
    # run silently becomes "train forever". Refuse rather than burn the GPU.
    raise SystemExit(
        "--until_converged scores the legacy 26-gate contract, whose "
        "ground_contact 0.90 / worst_family_contact 0.85 / "
        "hard_impact_event_count 0 thresholds the DLS teacher itself fails. "
        "It cannot pass. Train for a fixed --stage_iterations and score the "
        "result with scripts/evaluate_gates.py, which implements the 8-gate "
        "multi-mesh contract."
    )

std_clamp_bounds = None
if a.std_clamp is not None:
    try:
        std_lo, std_hi = (float(part) for part in a.std_clamp.split(","))
    except ValueError:
        p.error("--std_clamp must be 'min,max', e.g. '0.03,0.15'")
    if not 0.0 < std_lo < std_hi:
        p.error("--std_clamp must satisfy 0 < min < max")
    std_clamp_bounds = (std_lo, std_hi)
if a.resume is None and not a.from_scratch:
    p.error(
        "production ppo3_1 training requires a seed checkpoint (--resume) or "
        "an explicitly labelled --from_scratch ablation"
    )

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO_ROM  # noqa: F401,E402  (registers the gym task)

# Resolve the task and its actuator HERE, at module scope: main() validates the
# resumed checkpoint before it builds the env, so a resolution placed further
# down is read before it is assigned (UnboundLocalError, caught 2026-08-06).
if a.task.startswith("RRLAB-Mulag-Verge-IDFD"):
    import importlib

    importlib.import_module("PPO_ID_FD")
    TASK_ID = a.task
    EXPECTED_ACTUATOR = "idfd"
else:
    TASK_ID = a.task or PPO_ROM.TERRAIN_TASK_ID
    EXPECTED_ACTUATOR = "rom"
print(f"[PPO3-1-SOLE] task={TASK_ID} actuator={EXPECTED_ACTUATOR}", flush=True)
from PPO_ROM.terrain_env_cfg import (  # noqa: E402
    TerrainC4AuthorityPPORunnerCfg,  # noqa: F401  (kept for compatibility)
    apply_c4_authority_stage,
)
from verge_task.terrain_env_cfg import (  # noqa: E402
    configure_terrain_gpu_physx_memory,
)
from verge_task.distill import assert_teacher_disabled  # noqa: E402
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
    override_cfg,
)
from verge_task.terrain_convergence import summarize_convergence  # noqa: E402


def apply_plant_overrides(cfg) -> None:
    """Re-assert the CLI c4-authority and plant overrides on top of a stage."""
    if a.cylinder_target_lead_c4_m is not None:
        override_cfg(cfg, "cylinder_target_lead_c4_m", float(a.cylinder_target_lead_c4_m))
    if a.c4_action_scale_gain is not None:
        override_cfg(cfg, "c4_action_scale_gain", float(a.c4_action_scale_gain))


@torch.inference_mode()
def convergence_rollout(raw, policy, steps: int) -> dict:
    """Deterministic rollout used as the stop condition for persistent PPO."""
    obs, _ = raw.reset(seed=int(a.seed) + 100_003)
    errors, accelerations, alignments = [], [], []
    laterals, scan_validity, contacts = [], [], []
    wheel_support, wheel_forces = [], []
    truck_heading_errors = []
    terminations = 0
    for _ in range(int(steps)):
        action = policy(obs)
        obs, _, terminated, _, _ = raw.step(action)
        errors.append(raw.cut_height_error().abs())
        accelerations.append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)
        )
        alignments.append(raw._measure_head_down_alignment())
        laterals.append(raw.verge_lateral_error().abs())
        scan_validity.append(raw.scan_valid_fraction())
        contacts.append(raw.cutter_ground_contact().float())
        wheel_support.append(raw.wheel_road_contact_fraction())
        wheel_forces.append(raw.wheel_road_contact_forces_n())
        truck_heading_errors.append(raw._truck_heading_error_rad)
        terminations += int(terminated.sum())

    error_steps = torch.stack(errors)
    lateral_steps = torch.stack(laterals)
    contact_steps = torch.stack(contacts)
    error = error_steps.reshape(-1)
    metrics = {
        "mean_abs_cut_height_error_m": float(error.mean()),
        "p95_abs_cut_height_error_m": float(torch.quantile(error, 0.95)),
        "p95_tool_acceleration_mps2": float(
            torch.quantile(torch.cat(accelerations), 0.95)
        ),
        "min_head_alignment": float(torch.cat(alignments).min()),
        "p01_head_alignment": float(torch.quantile(torch.cat(alignments), 0.01)),
        "mean_abs_lateral_error_m": float(lateral_steps.mean()),
        "scan_valid_fraction": float(torch.cat(scan_validity).mean()),
        "ground_contact_fraction": float(contact_steps.mean()),
        "min_wheel_road_contact_duty_cycle": float(
            (
                torch.stack(wheel_forces)
                > float(getattr(raw.cfg, "road_contact_force_threshold_n", 10.0))
            ).float().mean(dim=(0, 1)).min()
        ),
        "max_truck_post_constraint_drift_m": float(
            raw._max_truck_post_constraint_drift.max()
        ),
        "max_truck_heading_error_deg": float(
            torch.rad2deg(torch.cat(truck_heading_errors).max())
        ),
        "safety_termination_rate": terminations / max(steps * raw.num_envs, 1),
    }
    family_rows = []
    for family in sorted(set(raw._tracking_path_family_names)):
        family_mask = torch.as_tensor(
            [name == family for name in raw._tracking_path_family_names],
            dtype=torch.bool,
            device=raw.device,
        )
        family_rows.append(
            {
                "family": family,
                "num_envs": int(family_mask.sum()),
                "mean_abs_cut_height_error_m": float(
                    error_steps[:, family_mask].mean()
                ),
                "mean_abs_lateral_error_m": float(
                    lateral_steps[:, family_mask].mean()
                ),
                "ground_contact_fraction": float(
                    contact_steps[:, family_mask].mean()
                ),
            }
        )
    metrics.update(
        {
            "worst_family_mean_abs_cut_height_error_m": max(
                row["mean_abs_cut_height_error_m"] for row in family_rows
            ),
            "worst_family_mean_abs_lateral_error_m": max(
                row["mean_abs_lateral_error_m"] for row in family_rows
            ),
            "worst_family_ground_contact_fraction": min(
                row["ground_contact_fraction"] for row in family_rows
            ),
        }
    )
    result = summarize_convergence(metrics)
    result["path_family_results"] = family_rows
    return result


def main() -> None:
    resume_path = Path(a.resume).resolve() if a.resume else None
    resume_payload = None
    resume_source = "random ablation"
    teacher_handoff = False
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        infos = resume_payload.get("infos") or {}
        resume_source = str(infos.get("source", ""))
        schema = str(infos.get("schema", ""))
        if schema not in {"terrain_teacher_student_v1", "terrain_shared_v2"} or resume_source not in {
            "PPO3 terrain teacher distillation v1",
            "PPO3 terrain PPO",
            "PPO3-1 terrain PPO",
        }:
            raise RuntimeError(
                "checkpoint is outside the supported ppo3_1 terrain contracts "
                f"(source={resume_source!r}, schema={schema!r})"
            )
        teacher_handoff = resume_source == "PPO3 terrain teacher distillation v1"
        # Refuse a checkpoint trained on a different actuator. Without this the
        # script happily fine-tunes an ID/FD-distilled student on the ROM plant
        # (measured 2026-08-06: run manifest read actuator=rom for a PPO4_1
        # checkpoint). The two are structurally interchangeable -- same 66-wide
        # observation, same 4-wide action -- so nothing else would catch it.
        from verge_task.actuators import assert_checkpoint_actuator

        assert_checkpoint_actuator(
            infos, EXPECTED_ACTUATOR, context="PPO3-1 training resume"
        )

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=a.device,
        num_envs=a.num_envs,
    )
    apply_plant_overrides(env_cfg)
    env_cfg.observation_space = observation_space_for_cfg(env_cfg)
    env_cfg.enable_contact_tracking_demo = False
    if resume_payload is not None:
        resume_infos = resume_payload.get("infos") or {}
        checkpoint_width = int(resume_infos.get("observation_width", -1))
        if checkpoint_width != int(env_cfg.observation_space):
            raise RuntimeError(
                f"checkpoint observation width {checkpoint_width} does not match "
                f"the ppo3_1 contract {env_cfg.observation_space}; migrate with "
                "verge_task/scripts/migrate_shared_policy.py --mode "
                "widen_observation --target_observations 66"
            )
        checkpoint_action_width = int(resume_infos.get("action_width", 4))
        if checkpoint_action_width != int(env_cfg.action_space):
            raise RuntimeError(
                f"checkpoint action width {checkpoint_action_width} does not "
                f"match environment width {env_cfg.action_space}"
            )
    if a.vehicle_speed_min_kmh is not None:
        # Per-EPISODE speed randomization, ported from the teacher_student
        # repo's `random_driving_speed` (uniform draw, constant for the
        # episode). That repo wrote the sample to the wheel joints; this asset
        # is advanced kinematically (`_truck_travel_m += speed/3.6 * dt`) and
        # its wheels will not drive it, so the same draw is applied to the
        # kinematic speed instead. `_reset_idx` already redraws per episode
        # from [min, max] -- previously a no-op because both bounds were the
        # same number.
        env_cfg.vehicle_speed_min_kmh = float(a.vehicle_speed_min_kmh)
        env_cfg.vehicle_speed_max_kmh = float(a.vehicle_speed_max_kmh)
        # The PPO2 obstacle task ALSO resamples every 10 steps mid-episode.
        # That is a different contract: it would make the truck accelerate and
        # decelerate under the arm within one rollout, which is not what the
        # teacher_student regime does and would confound the tracking signal.
        # Push the threshold past any episode so reset is the only resample.
        env_cfg.vehicle_speed_resample_step_threshold = 10**9
    else:
        env_cfg.vehicle_speed_min_kmh = float(a.vehicle_speed_kmh)
        env_cfg.vehicle_speed_max_kmh = float(a.vehicle_speed_kmh)
    env_cfg.terrain_base_seed = int(a.seed)
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))

    # Ppo31PPORunnerCfg, NOT TerrainC4AuthorityPPORunnerCfg: the merged config
    # carries the Isaac Lab standard regime (lr 1e-3 ADAPTIVE, entropy 0.005,
    # init_std 0.30). Instantiating the parent here silently trained at the old
    # fine-tuning hyperparameters -- measured from a run manifest: lr 0.0003,
    # entropy 0.001, i.e. exactly the regime the rebuild exists to replace.
    from PPO_ROM.env_cfg import Ppo31PPORunnerCfg

    runner_cfg = Ppo31PPORunnerCfg()
    runner_cfg.seed = int(a.seed)
    runner_cfg.max_iterations = int(a.stage_iterations)
    runner_cfg.save_interval = int(a.save_interval)
    if a.learning_rate is not None:
        runner_cfg.algorithm.learning_rate = float(a.learning_rate)
    if a.entropy_coef is not None:
        runner_cfg.algorithm.entropy_coef = float(a.entropy_coef)
    if a.desired_kl is not None:
        runner_cfg.algorithm.desired_kl = float(a.desired_kl)
    if a.init_std is not None:
        runner_cfg.actor.distribution_cfg.init_std = float(a.init_std)

    apply_c4_authority_stage(env_cfg, stages[0])
    apply_plant_overrides(env_cfg)
    # Deliberately NOT overridden here any more. This line used to re-impose
    # (0.35 ... 1.15) on top of whatever the task config set, which would
    # silently undo the feasibility-derived terrain in Ppo31SoleEnvCfg and put
    # the run back on ground the plant cannot track. The task config owns this.
    assert_teacher_disabled(env_cfg, context="PPO3-1 PPO training")
    env = gym.make(
        TASK_ID, cfg=env_cfg, render_mode=None
    )
    raw = env.unwrapped
    raw.reset(seed=int(a.seed))
    reset_obs = raw._get_observations()["policy"]
    if tuple(reset_obs.shape) != (raw.num_envs, int(raw.cfg.observation_space)):
        raise RuntimeError(
            f"reset observation shape {tuple(reset_obs.shape)} violates the "
            "ppo3_1 contract"
        )
    if not bool(torch.isfinite(reset_obs).all()):
        raise RuntimeError("non-finite reset observation; refusing to start PPO")

    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    log_dir = (
        Path(a.log_dir).expanduser().resolve()
        if a.log_dir
        else ROOT
        / "logs/rsl_rl"
        / runner_cfg.experiment_name
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=str(log_dir), device=a.device
    )

    startup_manifest = {
        "schema": "terrain_shared_v2",
        "actuator": EXPECTED_ACTUATOR,
        "experiment": "ppo3_1_sole_c4_authority",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(log_dir),
        "initialization_source": resume_source,
        "resume": str(resume_path) if resume_path is not None else None,
        "teacher_mode": str(raw.cfg.sole_teacher_mode),
        "num_envs": int(raw.num_envs),
        "observation_width": int(raw.cfg.observation_space),
        "action_width": int(raw.cfg.action_space),
        "c4_authority": raw.c4_authority_report(),
        "path_families": sorted(set(raw._tracking_path_family_names)),
        "terrain_base_seed": int(raw._mow_strip_cfg().seed),
        "terrain_height_scales": sorted(
            set(float(value) for value in raw._terrain_height_scales)
        ),
        "learning_rate": float(runner_cfg.algorithm.learning_rate),
        "entropy_coef": float(runner_cfg.algorithm.entropy_coef),
        "desired_kl": float(runner_cfg.algorithm.desired_kl),
        "std_clamp": list(std_clamp_bounds) if std_clamp_bounds else None,
        "freeze_actor_iterations": int(a.freeze_actor_iterations),
        "seed": int(a.seed),
        "stages": list(stages),
    }
    (log_dir / "run_manifest.json").write_text(
        json.dumps(startup_manifest, indent=2, sort_keys=True)
    )
    print("[PPO3-1-RUN] " + json.dumps(startup_manifest, sort_keys=True), flush=True)

    if resume_path is not None:
        if teacher_handoff:
            load_cfg = {
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            }
        elif a.reset_optimizer_on_resume:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            }
        else:
            load_cfg = None
        runner.load(str(resume_path), load_cfg=load_cfg)
        if teacher_handoff or a.reset_critic_head_on_resume:
            critic_output = runner.alg.critic.mlp[-1]
            if not isinstance(critic_output, torch.nn.Linear):
                raise RuntimeError("terrain critic output is not Linear")
            with torch.no_grad():
                critic_output.weight.zero_()
                critic_output.bias.zero_()
        if a.resume_std is not None:
            distribution = runner.alg.actor.distribution
            with torch.no_grad():
                distribution.std_param[:] = float(a.resume_std)
            print(
                f"[PPO3-1-VERGE] set exploration std to {float(a.resume_std)} "
                "on all actions after resume",
                flush=True,
            )
        if a.resume_c4_std is not None:
            distribution = runner.alg.actor.distribution
            with torch.no_grad():
                distribution.std_param[3] = float(a.resume_c4_std)
            print(
                f"[PPO3-1-VERGE] set c4 exploration std to {float(a.resume_c4_std)} "
                "after resume",
                flush=True,
            )
        if a.learning_rate is None:
            resumed_lr = float(runner.alg.optimizer.param_groups[0]["lr"])
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = resumed_lr
            runner.alg.learning_rate = resumed_lr
        print(f"[PPO3-1-SOLE] resumed from {a.resume}", flush=True)
        if teacher_handoff:
            print(
                "[PPO3-1-TEACHER-HANDOFF] loaded actor+normalizer only; "
                "teacher is off, critic and PPO optimizer are new",
                flush=True,
            )
    else:
        bias = torch.as_tensor(
            tuple(getattr(raw.cfg, "terrain_initial_action_bias", (0.0, 0.0, 0.0, 0.0))),
            dtype=torch.float32,
            device=a.device,
        )
        output_layer = runner.alg.get_policy().mlp[-1]
        if not isinstance(output_layer, torch.nn.Linear) or output_layer.bias is None:
            raise RuntimeError("terrain actor output layer is not a biased Linear layer")
        with torch.no_grad():
            output_layer.bias.copy_(bias)
        print(f"[PPO3-1-VERGE] initialized actor action bias={bias.tolist()}", flush=True)

    # ------------------------------------------------------------------
    # Critic diagnostics: EXPLAINED VARIANCE, not raw value loss.
    #
    # Raw value loss is uninterpretable on its own because it scales with the
    # square of the return magnitude. This task's per-step reward is ~11.7 and
    # gamma is 0.99, so returns are ~1170 and a value loss of 16,000 is an RMS
    # residual of 128 -- 11% of the return. Isaac Lab's locomotion tasks reward
    # ~1.5/step, so their returns are ~150 and the SAME relative accuracy shows
    # up as a value loss of ~260. Comparing our 16,000 against their ~10 and
    # concluding the critic is broken is a units error, and it is the reasoning
    # the "value loss stays 6k-26k" claim in the handover rests on.
    #
    # Explained variance is scale-free and answers the question that actually
    # matters: 1 - Var(returns - values) / Var(returns).
    #   ~0  the critic predicts nothing; advantages ARE noise (the claimed bug)
    #   ~1  the critic explains the returns; residual is irreducible
    # A high value loss with EV ~0.9 is a healthy critic on a large-reward task,
    # and calls for no fix at all.
    # NOTE: this name must stay distinct from the std-clamp wrapper's. Both
    # wrappers rebind `runner.alg.update`, and a shared variable name would
    # make the inner closure resolve to the outer wrapper at call time --
    # infinite recursion on the first update.
    undiagnosed_alg_update = runner.alg.update

    def update_with_diagnostics(*args, **kwargs):
        # Read the storage BEFORE update() consumes it. `compute_returns` has
        # already run by this point in `runner.learn`, so returns are final.
        #
        # Every part of this is wrapped: an instrument must never be able to
        # kill the run it is measuring. A diagnostic that crashes at iteration
        # 400 of 600 would cost more than the diagnostic is worth.
        stats = None
        try:
            storage = runner.alg.storage
            with torch.no_grad():
                returns = storage.returns.flatten().float()
                values = storage.values.flatten().float()
                return_var = returns.var()
                if float(return_var) > 1.0e-8:
                    stats = {
                        "Diagnostics/explained_variance": float(
                            1.0 - (returns - values).var() / return_var
                        ),
                        "Diagnostics/return_mean": float(returns.mean()),
                        "Diagnostics/return_std": float(return_var.sqrt()),
                    }
        except Exception as exc:  # noqa: BLE001 - never fail the training step
            print(f"[PPO3-1-DIAG] skipped: {exc}", flush=True)

        # ONE-TIME CRITIC BIAS WARM-START.
        #
        # `teacher_handoff` zeroes the critic's output head, so the critic
        # starts predicting 0 while returns average ~121. Adam at lr 3e-4 moves
        # that output by only ~0.36 per iteration (measured), so reaching the
        # right magnitude takes ~340 iterations of pure ramp -- the frozen
        # window would end with the critic still predicting near zero, and the
        # actor would unfreeze onto exactly the noise-dominated advantages the
        # warm-up exists to prevent. Measured before this fix: value loss
        # 19,806 against a return variance of only 5,878, i.e. the error is
        # dominated by a constant offset, not by unexplained variance.
        #
        # Setting the output bias to the observed mean return skips the ramp
        # entirely and leaves the critic only the part it should be learning:
        # the variance. This is standard value-function initialization, and the
        # data needed is already in the storage we just read.
        if stats is not None and not update_with_diagnostics.bias_initialized:
            update_with_diagnostics.bias_initialized = True
            try:
                head = runner.alg.critic.mlp[-1]
                if isinstance(head, torch.nn.Linear) and head.bias is not None:
                    before = float(head.bias.detach().mean())
                    if abs(before) < 1.0e-6:  # only when it really was zeroed
                        # Target the TRUE value-function fixed point, not the
                        # return currently in storage. With a zeroed critic,
                        # GAE bootstraps off V=0, so the stored return is only
                        # r/(1 - gamma*lam) -- measured 100 here against a true
                        # V* of ~1160, a 12x underestimate. Seeding the bias
                        # with it left the critic still ramping (returns had
                        # only reached 222 after 26 iterations and were still
                        # climbing), which is the same slow ramp in miniature.
                        # The reward stream gives the fixed point directly:
                        #   V* = E[r] / (1 - gamma)
                        gamma = float(runner.alg.gamma)
                        mean_reward = float(runner.alg.storage.rewards.mean())
                        target = mean_reward / max(1.0 - gamma, 1.0e-6)
                        with torch.no_grad():
                            head.bias.fill_(target)
                        print(
                            f"[PPO3-1-WARMSTART] critic output bias {before:.4f} "
                            f"-> {target:.1f}  (mean reward {mean_reward:.3f} / "
                            f"(1-gamma={gamma:g})); the stored bootstrapped "
                            f"return was only "
                            f"{stats['Diagnostics/return_mean']:.1f}",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001 - never fail the run
                print(f"[PPO3-1-WARMSTART] skipped: {exc}", flush=True)

        result = undiagnosed_alg_update(*args, **kwargs)

        # The writer lives on runner.logger, NOT on runner. A plain
        # getattr(runner, "writer", None) returns None and drops every stat
        # with no exception and no log line -- the diagnostic appears to be
        # running while writing nothing, which is worse than not having it.
        writer = getattr(getattr(runner, "logger", None), "writer", None) or getattr(
            runner, "writer", None
        )
        if stats and writer is not None:
            try:
                step = int(runner.current_learning_iteration)
                for tag, value in stats.items():
                    writer.add_scalar(tag, value, step)
            except Exception as exc:  # noqa: BLE001
                print(f"[PPO3-1-DIAG] write skipped: {exc}", flush=True)
        elif stats and not update_with_diagnostics.warned:
            # Say so once, loudly, rather than producing a run with a silently
            # missing metric that only shows up as a blank section at 3am.
            update_with_diagnostics.warned = True
            print(
                "[PPO3-1-DIAG] WARNING: no TensorBoard writer found; "
                "explained_variance will NOT be recorded for this run",
                flush=True,
            )
        return result

    update_with_diagnostics.warned = False
    update_with_diagnostics.bias_initialized = False

    runner.alg.update = update_with_diagnostics
    print(
        "[PPO3-1-VERGE] logging Diagnostics/explained_variance -- the scale-free "
        "critic health metric (raw value loss scales with return magnitude)",
        flush=True,
    )

    if std_clamp_bounds is not None:
        clamp_distribution = runner.alg.actor.distribution
        if not hasattr(clamp_distribution, "std_param"):
            raise RuntimeError(
                "--std_clamp requires a scalar-std Gaussian actor "
                f"(got {type(clamp_distribution).__name__})"
            )
        original_alg_update = runner.alg.update

        def update_with_std_clamp(*args, **kwargs):
            result = original_alg_update(*args, **kwargs)
            with torch.no_grad():
                clamp_distribution.std_param.clamp_(*std_clamp_bounds)
            return result

        runner.alg.update = update_with_std_clamp
        print(
            f"[PPO3-1-VERGE] exploration std clamped to {list(std_clamp_bounds)} "
            "after every PPO update",
            flush=True,
        )

    original_runner_save = runner.save

    def save_with_terrain_provenance(path: str, infos=None) -> None:
        checkpoint_infos = {
            "source": "PPO3-1 terrain PPO",
            "schema": "terrain_shared_v2",
            "actuator": EXPECTED_ACTUATOR,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "iteration": int(runner.current_learning_iteration),
            "num_envs": int(raw.num_envs),
            "observation_width": int(raw.cfg.observation_space),
            "action_width": int(raw.cfg.action_space),
            "head_joint_action": bool(raw.cfg.sole_head_joint_action),
            "c4_authority": raw.c4_authority_report(),
            "initialization_source": resume_source,
            "teacher_enabled_during_ppo": False,
            "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
            "terrain_base_seed": int(raw._mow_strip_cfg().seed),
            "path_families": sorted(set(raw._tracking_path_family_names)),
            "terrain_height_scales": sorted(
                set(float(value) for value in raw._terrain_height_scales)
            ),
        }
        if infos:
            checkpoint_infos.update(infos)
        original_runner_save(path, infos=checkpoint_infos)

    runner.save = save_with_terrain_provenance

    def critic_warmup(freeze: int) -> None:
        """Critic warm-up with a frozen actor.

        With the actor's requires_grad=False, loss.backward() leaves its grads
        None and the shared Adam step() skips those parameters entirely, so the
        actor (mean net AND exploration std -- the distribution is an actor
        submodule) cannot move while the critic learns against its rollouts.
        The obs-normalizer buffers are unaffected and keep adapting.

        The adaptive KL scheduler is suspended for the window: the normalizer
        keeps moving between rollout and recompute, so the update KL of a
        frozen actor is tiny but NONZERO, and ``adaptive`` would ratchet the
        shared learning rate x1.5 per minibatch to its 1e-2 cap. The first
        unfrozen updates would then fire at ~30x the configured rate --
        exactly the student-destroying jump this warm-up exists to prevent.
        At unfreeze the schedule is restored and the learning rate reset to
        the configured value.
        """
        schedule = runner.alg.schedule
        base_lr = float(runner.alg.learning_rate)
        runner.alg.schedule = "fixed"
        runner.alg.actor.requires_grad_(False)
        # FREEZE THE OBSERVATION NORMALIZER TOO. requires_grad_(False) stops the
        # WEIGHTS moving, but the normalizer keeps adapting, and its statistics
        # are part of the policy: the actor sees (obs - mean) / std, so a moving
        # normalizer changes behaviour even with identical weights.
        #
        # Measured 2026-08-06, actor frozen throughout, lateral error per
        # iteration:
        #   ROM-A   0.034 0.039 0.037 0.030 ... 0.040   (frozen-window mean 0.0402)
        #   ID/FD   0.093 0.163 0.313 0.469 ... 1.13    (frozen-window mean 0.6996)
        # The same ID/FD checkpoint scores 0.0628 in evaluation, where the
        # normalizer is held fixed. So the drift is the normalizer, and a
        # "frozen" window that lets it move is not a controlled baseline -- the
        # policy it hands to the fine-tune is not the policy that was measured.
        # ID/FD is markedly more sensitive to this than ROM-A, which is itself a
        # result worth keeping.
        _norm_modules = [
            mod
            for owner in (runner.alg.actor, runner.alg.critic)
            for mod in owner.modules()
            if type(mod).__name__ == "EmpiricalNormalization"
        ]
        for _mod in _norm_modules:
            _mod.eval()  # update() is a no-op unless self.training
        print(
            f"[PPO3-1-WARMUP] froze {len(_norm_modules)} observation "
            "normalizer(s) for the window; a moving normalizer changes a "
            "frozen actor's behaviour",
            flush=True,
        )
        runner.learn(num_learning_iterations=freeze, init_at_random_ep_len=False)
        runner.alg.actor.requires_grad_(True)
        for _mod in _norm_modules:
            _mod.train()  # resume adapting once the actor is allowed to move
        runner.alg.schedule = schedule
        runner.alg.learning_rate = base_lr
        for param_group in runner.alg.optimizer.param_groups:
            param_group["lr"] = base_lr
        print(
            f"[PPO3-1-WARMUP] actor frozen for {freeze} iterations "
            f"(critic-only updates at fixed lr); actor unfrozen, schedule "
            f"{schedule!r} restored, lr reset to {base_lr}",
            flush=True,
        )

    final_stage = stages[-1]
    last_convergence_result = None
    for stage in stages:
        apply_c4_authority_stage(raw.cfg, stage)
        apply_plant_overrides(raw.cfg)
        raw.cfg.trajectory_shape = str(raw.cfg.terrain_deployment_trajectory_shape)
        raw.cfg.randomize_trajectory_shape = False
        raw.cfg.randomize_start_phase = False
        print(
            f"[PPO3-1-VERGE] === stage {stage} === "
            f"truck={raw.cfg.vehicle_speed_min_kmh:.2f} km/h "
            f"c4_lead={raw.cfg.cylinder_target_max_lead_per_axis_m[3]:.3f} m "
            f"c4_scale={raw.cfg.cylinder_action_scale_per_axis_m[3]:.4f} m/unit "
            f"filter={raw.cfg.sole_action_filter:.2f} "
            + (
                f"until convergence (checks every {a.convergence_check_interval} iterations)"
                if a.until_converged and stage == final_stage
                else f"for {a.stage_iterations} iterations"
            ),
            flush=True,
        )
        # Warm-up runs at the start of the FIRST stage in both branches. The
        # previous structure skipped it silently when the first stage was also
        # the --until_converged final stage, which is precisely the production
        # invocation (--stages 6 --until_converged).
        warmed_up = 0
        if stage == stages[0] and a.freeze_actor_iterations > 0:
            warmed_up = int(a.freeze_actor_iterations)
            critic_warmup(warmed_up)
        if not (a.until_converged and stage == final_stage):
            remaining_total = int(a.stage_iterations) - warmed_up
            if remaining_total > 0:
                runner.learn(
                    num_learning_iterations=remaining_total,
                    init_at_random_ep_len=False,
                )
            continue

        while True:
            runner.learn(
                num_learning_iterations=int(a.convergence_check_interval),
                init_at_random_ep_len=False,
            )
            policy = runner.get_inference_policy(device=a.device)
            result = convergence_rollout(raw, policy, int(a.convergence_steps))
            last_convergence_result = result
            runner.alg.train_mode()
            result.update(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "iteration": int(runner.current_learning_iteration),
                    "num_envs": int(raw.num_envs),
                    "stage": int(stage),
                }
            )
            convergence_report = log_dir / "convergence_latest.json"
            convergence_report.write_text(json.dumps(result, indent=2))
            runner.save(
                str(log_dir / f"model_{runner.current_learning_iteration}.pt"),
                infos={"convergence": result},
            )
            print(
                "[PPO3-1-CONVERGENCE] " + json.dumps(result, sort_keys=True),
                flush=True,
            )
            if result["passed"]:
                converged_path = log_dir / "model_converged.pt"
                runner.save(
                    str(converged_path),
                    infos={"convergence": result},
                )
                print(
                    f"[PPO3-1-CONVERGENCE] PASSED; wrote {converged_path}",
                    flush=True,
                )
                break

    if a.checkpoint_out:
        checkpoint_out = Path(a.checkpoint_out).expanduser().resolve()
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_info = {
            "source": "PPO3-1 terrain PPO",
            "schema": "terrain_shared_v2",
            "actuator": EXPECTED_ACTUATOR,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "iteration": int(runner.current_learning_iteration),
            "num_envs": int(raw.num_envs),
            "stage": int(final_stage),
            "seed": int(a.seed),
            "learning_rate": float(runner.alg.optimizer.param_groups[0]["lr"]),
            "convergence": last_convergence_result,
            "observation_width": int(raw.cfg.observation_space),
            "action_width": int(raw.cfg.action_space),
            "head_joint_action": bool(raw.cfg.sole_head_joint_action),
            "c4_authority": raw.c4_authority_report(),
            "initialization_source": resume_source,
            "teacher_enabled_during_ppo": False,
            "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
            "terrain_base_seed": int(raw._mow_strip_cfg().seed),
            "path_families": sorted(set(raw._tracking_path_family_names)),
            "terrain_height_scales": sorted(
                set(float(value) for value in raw._terrain_height_scales)
            ),
        }
        runner.save(str(checkpoint_out), infos=checkpoint_info)
        manifest = checkpoint_out.with_suffix(".json")
        manifest_row = {"checkpoint": str(checkpoint_out), **checkpoint_info}
        manifest.write_text(json.dumps(manifest_row, indent=2))
        print(
            "[PPO3-1-CHECKPOINT] " + json.dumps(manifest_row, sort_keys=True),
            flush=True,
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
