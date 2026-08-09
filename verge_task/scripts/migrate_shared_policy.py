#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Migrate PPO3/PPO4 checkpoint interfaces with explicit invariants.

Three migrations live here:

``migrate_checkpoint``      legacy 54x4 ROM policy  -> shared 66x5 contract
``widen_observation``       any width               -> a wider observation
``collapse_redundant_q5``   redundant five actions  -> four physical actions

The second is what the contact-safe experiment needs. Adding the six
contact-state channels (and raising the model-error history depth) changes the
actor and critic input layer, and silently loading a narrower checkpoint into
it would either raise deep inside rsl_rl or -- worse -- succeed against a
reshaped tensor and quietly permute every input the policy learned.

New input columns get ZERO weight and a neutral normalizer (mean 0, var 1,
std 1), so the migrated network computes exactly what the old one computed
until training moves those weights. That is the property the tests assert:
the migration must be a no-op on behaviour, not an approximation of one.

Observation widening and legacy expansion are behaviour-preserving. The q5
collapse exactly combines the actor's linear outputs before clipping, but the
old environment clipped c4 and q5 separately. It is therefore a principled
initialization, not a claim of full-rollout parity; simulator validation is
required.

Migration always sets ``migration_requires_optimizer_reset``. The Adam moments
are indexed by parameter shape; carrying them across a changed layer applies
stale moments to parameters with a new meaning or shape.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch


OLD_OBSERVATIONS = 54
SHARED_OBSERVATIONS = 66
OLD_ACTIONS = 4
SHARED_ACTIONS = 5

#: Normalizer buffers and the neutral value each must take on a NEW column.
#: Mean 0 / var 1 / std 1 is the identity transform, so a zero-weight column
#: stays zero-weight rather than being scaled by a stale statistic.
NORMALIZER_FILLS = (
    ("obs_normalizer._mean", 0.0),
    ("obs_normalizer._var", 1.0),
    ("obs_normalizer._std", 1.0),
)


def _append_normalizer_channels(
    state: dict[str, torch.Tensor], *, source: int, target: int
) -> None:
    for key, fill in NORMALIZER_FILLS:
        value = state[key]
        if tuple(value.shape) != (1, source):
            raise ValueError(
                f"{key} has shape {tuple(value.shape)}, expected (1, {source})"
            )
        extra = torch.full(
            (1, target - source), fill, dtype=value.dtype, device=value.device
        )
        state[key] = torch.cat((value, extra), dim=1)


def _append_input_weights(
    state: dict[str, torch.Tensor], *, source: int, target: int
) -> None:
    weight = state["mlp.0.weight"]
    if weight.shape[1] != source:
        raise ValueError(
            f"mlp.0.weight has {weight.shape[1]} inputs, expected {source}"
        )
    zeros = torch.zeros(
        (weight.shape[0], target - source), dtype=weight.dtype, device=weight.device
    )
    state["mlp.0.weight"] = torch.cat((weight, zeros), dim=1)


def _widen_inputs(
    payload: dict, *, source: int, target: int
) -> None:
    if target < source:
        raise ValueError(
            f"cannot narrow an observation: {source} -> {target}. Columns are "
            "positional and dropping them would permute every later channel."
        )
    for key in ("actor_state_dict", "critic_state_dict"):
        state = payload[key]
        _append_normalizer_channels(state, source=source, target=target)
        _append_input_weights(state, source=source, target=target)


def widen_observation(
    payload: dict,
    *,
    source_observations: int,
    target_observations: int,
) -> dict:
    """Return a copy whose actor/critic accept ``target_observations`` inputs.

    Actions are untouched. Use this for the contact-state channels, which add
    observations only -- the policy still emits the same five outputs.
    """
    migrated = copy.deepcopy(payload)
    if target_observations == source_observations:
        return migrated
    _widen_inputs(
        migrated, source=source_observations, target=target_observations
    )
    infos = dict(migrated.get("infos") or {})
    infos.update(
        {
            "observation_width": int(target_observations),
            "migrated_observation_from": int(source_observations),
            "migration_requires_optimizer_reset": True,
            "migrated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    migrated["infos"] = infos
    return migrated


def migrate_checkpoint(
    payload: dict,
    *,
    q5_std: float = 0.04,
    source_observations: int = OLD_OBSERVATIONS,
    target_observations: int = SHARED_OBSERVATIONS,
) -> dict:
    """Return a behavior-preserving 66x5 copy of a legacy ROM checkpoint."""
    if q5_std <= 0.0:
        raise ValueError("q5_std must be positive")
    migrated = copy.deepcopy(payload)
    _widen_inputs(
        migrated, source=source_observations, target=target_observations
    )
    actor = migrated["actor_state_dict"]

    output_weight = actor["mlp.6.weight"]
    output_bias = actor["mlp.6.bias"]
    std = actor["distribution.std_param"]
    if output_weight.shape[0] != OLD_ACTIONS or output_bias.numel() != OLD_ACTIONS:
        raise ValueError("actor output is not the legacy four-action contract")
    if std.numel() != OLD_ACTIONS:
        raise ValueError("actor exploration std is not four-wide")
    actor["mlp.6.weight"] = torch.cat(
        (output_weight, torch.zeros_like(output_weight[:1])), dim=0
    )
    actor["mlp.6.bias"] = torch.cat((output_bias, torch.zeros_like(output_bias[:1])))
    actor["distribution.std_param"] = torch.cat(
        (std, torch.full_like(std[:1], float(q5_std)))
    )

    infos = dict(migrated.get("infos") or {})
    infos.update(
        {
            "source": "PPO3 terrain PPO",
            "schema": "terrain_shared_v2",
            "actuator": "rom",
            "observation_width": int(target_observations),
            "action_width": SHARED_ACTIONS,
            "head_joint_action": True,
            "head_action_to_c4_gain": 0.4,
            "migration_requires_optimizer_reset": True,
            "migrated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    migrated["infos"] = infos
    return migrated


def collapse_redundant_q5(
    payload: dict,
    *,
    head_action_to_c4_gain: float = 0.4,
) -> dict:
    """Fold the redundant q5 output into c4 and return a four-action copy.

    The environment applies ``c4_effective = c4 + gain * q5``; the advertised
    q5 correction integrator is not called. Combining the two output rows
    preserves the deterministic actor before the old per-channel clipping. For
    independent Gaussian exploration, variances add, so the new c4 standard
    deviation is ``sqrt(std_c4**2 + (gain * std_q5)**2)``.
    """
    gain = float(head_action_to_c4_gain)
    if not torch.isfinite(torch.tensor(gain)):
        raise ValueError("head_action_to_c4_gain must be finite")

    migrated = copy.deepcopy(payload)
    actor = migrated["actor_state_dict"]
    output_weight = actor["mlp.6.weight"]
    output_bias = actor["mlp.6.bias"]
    std = actor["distribution.std_param"]
    if output_weight.shape[0] != SHARED_ACTIONS:
        raise ValueError("actor output is not the redundant five-action contract")
    if output_bias.numel() != SHARED_ACTIONS or std.numel() != SHARED_ACTIONS:
        raise ValueError("actor bias/exploration std is not five-wide")

    combined_weight = output_weight[3] + gain * output_weight[4]
    combined_bias = output_bias[3] + gain * output_bias[4]
    combined_std = torch.sqrt(std[3].square() + (gain * std[4]).square())
    actor["mlp.6.weight"] = torch.cat(
        (output_weight[:3], combined_weight.unsqueeze(0)), dim=0
    )
    actor["mlp.6.bias"] = torch.cat((output_bias[:3], combined_bias.unsqueeze(0)))
    actor["distribution.std_param"] = torch.cat(
        (std[:3], combined_std.unsqueeze(0))
    )

    infos = dict(migrated.get("infos") or {})
    infos.update(
        {
            "action_width": OLD_ACTIONS,
            "head_joint_action": False,
            "migrated_action_from": SHARED_ACTIONS,
            "collapsed_head_action_gain": gain,
            "migration_requires_optimizer_reset": True,
            "migrated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    migrated["infos"] = infos
    return migrated


def expand_direct_q5(
    payload: dict,
    *,
    source_observations: int,
    target_observations: int,
    q5_std: float = 0.04,
) -> dict:
    """Warm-start direct q5 control from a four-action terrain checkpoint."""
    if q5_std <= 0.0:
        raise ValueError("q5_std must be positive")
    migrated = widen_observation(
        payload,
        source_observations=source_observations,
        target_observations=target_observations,
    )
    actor = migrated["actor_state_dict"]
    weight = actor["mlp.6.weight"]
    bias = actor["mlp.6.bias"]
    std = actor["distribution.std_param"]
    if weight.shape[0] != OLD_ACTIONS or bias.numel() != OLD_ACTIONS:
        raise ValueError("actor output is not a four-action checkpoint")
    if std.numel() != OLD_ACTIONS:
        raise ValueError("actor exploration std is not four-wide")
    actor["mlp.6.weight"] = torch.cat((weight, torch.zeros_like(weight[:1])), dim=0)
    actor["mlp.6.bias"] = torch.cat((bias, torch.zeros_like(bias[:1])))
    actor["distribution.std_param"] = torch.cat(
        (std, torch.full_like(std[:1], float(q5_std)))
    )
    infos = dict(migrated.get("infos") or {})
    infos.update(
        {
            "action_width": SHARED_ACTIONS,
            "head_joint_action": True,
            "head_control_mode": "direct_joint_position",
            "head_action_to_c4_gain": 0.0,
            "migrated_action_from": OLD_ACTIONS,
            "migration_requires_optimizer_reset": True,
            "migrated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    migrated["infos"] = infos
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--q5_std", type=float, default=0.04)
    parser.add_argument(
        "--mode",
        default="legacy_54x4",
        choices=(
            "legacy_54x4",
            "widen_observation",
            "expand_direct_q5",
            "collapse_redundant_q5",
        ),
        help=(
            "legacy_54x4 migrates the retired 54x4 ROM interface to 66x5; "
            "widen_observation only adds observation columns and keeps the "
            "action width, which is what the contact-state channels need; "
            "expand_direct_q5 also adds a neutral direct q5 output; "
            "collapse_redundant_q5 folds c4 + gain*q5 into one c4 output."
        ),
    )
    parser.add_argument(
        "--head_action_to_c4_gain",
        type=float,
        default=0.4,
        help="Mixer gain recorded by the five-action checkpoint (default: 0.4).",
    )
    parser.add_argument(
        "--source_observations",
        type=int,
        default=None,
        help="Defaults to the checkpoint's recorded observation_width.",
    )
    parser.add_argument(
        "--target_observations",
        type=int,
        default=SHARED_OBSERVATIONS,
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source == output:
        raise ValueError("source and output must differ; legacy checkpoints are immutable")
    payload = torch.load(source, map_location="cpu", weights_only=False)

    recorded = int((payload.get("infos") or {}).get("observation_width", -1))
    source_obs = (
        int(args.source_observations)
        if args.source_observations is not None
        else recorded
    )
    if source_obs <= 0:
        raise ValueError(
            "source observation width is unknown; pass --source_observations"
        )
    actual = int(payload["actor_state_dict"]["mlp.0.weight"].shape[1])
    if actual != source_obs:
        raise ValueError(
            f"checkpoint actor accepts {actual} inputs but the recorded/"
            f"requested source width is {source_obs}. Refusing to guess."
        )

    if args.mode == "legacy_54x4":
        migrated = migrate_checkpoint(
            payload,
            q5_std=float(args.q5_std),
            source_observations=source_obs,
            target_observations=int(args.target_observations),
        )
        label = f"{source_obs}x{OLD_ACTIONS} -> {args.target_observations}x{SHARED_ACTIONS}"
    elif args.mode == "widen_observation":
        migrated = widen_observation(
            payload,
            source_observations=source_obs,
            target_observations=int(args.target_observations),
        )
        label = f"{source_obs} -> {args.target_observations} observations"
    elif args.mode == "expand_direct_q5":
        migrated = expand_direct_q5(
            payload,
            source_observations=source_obs,
            target_observations=int(args.target_observations),
            q5_std=float(args.q5_std),
        )
        label = f"{source_obs}x4 -> {args.target_observations}x5 direct q5"
    else:
        if int(args.target_observations) != source_obs:
            raise ValueError(
                "collapse_redundant_q5 does not change observations; "
                "target_observations must equal the source width"
            )
        migrated = collapse_redundant_q5(
            payload,
            head_action_to_c4_gain=float(args.head_action_to_c4_gain),
        )
        label = f"{source_obs}x{SHARED_ACTIONS} -> {source_obs}x{OLD_ACTIONS}"
    migrated["infos"]["migrated_from"] = str(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output)
    print(f"[PPO3-MIGRATE] {source} -> {output} ({label})", flush=True)


if __name__ == "__main__":
    main()
