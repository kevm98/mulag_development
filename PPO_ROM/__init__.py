"""PPO3-1: sole-controller verge policy with c4 authority decoupling.

PPO3 and PPO4 share one measured failure: holding a sustained lateral offset
(lane_change_hold, offset_return, boom_sweep) tilts the head, and neither a
generic rate penalty nor a head-specific penalty separates "follow the path"
from "tilt the head". The handover (verge_task/HANDOVER.md section 4)
records the coupling as plausibly kinematic and names the untested axis:

    give c4 more authority (larger lead and/or action scale) so it can hold
    the head level while c1-c3 hold the offset. c4 lead was only swept to 0.20.

PPO3-1 is exactly that experiment, on the same plant. Everything PPO3/PPO4
share is inherited: the verge task, sensors, observation, rewards, safety
logic, terrain, path families, evaluation gates, and ROM-A actuator. The only
deliberate difference is that cylinder 4's authority is raised on two
independent axes:

    cylinder_target_max_lead_per_axis_m   c4 lead cap, metres
    cylinder_action_scale_per_axis_m      c4 action scale, metres/unit action

The policy remains the sole controller (no residual layer, no teacher in the
scored loop), exactly as in PPO3.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry


TERRAIN_TASK_ID = "RRLAB-Mulag-Verge-ROM-v0"
TERRAIN_PLAY_TASK_ID = "RRLAB-Mulag-Verge-ROM-PLAY-v0"


def _register_once(
    task_id: str, *, entry_point: str, env_cfg: str, runner_cfg: str
) -> None:
    if task_id in registry:
        return
    gym.register(
        id=task_id,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": runner_cfg,
        },
    )


_register_once(
    TERRAIN_TASK_ID,
    entry_point="PPO_ROM.env:Ppo31SoleEnv",
    env_cfg="PPO_ROM.env_cfg:Ppo31SoleEnvCfg",
    runner_cfg="PPO_ROM.env_cfg:Ppo31PPORunnerCfg",
)
_register_once(
    TERRAIN_PLAY_TASK_ID,
    entry_point="PPO_ROM.terrain_env:TerrainC4AuthorityEnv",
    env_cfg="PPO_ROM.terrain_env_cfg:TerrainC4AuthorityEnvCfg_PLAY",
    runner_cfg="PPO_ROM.terrain_env_cfg:TerrainC4AuthorityPPORunnerCfg",
)


__all__ = ["TERRAIN_TASK_ID", "TERRAIN_PLAY_TASK_ID"]
