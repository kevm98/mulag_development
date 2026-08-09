"""Prove the 5th action has authority over Messerkopf_joint.

Commands a constant +1 and -1 head action with cylinders held at zero and
reports how far the head joint and the world-down alignment actually move.
A vacuous channel shows identical numbers for both signs.
"""
import argparse, torch
from isaaclab.app import AppLauncher
import sys; sys.path.insert(0, "/home/rrlab/rrlab_mulag")

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(["--headless", "--device", "cuda:0"])
app = AppLauncher(a).app

import gymnasium as gym
import verge_task
from verge_task.terrain_env_cfg import (
    TerrainSolePolicyEnvCfg, apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)

cfg = TerrainSolePolicyEnvCfg()
apply_terrain_curriculum_stage(cfg, 5)
cfg.scene.num_envs = 8
configure_terrain_gpu_physx_memory(cfg, 8)
env = gym.make(verge_task.TERRAIN_TASK_ID, cfg=cfg, render_mode=None)
raw = env.unwrapped
print("action_space =", cfg.action_space, "| head joint action =", cfg.sole_head_joint_action)

tgt = raw._joint6_coupling_target_ref_index
jid = raw._joint6_coupling_target_joint_id
print("Messerkopf_joint ref index =", tgt, "joint id =", jid)

def run(sign, steps=40):
    raw.reset(seed=17)
    act = torch.zeros((raw.num_envs, cfg.action_space), device=raw.device)
    act[:, 4] = sign
    start_ref = float(raw._joint_ref[:, tgt].mean())
    for _ in range(steps):
        raw.step(act)
    import torch as T
    q = raw.robot.data.joint_pos[:, jid]
    ax = raw._head_down_axis_world()
    tilt = T.atan2(ax[:, 1], -ax[:, 2])
    return (start_ref, float(raw._joint_ref[:, tgt].mean()), float(q.mean()),
            float(tilt.mean()))

for s in (+1.0, -1.0):
    s0, ref, q, align = run(s)
    print("head action %+.0f : ref %.5f -> %.5f | q=%.5f | signed_tilt=%+.5f rad"
          % (s, s0, ref, q, align))
app.close()
