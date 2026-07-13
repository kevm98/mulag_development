#!/usr/bin/env python3
"""PPO Stage-0 Y-Z training with checkpoint saving."""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage-0 Y-Z PPO training.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=100)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--action_scale_m", type=float, default=0.05)
parser.add_argument("--num_steps_per_env", type=int, default=8)
parser.add_argument("--save_interval", type=int, default=25)
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.stage0_yz_env_wrapper import Stage0YZEnvWrapper

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from isaaclab.utils.math import quat_apply_inverse
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
import gymnasium as gym

J_yz = torch.tensor([[-1.74,29.81,20.45,20.29],[-1.51,-24.73,-30.88,-31.66]],dtype=torch.float64)
LAMBDA = 1.0; MAX_CYL_DELTA = 0.005; STOL = 0.10; OBS_DIM = 4

class Stage0YZEnvWrapper(VecEnv):
    def __init__(self, cfg, horizon_steps, action_scale_m, device):
        cfg.scene.num_envs = cfg.scene.num_envs
        cfg.enable_scripted_vehicle_motion = cfg.enable_startup_summary = cfg.log_target_delta_xdot_clipping = False
        cfg.allow_debug_manual_cylinder_targets = True
        cfg.lowlevel_actuator_backend = "explicit_effort_pd"
        cfg.explicit_pd_gain_scale = 0.1; cfg.explicit_pd_damping_scale = 0.2; cfg.explicit_pd_effort_scale = 1.0
        cfg.explicit_force_rate_limit_fraction = 0.01; cfg.explicit_pd_velocity_filter_beta = 0.8
        if device: cfg.device = device; cfg.sim.device = device
        id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
        cfg.id_checkpoint_override_path = str(id_path)
        apply_kevin_sim_memory_optimizations(cfg.sim)
        self.cfg = cfg
        self._env = MulagUpperLevelEnv(cfg=cfg)
        self._uw = self._env.unwrapped if hasattr(self._env,'unwrapped') else self._env
        self._H = int(horizon_steps); self._action_scale = float(action_scale_m); self._device = device
        self._cyl_ids = self._uw._cylinder_joint_ids; self._tool_id = self._uw._tool_joint_id
        self._ee_id = self._uw._ee_body_id; self._dt = float(self._uw.step_dt)
        B = self.num_envs = self._uw.num_envs
        self.device = device; self.max_episode_length = 50; self.num_actions = 2; self.num_obs = OBS_DIM
        JT = J_yz.T.to(torch.float64); I2 = torch.eye(2,dtype=torch.float64)
        self._J_pinv = (JT @ torch.linalg.inv(J_yz.to(torch.float64) @ JT + LAMBDA**2 * I2)).float().to(device)
        self._target_yz = torch.zeros((B,2),device=device); self._prev_action = torch.zeros((B,2),device=device)
        self._ep_length = torch.zeros(B,dtype=torch.long,device=device)
        self._prev_distance = torch.zeros(B,device=device)
        self._root_pose = None; self._root_vel = torch.zeros((B,6),device=device); self._tool_target = None
        self.single_action_space = gym.spaces.Box(low=-1.0,high=1.0,shape=(2,),dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space,B)
    def _tcp_yz(self):
        ee_w=self._uw.robot.data.body_link_state_w[:,self._ee_id,:3]; bp=self._uw.robot.data.root_pos_w
        bq=self._uw.robot.data.root_quat_w; bq=bq/bq.norm(dim=-1,keepdim=True).clamp(min=1e-8)
        return quat_apply_inverse(bq,ee_w-bp)[:,1:3]
    def _pin_root(self):
        if self._root_pose is not None:
            self._uw.robot.write_root_pose_to_sim(self._root_pose); self._uw.robot.write_root_velocity_to_sim(self._root_vel)
    def _sample_target(self):
        B=self.num_envs; ee=self._tcp_yz()
        dy=(torch.rand(B,device=self._device)*0.10-0.05); dz=(torch.rand(B,device=self._device)*0.04-0.02)
        self._target_yz=ee+torch.stack([dy,dz],dim=-1)
    def _compute_reward(self,action):
        ee=self._tcp_yz(); d=torch.norm(ee-self._target_yz,dim=-1); p=self._prev_distance-d; an=torch.norm(action,dim=-1)
        r=10.0*p-1.0*d-0.001*an  # no success bonus (targets inside tolerance); self._prev_distance=d.detach(); return r,d,(d<STOL)
    def reset(self):
        self._env.reset(); B=self.num_envs
        self._root_pose=self._uw.robot.data.root_state_w[:,:7].clone(); self._root_vel.zero_()
        cqh=self._uw.robot.data.joint_pos[:,self._cyl_ids].clone()
        self._tool_target=self._uw.robot.data.joint_pos[:,self._tool_id].unsqueeze(-1).clone()
        for _ in range(10): self._pin_root(); self._uw.set_debug_manual_cylinder_targets(cqh,tool_position_target=self._tool_target); self._env.step(torch.zeros((B,5),device=self._device))
        self._sample_target(); self._prev_action.zero_()
        self._prev_distance=torch.norm(self._tcp_yz()-self._target_yz,dim=-1); self._ep_length.zero_()
        return self._get_obs(),{}
    def _get_obs(self): return torch.cat([self._target_yz - self._tcp_yz(), self._prev_action], dim=-1)
    def _to_tensordict(self,obs): return TensorDict({"policy":obs},batch_size=[self.num_envs],device=obs.device)
    def get_observations(self): return self._to_tensordict(self._get_obs())
    def step(self,action):
        B=self.num_envs; action=torch.nan_to_num(action,nan=0.0).clamp(-1.0,1.0); self._prev_action=action.detach().clone()
        dc=torch.clamp((self._J_pinv@(action*self._action_scale).T).T,-MAX_CYL_DELTA,MAX_CYL_DELTA)
        cqh=self._uw.robot.data.joint_pos[:,self._cyl_ids].clone()
        lim=self._uw.robot.data.soft_joint_pos_limits[:,self._cyl_ids]; cqt=torch.clamp(cqh+dc,lim[...,0],lim[...,1])
        ee0=self._tcp_yz().detach().clone()
        for _ in range(self._H): self._pin_root(); self._uw.set_debug_manual_cylinder_targets(cqt,tool_position_target=self._tool_target); self._env.step(torch.zeros((B,5),device=self._device))
        ee1=self._tcp_yz(); jump=torch.norm(ee1-ee0,dim=-1)
        reward,distance,success=self._compute_reward(action); self._ep_length+=1
        term=(jump>0.30)  # no success termination (targets are inside 0.10m tolerance); trunc=self._ep_length>=self.max_episode_length; done=term|trunc
        extras={"log":{"distance":distance.mean(),"success_rate":success.float().mean(),"ee_jump":jump.mean(),"action_norm":torch.norm(action,dim=-1).mean(),"reward_mean":reward.mean()}}
        dids=done.nonzero(as_tuple=False).squeeze(-1)
        if dids.numel()>0:
            if dids.ndim==0: dids=dids.unsqueeze(0)
            self._uw._reset_idx(dids.tolist()); self._prev_action[dids]=0.0
            self._prev_distance[dids]=torch.norm(self._tcp_yz()[dids]-self._target_yz[dids],dim=-1); self._ep_length[dids]=0
            self._sample_target()  # re-sample target relative to current EE
        return self._to_tensordict(self._get_obs()), reward, (term | trunc).to(torch.long), extras
    def seed(self,seed=-1): return self._env.seed(seed)
    @property
    def episode_length_buf(self): return self._ep_length
    @episode_length_buf.setter
    def episode_length_buf(self,v): self._ep_length=v
    def close(self): self._env.close()

# ---- MAIN ----
if __name__ == "__main__":
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    env_wrapper = Stage0YZEnvWrapper(cfg, horizon_steps=int(args_cli.horizon_steps),
                                     action_scale_m=float(args_cli.action_scale_m),
                                     device=args_cli.device)

    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg
    agent_cfg = MulagUpperLevelPPORunnerCfg()
    agent_cfg.max_iterations = int(args_cli.max_iterations)
    agent_cfg.num_steps_per_env = int(args_cli.num_steps_per_env)
    agent_cfg.save_interval = int(args_cli.save_interval)
    agent_cfg.device = args_cli.device; agent_cfg.seed = 42
    agent_cfg.actor.hidden_dims = [64, 32]; agent_cfg.critic.hidden_dims = [64, 32]
    agent_cfg.algorithm.learning_rate = 1.0e-3

    log_root = os.path.join("logs", "rsl_rl", "stage0_yz")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, run_name); os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env_wrapper, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args_cli.checkpoint:
        print(f"Resuming from: {args_cli.checkpoint}", flush=True)
        runner.load(args_cli.checkpoint)

    total_iters = int(args_cli.max_iterations)
    print(f"Training: {total_iters} iters, {args_cli.num_envs} envs, save every {args_cli.save_interval}", flush=True)
    print(f"Log: {log_dir}", flush=True)

    try:
        with torch.inference_mode(False):
            runner.learn(num_learning_iterations=total_iters, init_at_random_ep_len=True)
    finally:
        env_wrapper.close()
        simulation_app.close()
        print("Training complete.", flush=True)
