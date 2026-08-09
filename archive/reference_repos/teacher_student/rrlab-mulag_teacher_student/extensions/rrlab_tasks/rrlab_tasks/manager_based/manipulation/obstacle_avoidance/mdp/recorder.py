# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_apply_inverse

class EvaluationRecorderTerm(RecorderTerm):
    """
    Records and evaluates the Unimog Mulag.
    Saves data to JSON after specific steps.
    """
    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        
        self.log_dir = "/home/moe/code/mulag_eval/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.has_saved_json = False 

        # Robot and sensors
        self.robot: Articulation = self._env.scene["robot"]
        self.body_name = "Messerkopf"
        self.body_idx = self.robot.find_bodies(self.body_name)[0]
        
        self.sensor_left = self._env.scene["height_sensor_left"]
        self.sensor_right = self._env.scene["height_sensor_right"]
        self.tcp_transformer = self._env.scene["tcp_transformer"]
        self.contact_sensor = self._env.scene["contact_sensor_head"]

        # --- Specific Envs to Monitor ---
        #self.desired_indices = [1, 2, 7, 8]
        self.desired_indices = [0, 6]
        #self.desired_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # all
        valid_indices = [i for i in self.desired_indices if i < self._env.num_envs]
        
        self.target_indices = torch.tensor(valid_indices, device=self._env.device, dtype=torch.long)
        self.num_target_envs = len(self.target_indices)

        print(f"[Recorder] Monitoring Force, Deviation & Crashes ONLY for Env indices: {valid_indices}")

        # --- Metrics Config ---
        self.thresholds = {
            "lateral_deviation": 0.39,     
            "tracking_error_pos": 0.3,
            "tracking_error_rot": 0.16,    
            "avg_raycaster_height": 0.045,  
            "force_xy_magnitude": 4000.0,    
            "action_rate_l2": 0.2  
        }

        self.all_metric_keys = [
            "tracking_error_pos", 
            "tracking_error_rot",
            "avg_raycaster_height",
            "lateral_deviation", 
            "force_xy_magnitude",
            "action_rate_l2"
        ]

        self.plot_metric_names = [
            "tracking_error_pos", 
            "tracking_error_rot",
            "avg_raycaster_height",
            "action_rate_l2",
            "force_xy_magnitude",
            "lateral_deviation"
        ]
        
        self.data_history = {key: [] for key in self.all_metric_keys}
        
        self.violation_counts = {
            "force_xy_magnitude": 0,
            "lateral_deviation": 0,
            "crashes": 0 
        }
        self.total_observations = 0 
        self.step_counter = 0

        # --- Plot Setup ---
        print("[Recorder] Initializing Dashboard..")
        plt.ion()
        
        self.fig, self.axs_history = plt.subplots(len(self.plot_metric_names), 1, figsize=(10, 10), sharex=True)
        if len(self.plot_metric_names) == 1: self.axs_history = [self.axs_history]

        self.lines = {}
        self.y_labels = {
            "tracking_error_rot": "Rotation Error [rad]",
            "tracking_error_pos": "Position Error [m]",
            "avg_raycaster_height": "Height of TCP [m]",
            "action_rate_l2": "Action Rate",
        }

        for ax, key in zip(self.axs_history, self.plot_metric_names):
            ax.grid(True, alpha=0.5, linestyle=':')
            title = self.y_labels.get(key, key.replace("_", " ").title())
            ax.set_ylabel(title, fontsize=9, fontweight='bold')
            self.lines[key] = ax.plot([], [], color='black', linewidth=1.5)[0]
            
            if key in self.thresholds:
                val = self.thresholds[key]
                ax.axhline(val, color='red', linestyle='--', alpha=0.6, linewidth=1.2)

        self.axs_history[-1].set_xlabel("Simulation Time [steps]")
        plt.tight_layout()


    def record_post_step(self) -> tuple[str | None, dict | None]:
        metrics = {}
        
        self.total_observations += 4

        # ======================================================================
        # 1a. Metrics Calculation (in specific envs)
        # ======================================================================
        
        # --- Crashes Only ---
        dones = self._env.termination_manager.dones
        time_outs = self._env.termination_manager.time_outs 
        crashes = dones & ~time_outs

        if self.num_target_envs > 0:
            crashes_subset = crashes[self.target_indices]
            crash_count = torch.count_nonzero(crashes_subset)
            self.violation_counts["crashes"] += int(crash_count.item())

        # --- Lateral Deviation ---
        env_origins = self._env.scene.env_origins
        root_positions = self.robot.data.root_pos_w
        dev_all = torch.abs(torch.abs((env_origins[:, 1] - root_positions[:, 1])) - 4.3)
        
        if self.num_target_envs > 0:
            dev_subset = dev_all[self.target_indices]
            bad_dev_count = torch.count_nonzero(dev_subset > self.thresholds["lateral_deviation"])
            self.violation_counts["lateral_deviation"] += int(bad_dev_count.item())
            metrics["lateral_deviation"] = torch.mean(dev_subset)
        else:
            metrics["lateral_deviation"] = torch.tensor(0.0, device=self._env.device)

        # --- Force ---
        forces = self.contact_sensor.data.net_forces_w 
        rotated_forces = quat_apply_inverse(self.tcp_transformer.data.target_quat_w, forces)
        force_xy_all = torch.norm(rotated_forces[..., :2], dim=-1) 
        if force_xy_all.dim() > 1:
            force_xy_all = torch.max(force_xy_all, dim=1)[0]

        if self.num_target_envs > 0:
            force_subset = force_xy_all[self.target_indices]
            contact_count = torch.count_nonzero(force_subset > self.thresholds["force_xy_magnitude"])
            self.violation_counts["force_xy_magnitude"] += int(contact_count.item())
            metrics["force_xy_magnitude"] = torch.mean(force_subset)
        else:
            metrics["force_xy_magnitude"] = torch.tensor(0.0, device=self._env.device)

        # ======================================================================
        # 1b. Metrics Calculation (global)
        # ======================================================================
        tcp_data = self.tcp_transformer.data
        vec_left = self.sensor_left.data.ray_hits_w - tcp_data.target_pos_w
        hits_left_local = quat_apply_inverse(tcp_data.target_quat_w, vec_left)
        vec_right = self.sensor_right.data.ray_hits_w - tcp_data.target_pos_w
        hits_right_local = quat_apply_inverse(tcp_data.target_quat_w, vec_right)
        avg_h = (torch.mean(hits_left_local[..., 2], dim=-1) + 
                 torch.mean(hits_right_local[..., 2], dim=-1)) / 2.0
        metrics["avg_raycaster_height"] = torch.mean(avg_h)

        command = self._env.command_manager.get_command("ee_pose")
        des_pos_w, des_quat_w = combine_frame_transforms(
            self.robot.data.root_link_state_w[:, :3], 
            self.robot.data.root_link_state_w[:, 3:7], 
            command[:, :3], command[:, 3:7]
        )
        
        curr_pos_w = self.robot.data.body_link_state_w[:, self.body_idx[0], :3]
        reduced_des_pos_w = des_pos_w[:, :2]
        reduced_cur_pos_w = curr_pos_w[:, :2]
        curr_quat_w = self.robot.data.body_link_state_w[:, self.body_idx[0], 3:7]
        
        metrics["tracking_error_pos"] = torch.mean(torch.norm(reduced_cur_pos_w - reduced_des_pos_w, dim=-1))
        metrics["tracking_error_rot"] = torch.mean(quat_error_magnitude(curr_quat_w, des_quat_w))
        
        curr_action = self._env.action_manager.action
        prev_action = self._env.action_manager.prev_action
        if prev_action is not None:
            rate = torch.norm(curr_action - prev_action, dim=-1)
            metrics["action_rate_l2"] = torch.mean(rate)
        else:
            metrics["action_rate_l2"] = torch.tensor(0.0, device=curr_action.device)

        # ======================================================================
        # 2. Evaluation Report (Console)
        # ======================================================================
        eval_step = 10000
        if self.step_counter == eval_step:
            self._print_report(eval_step, self.total_observations, self.desired_indices)

        # ======================================================================
        # 3. Visualization & Storage
        # ======================================================================
        
        self._update_internal_storage(metrics)
        self.step_counter += 1


        if self.step_counter == eval_step and not self.has_saved_json:
            self.save_history_to_json(prefix="run_10k")
            self.has_saved_json = True

        if self.step_counter % 10 == 0:
            self._update_dashboard()
            plt.pause(0.001)

        return None, None

    def save_history_to_json(self, prefix="eval_run"): # Saves data to json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)

        # Struktur erstellen
        export_data = {
            "metadata": {
                "total_steps": self.step_counter,
                "timestamp": timestamp,
                "thresholds": self.thresholds
            },
            "metrics": self.data_history, 
            "violations": self.violation_counts
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=4)
            print(f"\n[Recorder] >>> Data successfully saved to: {filepath} <<<")
        except Exception as e:
            print(f"[Recorder] Error saving JSON: {e}")

    def _print_report(self, eval_step, total_obs, desired_indices):
        print(f"\n{'='*60}")
        print(f" EVALUATION REPORT (Steps: {self.step_counter})")
        print(f"{'='*60}")
        
        avg_metrics = {}
        for key in self.all_metric_keys:
            avg_val = np.mean(self.data_history[key])
            avg_metrics[key] = torch.tensor(avg_val)
            print(f" -> Avg {key:25s}: {avg_val:.4f}")
        
        # Force Stats
        f_count = self.violation_counts["force_xy_magnitude"]
        f_perc = (f_count / total_obs) * 100.0 if total_obs > 0 else 0
        print(f" [FORCE STATS (Envs {desired_indices})] threshold: > {self.thresholds['force_xy_magnitude']} N")
        print(f" -> Total Contacts Detected:  {f_count} / {total_obs}")
        print(f" -> Contact Percentage:       {f_perc:.2f}%")
        # Deviation Stats
        d_count = self.violation_counts["lateral_deviation"]
        d_perc = (d_count / total_obs) * 100.0 if total_obs > 0 else 0
        print(f" [LATERAL STATS (Envs {desired_indices})] threshold: > {self.thresholds['lateral_deviation']} m")
        print(f" -> Total Deviations:         {d_count} / {total_obs}")
        print(f" -> Deviation Percentage:     {d_perc:.2f}%")
        
        # Termination Stats 
        max_pos_crashes = int(np.floor(eval_step / 240 * 4))
        c_count = self.violation_counts["crashes"]
        c_perc = (c_count / max_pos_crashes) * 100.0 if total_obs > 0 else 0
        print(f" [CRASH STATS (Envs {desired_indices})]")
        print(f" -> Total Crashes:            {c_count} / {max_pos_crashes}")
        print(f" -> Crash Rate:    {c_perc:.2f}%")
        print("-" * 60)
        final_score = self._compute_total_score(avg_metrics, eval_step).item()
        print(f" FINAL PERFORMANCE SCORE: {final_score:.2f}%")
        print(f"{'='*60}\n")


    def _compute_total_score(self, metrics: dict, eval_step: int) -> torch.Tensor:
        def compute_sub_score(val, threshold, sensitivity=2.0):
            excess = torch.clamp(val - threshold, min=0.0)
            t = threshold if threshold > 1e-5 else 1.0
            normalized_excess = excess / t
            return torch.exp(-torch.square(normalized_excess) * sensitivity)

        s_val = 2.0 
        max_pos_crashes = int(np.floor(eval_step / 240 * 4))
        if max_pos_crashes == 0: max_pos_crashes = 1 # avoid div by zero

        score_rot = compute_sub_score(metrics["tracking_error_rot"], self.thresholds["tracking_error_rot"], s_val)
        score_pos = compute_sub_score(metrics["tracking_error_pos"], self.thresholds["tracking_error_pos"], s_val)
        score_collision = compute_sub_score(torch.tensor(self.violation_counts["crashes"] / max_pos_crashes), 0.05, s_val)
        score_h = compute_sub_score(metrics["avg_raycaster_height"], self.thresholds["avg_raycaster_height"], s_val)
        score_smooth = compute_sub_score(metrics["action_rate_l2"], self.thresholds["action_rate_l2"], s_val)

        print(f"Score Comp: Pos={score_pos:.2f}, Rot={score_rot:.2f}, H={score_h:.2f}, Collision={score_collision:.2f}, Smooth={score_smooth:.2f}")

        total = (0.15 * score_rot + 
                 0.2 * score_pos + 
                 0.3 * score_collision +
                 0.25 * score_h + 
                 0.1 * score_smooth)
        
        return total * 100.0
    

    def _update_internal_storage(self, metrics):
        for key in self.all_metric_keys:
            if key in metrics:
                self.data_history[key].append(float(metrics[key].item()))


    def _update_dashboard(self):
        data_len = len(self.data_history[self.plot_metric_names[0]])
        x_data = np.arange(data_len)

        for ax, key in zip(self.axs_history, self.plot_metric_names):
            y_data = np.array(self.data_history[key])
            self.lines[key].set_data(x_data, y_data)
            ax.relim()
            ax.autoscale_view(tight=True, scalex=True, scaley=False)
            
            if len(y_data) > 0:
                y_min, y_max = np.min(y_data), np.max(y_data)
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
                ax.set_ylim(min(y_min - margin, 0), max(y_max + margin, 0))