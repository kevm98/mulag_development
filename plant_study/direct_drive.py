"""Direct PhysX cylinder drive — ID/FD COMPLETELY BYPASSED.

Purpose: learn what the plant can actually do, so we know what the ID/FD models
must deliver for physically meaningful motion.

Normal pipeline:   action -> ID(valve) -> FD(delta) -> target integrator -> PhysX
This module:                                            cylinder x -> PhysX

We convert a desired cylinder extension (learned coords) straight to USD joint
angles and write them as PhysX position targets, then step the sim manually.
No action, no ID, no FD, no target integrator.
"""
from __future__ import annotations

import torch


class DirectCylinderDriver:
    """Command cylinder extensions straight to the PhysX position drives."""

    def __init__(self, env):
        self.env = env
        self.robot = env.robot
        self.scaling = env._cylinder_adapter.scaling
        self.cyl_ids = env._cylinder_joint_ids
        self.dt = float(env.sim.get_physics_dt())
        self.lo = self.scaling.learned_min.to(env.device, torch.float32).reshape(1, 4)
        self.hi = self.scaling.learned_max.to(env.device, torch.float32).reshape(1, 4)

    @torch.no_grad()
    def measured_x(self) -> torch.Tensor:
        x, _ = self.env._refresh_cylinder_state()
        return x

    @torch.no_grad()
    def measured_xdot(self) -> torch.Tensor:
        _, xd = self.env._refresh_cylinder_state()
        return xd

    @torch.no_grad()
    def ee_local(self) -> torch.Tensor:
        return self.env._compute_ee_pos_local()

    @torch.no_grad()
    def step(self, cylinder_x_target: torch.Tensor, substeps: int = 2) -> None:
        """Write the cylinder target straight to PhysX and advance the sim.

        `substeps` matches the env's decimation so wall-clock rates are comparable.
        """
        x_goal = torch.clamp(cylinder_x_target.reshape(-1, 4), self.lo, self.hi)
        q_goal = self.scaling.clamp_usd_q(self.scaling.learned_x_to_usd_q(x_goal))
        for _ in range(substeps):
            self.robot.set_joint_position_target(q_goal, joint_ids=self.cyl_ids)
            # Match the production environment's explicit passive tool-joint
            # coupling.  Omitting this target leaves the closed-loop head in a
            # different physical configuration and severely under-reports the
            # cutter's reachable ground band.
            target_id = getattr(self.env, "_joint6_coupling_target_joint_id", None)
            source_id = getattr(self.env, "_joint6_coupling_source_joint_id", None)
            if target_id is not None and source_id is not None:
                source_column = int((self.cyl_ids == int(source_id)).nonzero(as_tuple=False)[0, 0].item())
                coupled = (
                    float(getattr(self.env.cfg, "joint6_coupling_a", 0.76))
                    * q_goal[:, source_column : source_column + 1]
                    + float(getattr(self.env.cfg, "joint6_coupling_b", 0.0))
                )
                self.robot.set_joint_position_target(coupled, joint_ids=[int(target_id)])
            self.env.scene.write_data_to_sim()
            self.env.sim.step()
            self.env.scene.update(self.dt)

    @torch.no_grad()
    def settle(self, steps: int = 60) -> None:
        """Hold the current pose so the arm reaches equilibrium under gravity."""
        x = self.measured_x().clone()
        for _ in range(steps):
            self.step(x)
