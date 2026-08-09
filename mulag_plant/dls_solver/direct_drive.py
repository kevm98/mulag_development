"""Direct PhysX cylinder drive with the learned ID/FD actuator bypassed."""

from __future__ import annotations

import torch


class DirectCylinderDriver:
    """Command learned-coordinate cylinder extensions as PhysX position targets."""

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
        _, velocity = self.env._refresh_cylinder_state()
        return velocity

    @torch.no_grad()
    def ee_local(self) -> torch.Tensor:
        return self.env._compute_ee_pos_local()

    @torch.no_grad()
    def step(self, cylinder_x_target: torch.Tensor, substeps: int = 2) -> None:
        """Write a bounded cylinder target and advance the simulation."""
        x_goal = torch.clamp(cylinder_x_target.reshape(-1, 4), self.lo, self.hi)
        q_goal = self.scaling.clamp_usd_q(self.scaling.learned_x_to_usd_q(x_goal))
        for _ in range(substeps):
            self.robot.set_joint_position_target(q_goal, joint_ids=self.cyl_ids)
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
        """Hold the current pose until the arm settles under gravity."""
        extension = self.measured_x().clone()
        for _ in range(steps):
            self.step(extension)
