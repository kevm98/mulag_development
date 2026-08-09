"""Non-invasive kinematic helpers backed by the native PhysX Jacobian."""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply_inverse


@torch.no_grad()
def _truck_base_quat(env) -> torch.Tensor:
    """Quaternion of the TRUCK CHASSIS. The articulation root is the Drehzapfen
    (the arm's slew base) and ROTATES WITH THE ARM -- expressing the Jacobian in
    that frame zeroes the slew's own row (the base-X-null artifact)."""
    names = env.robot.body_names
    for cand in ("base_link", "Frontrahmen"):
        if cand in names:
            q = env.robot.data.body_link_state_w[:, names.index(cand), 3:7]
            return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q = env.robot.data.root_quat_w
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


@torch.no_grad()
def ee_jacobian_local_learned(env) -> torch.Tensor:
    """Return d(EE local xyz)/d(cylinder learned x), shape (N, 3, 4)."""
    jacobians = env.robot.root_physx_view.get_jacobians()
    body_index = max(int(env._ee_body_id) - 1, 0)
    j_world = jacobians[:, body_index, :3, :]
    dof_offset = max(int(j_world.shape[-1]) - int(env.robot.num_joints), 0)
    cylinder_dofs = env._cylinder_joint_ids + dof_offset
    columns_world = j_world[:, :, cylinder_dofs].transpose(1, 2)
    base_quat = _truck_base_quat(env)[:, None, :].expand(-1, 4, -1)
    columns_local = quat_apply_inverse(
        base_quat.reshape(-1, 4), columns_world.reshape(-1, 3)
    ).reshape(env.num_envs, 4, 3)
    j_local_usd = columns_local.transpose(1, 2)
    scale = env._cylinder_adapter.scaling.scale.to(
        env.device, torch.float32
    ).reshape(1, 1, 4)
    return j_local_usd * scale


@torch.no_grad()
def ee_xyz_jacobian(env) -> torch.Tensor:
    """Full d(EE truck-frame xyz)/d(cylinder learned x), shape (N, 3, 4).
    X must be commanded: the Drehzapfen slew moves the tool mostly in truck-X."""
    return ee_jacobian_local_learned(env)


@torch.no_grad()
def ee_yz_jacobian(env) -> torch.Tensor:
    """Return d(EE local yz)/d(cylinder learned x), shape (N, 2, 4)."""
    return ee_jacobian_local_learned(env)[:, 1:3, :]


@torch.no_grad()
def ee_xyz_jacobian_stepped(env, eps: float = 0.02, steps: int = 10) -> torch.Tensor:
    """TRUE input->EE Jacobian d(EE truck xyz)/d(cylinder x), shape (N, 3, 4),
    measured by perturbing the cylinder POSITION TARGETS and STEPPING PHYSICS.

    The native PhysX jacobian sliced to the 4 cylinder columns implicitly holds the
    passive closed-loop *_constraint_joint_* DOFs fixed -> it welds the linkage and
    reports the slew's X row as ~0 (third SV ~1e-7): the same artifact as frozen-FK,
    in analytic form. Only stepped physics solves the constraint loop.
    Snapshot/restore keeps the rollouts side-effect free.
    """
    robot = env.robot
    sc = env._cylinder_adapter.scaling
    ids = env._cylinder_joint_ids
    dt = float(env.sim.get_physics_dt())
    root = robot.data.root_state_w.detach().clone()
    jp = robot.data.joint_pos.detach().clone()
    jv = robot.data.joint_vel.detach().clone()
    tgt = robot.data.joint_pos_target.detach().clone()
    x0, _ = env._refresh_cylinder_state()
    x0 = x0.detach().clone()

    def rollout(x_goal):
        q = sc.clamp_usd_q(sc.learned_x_to_usd_q(x_goal))
        for _ in range(steps):
            robot.set_joint_position_target(q, joint_ids=ids)
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt)
        return env._compute_ee_pos_local().detach().clone()

    cols = []
    for c in range(4):
        sides = []
        for sgn in (1.0, -1.0):
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(jp, jv)
            xt = x0.clone(); xt[:, c] += sgn * eps
            sides.append(rollout(xt))
        cols.append((sides[0] - sides[1]) / (2.0 * eps))
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(jp, jv)
    robot.set_joint_position_target(tgt)
    env.scene.write_data_to_sim()
    env.sim.forward()
    return torch.stack(cols, dim=2)
