# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./rrlab.sh -p standalone/tutorials/controllers/run_diff_ik.py --robot saugbagger --num_envs 1 --enable_camera

"""

"""Launch Isaac Sim Simulator first."""

# Torch is located at /home/qili/Software/Miniconda3/envs/isaaclab/lib/python3.10/site-packages/torch/libtorch.so
# Import torch first to ensure the version compiled with -D_GLIBCXX_USE_CXX11_ABI=1 is used.
# The pip-installed torch, which was compiled with -D_GLIBCXX_USE_CXX11_ABI=0, is incompatible 
# with nvblox and other downstream libraries.
import torch

import argparse

from isaaclab.app import AppLauncher
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--command_type", type=str, default="position", help="Command type of the Differential IK controller, here only position and pose been used, not support relative mode")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.devices import Se3Keyboard, Se3SpaceMouse
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, matrix_from_quat
from isaaclab.sensors import CameraCfg, TiledCameraCfg

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from rrlab_assets import SAUGBAGGER_CFG, MULAG_CFG


##
# Ros bridage and publisher set up
##
#from isaacsim.core.utils import extensions
# Enable ROS2 bridge extension
#extensions.enable_extension("omni.isaac.ros2_bridge")

# from omni.isaac.ros_bridge import RosBridge
#from sensor_msgs.msg import Image
#import rclpy
#from rclpy.node import Node
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge

# import sys
# print("The current python: ", sys.executable)

# import os
# # Get the PYTHONPATH from environment variables
# pythonpath = os.environ.get("PYTHONPATH", "")
# print("Python path: ")
# # Split the PYTHONPATH by ':' and print each entry in a new line
# for path in pythonpath.split(':'):
#     print(path)

# # os.system("ldd /home/qili/Software/nvblox_handler/build/nvblox_handler.so")


# # Print the path to libtorch.so
# libtorch_path = os.path.join(os.path.dirname(torch.__file__), 'libtorch.so')
# print(f"libtorch.so is located at: {libtorch_path}")

# from nvblox_handler import NvbloxHandler
#from nvblox_torch.mapper import Mapper
"""
class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/rgb_image', 10)
        self.bridge = CvBridge()
        
    def publish_image(self, camera_data):
        if isinstance(camera_data, torch.Tensor):
            rgb_image = camera_data.cpu().numpy()
        else:
            rgb_image = np.array(camera_data)

        # Check the shape of the image
        # print("Image shape:", rgb_image.shape)

        # Handle batch dimension (batch size of 1)
        if rgb_image.shape[0] == 1:
            rgb_image = np.squeeze(rgb_image, axis=0)  # Remove batch dimension

        # If there are multiple images (batch size of 2), select one
        if rgb_image.ndim == 4 and rgb_image.shape[0] == 2:
            rgb_image = rgb_image[1]  # Select the first image

        # Ensure the image is in (height, width, 4) format for RGBA
        if rgb_image.ndim == 3 and rgb_image.shape[2] == 4:
            # Remove the alpha channel (RGBA to RGB)
            rgb_image = rgb_image[:, :, :3]

        # Ensure the image is in (height, width, 3) format for RGB
        if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
            # If it's a valid RGB image, use rgb8 encoding
            ros_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        else:
            raise ValueError(f"Unexpected image format after selection: {rgb_image.shape}")

        self.publisher_.publish(ros_image)
        # self.get_logger().info('Publishing image to /camera/rgb_image')
"""

def save_as_ply(tensor, filename):
    """
    Save a PyTorch tensor as a PLY file.

    Args:
        tensor (torch.Tensor): Tensor of shape [num_points, 3] or [num_points, 4].
        filename (str): Path to save the PLY file.
    """

    tensor_reshaped = tensor.view(-1, 4)

    # Ensure the tensor is on the CPU and convert to numpy
    tensor_np = tensor_reshaped.cpu().numpy()

    # Determine the number of properties based on the tensor's shape
    # Ensure the tensor can be reshaped to (n, 4)
    print("tensor shape: ", tensor.shape)
    if tensor.numel() % 4 != 0:
        raise ValueError("The total number of elements is not divisible by 4. Cannot reshape to (n, 4).")

    
    print("tensor np before save to ply shape: ", tensor_np.shape)
    num_points, num_properties = tensor_np.shape
    if num_properties not in [3, 4]:
        raise ValueError("Tensor must have 3 (x, y, z) or 4 (x, y, z, additional property) columns.")
    
    # Write PLY header
    header = f"""ply
        format ascii 1.0
        element vertex {num_points}
        property float x
        property float y
        property float z
        """
    if num_properties == 4:
        header += "property float additional_property\n"
    header += "end_header\n"

    # Save PLY file
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, tensor_np, fmt="%.6f")


# Initialize the ROS2 node
#rclpy.init()
#camera_publisher = CameraPublisher()


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)), # ground is below 0
    )



    # # mount
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    #     ),
    # )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "saugbagger":
        robot = SAUGBAGGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "mulag":
        robot = MULAG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")    
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, sagbagger, mulag")
    
    # sensors
    """
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/MTS_SteelPipe_link/StereoCamera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707), convention="ros"),
    )
    """
    # # add camera to the scene
    # camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/MTS_SteelPipe_link/StereoCamera",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.2, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707), convention="ros"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=80,
    #     height=80,
    # )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
#def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, nvblox_handler: Mapper):
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    #print(f"DEBUG: Actual robot joint names: {robot.joint_names}")

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type=args_cli.command_type, use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    # ee_goals = [
    #     [-1.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
    #     [-1.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
    #     [-1.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    # ]
    ee_goals = [
        [3.5, 7.7, -0.2, 0.0, 0.70711, 0.70711, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    # Assign only the first `diff_ik_controller.action_dim` dimensions from `ee_goals`
    ee_goal = ee_goals[current_goal_idx:current_goal_idx + 1, :diff_ik_controller.action_dim]
    ik_commands[:] = ee_goal

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    elif args_cli.robot == "saugbagger":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["Link_[0-4]_link_joint"], body_names=["TCP_link"])
    elif args_cli.robot == "mulag":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["Drehzapfen_joint", "Ausleger_I_joint", "Ausleger_II_joint", "Messerkopf_Schwenk_joint", "Messerkopf_joint"], body_names=["Messerkopf"])    
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, saugbagger, mulag")
    
    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=0.04, rot_sensitivity=0.08)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", scene.reset)
    # print helper
    print(teleop_interface)

    # reset interfaces
    teleop_interface.reset()

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    isinited = False
    # Simulation loop
    while simulation_app.is_running():
        # compute frame in root frame -- this is the current ee pose in robot root frame
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # reset
        if  isinited == False:
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ee_goal = ee_goals[current_goal_idx:current_goal_idx + 1, :diff_ik_controller.action_dim]
            ik_commands[:] = ee_goal
            # ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands, ee_pos_b, ee_quat_b)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            isinited = True
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
           
            # compute the new target ee pose from keyboard
            # get keyboard command
            ee_delta_pose, gripper_command = teleop_interface.advance()

            if np.any(ee_delta_pose):
                ee_delta_pose = torch.from_numpy(ee_delta_pose * 0.1).cuda()
                ee_delta_pos = ee_delta_pose[0:3]
                ee_delta_quat = quat_from_euler_xyz(roll=ee_delta_pose[3], pitch=ee_delta_pose[4], yaw=ee_delta_pose[5])

                ik_commands[0, 0:3] = ik_commands[0, 0:3] + ee_delta_pos
                if args_cli.command_type == "pose":
                    ik_commands[0, 3:7] = quat_mul(ik_commands[0, 3:7], ee_delta_quat)
                # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands, ee_pos_b, ee_quat_b)
            
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        if args_cli.command_type == "pose":
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
        else:
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins)

        R = matrix_from_quat(ee_quat_b)
 
        T_L_C_t = torch.eye(4, device=sim.device)  # Start with the identity matrix
        T_L_C_t[:3, :3] = R.squeeze(0)  # Assign the rotation matrix
        T_L_C_t[:3, 3] = ee_pos_b.squeeze(0).float()     # Assign the translation vector

        """
        # # Get camera data and publish to ROS
        depth_images = scene["camera"].data.output["depth"]
        rgb_images = scene["camera"].data.output["rgb"]
        intrinsic_matrices = scene["camera"].data.intrinsic_matrices
        # print("intrinsic matrix: ", intrinsic_matrices[0])
        # Get the first dimension (number of images in the batch)
        batch_size = rgb_images.size(0)

        # Loop through the batch and get each image
        for i in range(batch_size):
            rgb_image = rgb_images[i]  # Get the i-th image with shape [480, 640, 3]
            depth_image = depth_images[i].squeeze(-1).float()
            intrinsic_matrice = intrinsic_matrices[i].float()
            # mapper_id = torch.tensor(i, device=sim.device, dtype=torch.int64)
            
            
            nvblox_handler.full_update(depth_image, rgb_image, T_L_C_t, intrinsic_matrice, i)

            num_voxels = nvblox_handler.get_num_voxels_in_detectionBlock()
            outputs = torch.empty((num_voxels * 4), dtype=torch.float32, device="cuda")
            print("outputs shape: ", outputs.shape)
            print(f"Number of voxels: {num_voxels}, Initialized outputs tensor with shape: {outputs.shape}")

            nvblox_handler.query_tsdf_from_detectionBlock(T_L_C_t, i, outputs)
            outputs = outputs.to("cpu")  # Move to CPU if necessary
            save_as_ply(outputs, "pointcloud.ply")
            print(outputs.shape)

            # nvblox_handler.fullUpdate(depth_image)
            # print(f"RGB Image {i} shape: {rgb_image.shape}")
            # print(f"Depth Image {i} shape: {depth_image.shape}")

        # # Publish to ROS2
        # camera_publisher.publish_image(rgb_image)
        """

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()

    # nvblox_handler = NvbloxHandler(args_cli.num_envs, "/home/qili/Software/IsaacSim_Exts/rrlab/extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/envs/nvblox_cfg.yaml")
    # mapper = Mapper(voxel_sizes=[0.02], integrator_types=["tsdf"])
    #mapper = Mapper(args_cli.num_envs, "/home/qili/Software/IsaacSim_Exts/rrlab/extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/envs/nvblox_cfg.yaml")
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    #run_simulator(sim, scene, mapper)
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
