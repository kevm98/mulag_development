import argparse
import torch
from isaaclab.app import AppLauncher

"""
This script randomly generates joint position for "Drehzapfen_joint" to test Unimog Mulag functionality

.. code-block:: bash

    # Usage
    ./rrlab.sh -p standalone/tutorials/controllers/random_move_mulag.py --num_envs 1

"""

# add argparse arguments
parser = argparse.ArgumentParser(description="Testing random joint movement on Unimog Mulag")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from rrlab_assets import MULAG_CFG



@configclass
class PlaneScene(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    robot = MULAG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")    


    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    
    robot = scene["robot"]
    print(f"DEBUG: Actual robot joint names: {robot.joint_names}")

    sim_dt = sim.get_physics_dt()
    count = 0
    
    target_joint_positions = robot.data.default_joint_pos.clone()

    try:
        drehzapfen_idx = robot.find_joints("Drehzapfen_joint")[0]
        target_joint_positions[:, drehzapfen_idx] = 1.0
    except ValueError:
        print("[ERROR] Drehzapfen_joint not found. Check joint name.")


    print(f"DEBUG: Initial target joint positions: {target_joint_positions[0]}")

    while simulation_app.is_running():
        robot.set_joint_position_target(target_joint_positions)

        if count % 100 == 0: 
            print(f"DEBUG: Current joint position (Drehzapfen_joint): {robot.data.joint_pos[0, drehzapfen_idx].item()}")
            print(f"DEBUG: Current joint velocity (Drehzapfen_joint): {robot.data.joint_vel[0, drehzapfen_idx].item()}")

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)




def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([11, 8, 1.3], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = PlaneScene(num_envs=args_cli.num_envs, env_spacing=20.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
