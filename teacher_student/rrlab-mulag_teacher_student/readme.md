# Install IsaacSim and IsaacLab using Isaac Sim Pip Package
follow the installation according to `https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html#`.
This installs the newest IsaacSim 5.1.0 and IsaacLab 2.3.0, which support the Unimog Mulag Task and are recommended.
# Install RRLAB Extensions
1. clone the rrlab extension repository to your prefered local,
2. run `sh rrlab.sh -i` or `./rrlab.sh -i`

rrlab is a isaaclab extension, check the isaaclab extension template. It runs upound isaaclab, isolated from it, keep it clean.
# Run RRLAB MULAG TASK  in Conda Env. (for other models change checkpoint)
`python standalone/workflows/skrl/play.py --task RRLAB-Obstacle-Avoidance-Mulag-v0 --num_envs 12 --enable_cameras --checkpoint logs/skrl/reach_mulag/MulagMain2/checkpoints/best_agent.pt`