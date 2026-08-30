set -u
cd /home/kevinmurera/isaaclab-cloud/projects/rrlab_mulag
for ARM in adaptive base control nominal; do
  MESH=1337 ENVS=1 STEPS=900 FRAMES=600 bash PPO_AAM_FD/visualize_cloud.sh $ARM
done
echo VIDEOS_DONE
