# PPO1 upper-level policy

All policy-specific environment, observations, rewards, scripts, tests, and checkpoints
for the first residual PPO experiment live here. PPO1 is currently disabled.

When work resumes, the policy may output only a small cutter-setpoint residual on top of
the deterministic controller. Zero action must reproduce the controller exactly, and a
deterministic evaluation must beat the controller baseline. Do not use PPO to hide the
open truck wheel/brake transition or command raw cylinder actions.
