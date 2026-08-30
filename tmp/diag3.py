import os, sys
sys.path.insert(0, "/workspace/projects/rrlab_mulag")
from isaaclab.app import AppLauncher
import argparse
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
ns = p.parse_args([]); ns.headless = True
app = AppLauncher(ns).app
out=[]
try:
    import PPO_AAM_FD
    from isaaclab_tasks.utils import parse_env_cfg
    from verge_task.terrain_observations import observation_space_for_cfg
    c = parse_env_cfg("RRLAB-Mulag-Verge-ID-AAM-FD-v0", device="cpu", num_envs=2)
    out.append(f"privileged_e={c.online_aam_policy_privileged_e}")
    out.append(f"cfg.observation_space={c.observation_space}")
    out.append(f"observation_space_for_cfg={observation_space_for_cfg(c)}")
    out.append("MATCH" if int(c.observation_space)==int(observation_space_for_cfg(c)) else "MISMATCH")
except Exception:
    import traceback; out.append(traceback.format_exc())
open("/workspace/projects/rrlab_mulag/tmp/diag3_out.txt","w").write("\n".join(out))
app.close()
