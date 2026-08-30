import os, sys
sys.path.insert(0, "/workspace/projects/rrlab_mulag")
from isaaclab.app import AppLauncher
import argparse
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
ns = p.parse_args([]); ns.headless = True
app = AppLauncher(ns).app
out=[]
try:
    import PPO_AAM_FD  # registers the gym task
    from isaaclab_tasks.utils import parse_env_cfg
    from PPO_AAM_FD.env_cfg import PpoIDAAMFDEnvCfg
    direct = PpoIDAAMFDEnvCfg()
    out.append(f"direct.observation_space={direct.observation_space} priv={direct.online_aam_policy_privileged_e}")
    for tid in ("RRLAB-Mulag-Verge-ID-AAM-FD-v0",):
        try:
            c = parse_env_cfg(tid, device="cpu", num_envs=2)
            out.append(f"parse_env_cfg({tid}).observation_space={c.observation_space} "
                       f"priv={getattr(c,'online_aam_policy_privileged_e','MISSING')} "
                       f"class={type(c).__name__}")
        except Exception as e:
            out.append(f"parse_env_cfg({tid}) FAILED: {e!r}")
except Exception:
    import traceback; out.append(traceback.format_exc())
open("/workspace/projects/rrlab_mulag/tmp/diag2_out.txt","w").write("\n".join(out))
app.close()
