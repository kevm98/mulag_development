import os, sys
sys.path.insert(0, "/workspace/projects/rrlab_mulag")
from isaaclab.app import AppLauncher
import argparse
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
ns = p.parse_args([]); ns.headless = True
app = AppLauncher(ns).app
out = []
try:
    from PPO_AAM_FD.env_cfg import PpoIDAAMFDEnvCfg
    c = PpoIDAAMFDEnvCfg()
    out.append(f"env_var={os.environ.get('MULAG_AAM_PRIVILEGED_E')}")
    out.append(f"privileged_e={c.online_aam_policy_privileged_e}")
    out.append(f"rif_context={getattr(c,'online_aam_policy_rif_context','MISSING')}")
    out.append(f"observation_space={c.observation_space}")
except Exception:
    import traceback; out.append(traceback.format_exc())
open("/workspace/projects/rrlab_mulag/tmp/diag_out.txt","w").write("\n".join(out))
app.close()
