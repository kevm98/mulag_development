from __future__ import annotations
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a).app
from PPO_AAM_FD.env_cfg import PpoIDAAMFDEnvCfg
c = PpoIDAAMFDEnvCfg()
print(f"MULAG_AAM_PRIVILEGED_E={os.environ.get('MULAG_AAM_PRIVILEGED_E','0')}")
print(f"  privileged_e route : {c.online_aam_policy_privileged_e}")
print(f"  observation_space  : {c.observation_space}   (66 base + {c.observation_space-66})")
app.close()
