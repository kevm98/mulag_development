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
    c = parse_env_cfg("RRLAB-Mulag-Verge-ID-AAM-FD-v0", device="cpu", num_envs=2)
    out.append(f"mixed_plant_enabled={c.mixed_plant_enabled}")
    out.append(f"rom_fraction={getattr(c,'mixed_plant_rom_fraction','MISSING')}")
    out.append(f"command_rom_checkpoint={c.command_rom_checkpoint}")
    from pathlib import Path
    out.append(f"rom_ckpt_exists={Path(c.command_rom_checkpoint).is_file()}")
    out.append(f"dynamics={c.cylinder_dynamics_model}  obs={c.observation_space}")
except Exception:
    import traceback; out.append(traceback.format_exc())
open("/workspace/projects/rrlab_mulag/tmp/diag_mixed_out.txt","w").write("\n".join(out))
app.close()
