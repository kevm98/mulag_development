from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
a = p.parse_args([])
a.headless = True
app = AppLauncher(a).app

import isaaclab_rl.rsl_rl as m
print("rsl_rl exports:", sorted(n for n in dir(m) if not n.startswith("_")))
try:
    import isaaclab_rl
    print("isaaclab_rl version:", getattr(isaaclab_rl, "__version__", "unknown"))
except Exception as e:
    print("version lookup failed:", e)

app.close()
