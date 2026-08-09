"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages # changed module name from omni.isaac.lab_tasks.utils -> isaaclab_tasks.utils

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
