"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .manager_based import *

try:
    import kevin_integration.tasks  # noqa: F401
except ImportError:
    pass
