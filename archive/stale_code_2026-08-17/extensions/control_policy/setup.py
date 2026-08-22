import os
try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml
from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_PATH = os.path.join(EXTENSION_PATH, "config", "extension.toml")
with open(EXTENSION_TOML_PATH, "rb") as f:
    EXTENSION_TOML_DATA = toml.load(f)

INSTALL_REQUIRES = [
    "psutil",
]

setup(
    name="control_policy",
    packages=["control_policy"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
