"""
AIcrowd CLI
"""
import os

from .version import get_versions

__version__ = get_versions()["version"]
del get_versions

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
