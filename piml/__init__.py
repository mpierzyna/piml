import os
import pathlib

from .workspace import Workspace
from .base import PiSet

# Root path of package used to load files inside package
PKG_ROOT = pathlib.Path(os.path.dirname(__file__))

