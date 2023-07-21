import datetime
from typing import Tuple, Optional, Dict
import pathlib

from piml.config.base import BaseYAMLConfig


class DataConfig(BaseYAMLConfig):
    """ Store data related configuration. """
    path: pathlib.Path
    test_interval: Tuple[datetime.date, datetime.date]
    col_to_var: Optional[Dict[str, str]] = None


class Config(BaseYAMLConfig):
    data: DataConfig
