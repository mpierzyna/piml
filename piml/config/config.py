import datetime
from typing import Tuple, Optional, Dict
import pathlib

from piml.config.base import BaseYAMLConfig
from piml.config.flaml import ConfigFLAML


class DataConfig(BaseYAMLConfig):
    """ Store data related configuration. """
    path: pathlib.Path
    test_interval: Tuple[datetime.date, datetime.date]
    col_to_var: Optional[Dict[str, str]] = None


class Config(BaseYAMLConfig):
    data: DataConfig
    automl_base_settings: ConfigFLAML
    n_members: int
