import datetime
from typing import Tuple, Optional, Dict
import pathlib

from piml.config.base import BaseYAMLConfig
from piml.config.flaml import FLAMLConfig


class DatasetConfig(BaseYAMLConfig):
    """ Store data related configuration. """
    path: pathlib.Path
    test_interval: Tuple[datetime.date, datetime.date]
    col_to_var: Optional[Dict[str, str]] = None
    target_transformers: Dict[str, str] = {}

    @property
    def test_interval_str(self) -> str:
        """ Return string representation of test interval """
        return f"{self.test_interval[0]}_{self.test_interval[1]}"


class Config(BaseYAMLConfig):
    dataset: DatasetConfig
    flaml: FLAMLConfig
    n_members: int
