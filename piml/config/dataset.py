import datetime
import pathlib
from typing import Tuple, Optional, Dict

from piml.config.base import BaseYAMLConfig


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

    def _get_dataset_name(self, ds_type: str, with_suffix: bool) -> str:
        """Generate dataset name based on dataset type (TRAIN, TEST) and test set interval."""
        base_name = str(self.path.name)
        suffixes = []
        for s in self.path.suffixes:
            base_name = base_name.replace(s, "")
            suffixes.append(s)

        name = f"{base_name}_{ds_type}_{self.test_interval_str}"
        if with_suffix:
            name += "".join(suffixes)

        return name

    def get_train_name(self, with_suffix: bool) -> str:
        return self._get_dataset_name(ds_type="TRAIN", with_suffix=with_suffix)

    def get_test_name(self, with_suffix: bool) -> str:
        return self._get_dataset_name(ds_type="TEST", with_suffix=with_suffix)
