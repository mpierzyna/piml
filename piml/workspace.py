from typing import Union
import os
import sys
import pathlib

from piml.config import DimVars, Config


class Workspace:
    def __init__(self, root: Union[str, pathlib.Path]):
        self.root = pathlib.Path(root)

        # Populate workspace if it does not exist
        if not self.root.exists():
            print(f"{root} does not exist. Creating it now.")
            self.root.mkdir(parents=True, exist_ok=False)
            self.populate()

        # Cache for variables
        self._dim_vars = None
        self._config = None

        print(f"Using workspace {root}.")

    @classmethod
    def from_env(cls):
        """ Create workspace from PIML_WORKSPACE environment variable. """
        if (root := os.environ.get('PIML_WORKSPACE')) is None:
            raise ValueError("PIML_WORKSPACE environment variable is not set.")
        print("Loading workspace from PIML_WORKSPACE environment variable.")
        return cls(root=root)

    @classmethod
    def from_argv(cls):
        """ Create workspace from first argument passed to script. """
        if len(sys.argv) < 2:
            raise ValueError("No workspace path provided as argument.")
        print("Loading workspace from first argument passed to script.")
        return cls(root=sys.argv[1])

    @classmethod
    def auto(cls):
        """ Create workspace from PIML_WORKSPACE environment variable if set, otherwise from first argument. """
        try:
            return cls.from_env()
        except ValueError:
            return cls.from_argv()

    def populate(self):
        # Create data directories
        for p in [self.data_raw, self.data_extracted, self.data_processed, self.data_train_test]:
            p.mkdir(exist_ok=False)

        # todo: Create empty config files

    @property
    def data_raw(self) -> pathlib.Path:
        path = self.root / '1_raw'
        return path

    @property
    def data_extracted(self) -> pathlib.Path:
        path = self.root / '2_extracted'
        return path

    @property
    def data_processed(self) -> pathlib.Path:
        path = self.root / '3_processed'
        return path

    @property
    def data_train_test(self) -> pathlib.Path:
        path = self.root / '4_train_test'
        return path

    @property
    def dim_vars(self) -> DimVars:
        if self._dim_vars is None:
            self._dim_vars = DimVars.from_yaml((self.root / 'vars.yml').read_text())
        return self._dim_vars

    @property
    def config(self) -> Config:
        if self._config is None:
            self._config = Config.from_yaml((self.root / 'config.yml').read_text())
        return self._config
