import warnings
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

        # Internal variables
        self._config_path = self.root / "config.yml"
        self._dim_vars_path = self.root / "dim_vars.yml"
        self._custom_code = None

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
        for p in [self.data_raw, self.data_extracted, self.data_processed, self.data_train_test, self.data_trained]:
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
    def data_trained(self) -> pathlib.Path:
        path = self.root / '5_trained'
        return path

    @property
    def dim_vars(self) -> DimVars:
        if self._dim_vars is None:
            self._dim_vars = DimVars.from_yaml(self._dim_vars_path.read_text())
        return self._dim_vars

    @property
    def config(self) -> Config:
        if self._config is None:
            self._config = Config.from_yaml(self._config_path.read_text())
        return self._config

    @property
    def custom_code(self):
        """ Custom code from `custom_code.py` in workspace root, if it exists. """
        if self._custom_code is None:
            # Try to import custom code from workspace
            path = sys.path
            try:
                sys.path.append(str(self.root.absolute()))
                import custom_code
                self._custom_code = custom_code
            except ModuleNotFoundError:
                # Reset path but deliberately raise error again to force me to catch it outside
                sys.path = path
                raise

        return self._custom_code
