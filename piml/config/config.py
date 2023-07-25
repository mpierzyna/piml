from piml.config.base import BaseYAMLConfig
from piml.config.dataset import DatasetConfig
from piml.config.dim_vars import DimVarsConfig
from piml.config.flaml import FLAMLConfig


class Config(BaseYAMLConfig):
    """ Parent config class. Needs to be in separate file to avoid circular imports. """
    dim_vars: DimVarsConfig
    dataset: DatasetConfig
    flaml: FLAMLConfig
    n_members: int
