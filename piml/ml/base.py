import pathlib
from typing import Optional, Tuple
import flaml
import pydantic
import numpy as np

from piml.config import Config
from piml.pi import PiSet


class Experiment(pydantic.BaseModel):
    """ Track all settings for different experiments plus results. """
    # Base settings
    config: Config  # config used to run this experiment
    pi_set: PiSet
    target: str  # key of target column in dataset

    # The trained model with scores
    model: flaml.AutoML = None
    train_score: float = -1
    val_score: float = -1
    test_score: float = -1

    # For ensemble analysis
    train_idx: np.ndarray = None
    val_idx: np.ndarray = None

    # For feature importance
    features: np.ndarray = None
    algo_fi: np.ndarray = None  # Feature importance according to algorithm
    perm_fi: np.ndarray = None  # Permutation feature importance

    class Config:
        arbitrary_types_allowed = True

    def get_str(self) -> str:
        """ Return string representation of experiment """
        dataset = self.config.dataset.path.name
        algos = "_".join(self.config.flaml.estimator_list)
        return f"PiSet_{self.pi_set.id:03d}__{algos}__{dataset}"

    @property
    def target_dim(self):
        """ Shortcut to target variable name in dimensional space """
        return self.config.dim_vars.output.symbol.name
