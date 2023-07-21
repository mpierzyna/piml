from typing import List, Union, Callable, Optional, Dict
import pathlib
import pydantic

from sklearn.model_selection import KFold
from piml.config.base import BaseYAMLConfig


class ConfigFLAML(BaseYAMLConfig):
    """ Lists all relevant FLAML parameters.

    Defaults are reasonable and should not require adjustment.
    For details, see: https://microsoft.github.io/FLAML/docs/reference/automl/automl#automl-objects
    """
    estimator_list: List[str] = []  # will be filled automatically
    metric: Union[str, Callable]
    time_budget: int  # in seconds

    # Sensible defaults
    task: str = "regression"
    early_stop: bool = True
    sample: bool = True
    ensemble: bool = False
    retrain_full: bool = True

    # Model evaluation during training: UNSHUFFELD 5-fold CV to not have too correlated data
    eval_method: str = "cv"
    split_type = KFold(n_splits=5, shuffle=False)

    # Technical settings
    verbose: int = 3
    seed: int  # No default, require seed to be set explicitly for clarity
    log_file_name: Optional[pathlib.Path] = None
    n_jobs: int = -1

    @pydantic.validator("eval_method")
    def no_override_em(cls, v):
        raise KeyError("Eval method cannot be overridden from config file "
                       "because it is important that non-shuffling CV is used!")

    @pydantic.validator("split_type")
    def no_override_st(cls, v):
        raise KeyError("Split type cannot be overridden from config file "
                       "because it is important that non-shuffling CV is used!")

    def dict(self, *args, **kwargs) -> Dict:
        """ Return dict but hide internal fields with leading underscore (_). """
        self_dict = super().dict(**kwargs)
        self_dict = {
            k: v for k, v in self_dict.items()
            if not k.startswith("_")
        }
        return self_dict
