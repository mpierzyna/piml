from typing import Union, List

import flaml
import numpy as np
import pandas as pd

from piml.ml import Experiment
from piml.ml.fi import get_permutation_importance
from piml.ml.splitter import RandomDaysSplitter
from piml.utils.lazy_array import LazyArray


def train_ensemble(base_exp: Experiment, df_train: pd.DataFrame, features: np.ndarray,
                   result_array: Union[List, LazyArray]) -> None:
    """ Train ensemble of estimators based on `base_exp` on randomly selected subsets of days."""
    # Setup training data
    target = base_exp.target
    X_train = df_train[features]
    y_train = df_train[target]
    print(f"Training on {target}.")

    # Sanity check
    assert target not in features
    assert target not in X_train.columns

    # Set up splitter for ensemble
    ensemble_splitter = RandomDaysSplitter(
        df_train["DAY_YEAR"],
        n_splits=base_exp.config.n_members,
        n_intervals_per_split=2,
        n_days_per_interval=7
    )

    # Train one member of ensemble at a time
    i = 1
    for idx_train, idx_val in ensemble_splitter.split(X_train):
        # Create Experiment instance for this member
        exp = base_exp.copy(deep=True)
        exp.train_idx = idx_train
        exp.val_idx = idx_val

        # Get train and validation data. Pandas needs to be converted to numpy for KFold CV to work
        X_train_i = X_train.iloc[idx_train].to_numpy()
        y_train_i = y_train.iloc[idx_train].to_numpy()
        X_val_i = X_train.iloc[idx_val].to_numpy()
        y_val_i = y_train.iloc[idx_val].to_numpy()

        # If logging is enabled, create one log file per ensemble
        log_file_name = base_exp.config.flaml.log_file_name
        if log_file_name:
            log_file_name = log_file_name.parent / f"{log_file_name.stem}_{i:03d}{log_file_name.suffix}"

        # Get FLAML settings dict from config
        automl_settings_dict = base_exp.config.flaml.copy(update={
            "log_file_name": log_file_name
        }).dict()

        # Train it
        print(f"Training model {i} of ensemble...", end=" ")
        automl = flaml.AutoML(**automl_settings_dict)
        automl.fit(X_train_i, y_train_i)

        # Store and evaluate it
        exp.model = automl
        exp.train_score = automl.best_loss
        exp.val_score = automl.score(X=X_val_i, y=y_val_i)

        # Rank features
        print("Feature ranking...", end=" ")
        exp.features = features
        exp.algo_fi = automl.feature_importances_
        exp.perm_fi = get_permutation_importance(
            automl, X_train_i, y_train_i, n_jobs=base_exp.config.flaml.n_jobs, lower_is_better=True
        )

        result_array.append(exp)
        i += 1
        print("Done!")

    print("Ensemble training done!")
