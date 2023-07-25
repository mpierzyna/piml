from typing import List

import warnings
import joblib
import numpy as np
import pandas as pd
import datetime

import piml
from piml.pi.transform import apply_pi_set
from piml.pi.utils import pi_sets_to_latex
from piml.ml import Experiment
from piml.ml.ensemble import train_ensemble
from piml.utils.lazy_array import LazyArray


def train_pi_set(s: piml.PiSet, df_dim_train: pd.DataFrame) -> None:
    """ Prepare dimensional data for training using one PiSet and train ensemble """
    df_train = apply_pi_set(df_dim=df_dim_train, s=s, dim_vars=ws.dim_vars)
    df_train["DAY_YEAR"] = df_dim_train["DAY_YEAR"]  # For ensemble splitting
    features = df_train.columns[:len(s.feature_exprs)]
    df_train[s.target_id] = np.log10(df_train[s.target_id])  # todo: make universal

    if df_train[s.target_id].isna().any():
        raise ValueError(f"NaNs in target variable {s.target_id}! Skipping Pi set {s.id}.")

    # Set up base experiment from which individual member models will be created
    base_exp = Experiment(
        config=ws.config,
        pi_set=s,
        target=s.target_id
    )

    # Create LazyArray, which will hold pickled version of each trained member
    ensemble_name = f"{base_exp.get_str()}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    la = LazyArray(ws.data_trained / ensemble_name / "ensemble", overwrite=True)

    # Copy config files to ensemble folder for reference
    (ws.data_trained / ensemble_name / "config.yml").write_text(ws._config_path.read_text())
    (ws.data_trained / ensemble_name / "dim_vars.yml").write_text(ws._dim_vars_path.read_text())
    (ws.data_trained / ensemble_name / "pi_set.md").write_text(pi_sets_to_latex([s]))

    # Start training
    train_ensemble(base_exp=base_exp, df_train=df_train, features=features, result_array=la)


def train_all_pi_sets(pi_sets: List[piml.PiSet], df_dim_train: pd.DataFrame) -> None:
    for s in pi_sets:
        try:
            train_pi_set(s, df_dim_train)
        except ValueError as e:
            warnings.warn(str(e))
            continue


if __name__ == '__main__':
    ws = piml.Workspace.auto()

    # Load Pi sets and dimensional data
    pi_sets: List[piml.PiSet] = joblib.load(ws.data_extracted / "pi_sets_constrained.joblib")
    df_dim_train = pd.read_csv(
        ws.data_train_test / "MLO_Obs_Stacked_2006-07-01_2006-07-15_TRAIN.csv.gz",  # todo: make universal
        parse_dates=["TIME"],
    )

    # Train all Pi sets
    # train_all_pi_sets(pi_sets, df_dim_train)

    # Train only specific Pi set
    train_pi_set(pi_sets[13], df_dim_train)

