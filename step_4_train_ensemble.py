from typing import List

import warnings
import joblib
import numpy as np
import pandas as pd
import datetime

import piml
from piml.pi.transform import apply_pi_set
from piml.ml import Experiment
from piml.ml.ensemble import train_ensemble
from piml.utils.lazy_array import LazyArray

if __name__ == '__main__':
    ws = piml.Workspace.auto()

    # Load Pi sets and dimensional data
    pi_sets: List[piml.PiSet] = joblib.load(ws.data_extracted / "pi_sets_constrained.joblib")
    df_dim_train = pd.read_csv(
        ws.data_train_test / "MLO_Obs_Stacked_2006-07-01_2006-07-15_TRAIN.csv.gz",  # todo: make universal
        parse_dates=["TIME"],
    )

    for s in pi_sets:
        df_train = apply_pi_set(df_dim=df_dim_train, s=s, dim_vars=ws.dim_vars)
        df_train["DAY_YEAR"] = df_dim_train["DAY_YEAR"]  # For ensemble splitting
        features = df_train.columns[:len(s.feature_exprs)]
        df_train[s.target_id] = np.log10(df_train[s.target_id])  # todo: make universal

        if df_train[s.target_id].isna().any():
            warnings.warn(f"NaNs in target variable {s.target_id}! Skipping Pi set {s.id}.")
            continue

        # Set up base experiment from which individual member models will be created
        base_exp = Experiment(
            config=ws.config,
            pi_set=s,
            target=s.target_id
        )

        # Create LazyArray, which will hold pickled version of each trained member
        ensemble_name = f"{base_exp.get_str()}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        la = LazyArray(ws.data_trained / ensemble_name, overwrite=True)

        # Start training
        train_ensemble(base_exp=base_exp, df_train=df_train, features=features, result_array=la)
