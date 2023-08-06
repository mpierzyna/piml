import datetime
import warnings
from typing import List

import joblib
import pandas as pd

import piml
from piml.ml import Experiment
from piml.ml.ensemble import train_ensemble
from piml.ml.transform import DimToPiTransformer
from piml.pi.utils import pi_sets_to_latex
from piml.utils.lazy_array import LazyArray


def train_pi_set(ws: piml.Workspace, s: piml.PiSet, df_dim_train: pd.DataFrame) -> None:
    """ Prepare dimensional data for training using one PiSet and train ensemble """
    # Set up base experiment from which individual member models will be created
    base_exp = Experiment(
        config=ws.config,
        pi_set=s,
        target=s.target_id
    )

    # Set up dimensional data transformer. Specified pre-pi and pre-train transforms will be applied automatically.
    dim_to_pi_tf = DimToPiTransformer.from_workspace(ws=ws, pi_set=s)
    df_train = dim_to_pi_tf.fit(df_dim=df_dim_train).transform_X_y()

    # Create LazyArray, which will hold pickled version of each trained member
    ensemble_name = f"{base_exp.get_str()}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    la = LazyArray(ws.data_trained / ensemble_name / "ensemble", overwrite=True)

    # Write config and latex Pi set to ensemble folder for documentation.
    # Using the yaml() method of the config object ensures that also default values are written.
    (ws.data_trained / ensemble_name / "config.yml").write_text(ws.config.yaml())
    (ws.data_trained / ensemble_name / "pi_set.md").write_text(pi_sets_to_latex([s]))

    # Start training
    train_ensemble(base_exp=base_exp, df_train=df_train, features=dim_to_pi_tf.features_, result_array=la)


def train_all_pi_sets(ws: piml.Workspace, pi_sets: List[piml.PiSet], df_dim_train: pd.DataFrame) -> None:
    for s in pi_sets:
        try:
            train_pi_set(ws, s, df_dim_train)
        except ValueError as e:
            warnings.warn(str(e))
            continue


def main():
    ws = piml.Workspace.auto()

    # Load Pi sets and dimensional data
    pi_sets: List[piml.PiSet] = joblib.load(ws.data_extracted / "pi_sets_constrained.joblib")
    df_dim_train = pd.read_csv(
        ws.data_train_test / ws.config.dataset.get_train_name(with_suffix=True),
        parse_dates=["TIME"],
    )

    # Train all Pi sets
    # train_all_pi_sets(pi_sets, df_dim_train)

    # Train only specific Pi set
    train_pi_set(ws, pi_sets[13], df_dim_train)


if __name__ == '__main__':
    main()
