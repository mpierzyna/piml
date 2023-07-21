import joblib
import pandas as pd

import piml
from piml.transform import apply_pi_set

if __name__ == '__main__':
    ws = piml.Workspace.auto()

    # Load Pi sets and dimensional data
    pi_sets = joblib.load(ws.data_extracted / "pi_sets_constrained.joblib")
    df_dim_train = pd.read_csv(
        ws.data_train_test / "MLO_Obs_Stacked_2006-07-01_2006-07-15_TRAIN.csv.gz",  # todo: make universal
        parse_dates=["TIME"],
    )

    for s in pi_sets:
        df_train = apply_pi_set(df_dim=df_dim_train, s=s, dim_vars=ws.dim_vars)
