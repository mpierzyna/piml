from typing import List

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

import piml
from piml.ml import Experiment
from piml.ml.transform import DimToPiTransformer
from piml.utils.lazy_array import LazyArray

if __name__ == '__main__':
    ws = piml.Workspace.auto()

    # Load ensembles
    ensembles = ws.data_trained.glob("*/ensemble")
    ensembles = [ens_path for ens_path in ensembles if ens_path.is_dir()]
    ensembles = sorted(ensembles)

    # Load Pi sets and dimensional data
    pi_sets: List[piml.PiSet] = joblib.load(ws.data_extracted / "pi_sets_constrained.joblib")
    df_dim_test = pd.read_csv(
        ws.data_train_test / ws.config.dataset.get_test_name(with_suffix=True),
        parse_dates=["TIME"],
    )

    for ens_path in ensembles:
        # Load ensemble
        ens: List[Experiment] = LazyArray(ens_path, overwrite=False).gather_to_mem()
        features = ens[0].features
        target = ens[0].target
        target_dim = ens[0].target_dim[:-3]

        # Transform dimensional data to Pi space
        pi_tf = DimToPiTransformer.from_workspace(ws=ws, pi_set=ens[0].pi_set)
        pi_tf.fit(df_dim=df_dim_test)
        df_pi_test = pi_tf.transform()

        # Make predictions using ensemble
        X_pi_test = df_pi_test[features].to_numpy()
        y_pi_pred_ens = np.array([
            m.model.predict(X_pi_test) for m in ens
        ])

        # Inverse transform predictions to dimensional space
        y_dim_pred_ens = np.array([
            pi_tf.inverse_transform_y(y_pi=y_pi_pred)
            for y_pi_pred in y_pi_pred_ens
        ])

        # Compute scores
        log_y_dim_test = np.log10(df_dim_test[target_dim].to_numpy())
        scores = np.array([
            [
                r2_score(log_y_dim_test, log_y_i),
                np.sqrt(mean_squared_error(log_y_dim_test, log_y_i))
            ]
            for log_y_i in np.log10(y_dim_pred_ens)
        ])
        print(f"R2: {scores[:, 0].mean():.3f} +/- {scores[:, 0].std():.3f}")
        print(f"RMSE: {scores[:, 1].mean():.3f} +/- {scores[:, 1].std():.3f}")

        # Plot predictions
        fig, ax = plt.subplots()
        ax.plot(df_dim_test["TIME"], y_dim_pred_ens.T, color="gray", alpha=0.5)
        ax.plot(df_dim_test["TIME"], df_dim_test[target_dim], color="red", linewidth=.75)
        ax.set_ylabel(target_dim)
        ax.set_yscale("log")
        fig.show()

        # Plot feature importance
        perm_fi_ens = np.array([
            m.perm_fi
            for m in ens
        ])
        perm_fi_df = pd.DataFrame(data=perm_fi_ens, columns=features).melt(var_name="feature", value_name="perm_fi")

        fig, ax = plt.subplots()
        sns.boxplot(perm_fi_df, x="feature", y="perm_fi", ax=ax)
        fig.show()
