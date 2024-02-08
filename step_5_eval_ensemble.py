from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    # Dimensional test data
    df_dim_test = pd.read_csv(
        ws.data_train_test / ws.config.dataset.get_test_name(with_suffix=True),
        parse_dates=["TIME"],
    )

    # Store ensemble scores
    ens_scores = []
    pi_sets = []

    for ens_path in ensembles:
        # Load ensemble
        ens: List[Experiment] = LazyArray(ens_path, overwrite=False).gather_to_mem()
        features = np.array(ens[0].features)  # ensure np array
        pi_set = ens[0].pi_set
        pi_sets.append(pi_set)

        # Transform dimensional data to Pi space
        pi_tf = DimToPiTransformer.from_workspace(ws=ws, pi_set=pi_set)
        pi_tf.fit(df_dim=df_dim_test)
        df_pi_test = pi_tf.transform_X_y()

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
        dim_target = pi_tf.dim_target_tf  # DimToPiTransformer takes care of setting correct target name
        log_y_dim_test = np.log10(df_dim_test[dim_target].to_numpy())
        scores = np.array([
            [
                r2_score(log_y_dim_test, log_y_i),
                np.sqrt(mean_squared_error(log_y_dim_test, log_y_i))
            ]
            for log_y_i in np.log10(y_dim_pred_ens)
        ])
        print(f"R2: {scores[:, 0].mean():.3f} +/- {scores[:, 0].std():.3f}")
        print(f"RMSE: {scores[:, 1].mean():.3f} +/- {scores[:, 1].std():.3f}")
        ens_scores.append(scores)

        # Plot predictions
        fig, ax = plt.subplots()
        ax.plot(df_dim_test["TIME"], y_dim_pred_ens.T, color="gray", alpha=0.5)
        ax.plot(df_dim_test["TIME"], df_dim_test[dim_target], color="red", linewidth=.75)
        ax.set_ylabel(dim_target)
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

    # %% Plot ensemble score overview and complexity
    ens_scores = np.array(ens_scores)
    df_ens_scores = pd.DataFrame(
        data=ens_scores[:, :, 1].T,  # RMSE
        columns=[f"Set {s.id}" for s in pi_sets]
    )

    # Compute complexity
    complexity = np.array([
        [len(pi.free_symbols) for pi in s.all_exprs]
        for s in pi_sets
    ])

    fig, (ax_box, ax_complexity) = plt.subplots(ncols=2, figsize=(10, 5), sharey="row")
    sns.boxplot(
        data=df_ens_scores.melt(var_name="Set", value_name="RMSE"),
        x="RMSE", y="Set",
        ax=ax_box
    )
    sns.heatmap(complexity, ax=ax_complexity)