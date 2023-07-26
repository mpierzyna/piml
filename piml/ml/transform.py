import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import piml
from piml.config.dataset import DatasetConfig
from piml.config.dim_vars import DimVarsConfig
from piml.ml.utils import get_custom_tf
from piml.pi.transform import apply_pi_set


class DimToPiTransformer(TransformerMixin, BaseEstimator):
    """ Transform dimensional data to non-dimensional Pi data and apply pre-pi and pre-train transforms if specified """

    def __init__(self, pi_set: piml.PiSet, dim_vars: DimVarsConfig, dataset: DatasetConfig,
                 pre_pi_tf: TransformerMixin = None, pre_train_tf: TransformerMixin = None):
        self.pi_set = pi_set
        self.dim_vars = dim_vars
        self.dataset = dataset
        self.pre_pi_tf = pre_pi_tf
        self.pre_train_tf = pre_train_tf

    def fit_transform(self, df_dim: pd.DataFrame, y=None, **fit_params):
        # For compatibility with sklearn pipelines
        return self.fit(df_dim).transform(df_dim)

    def fit(self, df_dim: pd.DataFrame, y=None, **fit_kwargs) -> "DimToPiTransformer":
        # No fitting necessary, so just return self
        return self

    def transform(self, df_dim: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        # Apply pre-pi transform to DIMENSIONAL target
        self.pre_pi_tf_applied_ = False
        if self.pre_pi_tf:
            print("Applying pre-pi transform to dimensional target: ", self.pre_pi_tf)
            # Get expected target variable names. Very nested... Sorry.
            dim_target_tf = self.dim_vars.output.symbol.name
            dim_target = dim_target_tf[:-3]  # dim_target_tf is expected to have `_tf` suffix, so remove it

            df_dim[dim_target_tf] = self.pre_pi_tf.fit_transform(df_dim[dim_target])
            df_dim = df_dim.drop(columns=[dim_target])
            self.pre_pi_tf_applied_ = True

        # Apply pi set to DIMENSIONAL data
        df_pi = apply_pi_set(df_dim=df_dim, s=self.pi_set, dim_vars=self.dim_vars)
        df_pi["DAY_YEAR"] = df_dim["DAY_YEAR"]  # Retain for ensemble splitting

        # Apply pre-train transform to PI/NON-dim target
        self.pre_train_tf_applied_ = False
        pi_target = self.pi_set.target_id
        if self.pre_train_tf:
            print("Applying pre-train transform to PI target: ", self.pre_train_tf)
            df_pi[pi_target] = self.pre_train_tf.fit_transform(df_pi[pi_target])
            self.pre_train_tf_applied_ = True

        # Check for NaNs in target variable. Log trafo, for example, results in NaNs if input is negative.
        if df_pi[pi_target].isna().any():
            raise ValueError(f"NaNs in target variable {pi_target} after transform!")

        # Make features available after transform. Do it this way, to avoid adding meta columns
        self.features_ = df_pi.columns[:len(self.pi_set.feature_exprs)].to_numpy()

        return df_pi

    @classmethod
    def from_workspace(cls, ws: piml.Workspace, pi_set: piml.PiSet) -> "DimToPiTransformer":
        """Create transformer from config contained in workspace"""
        dim_vars = ws.config.dim_vars
        dataset = ws.config.dataset

        # Load custom pre_pi transformer, if specified
        pre_pi_tf = dataset.target_transformers.get("pre_pi")
        if pre_pi_tf:
            pre_pi_tf = get_custom_tf(ws, pre_pi_tf)

        # Load custom pre_train transformer, if specified
        pre_train_tf = dataset.target_transformers.get("pre_train")
        if pre_train_tf:
            pre_train_tf = get_custom_tf(ws, pre_train_tf)

        return cls(pre_pi_tf=pre_pi_tf, pre_train_tf=pre_train_tf, pi_set=pi_set, dim_vars=dim_vars, dataset=dataset)
