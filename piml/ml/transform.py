from typing import Protocol
import pandas as pd

import piml
from piml.config.dataset import DatasetConfig
from piml.config.dim_vars import DimVarsConfig
from piml.ml.utils import get_custom_tf
from piml.pi.transform import apply_pi_set, PiTargetTransformer


class InvertableTransformer(Protocol):
    """ Invertable sklearn-compatible transformer.
    Requires forward (fit_transform) and backward (inverse_transform) transforms to be implemented.

    Note: This Protocol aims to be a minimal interface for invertable transformers. It is not complete and does not
    contain all methods of sklearn transformers. It is only used for type checking and IDE support.
    """
    def fit_transform(self, X, y=None, **fit_params) -> pd.DataFrame:
        ...

    def inverse_transform(self, X, y=None, **fit_params) -> pd.DataFrame:
        ...


class DimToPiTransformer:
    """ Transform data between dimensional and Pi space. """
    def __init__(self, pi_set: piml.PiSet, dim_vars: DimVarsConfig, dataset: DatasetConfig,
                 pre_pi_tf: InvertableTransformer = None, pre_train_tf: InvertableTransformer = None):
        self.pi_set = pi_set
        self.dim_vars = dim_vars
        self.dataset = dataset
        self.pre_pi_tf = pre_pi_tf
        self.pre_train_tf = pre_train_tf

    def fit(self, *, df_dim: pd.DataFrame) -> "DimToPiTransformer":
        """ Provide dimensional data that serve as basis for transform and inverse transform. """
        self.df_dim_ = df_dim.copy()
        return self

    def transform(self) -> pd.DataFrame:
        """ Transform dimensional data to Pi space and apply pre-pi and pre-train transforms if specified """
        # Work on copy of data (@todo: Could be dangerous with big datasets!)
        df_dim = self.df_dim_.copy()

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

    def fit_transform(self, *, df_dim: pd.DataFrame) -> pd.DataFrame:
        """ Fit transformer and transform data. """
        return self.fit(df_dim=df_dim).transform()

    def inverse_transform_y(self, y_pi: pd.Series) -> pd.Series:
        """ Inverse-transform *target* from Pi space to original space.
        All transforms are inverted in reverse order.
        """
        def to_series(y) -> pd.Series:
            """ Make sure y is a series after transform. """
            if not isinstance(y, pd.Series):
                return pd.Series(y_pi, index=self.df_dim_.index)
            return y

        # Invert pre-train transform
        if self.pre_train_tf_applied_:
            print("Inverting pre-train transform: ", self.pre_train_tf)
            y_pi = self.pre_train_tf.inverse_transform(y_pi)
            y_pi = to_series(y_pi)

        # Invert pi set
        y_dim = PiTargetTransformer(
            pi_set=self.pi_set, dim_vars=self.dim_vars
        ).fit(
            df_dim=self.df_dim_
        ).inverse_transform(
            y_pi=y_pi
        )

        # Invert pre-pi transform
        if self.pre_pi_tf_applied_:
            print("Inverting pre-pi transform: ", self.pre_pi_tf)
            y_dim = self.pre_pi_tf.inverse_transform(y_dim)
            y_dim = to_series(y_dim)

        return y_dim

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
