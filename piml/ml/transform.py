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

        # Expected target column names. Very nested... Sorry.
        self.dim_target_tf = self.dim_vars.output.symbol.name
        self.dim_target = self.dim_target_tf[:-3]  # dim_target_tf is expected to have `_tf` suffix, so remove it
        self.pi_target = self.pi_set.target_id

    def fit(self, *, df_dim: pd.DataFrame) -> "DimToPiTransformer":
        """ Provide dimensional data that serve as basis for transform and inverse transform. """
        self.df_dim_ = df_dim.copy()
        return self

    def transform_X(self) -> pd.DataFrame:
        """ Transform dimensional INPUT (X) into Pi space (e.g., for predictions). """
        # Evaluate features/inputs but NOT target.
        df_X_pi = apply_pi_set(df_dim=self.df_dim_, s=self.pi_set, dim_vars=self.dim_vars, with_y=False)
        df_X_pi["DAY_YEAR"] = self.df_dim_["DAY_YEAR"]  # Retain for ensemble splitting
        self.features_ = df_X_pi.columns[:-1]
        return df_X_pi

    def transform_y(self) -> pd.Series:
        """ Transform dimensional OUTPUT/TARGET (y) into Pi space. """
        # Work on copy of target
        df_dim = self.df_dim_
        y_dim = df_dim[self.dim_target].copy()

        # 1) Apply pre-pi transform to DIMENSIONAL target
        if self.pre_pi_tf:
            print("Applying pre-pi transform to dimensional target: ", self.pre_pi_tf)
            y_dim = self.pre_pi_tf.fit_transform(y_dim)

            # Replace untransformed with transformed column, so that step 2 works with correct version!
            df_dim = pd.concat([
                df_dim.drop(columns=self.dim_target),
                pd.DataFrame.from_dict({self.dim_target_tf: y_dim}).set_index(df_dim.index, drop=True)
            ], axis=1)

        # 2) Transform target from dimensional to Pi/non-dim space
        y_pi = PiTargetTransformer(
            pi_set=self.pi_set, dim_vars=self.dim_vars
        ).fit(
            df_dim=df_dim
        ).transform(
            y_non_log=y_dim
        )

        # 3) Apply pre-train transform to Pi/non-dim target
        if self.pre_train_tf:
            print("Applying pre-train transform to PI target: ", self.pre_train_tf)
            y_pi = self.pre_train_tf.fit_transform(y_pi)

        # Check for NaNs in target variable. Log trafo, for example, results in NaNs if input is negative.
        if y_pi.isna().any():
            raise ValueError(f"NaNs in target variable {self.pi_target} after transform!")

        return y_pi

    def transform_X_y(self) -> pd.DataFrame:
        """ Transform full dataset (X, y) into Pi space (e.g., for training). """
        df_pi = self.transform_X()
        df_pi[self.pi_target] = self.transform_y()
        return df_pi

    def inverse_transform_y(self, y_pi: pd.Series) -> pd.Series:
        """ Inverse-transform TARGET (y) from Pi space to original space (e.g., prediction result).
        All transforms are inverted in reverse order.
        """
        def to_series(y) -> pd.Series:
            """ Make sure y is a series after transform. """
            if not isinstance(y, pd.Series):
                return pd.Series(y_pi, index=self.df_dim_.index)
            return y

        # 3) Invert pre-train transform
        if self.pre_train_tf:
            print("Inverting pre-train transform: ", self.pre_train_tf)
            y_pi = self.pre_train_tf.inverse_transform(y_pi)
            y_pi = to_series(y_pi)

        # 2) Invert pi set
        y_dim = PiTargetTransformer(
            pi_set=self.pi_set, dim_vars=self.dim_vars
        ).fit(
            df_dim=self.df_dim_
        ).inverse_transform(
            y_pi=y_pi
        )

        # 1) Invert pre-pi transform
        if self.pre_pi_tf:
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
