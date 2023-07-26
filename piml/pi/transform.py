from __future__ import annotations

import numpy as np
import pandas as pd
import sympy as sp

from piml.pi.base import PiSet, PI_Y_expr
from piml.config.dim_vars import DimVarsConfig


class PiTargetTransformer:
    """ Transform target variable between non-dimensional and dimensional form. """
    def __init__(self, pi_set: PiSet, dim_vars: DimVarsConfig):
        # Dimensional to non-dimensional (Pi_y -> y)
        eval_fn = sp.lambdify(
            args=dim_vars.all_strs,
            expr=pi_set.target_expr,
            modules="numpy"
        )
        # Non-dimensional to dimensional (Pi_y -> y)
        eval_inv_fn = sp.lambdify(
            args=[PI_Y_expr.name, *dim_vars.input_strs],
            expr=pi_set.target_inv_expr,
            modules="numpy"
        )

        self.pi_set = pi_set
        self.dim_vars = dim_vars
        self.eval_fn = eval_fn
        self.eval_inv_fn = eval_inv_fn

    def fit(self, *, df_dim: pd.DataFrame) -> PiTargetTransformer:
        """ Provide dimensioned dataframe that will be base for transform or inverse transform. """
        self.df_dim_ = df_dim
        return self

    def transform(self, *, y_non_log: pd.Series) -> pd.Series:
        """ Transform dimensional target variable to non-dimensional form. """
        # If lengths of y and stored X (df_dim) do not align, raise error
        df_dim = self.df_dim_
        if len(y_non_log) != len(df_dim):
            raise ValueError(f"Dimensioned base dataframe ({len(df_dim)}) "
                             f"and transformation target ({len(y_non_log)} do not have the same length!")

        # Apply pi transformation
        y_pi = self.eval_fn(**{
            v: df_dim[v]
            for v in self.dim_vars.all_strs
        })
        return pd.Series(y_pi, index=y_non_log.index)

    def inverse_transform(self, *, y_pi: pd.Series) -> pd.Series:
        """ Transform non-dimensional target variable to dimensional form. """
        df_dim = self.df_dim_
        if len(y_pi) != len(df_dim):
            raise ValueError(f"Dimensioned base dataframe ({len(df_dim)}) "
                             f"and transformation target ({len(y_pi)} do not have the same length!")

        # Invert pi transformation
        y_non_log = self.eval_inv_fn(**{
            PI_Y_expr.name: y_pi,
            **{
                v: df_dim[v]
                for v in self.dim_vars.input_strs
            }
        })
        return pd.Series(y_non_log, index=y_pi.index)


def apply_pi_var(df_dim: pd.DataFrame, pi_expr: sp.Expr, dim_vars: DimVarsConfig) -> np.ndarray:
    """ Numerically evaluate Pi variable on dataframe.
    This function explicitly makes no use of target/output to avoid data leakage into features/inputs.
    That means it will fail when Pi_y group is provided! Use transformer instead.
    """
    eval_fn = sp.lambdify(dim_vars.input_strs, pi_expr, "numpy")
    return eval_fn(**{
        v: df_dim[v]
        for v in dim_vars.input_strs
    })


def apply_pi_set(df_dim: pd.DataFrame, s: PiSet, dim_vars: DimVarsConfig) -> pd.DataFrame:
    """ Numerically evaluate Pi set on dataframe. """
    # Evaluate features/inputs
    pi_eval = {
        f"s-{s.id}_pi-{i:02d}": apply_pi_var(df_dim, pi, dim_vars)
        for i, pi in enumerate(s.feature_exprs)
    }

    # Evaluate target/output
    pi_eval[s.target_id] = PiTargetTransformer(
        pi_set=s, dim_vars=dim_vars
    ).fit(df_dim=df_dim).transform(y_non_log=df_dim[dim_vars.output.symbol.name])

    return pd.DataFrame.from_dict(pi_eval)
