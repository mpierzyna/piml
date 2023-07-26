from typing import List

import sympy as sp
import pydantic

# Define signed and unsigned constants
SIGNED = True
UNSIGNED = False

# Pi target expression used to convert between dimensional and non-dimensional space
PI_Y_expr = sp.symbols("PI_Y")


class PiSet(pydantic.BaseModel):
    id: int
    feature_exprs: List[sp.Expr]
    target_id: str
    target_expr: sp.Expr
    target_inv_expr: sp.Expr

    class Config:
        arbitrary_types_allowed = True

    @property
    def all_exprs(self) -> List[sp.Expr]:
        """ Feature and target expressions combined into a single list. """
        return self.feature_exprs + [self.target_expr]

