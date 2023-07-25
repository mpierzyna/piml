from typing import List
import dataclasses

import sympy as sp
import pydantic

# Define signed and unsigned constants
SIGNED = True
UNSIGNED = False

# Pi target expression used to convert between dimensional and non-dimensional space
PI_Y_expr = sp.symbols("PI_Y")


@dataclasses.dataclass
class DimSymbol:
    """ Dataclass for dimensioned symbol, linking symbol to its sign and dimensions. """
    symbol: sp.Expr
    signed: bool
    dimensions: str

    @pydantic.validator("symbol", pre=True)
    def to_sp_symbol(cls, v):
        """ Convert input to symbol if necessary. """
        if not isinstance(v, sp.Expr):
            return sp.symbols(v)
        return v

    def __str__(self):
        return str(self.symbol)


class PiSet(pydantic.BaseModel):
    id: str
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

