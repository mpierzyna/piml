from typing import List

import pydantic
import sympy as sp

from piml.config.base import BaseYAMLConfig


class DimSymbol(pydantic.BaseModel):
    """ Dataclass for dimensioned symbol, linking symbol to its sign and dimensions. """
    symbol: sp.Expr
    signed: bool
    dimensions: str

    class Config:
        arbitrary_types_allowed = True

    @pydantic.validator("symbol", pre=True)
    def to_sp_symbol(cls, v):
        """ Convert input to symbol if necessary. """
        if not isinstance(v, sp.Expr):
            return sp.symbols(v)
        return v

    def __str__(self):
        return str(self.symbol)


class DimVarsConfig(BaseYAMLConfig):
    inputs: List[DimSymbol]
    output: DimSymbol

    def __getitem__(self, item) -> DimSymbol:
        for v in self.inputs + [self.output]:
            if v.symbol == item:
                return v
            if v.symbol.name == item:
                return v
        raise KeyError(f"Symbol {item} not found.")

    @property
    def all_strs(self) -> List[str]:
        """ Return list of all symbols in config as list of strings """
        return [v.symbol.name for v in self.inputs + [self.output]]

    @property
    def input_strs(self) -> List[str]:
        """ Return list of input symbols in config as list of strings """
        return [v.symbol.name for v in self.inputs]

