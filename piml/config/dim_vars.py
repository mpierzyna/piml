from typing import List

from piml.base import DimSymbol
from piml.config.base import BaseYAMLConfig


class DimVars(BaseYAMLConfig):
    dim_inputs: List[DimSymbol]
    dim_output: DimSymbol

    def __getitem__(self, item) -> DimSymbol:
        for v in self.dim_inputs + [self.dim_output]:
            if v.symbol == item:
                return v
            if v.symbol.name == item:
                return v
        raise KeyError(f"Symbol {item} not found.")
