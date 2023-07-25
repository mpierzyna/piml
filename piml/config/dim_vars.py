from typing import List

from piml.pi.base import DimSymbol
from piml.config.base import BaseYAMLConfig


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
