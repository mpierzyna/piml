"""
Set up yaml serializer to read and write config files (based on ``pydantic.BaseModel``).
"""
import pathlib
from typing import Dict, Any

import sympy as sp
import pydantic
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def path_representer(dumper: yaml.Dumper, path: pathlib.Path):
    """ Represent ``pathlib.Path`` as string in yaml. """
    return dumper.represent_scalar(tag="!path", value=str(path))


def path_constructor(loader: yaml.loader, node) -> pathlib.Path:
    """ Convert string back to ``pathlib.Path`` object. """
    return pathlib.Path(
        loader.construct_scalar(node)
    )


def sp_symbol_representer(dumper: yaml.Dumper, symbol: sp.Expr):
    """ Represent ``sympy.Expr`` as string in yaml. """
    # Do not use tag here because pydantic validator takes care of str to sympy conversion on deserialization.
    return dumper.represent_str(str(symbol))


def tuple_representer(dumper: yaml.Dumper, t: tuple):
    """ Convert tuple to yaml list. Attention! Deserialisation will be list not tuple! Pydantic will fix that. """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', t, flow_style=True)


# Register path representer and constructor for Posix and Windows to Dumper
yaml.add_representer(pathlib.Path, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.PosixPath, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.WindowsPath, path_representer, Dumper=Dumper)
yaml.add_constructor('!path', path_constructor, Loader=Loader)

# Register representer and constructor to convert tuple between Python and yaml.
yaml.add_representer(tuple, tuple_representer, Dumper=Dumper)

# Register representer for sympy symbols
yaml.add_representer(sp.Symbol, sp_symbol_representer, Dumper=Dumper)


class BaseYAMLConfig(pydantic.BaseModel):
    """ Mixin to add yaml dumping and loading support to pydantic ``BaseModel``. """

    @staticmethod
    def _yaml_load(yaml_str: str) -> Dict:
        """ Only convert yaml back to dict but don't process it further. """
        return yaml.load(yaml_str, Loader=Loader)

    def yaml(self, exclude: Dict[str, Any] = None) -> str:
        """ Convert model to yaml string. """
        if exclude is None:
            exclude = {}
        return yaml.dump(self.dict(exclude=exclude), Dumper=Dumper)

    @classmethod
    def from_yaml(cls, yaml_str: str):
        """ Load model from yaml string. """
        return cls(**cls._yaml_load(yaml_str))  # noqa

    class Config:
        # Allow, e.g., numpy arrays or custom types
        arbitrary_types_allowed = True
