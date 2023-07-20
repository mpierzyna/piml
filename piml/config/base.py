from typing import Dict, Any

import pydantic
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


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
