import importlib
import json
import typing as tp
from dataclasses import dataclass, fields, is_dataclass
from typing import ClassVar

import yaml

BaseTypes = tp.Union[str, int, float, bool, None, list, dict]


class BaseConfig:
    strict_field_types: ClassVar[bool] = True

    def __post_init__(self):
        if not is_dataclass(self):
            raise TypeError(
                f"Class {type(self).__name__} must be decorated with @dataclass(frozen=True)"
            )

        # Access the dataclass parameters to check if frozen=True was set
        dataclass_params = self.__dataclass_params__
        if not dataclass_params.frozen:
            raise TypeError(
                f"Class {type(self).__name__} must use @dataclass(frozen=True)"
            )

        if self.strict_field_types:
            for f in fields(self):
                value = getattr(self, f.name)
                self._validate_field_type(f.name, value)

    def _validate_field_type(self, name, value):
        """
        Validate that a field value is either a basic type, a ConfigBase, or a collection of them.
        :param name: Name of the field, used for error messages only.
        :param value: Value of the field.
        :return: None, raises an error if the field is invalid.
        """
        if isinstance(value, (str, int, float, bool, type(None), BaseConfig)):
            return
        elif isinstance(value, list):
            for i, v in enumerate(value):
                self._validate_field_type(f"{name}[{i}]", v)
        elif isinstance(value, dict):
            for k, v in value.items():
                assert isinstance(k, str), f"Key {k} in {name} is not a string"
                self._validate_field_type(f"{name}[{k}]", v)
        else:
            raise ValueError(
                f"Field '{name}' in {self.__class__.__name__} has invalid type '{type(value).__name__}'. "
                "Fields should be of basic type, ConfigBase, or collections of them. "
                "The only exception of this rule is the priors field."
            )

    def to_dict(self):
        out_dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseConfig):
                out_dict[f.name] = value.to_dict()
            elif isinstance(value, list):
                out_dict[f.name] = [
                    v.to_dict() if isinstance(v, BaseConfig) else v
                    for v in value
                ]
            elif isinstance(value, dict):
                out_dict[f.name] = {
                    k: v.to_dict() if isinstance(v, BaseConfig) else v
                    for k, v in value.items()
                }
            else:
                out_dict[f.name] = value
        module_name = self.__module__
        class_name = self.__class__.__name__
        out_dict["type"] = f"{module_name}:{class_name}"
        return out_dict

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_yaml(self):
        return yaml.dump(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, yaml_str: str):
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @staticmethod
    def from_dict(data: dict):
        module_name, class_name = data.pop("type").split(":")
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)

        for k, v in data.items():
            if isinstance(v, dict) and "type" in v:
                data[k] = BaseConfig.from_dict(
                    v
                )  # Use ConfigBase instead of globals()
            elif isinstance(v, list):
                data[k] = [
                    BaseConfig.from_dict(item)
                    if isinstance(item, dict) and "type" in item
                    else item
                    for item in v
                ]
            elif isinstance(v, dict):
                data[k] = {
                    key: BaseConfig.from_dict(value)
                    if isinstance(value, dict) and "type" in value
                    else value
                    for key, value in v.items()
                }

        return cls(**data)
