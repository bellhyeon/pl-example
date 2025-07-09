from abc import ABCMeta, abstractmethod
from typing import Dict, Any

from torch import nn

_MODEL_REGISTER: Dict[str, Any] = {}


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @classmethod
    def register(cls, model_name: str):
        def decorator(model_cls: Any):
            if model_name in _MODEL_REGISTER:
                raise ValueError(f"Model {model_name} already registered")
            _MODEL_REGISTER[model_name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get_model(cls, **kwargs):
        model_name = kwargs.pop("name")
        if model_name not in _MODEL_REGISTER:
            raise KeyError(f"{model_name} is not registered")
        return _MODEL_REGISTER[model_name](**kwargs)

    @classmethod
    def get_model_names(cls):
        return list(_MODEL_REGISTER.keys())
