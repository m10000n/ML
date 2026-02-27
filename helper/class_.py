from __future__ import annotations

import copy
import importlib
import inspect
import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from typing import (
    Any,
    ClassVar,
    Literal,
    Type,
    TypeVar,
    cast,
    final,
    get_args,
    get_type_hints,
)

import torch
import torch.nn as nn
from torch.utils import data

from helper import path, system
from helper.validator import validator

_Module = TypeVar("_Module", bound="Module_")
_ModuleConfig = TypeVar("_ModuleConfig", bound="ModuleConfig_")


# abstract classes
class Module_(nn.Module, ABC):
    config: ModuleConfig_

    @abstractmethod
    def __init__(self, config: ModuleConfig_) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)

    @classmethod
    def create(cls: Type[_Module], config: ModuleConfig_) -> _Module:
        if not issubclass(config.class_, cls):
            raise ValueError(
                f"Failed to create module. `{config.class_.__name__}` is not a subclass of `{cls.__name__}`."
            )
        return cast(_Module, config.class_(config))


class Model_(nn.Module, ABC):
    config: ModelConfig_

    @abstractmethod
    def __init__(self, config: ModelConfig_) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)

    @staticmethod
    def create(config: ModelConfig_) -> Model_:
        return config.class_(config)


class Data_(ABC):
    config: DataConfig_

    @abstractmethod
    def __init__(self, config: DataConfig_) -> None:
        self.config = copy.deepcopy(config)

    @staticmethod
    def create(config: DataConfig_) -> Data_:
        return config.class_(config)

    @abstractmethod
    def size_info(self) -> None:
        pass

    @abstractmethod
    def download(self) -> None:
        pass


class Datasets_(ABC):
    config: DatasetsConfig_
    datasets: dict[str, Dataset_]

    @abstractmethod
    def __init__(self, config: DatasetsConfig_, datasets: dict[str, Dataset_]) -> None:
        self.config = copy.deepcopy(config)
        self.datasets = datasets

    @staticmethod
    def create(config: DatasetsConfig_) -> Datasets_:
        return config.class_(config=config)  # type: ignore[call-arg]

    @staticmethod
    def get_split(ids: list[str], split: list[float]) -> list[list[str]]:
        if not math.isclose(sum(split), 1):
            raise ValueError("`split` must sum up to 1.")

        system.get_rng().shuffle(ids)

        datasets_size = [int(len(ids) * fraction) for fraction in split]
        datasets_size[-1] = len(ids) - sum(datasets_size[:-1])

        if 0 in datasets_size:
            raise ValueError("At least one dataset contains no subjects.")

        datasets_ids = []
        start = 0

        for dataset_size in datasets_size:
            datasets_ids.append(sorted(ids[start : start + dataset_size]))
            start += dataset_size

        return datasets_ids

    @abstractmethod
    def get_dataset(self, name: str) -> Dataset_:
        pass

    @abstractmethod
    def get_class_distribution(self) -> list[int]:
        pass

    @abstractmethod
    def get_metadata(self) -> list[str]:
        pass

    @abstractmethod
    def size_info(self) -> None:
        pass

    @abstractmethod
    def download(self) -> None:
        pass

    def get_dataset_names(self) -> list[str]:
        return list(self.datasets.keys())

    def get_datasets(self, names: list[str] | None = None) -> tuple[Dataset_, ...]:
        names_ = self.get_dataset_names() if names is None else names
        return tuple(self.get_dataset(name) for name in names_)


class Dataset_(data.Dataset, ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_class_distribution(self) -> list[int]:
        pass


# abstract data classes
@dataclass
class ModuleConfig_(ABC):
    class_: ClassVar[type[Module_]]

    @classmethod
    @final
    def from_dict(cls: Type[_ModuleConfig], dict_: dict[str, Any]) -> _ModuleConfig:
        return from_dict(class_=cls, dict_=dict_)

    @classmethod
    def _from_dict(cls: Type[_ModuleConfig], dict_: dict[str, Any]) -> _ModuleConfig:
        return cls(**dict_)

    @final
    def as_dict(self) -> dict[str, Any]:
        if not self.is_initialized(mode="as_dict"):
            raise ValueError(f"Failed to create dictionary. `{self.__class__.__name__}` is not initialized.")

        return as_dict(self)

    def _as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_initialized(self, mode: Literal["as_dict", "use", "full"] = "full") -> bool:
        return True

    def attr_is_initialized(self, attribute: str) -> bool:
        return hasattr(self, attribute)


@dataclass(init=False)
class ModelConfig_(ABC):
    class_: ClassVar[type[Model_]]

    description: str
    input_shape: tuple[int, ...]

    @validator.constraints("input_shape", "x > 0")
    def __init__(self, description: str, input_shape: tuple[int, ...] | list[int]):
        input_shape_hint = get_type_hints(self.__class__).get("input_shape")

        len_expected = len(get_args(input_shape_hint))
        len_actual = len(input_shape)

        if len_expected != len_actual:
            raise ValueError(f"`input_shape` must be of length {len_expected}, got {len_actual}.")

        self.description = description
        self.input_shape = tuple(input_shape)

    def __init_subclass__(cls) -> None:
        if get_type_hints(cls).get("input_shape") == tuple[int, ...]:
            raise TypeError(
                f"`{cls.__name__}` must define `input_shape` with a concrete, fixed‐length tuple type in its annotation."
            )

    @staticmethod
    @final
    def from_dict(dict_: dict[str, Any]) -> ModelConfig_:
        return from_dict(class_=ModelConfig_, dict_=dict_)

    @classmethod
    def _from_dict(cls: Type[ModelConfig_], dict_: dict[str, Any]) -> ModelConfig_:
        return cls(**dict_)

    @final
    def as_dict(self) -> dict[str, Any]:
        return as_dict(self)

    def _as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig_(ABC):
    class_: ClassVar[type[Data_]]

    name: ClassVar[str]
    description: str
    ids: list[str]

    def __init_subclass__(cls) -> None:
        if not cls.name.strip():
            raise TypeError(f"Subclass {cls.__name__} must define a non-empty class variable `name`.")


@dataclass
class DatasetsConfig_(ABC):
    class_: ClassVar[type[Datasets_]]

    description: str
    data_config: DataConfig_
    dataset_ids: dict[str, list[str]]
    split: list[float]

    @staticmethod
    @final
    def from_dict(dict_: dict[str, Any]) -> DatasetsConfig_:
        return from_dict(class_=DatasetsConfig_, dict_=dict_)

    @classmethod
    @abstractmethod
    def _from_dict(cls: Type[DatasetsConfig_], dict_: dict[str, Any]) -> DatasetsConfig_:
        pass

    @final
    def as_dict(self) -> dict[str, Any]:
        return as_dict(self)

    @abstractmethod
    def _as_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_classes(self, pretty: bool = False) -> list[str]:
        pass

    @abstractmethod
    def get_dataset_size(self, name: str) -> int:
        pass

    def get_ids(self) -> list[str]:
        return self.data_config.ids

    def is_for_exp(self) -> bool:
        from model.experiment import EXPERIMENT_DATASETS

        return Counter(self.dataset_ids.keys()) == Counter(EXPERIMENT_DATASETS)


# helper functions
def from_dict(class_: type, dict_: dict[str, Any]) -> Any:
    try:
        module_name = dict_.pop("module")
        module = importlib.import_module(module_name)
    except KeyError:
        raise ValueError(f"Failed to load config from dictionary. `dict_` must contain the key `module`.")
    except ImportError as e:
        raise ImportError(f"Failed to import module `{module_name}`.") from e

    try:
        class_name = dict_.pop("class") + "Config"
        class_config = getattr(module, class_name)
    except KeyError:
        raise ValueError(f"Failed to load config from dictionary. `dict_` must contain the key `class`.")
    except AttributeError as e:
        raise AttributeError(f"Failed to get class config `{class_name}` from module `{module_name}`.") from e

    if not issubclass(class_config, class_):
        raise ValueError(
            f"Failed to load config from dictionary. `{class_config.__name__}` is not a subclass of `{class_.__name__}`."
        )

    if not hasattr(class_config, "_from_dict") or not callable(getattr(class_config, "_from_dict")):
        raise ValueError(
            f"Failed to load config from dictionary. `{class_config.__name__}` must define the method `_from_dict`."
        )

    return class_config._from_dict(dict_)


def as_dict(obj: Any) -> dict[str, Any]:
    try:
        class_ = obj.class_
    except AttributeError:
        raise ValueError(f"Failed to create dictionary. `{obj.__class__.__name__}` must define the variable `class_`.")

    try:
        module = path.make_module(inspect.getfile(class_))
    except TypeError as e:
        raise TypeError(f"Failed to create dictionary.") from e

    if not hasattr(obj, "_as_dict") or not callable(getattr(obj, "_as_dict")):
        raise ValueError(f"Failed to create dictionary. `{obj.__class__.__name__}` must define the method `_as_dict`.")

    dict_ = {
        "class": class_.__name__,
        "module": module,
    }

    for key, value in obj._as_dict().items():
        if isinstance(value, torch.Tensor):
            dict_[key] = value.tolist()
        else:
            dict_[key] = value

    return dict_
