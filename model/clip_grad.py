from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Type, final

from torch import nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from helper import class_ as class_helper


# abstract class
class ClipGrad_(ABC):
    config: ClipGradConfig_

    @abstractmethod
    def __init__(self, config: ClipGradConfig_, module: nn.Module):
        self.config = copy.deepcopy(config)
        self.module = module

    @staticmethod
    def create(config: ClipGradConfig_, module: nn.Module) -> ClipGrad_:
        return config.class_(config=config, module=module)

    @abstractmethod
    def clip(self) -> None:
        pass


class ClipGradNorm(ClipGrad_):
    config: ClipGradConfigNorm

    def __init__(self, config: ClipGradConfigNorm, module: nn.Module):
        super().__init__(config, module)

    def clip(self) -> None:
        clip_grad_norm_(
            parameters=self.module.parameters(),
            max_norm=self.config.max_norm,
            norm_type=self.config.norm_type,
            error_if_nonfinite=self.config.error_if_nonfinite,
            foreach=self.config.foreach,
        )


class ClipGradValue(ClipGrad_):
    config: ClipGradConfigValue

    def __init__(self, config: ClipGradConfigValue, module: nn.Module):
        super().__init__(config, module)

    def clip(self) -> None:
        clip_grad_value_(
            parameters=self.module.parameters(), clip_value=self.config.clip_value, foreach=self.config.foreach
        )


# abstract data class
@dataclass
class ClipGradConfig_(ABC):
    class_: ClassVar[Type[ClipGrad_]]

    @staticmethod
    @final
    def from_dict(dict_: dict[str, Any]) -> ClipGradConfig_:
        return class_helper.from_dict(class_=ClipGradConfig_, dict_=dict_)

    @classmethod
    def _from_dict(cls: Type[ClipGradConfig_], dict_: dict[str, Any]) -> ClipGradConfig_:
        return cls(**dict_)

    @final
    def as_dict(self) -> dict[str, Any]:
        return class_helper.as_dict(self)

    def _as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClipGradConfigNorm(ClipGradConfig_):
    class_: ClassVar[Type[ClipGradNorm]] = ClipGradNorm

    max_norm: float
    norm_type: float = 2.0
    error_if_nonfinite: bool = False
    foreach: bool | None = None


@dataclass
class ClipGradConfigValue(ClipGradConfig_):
    class_: ClassVar[Type[ClipGradValue]] = ClipGradValue

    clip_value: float
    foreach: bool | None = None
