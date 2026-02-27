from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Type, final

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from helper import class_ as class_helper


# abstract class
class Criterion_(ABC):
    config: CriterionConfig_

    @abstractmethod
    def __init__(self, config: CriterionConfig_, criterion: _Loss):
        self.config = copy.deepcopy(config)
        self.criterion = criterion

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(input, target)

        if self.config.class_weights is not None:
            loss = loss * self.config.class_weights[target]

        if self.config.reduction == "none":
            return loss
        elif self.config.reduction == "mean":
            return loss.mean()
        elif self.config.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.config.reduction}")

    @staticmethod
    def create(config: CriterionConfig_) -> Criterion_:
        if not config.is_fully_initialized():
            raise ValueError("`config` is not fully initialized.")

        return config.class_(config)  # type: ignore[call-arg]

    @classmethod
    def supports_class_weighting(cls) -> bool:
        return False

    def class_weights_to(self, pu: torch.device) -> None:
        if self.config.class_weights is None:
            raise ValueError("This `MLCriterion` does not use class weighting.")
        else:
            self.config.class_weights = self.config.class_weights.to(pu)


# concrete classes
class CrossEntropyLoss(Criterion_):
    config: CrossEntropyLossConfig

    def __init__(self, config: CrossEntropyLossConfig):
        loss = nn.CrossEntropyLoss(
            ignore_index=config.ignore_index,
            reduction="none",
            label_smoothing=config.label_smoothing,
        )
        super().__init__(config=config, criterion=loss)

    @classmethod
    def supports_class_weighting(cls: Type[CrossEntropyLoss]) -> bool:
        return True


class MSELoss(Criterion_):
    config: MSELossConfig

    def __init__(self, config: MSELossConfig):
        loss = nn.MSELoss(reduction="none")
        super().__init__(config=config, criterion=loss)


# abstract data class
@dataclass
class CriterionConfig_(ABC):
    class_: ClassVar[Type[Criterion_]]

    reduction: Literal["none", "mean", "sum"] = "mean"
    class_weighting: bool = False
    _class_weights: torch.Tensor | None = field(default=None)

    @property
    def class_weights(self) -> torch.Tensor | None:
        if self.class_weighting and self._class_weights is None:
            raise ValueError("`class_weights` is not initialized.")

        return self._class_weights

    @class_weights.setter
    def class_weights(self, class_weights: torch.Tensor) -> None:
        if self.class_weighting is False:
            raise ValueError("Class weights must not be set if `class_weighting` is False.")

        self._class_weights = class_weights

    def __post_init__(self) -> None:
        if self.class_weighting and not self.class_.supports_class_weighting():
            raise NotImplementedError(f"Class weighting is not supported by `{self.class_.__name__}`.")

    @staticmethod
    @final
    def from_dict(dict_: dict[str, Any]) -> CriterionConfig_:
        return class_helper.from_dict(class_=CriterionConfig_, dict_=dict_)

    @classmethod
    def _from_dict(cls: Type[CriterionConfig_], dict_: dict[str, Any]) -> CriterionConfig_:
        config = object.__new__(cls)
        config.reduction = dict_["reduction"]
        config.class_weighting = dict_["class_weighting"]
        config._class_weights = dict_["class_weights"]
        return config

    @final
    def as_dict(self) -> dict[str, Any]:
        return class_helper.as_dict(self)

    def _as_dict(self) -> dict[str, Any]:
        return {
            "reduction": self.reduction,
            "class_weighting": self.class_weighting,
            "class_weights": self.class_weights,
        }

    def set_class_weights(self, class_distribution: torch.Tensor | list[int]) -> None:
        if self.class_weighting is False:
            raise ValueError("Class weights must not be set if `class_weighting` is False.")

        class_distribution = torch.tensor(class_distribution)
        inv_class_distribution = 1 / class_distribution
        self._class_weights = inv_class_distribution / inv_class_distribution.mean()

    def is_fully_initialized(self) -> bool:
        return not self.class_weighting or self._class_weights is not None


# concrete data classes
@dataclass
class CrossEntropyLossConfig(CriterionConfig_):
    class_: ClassVar[Type[CrossEntropyLoss]] = CrossEntropyLoss

    ignore_index: int = -100
    label_smoothing: float = 0.0

    def _as_dict(self) -> dict[str, Any]:
        return super()._as_dict() | {
            "ignore_index": self.ignore_index,
            "label_smoothing": self.label_smoothing,
        }


@dataclass
class MSELossConfig(CriterionConfig_):
    class_: ClassVar[Type[MSELoss]] = MSELoss
