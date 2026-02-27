from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Type, final

from helper import class_ as class_helper


# abstract class
class StopCriterion_(ABC):
    config: StopCriterionConfig_

    @abstractmethod
    def __init__(self, config: StopCriterionConfig_):
        self.config = copy.deepcopy(config)

    @staticmethod
    def create(config: StopCriterionConfig_) -> StopCriterion_:
        return config.class_(config=config)

    @abstractmethod
    def stop(self) -> bool:
        pass

    @abstractmethod
    def step(self, metric: float) -> None:
        pass


# concrete classes
class StopEpoch(StopCriterion_):
    config: StopEpochConfig

    def __init__(self, config: StopEpochConfig):
        super().__init__(config)
        self.epoch = 0

    def stop(self) -> bool:
        return self.epoch >= self.config.n_epochs

    def step(self, metric: float) -> None:
        self.epoch += 1


class StopPatience(StopCriterion_):
    config: StopPatienceConfig

    def __init__(self, config: StopPatienceConfig):
        super().__init__(config)
        self.counter = 0
        self.smallest = float("inf")

    def stop(self) -> bool:
        return self.counter >= self.config.patience

    def step(self, metric: float) -> None:
        if not isinstance(metric, float):
            raise ValueError(f"`metric` must be a float. Got {type(metric)}.")

        if metric < self.smallest:
            self.smallest = metric
            self.counter = 0
        else:
            self.counter += 1


class StopThreshold(StopCriterion_):
    config: StopThresholdConfig

    def __init__(self, config: StopThresholdConfig):
        super().__init__(config)
        self.current = float("inf")

    def stop(self) -> bool:
        return self.current <= self.config.threshold

    def step(self, metric: float) -> None:
        if not isinstance(metric, float):
            raise ValueError(f"`metric` must be a float. Got {type(metric)}.")

        self.current = metric


# abstract data class
@dataclass
class StopCriterionConfig_(ABC):
    class_: ClassVar[Type[StopCriterion_]]

    @staticmethod
    @final
    def from_dict(dict_: dict[str, Any]) -> StopCriterionConfig_:
        return class_helper.from_dict(class_=StopCriterionConfig_, dict_=dict_)

    @classmethod
    def _from_dict(cls: Type[StopCriterionConfig_], dict_: dict[str, Any]) -> StopCriterionConfig_:
        return cls(**dict_)

    @final
    def as_dict(self) -> dict[str, Any]:
        return class_helper.as_dict(self)

    def _as_dict(self) -> dict[str, Any]:
        return asdict(self)


# concrete data classes
@dataclass
class StopEpochConfig(StopCriterionConfig_):
    class_: ClassVar[Type[StopEpoch]] = StopEpoch

    n_epochs: int


@dataclass
class StopPatienceConfig(StopCriterionConfig_):
    class_: ClassVar[Type[StopPatience]] = StopPatience

    patience: int


@dataclass
class StopThresholdConfig(StopCriterionConfig_):
    class_: ClassVar[Type[StopThreshold]] = StopThreshold

    threshold: float
