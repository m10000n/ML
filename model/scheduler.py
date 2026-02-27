from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal, Type, final

from torch.optim.lr_scheduler import CosineAnnealingLR as CosineAnnealingLR_
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau_
from torch.optim.lr_scheduler import StepLR as StepLR_

from helper import class_ as class_helper
from model.optimizer import Optimizer_

_STEP_TYPE_ = StepLR_ | CosineAnnealingLR_
_METRIC_TYPE_ = ReduceLROnPlateau_


# abstract classes
class Scheduler_(ABC):
    config: SchedulerConfig_
    optimizer: Optimizer_
    scheduler: _STEP_TYPE_ | _METRIC_TYPE_

    @abstractmethod
    def __init__(self, config: SchedulerConfig_, optimizer: Optimizer_, scheduler: _STEP_TYPE_ | _METRIC_TYPE_):
        self.config = copy.deepcopy(config)
        self.optimizer = optimizer
        self.scheduler = scheduler

    @staticmethod
    def create(config: SchedulerConfig_, optimizer: Optimizer_) -> Scheduler_:
        return config.class_(config=config, optimizer=optimizer)  # type: ignore[call-arg]


class StepLRScheduler_(Scheduler_):
    config: StepLRSchedulerConfig_
    scheduler: _STEP_TYPE_
    step_count: int = 0

    @abstractmethod
    def __init__(self, config: StepLRSchedulerConfig_, optimizer: Optimizer_, scheduler: _STEP_TYPE_):
        super().__init__(config=config, optimizer=optimizer, scheduler=scheduler)

    def step(self, type: Literal["step", "epoch"]) -> None:
        if self.config.type == type:
            self.scheduler.step()
            self.step_count += 1


class MetricLRScheduler_(Scheduler_):
    config: MetricLRSchedulerConfig_
    scheduler: _METRIC_TYPE_

    @abstractmethod
    def __init__(self, config: MetricLRSchedulerConfig_, optimizer: Optimizer_, scheduler: _METRIC_TYPE_):
        super().__init__(config=config, optimizer=optimizer, scheduler=scheduler)

    def step(self, metric: float) -> None:
        self.scheduler.step(metric)


# concrete classes
class StepLR(StepLRScheduler_):
    config: StepLRConfig

    def __init__(
        self,
        config: StepLRConfig,
        optimizer: Optimizer_,
    ):
        scheduler = StepLR_(optimizer=optimizer.optimizer, step_size=config.step_size, gamma=config.gamma)
        super().__init__(config=config, optimizer=optimizer, scheduler=scheduler)


class CosineAnnealingLR(StepLRScheduler_):
    config: CosineAnnealingLRConfig

    def __init__(self, config: CosineAnnealingLRConfig, optimizer: Optimizer_):
        scheduler = CosineAnnealingLR_(
            optimizer=optimizer.optimizer,
            T_max=config.t_max,
            eta_min=config.eta_min,
        )
        super().__init__(config=config, optimizer=optimizer, scheduler=scheduler)

    def step(self, type: Literal["step", "epoch"]) -> None:
        if self.config.periodic or self.step_count < self.config.t_max:
            super().step(type)


class ReduceLROnPlateau(MetricLRScheduler_):
    config: ReduceLROnPlateauConfig

    def __init__(
        self,
        config: ReduceLROnPlateauConfig,
        optimizer: Optimizer_,
    ):
        scheduler = ReduceLROnPlateau_(
            optimizer=optimizer.optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps,
        )
        super().__init__(config=config, optimizer=optimizer, scheduler=scheduler)


# abstract data calsses
@dataclass
class SchedulerConfig_(ABC):
    class_: ClassVar[Type[Scheduler_]]

    @staticmethod
    @final
    def from_dict(dict_: dict[str, Any]) -> SchedulerConfig_:
        return class_helper.from_dict(class_=SchedulerConfig_, dict_=dict_)

    @classmethod
    def _from_dict(cls: Type[SchedulerConfig_], dict_: dict[str, Any]) -> SchedulerConfig_:
        return cls(**dict_)

    @final
    def as_dict(self) -> dict[str, Any]:
        return class_helper.as_dict(self)

    def _as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepLRSchedulerConfig_(SchedulerConfig_):
    class_: ClassVar[Type[StepLRScheduler_]]

    type: Literal["batch", "epoch"]


@dataclass
class MetricLRSchedulerConfig_(SchedulerConfig_):
    class_: ClassVar[Type[MetricLRScheduler_]]


# concrete data classes
@dataclass
class StepLRConfig(StepLRSchedulerConfig_):
    class_: ClassVar[Type[StepLR]] = StepLR

    step_size: int
    gamma: float = 0.1


@dataclass
class CosineAnnealingLRConfig(StepLRSchedulerConfig_):
    class_: ClassVar[Type[CosineAnnealingLR]] = CosineAnnealingLR

    t_max: int
    eta_min: float = 0.0
    periodic: bool = False


@dataclass
class ReduceLROnPlateauConfig(MetricLRSchedulerConfig_):
    class_: ClassVar[Type[ReduceLROnPlateau]] = ReduceLROnPlateau

    mode: Literal["min", "max"] = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 0.0001
    threshold_mode: Literal["rel", "abs"] = "rel"
    cooldown: int = 0
    min_lr: float | list[float] = 0
    eps: float = 1e-08
