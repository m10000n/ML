from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, List, Literal, Type, cast, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD as SGD_
from torch.optim import Adam as Adam_
from torch.optim import AdamW as AdamW_
from torch.optim import Optimizer

from helper import class_ as class_helper
from model.Dosovitskiy_ViT.architecture.add_class_token import AddClassToken
from model.Dosovitskiy_ViT.architecture.pos_embed import PosEmbed


# abstract class
class Optimizer_(ABC):
    config: OptimizerConfig_

    @abstractmethod
    def __init__(self, config: OptimizerConfig_, optimizer: Optimizer, module: nn.Module):
        self.config = copy.deepcopy(config)
        self.optimizer = optimizer
        self.module = module

    @staticmethod
    def create(config: OptimizerConfig_, module: nn.Module) -> Optimizer_:
        return config.class_(config=config, module=module)  # type: ignore[call-arg]

    @staticmethod
    def get_params(
        module: nn.Module, exclude: List[Literal["bias", "norm", "class_token", "pos_embed"]] | None
    ) -> tuple[List[nn.Parameter], List[nn.Parameter]]:
        if exclude is None:
            return [p for p in module.parameters() if p.requires_grad], []

        norm_classes = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        )

        class_token_classes = (AddClassToken,)

        pos_embed_classes = (PosEmbed,)

        included: List[nn.Parameter] = []
        excluded: List[nn.Parameter] = []
        seen = set()

        for module_ in module.modules():
            for param_name, param in module_.named_parameters(recurse=False):
                if not param.requires_grad or id(param) in seen:
                    continue

                seen.add(id(param))

                layer_type = type(module_)

                if (
                    ("bias" in exclude and param_name == "bias")
                    or ("norm" in exclude and layer_type in norm_classes)
                    or ("class_token" in exclude and layer_type in class_token_classes)
                    or ("pos_embed" in exclude and layer_type in pos_embed_classes)
                ):
                    excluded.append(param)
                else:
                    included.append(param)

        return included, excluded

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def get_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def set_learning_rate(self, lr: float) -> None:
        self.optimizer.param_groups[0]["lr"] = lr


# concrete classes
class Adam(Optimizer_):
    config: AdamConfig

    def __init__(self, config: AdamConfig, module: nn.Module):
        included, excluded = self.get_params(module=module, exclude=config.exclude)
        params = [
            {"params": included, "weight_decay": config.weight_decay},
            {"params": excluded, "weight_decay": 0.0},
        ]

        optimizer = Adam_(
            params=params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            amsgrad=config.amsgrad,
            foreach=config.foreach,
            maximize=config.maximize,
            capturable=config.capturable,
            differentiable=config.differentiable,
            fused=config.fused,
        )
        super().__init__(config=config, optimizer=optimizer, module=module)


class AdamW(Optimizer_):
    config: AdamWConfig

    def __init__(self, config: AdamWConfig, module: nn.Module):
        included, excluded = self.get_params(module=module, exclude=config.exclude)
        params = [
            {"params": included, "weight_decay": config.weight_decay},
            {"params": excluded, "weight_decay": 0.0},
        ]

        optimizer = AdamW_(
            params=params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            amsgrad=config.amsgrad,
            maximize=config.maximize,
            foreach=config.foreach,
            capturable=config.capturable,
            differentiable=config.differentiable,
            fused=config.fused,
        )
        super().__init__(config=config, optimizer=optimizer, module=module)


class SGD(Optimizer_):
    config: SGDConfig

    def __init__(self, config: SGDConfig, module: nn.Module):
        optimizer = SGD_(
            params=module.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            dampening=config.dampening,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
            maximize=config.maximize,
            foreach=config.foreach,
            differentiable=config.differentiable,
            fused=config.fused,
        )
        super().__init__(config=config, optimizer=optimizer, module=module)


# abstract data class
@dataclass
class OptimizerConfig_(ABC):
    class_: ClassVar[Type[Optimizer_]]

    @staticmethod
    @final
    def from_dict(dict_: dict[str, Any]) -> OptimizerConfig_:
        return class_helper.from_dict(class_=OptimizerConfig_, dict_=dict_)

    @classmethod
    def _from_dict(cls: Type[OptimizerConfig_], dict_: dict[str, Any]) -> OptimizerConfig_:
        return cls(**dict_)

    @final
    def as_dict(self) -> dict[str, Any]:
        return class_helper.as_dict(self)

    def _as_dict(self) -> dict[str, Any]:
        return asdict(self)


# concrete data classes
@dataclass(init=False)
class AdamConfig(OptimizerConfig_):
    class_: ClassVar[Type[Adam]] = Adam

    lr: float | Tensor
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    foreach: bool | None
    maximize: bool
    capturable: bool
    differentiable: bool
    fused: bool | None
    decoupled_weight_decay: bool | None
    exclude: List[Literal["bias", "norm", "class_token", "pos_embed"]] | None

    def __init__(
        self,
        lr: float | Tensor | list[float] = 0.001,
        betas: tuple[float, float] | list[float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        decoupled_weight_decay: bool | None = False,
        exclude: List[Literal["bias", "norm", "class_token", "pos_embed"]] | None = None,
    ):
        super().__init__()
        self.lr = torch.tensor(lr) if isinstance(lr, list) else lr
        self.betas = cast(tuple[float, float], tuple(betas) if isinstance(betas, list) else betas)
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.foreach = foreach
        self.maximize = maximize
        self.capturable = capturable
        self.differentiable = differentiable
        self.fused = fused
        self.decoupled_weight_decay = decoupled_weight_decay
        self.exclude = exclude


@dataclass(init=False)
class AdamWConfig(OptimizerConfig_):
    class_: ClassVar[Type[AdamW]] = AdamW

    lr: float | Tensor
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    maximize: bool
    foreach: bool | None
    capturable: bool
    differentiable: bool
    fused: bool | None
    exclude: List[Literal["bias", "norm", "class_token", "pos_embed"]] | None

    def __init__(
        self,
        lr: float | Tensor | list[float] = 0.001,
        betas: tuple[float, float] | list[float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        exclude: List[Literal["bias", "norm", "class_token", "pos_embed"]] | None = [
            "bias",
            "norm",
            "class_token",
            "pos_embed",
        ],
    ):
        super().__init__()
        self.lr = torch.tensor(lr) if isinstance(lr, list) else lr
        self.betas = cast(tuple[float, float], tuple(betas) if isinstance(betas, list) else betas)
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        self.differentiable = differentiable
        self.fused = fused
        self.exclude = exclude


@dataclass(init=False)
class SGDConfig(OptimizerConfig_):
    class_: ClassVar[Type[SGD]] = SGD

    lr: float | Tensor
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool
    maximize: bool
    foreach: bool | None
    differentiable: bool
    fused: bool | None

    def __init__(
        self,
        lr: float | Tensor | list[float] = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        differentiable: bool = False,
        fused: bool | None = None,
    ):
        super().__init__()
        self.lr = torch.tensor(lr) if isinstance(lr, list) else lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        self.foreach = foreach
        self.differentiable = differentiable
        self.fused = fused
