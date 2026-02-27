from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from torch.optim.lr_scheduler import LambdaLR

from helper.validator import validator
from model.optimizer import Optimizer_


# class
class Warmup:
    config: WarmupConfig

    def __init__(self, config: WarmupConfig, optimizer: Optimizer_):
        self.config = copy.deepcopy(config)
        self.scheduler = LambdaLR(
            optimizer=optimizer.optimizer, lr_lambda=Warmup._get_lr_lambda(config.function, config.n_steps)
        )
        self.n_steps = 0

    @staticmethod
    def _get_lr_lambda(
        function: Literal["linear", "cosine", "sqrt", "quadratic", "cubic", "log", "exp"], warmup_steps: int
    ) -> Callable:
        if function == "linear":
            return lambda step: (step + 1) / (warmup_steps + 1)
        elif function == "cosine":
            return lambda step: 0.5 * (1 - math.cos(math.pi * (step + 1) / (warmup_steps + 1)))
        elif function == "sqrt":
            return lambda step: ((step + 1) / (warmup_steps + 1)) ** 0.5
        elif function == "quadratic":
            return lambda step: ((step + 1) / (warmup_steps + 1)) ** 2
        elif function == "cubic":
            return lambda step: ((step + 1) / (warmup_steps + 1)) ** 3
        elif function == "log":
            return lambda step: math.log(step + 2) / math.log(warmup_steps + 2)
        elif function == "exp":
            return lambda step: (math.exp((step + 1) / (warmup_steps + 1)) - 1) / (math.e - 1)
        else:
            raise ValueError(f"Invalid warmup function: {function}")

    def get_n_total_steps(self) -> int:
        return self.config.n_steps

    def step(self) -> None:
        if self.done():
            raise RuntimeError("Failed to update the warmup scheduler. The warmup is already complete.")

        self.scheduler.step()
        self.n_steps += 1

    def done(self) -> bool:
        return self.n_steps >= self.config.n_steps


# data class
@dataclass
class WarmupConfig:
    n_steps: int
    function: Literal["linear", "cosine", "sqrt", "quadratic", "cubic", "log", "exp"]

    @validator.constraints("n_steps", "x > 0")
    def __init__(
        self, n_steps: int, function: Literal["linear", "cosine", "sqrt", "quadratic", "cubic", "log", "exp"] = "linear"
    ):
        self.n_steps = n_steps
        self.function = function

    @staticmethod
    def from_dict(dict_: dict[str, Any]) -> WarmupConfig:
        return WarmupConfig(**dict_)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
