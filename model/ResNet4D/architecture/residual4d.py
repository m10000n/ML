from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Type

import torch
import torch.nn as nn

from helper.class_ import Module_, ModuleConfig_
from model.model_helper import Add


class ResidualBlock4D_(Module_):
    """
    Abstract base class for 4D residual blocks. Subclasses must implement the forward pass.

    Attributes:
        config (ResidualBlock4DConfig_): Configuration of the module.
    """

    config: ResidualBlock4DConfig_

    @abstractmethod
    def __init__(self, config: ResidualBlock4DConfig_):
        if not config.is_initialized(mode="use"):
            raise ValueError(
                f"Failed to create an instance of `{self.__class__.__name__}`, because`config` is not initialized."
            )

        super().__init__(config)


class ResidualBlock4DB(ResidualBlock4D_):
    """
    Implements a 4D residual block using Option B from the ResNet paper.

    This block applies two 3x3 convolutional layers in the residual branch. The first is followed by Batch Normalization
    and a ReLU activation, while the second is followed only by Batch Normalization. When the number of input and output
    channels differ, the identity branch uses a 1x1 convolution with stride 2 followed by Batch Normalization to match
    spatial and channel dimensions, following Option B from “Deep Residual Learning for Image Recognition”
    (He et al., 2016). The output is computed by adding the residual branch and the identity branch, followed by a ReLU
    activation.

    Attributes:
        config (ResidualBlock4DBConfig): Configuration of the module.
    """

    config: ResidualBlock4DBConfig

    def __init__(self, config: ResidualBlock4DBConfig):
        """
        Initialize the ResidualBlock4DB class.

        Args:
            config (ResidualBlock4DBConfig): Configuration of the module.
        """
        super().__init__(config)

        down = config.in_channels < config.out_channels

        self._residual = nn.Sequential(
            nn.Conv3d(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                kernel_size=3,
                stride=2 if down else 1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(config.out_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=config.out_channels,
                out_channels=config.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(config.out_channels),
        )

        self._identity: nn.Sequential | nn.Identity
        if down:
            self._identity = nn.Sequential(
                nn.Conv3d(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm3d(config.out_channels),
            )
        else:
            self._identity = nn.Identity()

        self._add = Add()
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self._residual(x)
        x = self._identity(x)
        x = self._add(residual, x)
        x = self._relu(x)
        return x


@dataclass()
class ResidualBlock4DConfig_(ModuleConfig_):
    """
    Abstract base configuration class for ResidualBlock4D_ classes.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels
    """

    class_: ClassVar[Type[ResidualBlock4D_]]

    _in_channels: int = field(init=False)
    _out_channels: int = field(init=False)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @in_channels.setter
    def in_channels(self, value: int) -> None:
        if value <= 0:
            raise ValueError("`in_channels` must be positive.")
        self._in_channels = value

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @out_channels.setter
    def out_channels(self, value: int) -> None:
        if value <= 0:
            raise ValueError("`out_channels` must be positive.")
        self._out_channels = value

    def _as_dict(self) -> dict[str, Any]:
        return {}

    def is_initialized(self, mode: Literal["as_dict", "use", "full"] = "full") -> bool:
        if mode in ["use", "full"]:
            for attribute in ["in_channels", "out_channels"]:
                if not self.attr_is_initialized(attribute):
                    return False
            return True
        else:
            return True


@dataclass()
class ResidualBlock4DBConfig(ResidualBlock4DConfig_):
    """
    Configuration class for the ResidualBlock4DB class.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        class_ (Type[ResidualBlock4DB]): Class of the module.

    Note:
        - All instance variables must be set to non-None values befor using an instance of this class.
    """

    class_: ClassVar[Type[ResidualBlock4DB]] = ResidualBlock4DB
