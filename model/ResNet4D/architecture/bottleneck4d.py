from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Type

import torch
import torch.nn as nn

from model.model_helper import Add
from model.ResNet4D.architecture.residual4d import (
    ResidualBlock4D_,
    ResidualBlock4DConfig_,
)


class Bottleneck4D(ResidualBlock4D_):
    """
    Implements a 4D residual block using the bottleneck architecture from the ResNet paper.

    This block applies a sequence of three 3D convolutions in the residual branch: a 1x1 convolution for dimensionality
    reduction, a 3x3 convolution for spatial feature extraction, and a final 1x1 convolution for dimensionality
    restoration. Each convolution is followed by Batch Normalization and a ReLU activation, except the last, which is
    followed only by Batch Normalization. When the number of input and output channels differ, the identity branch uses
    a 1x1 convolution with stride 2 followed by Batch Normalization to match spatial and channel dimensions, following
    the bottleneck design from “Deep Residual Learning for Image Recognition” (He et al., 2015).
    The output is computed by adding the residual branch and the identity branch,followed by a ReLU activation.

    Attributes:
        config (Bottleneck4DConfig): Configuration of the module.
    """

    config: Bottleneck4DConfig

    def __init__(self, config: Bottleneck4DConfig):
        """
        Initialize the Bottleneck4D class.

        Args:
            config (Bottleneck4DConfig): Configuration of the module.
        """
        super().__init__(config)

        down = config.in_channels < config.out_channels
        hidden_channels = config.out_channels // 4

        self._residual = nn.Sequential(
            nn.Conv3d(
                in_channels=config.in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                # stride=2 if down else 1,
                stride=2 if down and not config.late_stride else 1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=2 if down and config.late_stride else 1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=config.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
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
        """
        Forward pass of the Bottleneck4D class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_channels`, width_out, height_out, depth_out).
        """
        residual = self._residual(x)
        identity = self._identity(x)
        x = self._add(residual, identity)
        x = self._relu(x)
        return x


@dataclass()
class Bottleneck4DConfig(ResidualBlock4DConfig_):
    """
    Configuration class for the Bottleneck4D class.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        late_stride (bool): Whether to apply stride 2 to the identity branch.
        class_ (Type[Bottleneck4D]): Class of the module.

    Note:
        - All instance variables must be set to non-None values before calling `as_dict`.
    """

    class_: ClassVar[Type[Bottleneck4D]] = Bottleneck4D

    late_stride: bool = False

    def __init__(self, late_stride: bool = False):
        """
        Initialize the Bottleneck4DConfig class.

        Args:
            late_stride (bool): Whether to apply spatial downsampling in the second convolution in the residual branch.
                Default: False.
        """
        self.late_stride = late_stride

    def _as_dict(self) -> dict[str, Any]:
        return super()._as_dict() | {
            "late_stride": self.late_stride,
        }
