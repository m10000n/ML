import torch
import torch.nn as nn

from model.He_ResNet.architecture.residual import ResidualBlock_
from model.model_helper import Add


class Bottleneck(ResidualBlock_):
    """
    Implements a residual block using the bottleneck architecture from the ResNet paper.

    This block applies a sequence of three convolutions in the residual branch: a 1x1 convolution for dimensionality
    reduction, a 3x3 convolution for spatial feature extraction, and a final 1x1 convolution for dimensionality
    restoration. Each convolution is followed by Batch Normalization and a ReLU activation, except the last, which is
    followed only by Batch Normalization. When the number of input and output channels differ, the identity branch uses
    a 1x1 convolution with stride 2 followed by Batch Normalization to match spatial and channel dimensions, following
    the bottleneck design from “Deep Residual Learning for Image Recognition” (He et al., 2015). The output is computed
    by adding the residual branch and the identity branch, followed by a ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the Bottleneck class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super().__init__(in_channels=in_channels, out_channels=out_channels)

        down = in_channels < out_channels
        hidden_channels = out_channels // 4

        self._residual = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=2 if down else 1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        self._identity: nn.Sequential | nn.Identity
        if down:
            self._identity = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self._identity = nn.Identity()

        self._add = Add()
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bottleneck class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_channels`, height_out, width_out).
        """
        residual = self._residual(x)
        identity = self._identity(x)
        x = self._add(residual, identity)
        x = self._relu(x)
        return x
