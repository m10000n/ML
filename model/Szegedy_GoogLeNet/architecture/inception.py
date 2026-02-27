from __future__ import annotations

import torch
import torch.nn as nn

from helper.validator import validator
from model.model_helper import Concat


class Inception(nn.Module):
    """
    Implements the Inception module.

    This module captures multi-scale spatial features through four parallel branches:
        - A 1x1 convolution for channel reduction.
        - A 1x1 convolution followed by a 3x3 convolution for medium-scale context.
        - A 1x1 convolution followed by a 5x5 convolution for larger receptive fields.
        - A 3x3 max-pooling followed by a 1x1 convolution for local aggregation.

    Each convolution is immediately followed by a ReLU activation. The outputs of all branches are concatenated along
    the channel dimension. This implementation follows the design from the paper “Going Deeper with Convolutions”
    (Szegedy et al., 2015).

    Attributes:
        in_channels (int): Number of input channels.
        out_1x1 (int): Number of output channels of the 1x1 branch.
        out_3x3_reduce (int): Number of channels in the 1x1 reduction before the 3x3 branch.
        out_3x3 (int): Number of output channels of the 3x3 branch.
        out_5x5_reduce (int): Number of channels in the 1x1 reduction before the 5x5 branch.
        out_5x5 (int): Number of output channels of the 5x5 branch.
        out_pool (int): Number of output channels of the pooling branch's 1x1 convolution.
    """

    @validator.constraints("in_channels", "x > 0")
    @validator.constraints("out_1x1", "x > 0")
    @validator.constraints("out_3x3_reduce", "x > 0")
    @validator.constraints("out_3x3", "x > 0")
    @validator.constraints("out_5x5_reduce", "x > 0")
    @validator.constraints("out_5x5", "x > 0")
    @validator.constraints("out_pool", "x > 0")
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        out_3x3_reduce: int,
        out_3x3: int,
        out_5x5_reduce: int,
        out_5x5: int,
        out_pool: int,
    ):
        """
        Initializes the Inception class.

        Args:
            in_channels (int): Number of input channels. Must be a positive integer.
            out_1x1 (int): Number of output channels of the 1x1 convolutional branch. Must be a positive integer.
            out_3x3_reduce (int):
                Number of output channels of the 1x1 convolution in the 3x3 branch. Must be a positive integer.
            out_3x3 (int): Number of output channels of the 3x3 convolutional branch. Must be a positive integer.
            out_5x5_reduce (int):
                Number of output channels of the 1x1 convolution in the 5x5 branch. Must be a positive integer.
            out_5x5 (int): Number of output channels of the 5x5 convolutional branch. Must be a positive integer.
            out_pool (int): Number of output channels of the max pooling branch. Must be a positive integer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_1x1 = out_1x1
        self.out_3x3_reduce = out_3x3_reduce
        self.out_3x3 = out_3x3
        self.out_5x5_reduce = out_5x5_reduce
        self.out_5x5 = out_5x5
        self.out_pool = out_pool

        self._1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_1x1, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self._3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_3x3_reduce, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_3x3_reduce, out_channels=out_3x3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self._5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_5x5_reduce, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_5x5_reduce, out_channels=out_5x5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self._pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_pool, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self._concat = Concat(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Inception class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_1x1` + `out_3x3` + `out_5x5` + `out_pool`, width, height).
        """
        x1 = self._1x1(x)
        x2 = self._3x3(x)
        x3 = self._5x5(x)
        x4 = self._pool(x)

        return self._concat([x1, x2, x3, x4])
