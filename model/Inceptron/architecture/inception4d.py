from __future__ import annotations

import torch
import torch.nn as nn

from model.model_helper import Concat


class Inception4d(nn.Module):
    """
    Implements the Inception module for four-dimensional input.

    This module captures multi-scale spatial features through four parallel branches:
        - A 1x1 convolution for channel reduction.
        - A 1x1 convolution followed by a 3x3 convolution for medium-scale context.
        - A 1x1 convolution followed by a 5x5 convolution for larger receptive fields.
        - A 3x3 max-pooling followed by a 1x1 convolution for local aggregation.

    Each convolution is immediately followed by batch normalization and ReLU. The outputs of all
    branches are concatenated along the channel dimension. This implementation follows the original Inception design
    from “Going Deeper with Convolutions” (Szegedy et al., 2015), extended to handle 4D tensors.


    Attributes:
        in_channels (int): Number of input channels.
        out_1x1 (int): Number of output channels of the 1x1 convolutional branch.
        out_3x3_reduce (int): Number of output channels of the 1x1 convolution in the 3x3 branch.
        out_3x3 (int): Number of output channels of the 3x3 convolutional branch.
        out_5x5_reduce (int): Number of output channels of the 1x1 convolution in the 5x5 branch.
        out_5x5 (int): Number of output channels of the 5x5 convolutional branch.
        out_pool (int): Number of output channels of the max pooling branch.
    """

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
        Initializes the Inception4d class.

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

        self._1x1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_1x1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(num_features=out_1x1),
            nn.ReLU(),
        )
        self._3x3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_3x3_reduce, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(num_features=out_3x3_reduce),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_3x3_reduce, out_channels=out_3x3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=out_3x3),
            nn.ReLU(),
        )
        self._5x5 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_5x5_reduce, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(num_features=out_5x5_reduce),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_5x5_reduce, out_channels=out_5x5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(num_features=out_5x5),
            nn.ReLU(),
        )
        self._pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=in_channels, out_channels=out_pool, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(num_features=out_pool),
            nn.ReLU(),
        )
        self._concat = Concat(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Inception4d module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, x, y, z).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_1x1` + `out_3x3` + `out_5x5` + `out_pool`, x, y, z).
        """
        x1 = self._1x1(x)
        x2 = self._3x3(x)
        x3 = self._5x5(x)
        x4 = self._pool(x)

        return self._concat([x1, x2, x3, x4])
