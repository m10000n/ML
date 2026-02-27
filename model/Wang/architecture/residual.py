import torch.nn as nn
from torch import Tensor

from helper.validator import validator
from model.model_helper import Add


class Residual(nn.Module):
    """
    Residual block for 3D convolutional feature extraction with identity mapping.

    This block implements a 3D residual unit inspired by the ResNet architecture, as described in the paper
    "Deep Residual Learning for Image Recognition" by He et al. (2015). It enables effective feature extraction while
    preserving gradient flow in deep convolutional networks. The block consists of two 3D convolutional layers with
    batch normalization and ReLU activation in the residual path, along with an identity shortcut connection. If the
    input and output channel dimensions differ, or if a stride other than one is used, the identity path includes a
    1x1x1 convolution with matching stride and batch normalization to align shapes before addition. The output is
    obtained by adding the residual and identity paths, followed by a ReLU activation.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride used in the first convolution of the residual path and identity mapping.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        Initializes the Residual class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride used in the first convolution of the residual path and identity mapping.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        super().__init__()

        self._residual = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        )

        self._identity: nn.Sequential | nn.Identity
        if in_channels != out_channels or stride != 1:
            self._identity = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self._identity = nn.Identity()

        self._add = Add()
        self._relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Residual class.

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


class ResidualStack(nn.Module):
    """
    Stack of residual blocks for hierarchical feature extraction in fMRI data.

    This module applies four sequential groups of residual blocks to extract multi-scale spatiotemporal features from
    4D fMRI volumes. Each group starts with a residual block that adjusts channel dimensions and spatial resolution,
    followed by a variable number of additional residual blocks for deeper feature extraction while preserving gradient
    flow. The number of additional residual repetitions per group is configurable via `n_res_reps`, following the design
    used in the authors' reference implementation to control model depth for different experimental needs. This
    implementation follows the architecture described in the paper "Decoding and Mapping Task States of the Human Brain
    via Deep Learning" by Wang et al. (2020) and is based on the reference implementation available at
    https://github.com/ustc-bmec/Whole-Brain-Conv.

    Attributes:
        n_res_reps (int): Number of additional residual repetitions per group.
    """

    @validator.constraints("n_res_reps", "x >= 0")
    def __init__(self, n_res_reps: int):
        """
        Initializes the ResidualStack class.

        Args:
            n_res_reps (int): Number of additional residual repetitions per group.
        """
        super().__init__()

        res1 = [Residual(in_channels=24, out_channels=32, stride=1)]
        res2 = [Residual(in_channels=32, out_channels=64, stride=2)]
        res3 = [Residual(in_channels=64, out_channels=64, stride=2)]
        res4 = [Residual(in_channels=64, out_channels=128, stride=2)]

        for _ in range(n_res_reps):
            res1.append(Residual(in_channels=32, out_channels=32, stride=1))
            res2.append(Residual(in_channels=64, out_channels=64, stride=1))
            res3.append(Residual(in_channels=64, out_channels=64, stride=1))
            res4.append(Residual(in_channels=128, out_channels=128, stride=1))

        self._res1 = nn.Sequential(*res1)
        self._res2 = nn.Sequential(*res2)
        self._res3 = nn.Sequential(*res3)
        self._res4 = nn.Sequential(*res4)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResidualStack class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_channels`, width_out, height_out, depth_out).
        """
        x = self._res1(x)
        x = self._res2(x)
        x = self._res3(x)
        x = self._res4(x)
        return x
