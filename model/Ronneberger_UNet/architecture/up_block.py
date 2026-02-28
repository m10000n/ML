from typing import Literal

import torch
import torch.nn as nn

from model.model_helper import CenterCrop, Concat, Pad
from model.Ronneberger_UNet.architecture.conv_block import ConvBlock


class UpBlock(nn.Module):
    """
    Upsampling block for feature refinement and localization in U-Net.

    UpBlock increases the spatial resolution of feature maps during the expansive path of U-Net, enabling precise
    localization by integrating encoder features via skip connections. It supports either transposed convolution-based
    upsampling or interpolation followed by convolution. After upsampling, the corresponding encoder feature map is
    center-cropped for spatial alignment and concatenated with the upsampled tensor. A two-layer convolutional block
    with ReLU activations is then applied to refine the combined feature representations.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        size (int): Size of the center-cropped encoder feature map.
        type_ (Literal["conv", "interpolate"]): Type of upsampling.

    Note:
        The upsampling is determined by `type`:
        - "conv": Uses a transposed convolution with stride 2 and padding 0 to increase spatial dimensions.
        - "interpolate": Uses nearest neighbor interpolation followed by a 2x2 convolution with stride 1 and padding 0
            to increase spatial dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, size: int, type_: Literal["conv", "interpolate"] = "conv"):
        """
        Initialize the UpBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            size (int): Size of the center-cropped encoder feature map.
            type_ (Literal["conv", "interpolate"]): Type of upsampling. Default: "conv".
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.type_ = type_

        self._center_crop = CenterCrop(target_size=[None, None, size, size])

        self._up: nn.ConvTranspose2d | nn.Sequential
        if self.type_ == "conv":
            self._up = nn.ConvTranspose2d(
                in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=2, stride=2, padding=0
            )
        else:
            self._up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                Pad(padding=[None, None, (1, 0), (1, 0)]),
                nn.Conv2d(
                    in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=2, stride=1, padding=0
                ),
            )

        self._concat = Concat(dim=1)
        self._conv_block = ConvBlock(in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UpBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, width_in, height_in).
            x_skip (torch.Tensor):
                Encoder feature map tensor of shape (batch_size, `in_channels`, width_skip, height_skip).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_channels`, `size`, `size`).
        """
        x_skip_cropped = self._center_crop(x_skip)
        x = self._up(x)
        x = self._concat([x_skip_cropped, x])
        x = self._conv_block(x)
        return x
