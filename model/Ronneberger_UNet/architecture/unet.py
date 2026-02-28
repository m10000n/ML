from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type

import torch
import torch.nn as nn

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator
from model.Ronneberger_UNet.architecture.conv_block import ConvBlock
from model.Ronneberger_UNet.architecture.up_block import UpBlock


class UNet(Model_):
    """
    Deep learning model for the segmentation of biomedical images.

    U-Net is a convolutional neural network architecture designed for efficient and precise biomedical image
    segmentation, featuring a symmetric encoder-decoder (“U-shaped”) structure. The contracting path captures context
    using repeated unpadded 3x3 convolutions with ReLU activations, followed by 2x2 max pooling for downsampling while
    doubling feature channels. Dropout is applied at the end of the contracting path for implicit data augmentation and
    regularization. The expansive path enables precise localization through the integration of encoder features and
    progressive upsampling, refining segmentation predictions at each resolution level. A final 1x1 convolution reduces
    the output to the desired number of classes. This implementation follows the methodology introduced in the paper
    “U-Net: Convolutional Networks for Biomedical Image Segmentation” (Ronneberger et al., 2015).

    Attributes:
        config (UNetConfig): Configuration of the model.
    """

    config: UNetConfig

    def __init__(self, config: UNetConfig):
        """
        Initialize the UNet model.

        Args:
            config (UNetConfig): Configuration of the model.
        """
        super().__init__(config)

        channels = self.config.input_shape[0]
        out1_shape = self.config.input_shape[1] - 4
        out2_shape = out1_shape // 2 - 4
        out3_shape = out2_shape // 2 - 4
        out4_shape = out3_shape // 2 - 4

        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self._conv1 = ConvBlock(in_channels=channels, out_channels=64)
        self._conv2 = ConvBlock(in_channels=64, out_channels=128)
        self._conv3 = ConvBlock(in_channels=128, out_channels=256)
        self._conv4 = ConvBlock(in_channels=256, out_channels=512)
        # paper: dropout probability not mentioned
        self._dropout = nn.Dropout(p=0.5)
        self._conv5 = ConvBlock(in_channels=512, out_channels=1024)
        self._up1 = UpBlock(in_channels=1024, out_channels=512, size=out4_shape - 8)
        self._up2 = UpBlock(in_channels=512, out_channels=256, size=out3_shape - 32)
        self._up3 = UpBlock(in_channels=256, out_channels=128, size=out2_shape - 80)
        self._up4 = UpBlock(in_channels=128, out_channels=64, size=out1_shape - 176)
        self._conv6 = nn.Conv2d(in_channels=64, out_channels=config.n_classes, kernel_size=1, stride=1, padding=0)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `channels`, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `n_classes`, width - 176, height - 176).
        """
        out1 = self._conv1(x)
        x = self._max_pool(out1)
        out2 = self._conv2(x)
        x = self._max_pool(out2)
        out3 = self._conv3(x)
        x = self._max_pool(out3)
        out4 = self._conv4(x)
        x = self._dropout(out4)
        x = self._max_pool(out4)
        x = self._conv5(x)
        x = self._up1(x=x, x_skip=out4)
        x = self._up2(x=x, x_skip=out3)
        x = self._up3(x=x, x_skip=out2)
        x = self._up4(x=x, x_skip=out1)
        x = self._conv6(x)
        return x


@dataclass
class UNetConfig(ModelConfig_):
    """
    Configuration class for the UNet model.

    Attributes:
        description (str): Description of the model. Static: "".
        input_shape (tuple[int, int, int]): Shape of the input image as (channels, width, height).
        n_classes (int): Number of output classes.
        class_ (Type[UNet]): Class of the model.
    """

    class_: ClassVar[Type[UNet]] = UNet

    input_shape: tuple[int, int, int]
    n_classes: int

    @validator.constraints("n_classes", "x > 1")
    def __init__(self, input_shape: tuple[int, int, int] = (1, 572, 572), n_classes: int = 2):
        """
        Initialize the UNetConfig class.

        Args:
            input_shape (tuple[int, int, int]): Shape of the input image as (channels, width, height).
                channels must be 1. width and height must be equal, even and greater than 300. Default: (1, 572, 572)
            n_classes (int): Number of output classes. Must be greater than 1. Default: 2.
        """
        if not input_shape[0] == 1:
            raise ValueError("The number of channels in the input shape must be 1.")

        if not input_shape[1] == input_shape[2]:
            raise ValueError("The width and height of the input shape must be equal.")

        if not all(x % 2 == 0 for x in input_shape[1:]):
            raise ValueError("The width and height of the input shape must be even.")

        if input_shape[1] < 300:
            raise ValueError("The width and height of the input shape must be greater than 300.")

        super().__init__(description="", input_shape=input_shape)
        self.n_classes = n_classes
