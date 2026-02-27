from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Type

import torch
import torch.nn as nn

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator
from model.He_ResNet.architecture.bottleneck import Bottleneck
from model.He_ResNet.architecture.residual import (
    ResidualBlock_,
    ResidualBlockA,
    ResidualBlockB,
    ResidualBlockC,
)


class ResNet(Model_):
    """
    Deep learning model for the classification of images.

    ResNet (Residual Network) is a deep convolutional neural network designed for large-scale image classification
    tasks, introducing residual connections that allow the network to learn residual functions with reference to the
    layer inputs, effectively enabling the training of very deep networks by mitigating the vanishing gradient problem.
    The architecture consists of stacked residual blocks, each containing batch normalization, ReLU activations,
    and convolutional layers, with residual connections that add the input of the block to its output to facilitate
    efficient gradient flow during training. Global average pooling is applied before the final fully connected
    classification layer to reduce parameters and overfitting. This implementation follows the methodology introduced
    in the paper “Deep Residual Learning for Image Recognition” (He et al., 2016).

    Attributes:
        config (ResNetConfig): Configuration of the model.
    """

    config: ResNetConfig

    def __init__(self, config: ResNetConfig):
        """
        Initialize the ResNet model.

        Args:
            config (ResNetConfig): Configuration of the model.
        """
        super().__init__(config)

        ResidualBlock: Type[ResidualBlock_]
        if self.config.res_type == "A":
            ResidualBlock = ResidualBlockA
        elif self.config.res_type == "B":
            ResidualBlock = ResidualBlockB
        elif self.config.res_type == "C":
            ResidualBlock = ResidualBlockC
        else:
            ResidualBlock = Bottleneck

        self._relu = nn.ReLU()

        self._conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self._batch_norm = nn.BatchNorm2d(64)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._conv2 = nn.Sequential(
            *[ResidualBlock(in_channels=64, out_channels=64) for _ in range(self.config.n_conv2)]
        )

        self._conv3 = nn.Sequential(
            *[
                ResidualBlock(in_channels=64, out_channels=128),
                *[ResidualBlock(in_channels=128, out_channels=128) for _ in range(self.config.n_conv3 - 1)],
            ]
        )

        self._conv4 = nn.Sequential(
            *[
                ResidualBlock(in_channels=128, out_channels=256),
                *[ResidualBlock(in_channels=256, out_channels=256) for _ in range(self.config.n_conv4 - 1)],
            ]
        )

        self._conv5 = nn.Sequential(
            *[
                ResidualBlock(in_channels=256, out_channels=512),
                *[ResidualBlock(in_channels=512, out_channels=512) for _ in range(self.config.n_conv5 - 1)],
            ]
        )

        self._avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self._flatten = nn.Flatten(start_dim=1)
        self._linear = nn.Linear(in_features=512, out_features=self.config.n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *`input_shape`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `n_classes`).
        """
        x = self._conv1(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        x = self._max_pool(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)
        x = self._avg_pool(x)
        x = self._flatten(x)
        x = self._linear(x)
        return x


@dataclass(init=False)
class ResNetConfig(ModelConfig_):
    """
    Configuration class for the ResNet model.

    Attributes:
        description (str): Description of the model.
        input_shape (tuple[int, int, int]): Shape of the input image as (channels, width, height). Static: (3, 224, 224).
        res_type (Literal["A", "B", "C", "bottleneck"]): Type of residual block to use.
        n_conv2 (int): Number of residual blocks in the first stage of the network.
        n_conv3 (int): Number of residual blocks in the second stage of the network.
        n_conv4 (int): Number of residual blocks in the third stage of the network.
        n_conv5 (int): Number of residual blocks in the fourth stage of the network.
        n_classes (int): Number of output classes.
        class_ (ClassVar[Type[ResNet]]): Class of the model.
    """

    class_: ClassVar[Type[ResNet]] = ResNet

    input_shape: tuple[int, int, int]
    res_type: Literal["A", "B", "C", "bottleneck"]
    n_conv2: int
    n_conv3: int
    n_conv4: int
    n_conv5: int
    n_classes: int

    @validator.constraints("n_classes", "x > 0")
    @validator.constraints("n_conv2", " x > 0")
    @validator.constraints("n_conv3", " x > 0")
    @validator.constraints("n_conv4", " x > 0")
    @validator.constraints("n_conv5", " x > 0")
    @validator.constraints("n_classes", " x > 1")
    def __init__(
        self,
        description: str,
        res_type: Literal["A", "B", "C", "bottleneck"],
        n_conv2: int,
        n_conv3: int,
        n_conv4: int,
        n_conv5: int,
        n_classes: int = 1000,
    ) -> None:
        """
        Initialize the ResNetConfig class.

        Args:
            description (str): Description of the model.
            res_type (Literal["A", "B", "C", "bottleneck"]): Type of residual block to use.
            n_conv2 (int): Number of residual blocks in the first stage of the network. Must be a positive integer.
            n_conv3 (int): Number of residual blocks in the second stage of the network. Must be a positive integer.
            n_conv4 (int): Number of residual blocks in the third stage of the network. Must be a positive integer.
            n_conv5 (int): Number of residual blocks in the fourth stage of the network. Must be a positive integer.
            n_classes (int): Number of output classes. Must be greater than 1. Default: 1000
        """
        super().__init__(description=description, input_shape=(3, 224, 224))
        self.res_type = res_type
        self.n_conv2 = n_conv2
        self.n_conv3 = n_conv3
        self.n_conv4 = n_conv4
        self.n_conv5 = n_conv5
        self.n_classes = n_classes
