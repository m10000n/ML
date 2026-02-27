from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type

import torch
import torch.nn as nn

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator
from model.Inceptron.architecture.inception4d import Inception4d
from model.model_helper import Interpolate3d


class Inceptron(Model_):
    """
    Deep learning model for the classification of fMRI data.

    Inceptron is a deep convolutional neural network inspired by GoogLeNet, introduced in the paper "Going deeper
    with convolutions" (Szegedy et al., 2015). In this implementation, each input is resized via trilinear interpolation
    to a fixed spatial shape. Overall, the architecture is similar to the original design but adapted for
    four-dimensional input. The following modifications were made:
        - The first max pooling layer is dropped to compensate for the decrease in spatial dimensions.
        - Local response normalization is replaced by batch normalization.
        - Normalization is applied after each convolution.
        - The activation function is applied after normalization.

    Attributes:
        config (InceptronConfig): Configuration of the model.
    """

    config: InceptronConfig
    spatial_shape = (112, 112, 112)

    def __init__(self, config: InceptronConfig):
        """
        Initializes the Inceptron class.

        Args:
            config (InceptronConfig): Configuration of the model.
        """
        super().__init__(config=config)

        self._relu = nn.ReLU()
        self._max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self._interpolate = Interpolate3d(size=self.spatial_shape)
        self._conv1 = nn.Conv3d(
            in_channels=self.config.input_shape[0], out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self._batch_norm1 = nn.BatchNorm3d(num_features=64)
        self._conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self._batch_norm2 = nn.BatchNorm3d(num_features=64)
        self._conv3 = nn.Conv3d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self._batch_norm3 = nn.BatchNorm3d(num_features=192)
        self._inception1a = Inception4d(
            in_channels=192, out_1x1=64, out_3x3_reduce=96, out_3x3=128, out_5x5_reduce=16, out_5x5=32, out_pool=32
        )
        self._inception1b = Inception4d(
            in_channels=256, out_1x1=128, out_3x3_reduce=128, out_3x3=192, out_5x5_reduce=32, out_5x5=96, out_pool=64
        )
        self._inception2a = Inception4d(
            in_channels=480, out_1x1=192, out_3x3_reduce=96, out_3x3=208, out_5x5_reduce=16, out_5x5=48, out_pool=64
        )
        self._inception2b = Inception4d(
            in_channels=512, out_1x1=160, out_3x3_reduce=112, out_3x3=224, out_5x5_reduce=24, out_5x5=64, out_pool=64
        )
        self._inception2c = Inception4d(
            in_channels=512, out_1x1=128, out_3x3_reduce=128, out_3x3=256, out_5x5_reduce=24, out_5x5=64, out_pool=64
        )
        self._inception2d = Inception4d(
            in_channels=512, out_1x1=112, out_3x3_reduce=144, out_3x3=288, out_5x5_reduce=32, out_5x5=64, out_pool=64
        )
        self._inception2e = Inception4d(
            in_channels=528, out_1x1=256, out_3x3_reduce=160, out_3x3=320, out_5x5_reduce=32, out_5x5=128, out_pool=128
        )
        self._inception3a = Inception4d(
            in_channels=832, out_1x1=256, out_3x3_reduce=160, out_3x3=320, out_5x5_reduce=32, out_5x5=128, out_pool=128
        )
        self._inception3b = Inception4d(
            in_channels=832, out_1x1=384, out_3x3_reduce=192, out_3x3=384, out_5x5_reduce=48, out_5x5=128, out_pool=128
        )
        self._avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self._dropout = nn.Dropout(p=config.p_drop)
        self._flatten = nn.Flatten(start_dim=1)
        self._linear = nn.Linear(in_features=1024, out_features=self.config.n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Inceptron model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, volumes, x, y, z).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `n_classes`).
        """
        x = self._interpolate(x)
        x = self._conv1(x)
        x = self._batch_norm1(x)
        x = self._relu(x)
        x = self._conv2(x)
        x = self._batch_norm2(x)
        x = self._relu(x)
        x = self._conv3(x)
        x = self._batch_norm3(x)
        x = self._relu(x)
        x = self._max_pool(x)
        x = self._inception1a(x)
        x = self._inception1b(x)
        x = self._max_pool(x)
        x = self._inception2a(x)
        x = self._inception2b(x)
        x = self._inception2c(x)
        x = self._inception2d(x)
        x = self._inception2e(x)
        x = self._max_pool(x)
        x = self._inception3a(x)
        x = self._inception3b(x)
        x = self._avg_pool(x)
        x = self._dropout(x)
        x = self._flatten(x)
        x = self._linear(x)
        return x


@dataclass
class InceptronConfig(ModelConfig_):
    """
    Configuration class for the Inceptron model.

    Attributes:
        description (str): Description of the model.
        input_shape (tuple[int, int, int, int]): Shape of the input fMRI data as (volumes, x, y, z).
        n_classes (int): Number of output classes.
        p_drop (float): Dropout probability.
    """

    class_: ClassVar[Type[Inceptron]] = Inceptron

    input_shape: tuple[int, int, int, int]
    n_classes: int
    p_drop: float

    @validator.constraints("n_classes", "x > 1")
    @validator.constraints("p_drop", "x >= 0 and x <= 1")
    def __init__(
        self,
        description: str,
        input_shape: tuple[int, int, int, int] | list[int],
        n_classes: int,
        p_drop: float = 0.4,
    ):
        """
        Initializes the InceptronConfig class.

        Args:
            description (str): Description of the model.
            input_shape (tuple[int, int, int, int] | list[int]):
                Shape of the input fMRI data as (volumes, x, y, z). Must be a tuple or list of 4 positive integers.
            n_classes (int): Number of output classes. Must be greater than 1.
            p_drop (float): Must be a float in range [0, 1]. Default: 0.4
        """
        super().__init__(description=description, input_shape=input_shape)
        self.n_classes = n_classes
        self.p_drop = p_drop
