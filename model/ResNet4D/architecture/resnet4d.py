from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, ClassVar, Type

import torch
import torch.nn as nn

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator
from model.model_helper import Interpolate3d
from model.ResNet4D.architecture.residual4d import (
    ResidualBlock4D_,
    ResidualBlock4DConfig_,
)


class ResNet4D(Model_):
    """
    Deep learning model for the classification of fMRI data.

    ResNet4D is a deep convolutional neural network inspired by the ResNet architecture introduced in the paper
    "Deep Residual Learning for Image Recognition" by He et al. (2015), adapted for the classification of
    four-dimensional fMRI data. The model begins by resizing each input volume via trilinear interpolation to a
    fixed spatial shape, followed by an initial 3D convolutional layer for early feature extraction.
    It then applies a series of stacked 3D residual blocks to capture hierarchical spatiotemporal features while
    preserving gradient flow, enabling the training of deep networks on volumetric data.
    The architecture supports flexibility in depth through configurable numbers of residual blocks at each
    stage and allows the use of either basic or bottleneck residual block types for efficiency or capacity depending on
    the experimental setting. Global average pooling is applied before the final fully connected classification layer to
    reduce parameters and overfitting.

    Attributes:
        config (ResNet4DConfig): Configuration of the model.
        spatial_shape (tuple[int, int, int]): Fixed spatial shape to which each input volume is resized via trilinear
            interpolation. Static: (112, 112, 112).
    """

    config: ResNet4DConfig
    spatial_shape = (112, 112, 112)

    def __init__(self, config: ResNet4DConfig):
        """
        Initializes the ResNet4D class.

        Args:
            config (ResNet4DConfig): Configuration of the model.
        """
        super().__init__(config)

        self._relu = nn.ReLU()

        self._interpolate = Interpolate3d(size=self.spatial_shape)
        self._conv1 = nn.Conv3d(
            in_channels=self.config.input_shape[0], out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self._batch_norm = nn.BatchNorm3d(64)

        res_block_config_conv2 = copy.deepcopy(self.config.res_block)
        res_block_config_conv2.in_channels = 64
        res_block_config_conv2.out_channels = 64
        self._conv2 = nn.Sequential(
            *[ResidualBlock4D_.create(config=res_block_config_conv2) for _ in range(self.config.n_conv2)],
        )

        res_block_config_conv3 = copy.deepcopy(self.config.res_block)
        res_block_config_conv3.in_channels = 64
        res_block_config_conv3.out_channels = 128
        res_block_configs_conv3 = [copy.deepcopy(self.config.res_block) for _ in range(self.config.n_conv3 - 1)]
        for res_block_config in res_block_configs_conv3:
            res_block_config.in_channels = 128
            res_block_config.out_channels = 128
        self._conv3 = nn.Sequential(
            *[
                ResidualBlock4D_.create(config=res_block_config_conv3),
                *[ResidualBlock4D_.create(config=config) for config in res_block_configs_conv3],
            ]
        )

        res_block_config_conv4 = copy.deepcopy(self.config.res_block)
        res_block_config_conv4.in_channels = 128
        res_block_config_conv4.out_channels = 256
        res_block_configs_conv4 = [copy.deepcopy(self.config.res_block) for _ in range(self.config.n_conv4 - 1)]
        for res_block_config in res_block_configs_conv4:
            res_block_config.in_channels = 256
            res_block_config.out_channels = 256
        self._conv4 = nn.Sequential(
            *[
                ResidualBlock4D_.create(config=res_block_config_conv4),
                *[ResidualBlock4D_.create(config=config) for config in res_block_configs_conv4],
            ]
        )

        res_block_config_conv5 = copy.deepcopy(self.config.res_block)
        res_block_config_conv5.in_channels = 256
        res_block_config_conv5.out_channels = 512
        res_block_configs_conv5 = [copy.deepcopy(self.config.res_block) for _ in range(self.config.n_conv5 - 1)]
        for res_block_config in res_block_configs_conv5:
            res_block_config.in_channels = 512
            res_block_config.out_channels = 512
        self._conv5 = nn.Sequential(
            *[
                ResidualBlock4D_.create(config=res_block_config_conv5),
                *[ResidualBlock4D_.create(config=config) for config in res_block_configs_conv5],
            ]
        )

        self._avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self._flatten = nn.Flatten(start_dim=1)
        self._linear = nn.Linear(in_features=512, out_features=self.config.n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet4D class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *`input_shape`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `n_classes`).
        """
        x = self._interpolate(x)
        x = self._conv1(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)
        x = self._avg_pool(x)
        x = self._flatten(x)
        x = self._linear(x)
        return x


@dataclass(init=False)
class ResNet4DConfig(ModelConfig_):
    """
    Configuration class for the ResNet4D model.

    Attributes:
        description (str): Description of the model.
        input_shape (tuple[int, int, int, int]): Shape of the input fMRI data as (volumes, x, y, z).
        res_type (Literal["B", "bottleneck"]): Type of residual block to use.
        n_conv2 (int): Number of residual blocks in the second stage of the network.
        n_conv3 (int): Number of residual blocks in the third stage of the network.
        n_conv4 (int): Number of residual blocks in the fourth stage of the network.
        n_conv5 (int): Number of residual blocks in the fifth stage of the network.
        n_classes (int): Number of output classes.
    """

    class_: ClassVar[Type[ResNet4D]] = ResNet4D

    input_shape: tuple[int, int, int, int]
    n_classes: int
    res_block: ResidualBlock4DConfig_
    n_conv2: int
    n_conv3: int
    n_conv4: int
    n_conv5: int

    @validator.constraints("n_classes", "x > 1")
    @validator.constraints("n_conv2", "x > 0")
    @validator.constraints("n_conv3", "x > 0")
    @validator.constraints("n_conv4", "x > 0")
    @validator.constraints("n_conv5", "x > 0")
    def __init__(
        self,
        description: str,
        input_shape: tuple[int, int, int, int] | list[int],
        n_classes: int,
        res_block: ResidualBlock4DConfig_,
        n_conv2: int,
        n_conv3: int,
        n_conv4: int,
        n_conv5: int,
    ):
        """
        Initializes the ResNet4DConfig class.

        Args:
            description (str): Description of the model.
            input_shape (tuple[int, int, int, int] | list[int]): Shape of the input fMRI data as (volumes, x, y, z).
            res_block (ResidualBlock4DConfig_): Configuration of the 4D residual block.
            n_conv2 (int): Number of residual blocks in the second stage of the network.
            n_conv3 (int): Number of residual blocks in the third stage of the network.
            n_conv4 (int): Number of residual blocks in the fourth stage of the network.
            n_conv5 (int): Number of residual blocks in the fifth stage of the network.
            n_classes (int): Number of output classes. Default: 1000.
        """
        super().__init__(description=description, input_shape=input_shape)
        self.res_block = res_block
        self.n_classes = n_classes
        self.n_conv2 = n_conv2
        self.n_conv3 = n_conv3
        self.n_conv4 = n_conv4
        self.n_conv5 = n_conv5

    @classmethod
    def _from_dict(cls: Type[ResNet4DConfig], dict_: dict[str, Any]) -> ResNet4DConfig:
        return cls(
            description=dict_["description"],
            input_shape=dict_["input_shape"],
            n_classes=dict_["n_classes"],
            res_block=ResidualBlock4DConfig_.from_dict(dict_["res_block"]),
            n_conv2=dict_["n_conv2"],
            n_conv3=dict_["n_conv3"],
            n_conv4=dict_["n_conv4"],
            n_conv5=dict_["n_conv5"],
        )

    def _as_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "input_shape": self.input_shape,
            "n_classes": self.n_classes,
            "res_block": self.res_block.as_dict(),
            "n_conv2": self.n_conv2,
            "n_conv3": self.n_conv3,
            "n_conv4": self.n_conv4,
            "n_conv5": self.n_conv5,
        }
