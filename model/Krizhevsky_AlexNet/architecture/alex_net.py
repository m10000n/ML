from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type

import torch
import torch.nn as nn

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator


class AlexNet(Model_):
    """
    Deep learning model for the classification of images.

    AlexNet is a deep convolutional neural network designed for large-scale image classification tasks, utilizing a
    series of convolutional and pooling layers followed by fully connected layers to extract hierarchical feature
    representations from input images. By leveraging ReLU activations, local response normalization, and dropout,
    AlexNet effectively captures spatial hierarchies in images while reducing overfitting, enabling robust and scalable
    image classification. This implementation follows the methodology introduced in the paper "ImageNet Classification
    with Deep Convolutional Neural Networks" by Krizhevsky et al. (2012). While this implementation does not utilize the
    dual-GPU setup described in the original paper, it preserves the same layer connections and architecture.

    Attributes:
        config (AlexNetConfig): Configuration of the model.
    """

    config: AlexNetConfig

    def __init__(self, config: AlexNetConfig):
        """
        Initialize the AlexNet model.

        Args:
            config (AlexNetConfig): Configuration of the model.
        """
        super().__init__(config)

        self._relu = nn.ReLU()
        self._response_norm = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self._dropout = nn.Dropout(p=0.5)

        self._conv1 = nn.Conv2d(
            in_channels=self.config.input_shape[0], out_channels=96, kernel_size=11, stride=4, padding=2
        )
        self._conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, groups=2)
        self._conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self._conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=2)
        self._conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, groups=2)
        self._flatten = nn.Flatten(start_dim=1)
        self._fc1 = nn.Linear(in_features=9216, out_features=4096)
        self._fc2 = nn.Linear(in_features=4096, out_features=4096)
        self._fc3 = nn.Linear(in_features=4096, out_features=self.config.n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0, std=0.01)

                if module.bias is not None:
                    if name in ["_conv1", "_conv3"]:
                        bias = 0
                    else:
                        bias = 1

                    nn.init.constant_(tensor=module.bias, val=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AlexNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *`input_shape`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `n_classes`).
        """
        x = self._conv1(x)
        x = self._relu(x)
        x = self._response_norm(x)
        x = self._max_pool(x)
        x = self._conv2(x)
        x = self._relu(x)
        x = self._response_norm(x)
        x = self._max_pool(x)
        x = self._conv3(x)
        x = self._relu(x)
        x = self._conv4(x)
        x = self._relu(x)
        x = self._conv5(x)
        x = self._relu(x)
        x = self._max_pool(x)
        x = self._flatten(x)
        x = self._fc1(x)
        x = self._relu(x)
        x = self._dropout(x)
        x = self._fc2(x)
        x = self._relu(x)
        x = self._dropout(x)
        x = self._fc3(x)
        return x


@dataclass
class AlexNetConfig(ModelConfig_):
    """
    Configuration class for the AlexNet class.

    Attributes:
        description (str): Description of the model.
        input_shape (tuple[int, int, int]): Shape of the input image as (channels, width, height). Static: (3, 224, 224).
        n_classes (int): Number of output classes.
        class_ (Type[AlexNet]): Class of the model.
    """

    class_: ClassVar[Type[AlexNet]] = AlexNet

    input_shape: tuple[int, int, int] = (3, 224, 224)
    n_classes: int = 1000

    @validator.constraints("n_classes", "x > 1")
    def __init__(
        self,
        description: str = "",
        n_classes: int = 1000,
    ):
        """
        Initialize the AlexNetConfig class.

        Args:
            description (str): Description of the model. Default: "".
            n_classes (int): Number of output classes. Must be greater than 1. Default: 1000.
        """
        super().__init__(description=description, input_shape=self.input_shape)
        self.n_classes = n_classes
