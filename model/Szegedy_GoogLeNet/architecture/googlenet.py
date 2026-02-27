from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type

import torch
import torch.nn as nn

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator
from model.Szegedy_GoogLeNet.architecture.auxillary import Auxillary
from model.Szegedy_GoogLeNet.architecture.inception import Inception


class GoogLeNet(Model_):
    """
    Deep learning model for the classification of images.

    GoogLeNet is a deep convolutional neural network designed for large-scale image classification tasks,
    introducing the Inception module to capture multi-scale spatial features via parallel 1x1, 3x3, and 5x5
    convolutions with dimensionality reduction. It employs ReLU activations, Local Response Normalization to
    encourage lateral inhibition, auxiliary classifiers at intermediate depths to improve gradient flow, and
    global average pooling in place of large fully connected layers to reduce parameters and overfitting.
    Dropout is used for regularization. This implementation follows the methodology introduced in the paper “Going
    deeper with convolutions” (Szegedy et al., 2015).

    Attributes:
        config (GoogLeNetConfig): Configuration of the model.
    """

    config: GoogLeNetConfig

    def __init__(self, config: GoogLeNetConfig):
        """
        Initialize the GoogLeNet model.

        Args:
            config (GoogLeNetConfig): Configuration of the model.
        """
        super().__init__(config)

        self._relu = nn.ReLU()
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # paper: no parameters given for local response normalization
        self._response_norm = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)

        self._conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self._conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self._conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self._inception3a = Inception(
            in_channels=192, out_1x1=64, out_3x3_reduce=96, out_3x3=128, out_5x5_reduce=16, out_5x5=32, out_pool=32
        )
        self._inception3b = Inception(
            in_channels=256, out_1x1=128, out_3x3_reduce=128, out_3x3=192, out_5x5_reduce=32, out_5x5=96, out_pool=64
        )
        self._inception4a = Inception(
            in_channels=480, out_1x1=192, out_3x3_reduce=96, out_3x3=208, out_5x5_reduce=16, out_5x5=48, out_pool=64
        )
        self._aux4a = Auxillary(in_channels=512, n_classes=self.config.n_classes)
        self._inception4b = Inception(
            in_channels=512, out_1x1=160, out_3x3_reduce=112, out_3x3=224, out_5x5_reduce=24, out_5x5=64, out_pool=64
        )

        self._inception4c = Inception(
            in_channels=512, out_1x1=128, out_3x3_reduce=128, out_3x3=256, out_5x5_reduce=24, out_5x5=64, out_pool=64
        )
        self._inception4d = Inception(
            in_channels=512, out_1x1=112, out_3x3_reduce=144, out_3x3=288, out_5x5_reduce=32, out_5x5=64, out_pool=64
        )
        self._aux4d = Auxillary(in_channels=528, n_classes=self.config.n_classes)
        self._inception4e = Inception(
            in_channels=528, out_1x1=256, out_3x3_reduce=160, out_3x3=320, out_5x5_reduce=32, out_5x5=128, out_pool=128
        )

        self._inception5a = Inception(
            in_channels=832, out_1x1=256, out_3x3_reduce=160, out_3x3=320, out_5x5_reduce=32, out_5x5=128, out_pool=128
        )
        self._inception5b = Inception(
            in_channels=832, out_1x1=384, out_3x3_reduce=192, out_3x3=384, out_5x5_reduce=48, out_5x5=128, out_pool=128
        )

        self._avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self._dropout = nn.Dropout(p=0.4)
        self._flatten = nn.Flatten(start_dim=1)
        self._linear = nn.Linear(in_features=1024, out_features=self.config.n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | torch.Tensor:
        """
        Forward pass of the GoogLeNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *`input_shape`).

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                Output tensors of shape (batch_size, `n_classes`).
        """
        x = self._conv1(x)
        x = self._relu(x)
        x = self._max_pool(x)
        x = self._response_norm(x)
        x = self._conv2(x)
        x = self._relu(x)
        x = self._conv3(x)
        x = self._relu(x)
        x = self._response_norm(x)
        x = self._max_pool(x)
        x = self._inception3a(x)
        x = self._inception3b(x)
        x = self._max_pool(x)
        x = self._inception4a(x)
        aux4a_input = x
        x = self._inception4b(x)
        x = self._inception4c(x)
        x = self._inception4d(x)
        aux4d_input = x
        x = self._inception4e(x)
        x = self._max_pool(x)
        x = self._inception5a(x)
        x = self._inception5b(x)
        x = self._avg_pool(x)
        x = self._dropout(x)
        x = self._flatten(x)
        x = self._linear(x)

        if self.config.training:
            x_aux4a = self._aux4a(aux4a_input)
            x_aux4d = self._aux4d(aux4d_input)
            return x, x_aux4a, x_aux4d
        else:
            return x


@dataclass
class GoogLeNetConfig(ModelConfig_):
    """
    Configuration class for the GoogLeNet model.

    Attributes:
        description (str): Description of the model.
        input_shape (tuple[int, int, int]): Shape of the input image as (channels, width, height). Static: (3, 224, 224).
        n_classes (int): Number of output classes.
        training (bool): Wether the model is in training mode. When True, the auxiliary classifier heads are applied
                and `forward` returns `(main_logits, aux1_logits, aux2_logits)`, otherwise only `main_logits` is
                returned.
        class_ (Type[GoogLeNet]): Class of the model.
    """

    class_: ClassVar[Type[GoogLeNet]] = GoogLeNet

    input_shape: tuple[int, int, int] = (3, 224, 224)
    n_classes: int = 1000
    training: int = True

    @validator.constraints("n_classes", "x > 1")
    def __init__(
        self,
        description: str = "",
        n_classes: int = 1000,
        training: bool = True,
    ):
        """
        Initialize the GoogLeNetConfig class.

        Args:
            description (str): Description of the model. Default: "".
            n_classes (int): Number of output classes. Must be greater than 1. Default: 1000.
            training (bool): Wether the model is in training mode. When True, the auxiliary classifier heads are applied
                and `forward` returns `(main_logits, aux1_logits, aux2_logits)`, otherwise only `main_logits` is
                returned. Default: True
        """
        super().__init__(description=description, input_shape=self.input_shape)
        self.n_classes = n_classes
        self.training = training
