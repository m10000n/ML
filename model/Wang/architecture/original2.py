from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type, cast

import torch
import torch.nn as nn

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator
from model import model_helper
from model.Wang.architecture.residual import ResidualStack


class Original2(Model_):
    """
    Deep learning model for the classification of fMRI data.

    This model consists of two convolutional layers, followed by 8 residual blocks and three fully connected layers,
    designed to decode and map fMRI data into classes. The first convolutional layer uses a kernel size of 1 to
    enhance nonlinearity without affecting the receptive field. The second convolutional layer and subsequent residual
    blocks extract high-level features. The residual blocks are inspired by the ResNet architecture, as described in the
    paper "Deep Residual Learning for Image Recognition" by He et al. (2015). The final fully connected layers project
    the extracted features into class scores corresponding to the task states. This implementation follows the
    methodology introduced in the paper "Decoding and Mapping Task States of the Human Brain via Deep Learning" by Wang
    et al. (2020) and is based on the authors' reference implementation available at
    https://github.com/ustc-bmec/Whole-Brain-Conv. The differences between the network described in the paper and the
    authors' implementation are the use of 8 residual blocks instead of 4 and the addition of a dropout layer
    before the final fully connected layer.

    Attributes:
        config (Original2Config): Configuration of the model.
    """

    config: Original2Config

    def __init__(self, config: Original2Config) -> None:
        """
        Initializes the Original2 class.

        Args:
            config (Original2Config): Configuration of the model.
        """
        super().__init__(config)

        self._relu = nn.ReLU()

        self._conv1 = nn.Conv3d(in_channels=27, out_channels=3, kernel_size=1, stride=1, padding=0)
        self._batch_norm1 = nn.BatchNorm3d(3)
        self._conv2 = nn.Conv3d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1)
        self._batch_norm2 = nn.BatchNorm3d(24)

        self._res_stack = ResidualStack(n_res_reps=1)

        shape = model_helper.get_output_shape(
            input_shape=self.config.input_shape,
            module=[self._conv1, self._conv2, self._res_stack],
        )

        self._conv3 = nn.Conv3d(
            in_channels=128,
            out_channels=64,
            kernel_size=cast(tuple[int, int, int], tuple(shape[1:])),
            stride=1,
            padding=0,
        )
        self._flatten = nn.Flatten(start_dim=1)
        self._fc1 = nn.Linear(64, 64)
        self._dropout = nn.Dropout(p=0.5) if self.config.drop_out else None
        self._fc2 = nn.Linear(64, self.config.n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Original2 (Wang) class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *`input_shape`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `n_classes`).
        """
        x = self._conv1(x)
        x = self._batch_norm1(x)
        x = self._relu(x)
        x = self._conv2(x)
        x = self._batch_norm2(x)
        x = self._relu(x)
        x = self._res_stack(x)
        x = self._conv3(x)
        x = self._relu(x)
        x = self._flatten(x)
        x = self._fc1(x)
        x = self._relu(x)
        x = self._dropout(x) if self._dropout is not None else x
        x = self._fc2(x)
        return x


@dataclass(init=False)
class Original2Config(ModelConfig_):
    """
    Configuration class for the Original2 (Wang) class.

    Attributes:
        class_: (Type[Original2]): Class of the model.
        input_shape: (tuple[int, int, int, int]): Shape of the input fMRI data as (volumes, x, y, z).
        n_classes: (int): Number of output classes.
    """

    class_: ClassVar[Type[Original2]] = Original2

    input_shape: tuple[int, int, int, int]
    drop_out: bool
    n_classes: int

    @validator.constraints("n_classes", "x > 1")
    def __init__(
        self, description: str, input_shape: tuple[int, int, int, int] | list[int], drop_out: bool, n_classes: int = 7
    ):
        """
        Initializes the Original2Config class.

        Args:
            description (str): Description of the model.
            input_shape (tuple[int, int, int, int] | list[int]): Shape of the input fMRI data as (volumes, x, y, z).
            drop_out (bool): Whether to use dropout.
            n_classes (int): Number of output classes. Default: 7.
        """
        super().__init__(description=description, input_shape=tuple(input_shape))

        self.drop_out = drop_out
        self.n_classes = n_classes
