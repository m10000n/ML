import torch
import torch.nn as nn

from helper.validator import validator


class Auxillary(nn.Module):
    """
    Auxillary classifier for the GoogLeNet model.

    This module is used to classify the output of the Inception modules at intermediate depths.
    It consists of an average pooling layer, a convolutional layer, a flattening layer, and two linear layers.

    Attributes:
        in_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
    """

    @validator.constraints("in_channels", "x > 0")
    @validator.constraints("n_classes", "x > 0")
    def __init__(self, in_channels: int, n_classes: int):
        """
        Initialize the Auxillary class.

        Args:
            in_channels (int): Number of input channels. Must be a positive integer.
            n_classes (int): Number of output classes. Must be a positive integer.
        """
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes

        self._relu = nn.ReLU()

        self._avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
        self._flatten = nn.Flatten(start_dim=1)
        self._linear1 = nn.Linear(in_features=2048, out_features=1024)
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(p=0.7)
        self._linear2 = nn.Linear(in_features=1024, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Auxillary class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, 14, 14).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `n_classes`).
        """
        x = self._avg_pool(x)
        x = self._conv(x)
        x = self._relu(x)
        x = self._flatten(x)
        x = self._linear1(x)
        x = self._relu(x)
        x = self._dropout(x)
        x = self._linear2(x)
        return x
