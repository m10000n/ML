import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Two-layer convolutional block with ReLU activations for feature extraction.

    This block applies two consecutive unpadded 3x3 convolutional layers with stride 1, each followed by a ReLU
    activation, to extract spatial features while reducing the spatial dimensions of the input tensor. It is used
    throughout the contracting and expansive paths of U-Net to refine and increase the expressiveness of feature
    representations.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()
        self._relu = nn.ReLU()

        self._conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self._conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_channels`, width - 4, height - 4).
        """
        x = self._conv1(x)
        x = self._relu(x)
        x = self._conv2(x)
        x = self._relu(x)
        return x
