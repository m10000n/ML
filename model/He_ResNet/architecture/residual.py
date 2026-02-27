import torch
import torch.nn as nn

from model.model_helper import Add, Pad, Subsample


class ResidualBlock_(nn.Module):
    """
    Abstract base class for residual blocks. Subclasses must implement the forward pass.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


class ResidualBlockA(ResidualBlock_):
    """
    Implements a residual block using Option A from the ResNet paper.

    This block applies two 3x3 convolutional layers in the residual branch. The first is followed by Batch Normalization
    and a ReLU activation, while the second is followed only by Batch Normalization. When the number of input and output
    channels differ, spatial downsampling and zero-padding are applied to the identity branch to match dimensions,
    following Option A from “Deep Residual Learning for Image Recognition” (He et al., 2016). The output is computed by
    adding the residual branch and the identity branch, followed by a ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the ResidualBlockA class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        down = in_channels < out_channels

        self._residual = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2 if down else 1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        self._identity: nn.Sequential | nn.Identity
        if down:
            self._identity = nn.Sequential(
                Subsample(strides=[None, None, 2, 2]), Pad(padding=[None, (0, out_channels - in_channels), None, None])
            )
        else:
            self._identity = nn.Identity()

        self._add = Add()
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlockA class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_channels`, height_out, width_out).
        """
        residual = self._residual(x)
        identity = self._identity(x)
        x = self._add(residual, identity)
        x = self._relu(x)
        return x


class ResidualBlockB(ResidualBlock_):
    """
    Implements a residual block using Option B from the ResNet paper.

    This block applies two 3x3 convolutional layers in the residual branch. The first is followed by Batch Normalization
    and a ReLU activation, while the second is followed only by Batch Normalization. When the number of input and output
    channels differ, the identity branch uses a 1x1 convolution with stride 2 followed by Batch Normalization to match
    spatial and channel dimensions, following Option B from “Deep Residual Learning for Image Recognition”
    (He et al., 2016). The output is computed by adding the residual branch and the identity branch, followed by a ReLU
    activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the ResidualBlockB class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        down = in_channels < out_channels

        self._residual = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2 if down else 1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        self._identity: nn.Sequential | nn.Identity
        if down:
            self._identity = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self._identity = nn.Identity()

        self._add = Add()
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlockB class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_channels`, height_out, width_out).
        """
        residual = self._residual(x)
        identity = self._identity(x)
        x = self._add(residual, identity)
        x = self._relu(x)
        return x


class ResidualBlockC(ResidualBlock_):
    """
    Implements a residual block using Option C from the ResNet paper.

    This block applies two 3x3 convolutional layers in the residual branch. The first is followed by Batch Normalization
    and a ReLU activation, while the second is followed only by Batch Normalization. The identity branch always uses a
    1x1 convolution followed by Batch Normalization, regardless of whether the number of input and output channels
    differ, following Option C from “Deep Residual Learning for Image Recognition” (He et al., 2016). The output is
    computed by adding the residual branch and the identity branch, followed by a ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the ResidualBlockC class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        stride = 2 if in_channels < out_channels else 1

        self._residual = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        self._identity = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        self._add = Add()
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlockC class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `in_channels`, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `out_channels`, height_out, width_out).
        """
        residual = self._residual(x)
        identity = self._identity(x)
        x = self._add(residual, identity)
        x = self._relu(x)
        return x
