import torch.nn as nn

from model.Wang.shortcut import Shortcut


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = Shortcut(
            in_channels=in_channels, out_channels=out_channels, stride=stride
        )

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.batch_norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = self.shortcut(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x += res
        x = self.relu(x)
        return x
