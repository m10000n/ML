import torch.nn as nn


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Shortcut, self).__init__()
        self.c = None
        if in_channels != out_channels or stride != 1:
            self.c = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )

    def forward(self, x):
        if self.c:
            x = self.c(x)
        return x
