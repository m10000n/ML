import torch
import torch.nn as nn
from torchinfo import summary


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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(
            in_channels=27, out_channels=3, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm1 = nn.BatchNorm3d(3)
        self.conv2 = nn.Conv3d(
            in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1
        )
        self.batch_norm2 = nn.BatchNorm3d(24)
        self.res1 = Residual(in_channels=24, out_channels=32, stride=1)
        self.res2 = Residual(in_channels=32, out_channels=64, stride=2)
        self.res3 = Residual(in_channels=64, out_channels=64, stride=2)
        self.res4 = Residual(in_channels=64, out_channels=128, stride=2)
        self.conv3 = nn.Conv3d(
            in_channels=128, out_channels=64, kernel_size=(5, 6, 6), stride=1, padding=0
        )
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 7)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.fc1(torch.flatten(x, 1))
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    summary(
        Model(),
        input_size=(1, 27, 75, 93, 81),
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "trainable",
        ),
    )
