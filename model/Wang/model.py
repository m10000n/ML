import torch
import torch.nn as nn
from torchinfo import summary

from model.Wang.residual import Residual


# Trainable params: 2,597,473
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

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
