import torch
import torch.nn as nn
from torchinfo import summary

from model.Wang.residual import Residual


# Trainable params: 3,981,601
class ModelLarge(nn.Module):
    def __init__(self, rep=1):
        super(ModelLarge, self).__init__()
        self.rep = rep

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(
            in_channels=27, out_channels=3, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm1 = nn.BatchNorm3d(3)
        self.conv2 = nn.Conv3d(
            in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1
        )
        self.batch_norm2 = nn.BatchNorm3d(24)

        self.res1_1 = Residual(in_channels=24, out_channels=32, stride=1)
        for i in range(2, rep + 2):
            setattr(
                self, f"res1_{i}", Residual(in_channels=32, out_channels=32, stride=1)
            )
        self.res2_1 = Residual(in_channels=32, out_channels=64, stride=2)
        for i in range(2, rep + 2):
            setattr(
                self, f"res2_{i}", Residual(in_channels=64, out_channels=64, stride=1)
            )
        self.res3_1 = Residual(in_channels=64, out_channels=64, stride=2)
        for i in range(2, rep + 2):
            setattr(
                self, f"res3_{i}", Residual(in_channels=64, out_channels=64, stride=1)
            )
        self.res4_1 = Residual(in_channels=64, out_channels=128, stride=2)
        for i in range(2, rep + 2):
            setattr(
                self, f"res4_{i}", Residual(in_channels=128, out_channels=128, stride=1)
            )
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
        x = self.res1_1(x)
        for i in range(2, self.rep + 2):
            x = getattr(self, f"res1_{i}")(x)
        x = self.res2_1(x)
        for i in range(2, self.rep + 2):
            x = getattr(self, f"res2_{i}")(x)
        x = self.res3_1(x)
        for i in range(2, self.rep + 2):
            x = getattr(self, f"res3_{i}")(x)
        x = self.res4_1(x)
        for i in range(2, self.rep + 2):
            x = getattr(self, f"res4_{i}")(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.fc1(torch.flatten(x, 1))
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Trainable params: 5,365,729
class ModelLarge2(ModelLarge):
    def __init__(self):
        super().__init__(rep=2)


# Trainable params: 8,133,985
class ModelLarge4(ModelLarge):
    def __init__(self):
        super().__init__(rep=4)


# Trainable params: 13,670,497
class ModelLarge8(ModelLarge):
    def __init__(self):
        super().__init__(rep=8)


if __name__ == "__main__":
    summary(
        ModelLarge(rep=1),
        input_size=(1, 27, 75, 93, 81),
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "trainable",
        ),
    )
