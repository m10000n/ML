from model.He_ResNet.architecture.resnet import ResNetConfig


def resnet_34_a() -> ResNetConfig:
    return ResNetConfig(
        description="ResNet-34 A",
        res_type="A",
        n_conv2=3,
        n_conv3=4,
        n_conv4=6,
        n_conv5=3,
    )


def resnet_34_b() -> ResNetConfig:
    return ResNetConfig(
        description="ResNet-34 B",
        res_type="B",
        n_conv2=3,
        n_conv3=4,
        n_conv4=6,
        n_conv5=3,
    )


def resnet_34_c() -> ResNetConfig:
    return ResNetConfig(
        description="ResNet-34 C",
        res_type="C",
        n_conv2=3,
        n_conv3=4,
        n_conv4=6,
        n_conv5=3,
    )


def resnet_50() -> ResNetConfig:
    return ResNetConfig(
        description="ResNet-50",
        res_type="bottleneck",
        n_conv2=3,
        n_conv3=4,
        n_conv4=6,
        n_conv5=3,
    )


def resnet_101() -> ResNetConfig:
    return ResNetConfig(
        description="ResNet-50",
        res_type="bottleneck",
        n_conv2=3,
        n_conv3=4,
        n_conv4=23,
        n_conv5=3,
    )


def resnet_152() -> ResNetConfig:
    return ResNetConfig(
        description="ResNet-50",
        res_type="bottleneck",
        n_conv2=3,
        n_conv3=8,
        n_conv4=36,
        n_conv5=3,
    )
