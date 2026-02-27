from model.ResNet4D.architecture.bottleneck4d import Bottleneck4DConfig
from model.ResNet4D.architecture.residual4d import ResidualBlock4DBConfig
from model.ResNet4D.architecture.resnet4d import ResNet4DConfig

_WANG_N_VOLUMES = 27
_N_CLASSES = 7

# ResNet4D supports arbitrary 4D inputs by automatically resampling the volume to 112×112×112.


# residual block B
def resnet4d_18_b() -> ResNet4DConfig:  # 33,734,343 params, 483.138 GFLOP
    return ResNet4DConfig(
        description="ResNet4D-18 B",
        input_shape=(_WANG_N_VOLUMES, 112, 112, 112),
        n_classes=_N_CLASSES,
        res_block=ResidualBlock4DBConfig(),
        n_conv2=2,
        n_conv3=2,
        n_conv4=2,
        n_conv5=2,
    )


def resnet4d_34_b() -> ResNet4DConfig:  # 64,043,975 params, 725.909 GFLOP
    return ResNet4DConfig(
        description="ResNet4D-34 B",
        input_shape=(_WANG_N_VOLUMES, 112, 112, 112),
        n_classes=_N_CLASSES,
        res_block=ResidualBlock4DBConfig(),
        n_conv2=3,
        n_conv3=4,
        n_conv4=6,
        n_conv5=3,
    )


# bottleneck block
def resnet4d_26_bn() -> ResNet4DConfig:  # 2,256,199 params, 221.552 GFLOP
    return ResNet4DConfig(
        description="ResNet4D-18 bottleneck, late stride",
        input_shape=(_WANG_N_VOLUMES, 112, 112, 112),
        n_classes=_N_CLASSES,
        res_block=Bottleneck4DConfig(late_stride=True),
        n_conv2=2,
        n_conv3=2,
        n_conv4=2,
        n_conv5=2,
    )


def resnet4d_50_bn() -> ResNet4DConfig:  # 3,489,287 params, 231.386 GFLOP
    return ResNet4DConfig(
        description="ResNet4D-50 bottleneck, late stride",
        input_shape=(_WANG_N_VOLUMES, 112, 112, 112),
        n_classes=_N_CLASSES,
        res_block=Bottleneck4DConfig(late_stride=True),
        n_conv2=3,
        n_conv3=4,
        n_conv4=6,
        n_conv5=3,
    )


def resnet4d_101_bn() -> ResNet4DConfig:  # 6,448,872 params, 244.761 GFLOP
    return ResNet4DConfig(
        description="ResNet4D-101 bottleneck, late stride",
        input_shape=(_WANG_N_VOLUMES, 112, 112, 112),
        n_classes=_N_CLASSES,
        res_block=Bottleneck4DConfig(late_stride=True),
        n_conv2=3,
        n_conv3=4,
        n_conv4=23,
        n_conv5=3,
    )


def resnet4d_152_bn() -> ResNet4DConfig:  # 8,467,432 params, 261.283 GFLOP
    return ResNet4DConfig(
        description="ResNet4D-152 bottleneck, late stride",
        input_shape=(_WANG_N_VOLUMES, 112, 112, 112),
        n_classes=_N_CLASSES,
        res_block=Bottleneck4DConfig(late_stride=True),
        n_conv2=3,
        n_conv3=8,
        n_conv4=36,
        n_conv5=3,
    )
