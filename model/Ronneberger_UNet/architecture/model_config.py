from model.Ronneberger_UNet.architecture.unet import UNetConfig


def unet() -> UNetConfig:
    return UNetConfig(input_shape=(1, 572, 572))
