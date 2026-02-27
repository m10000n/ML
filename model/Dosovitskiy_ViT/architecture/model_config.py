from model.Dosovitskiy_ViT.architecture.vision_transformer import (
    VisionTransformerConfig,
)


def vit_original() -> VisionTransformerConfig:
    description = "original ViT"
    return VisionTransformerConfig(
        description=description, input_shape=(3, 224, 224), n_classes=100, patch_size=(16, 16)
    )
