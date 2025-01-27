from argparse import ArgumentParser

from torch.nn import (
    Dropout,
    LayerNorm,
    Linear,
    Module,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torchinfo import summary

from model.Dosovitskiy_ViT.embed import AddClassToken, PatchEmbed, PosEmbed


class ViT(Module):
    """
    Vision Transformer (ViT) model for image classification.

    This model divides an input image into fixed-size patches, encodes each patch into a learnable embedding, and processes the sequence of embeddings through a
    stack of Transformer encoders. By utilizing self-attention mechanisms, the Vision Transformer (ViT) effectively captures both local and global dependencies
    between patches, enabling accurate and robust image classification. This implementation follows the methodology introduced in the paper "An Image is Worth
    16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020).

    Attributes:
        n_classes (int): Number of output classes for classification.
        img_channels (int): Number of channels in the input image.
        patch_size (int): Size of each patch.
        embed_dim (int): Dimensionality of the patch embeddings.
        n_heads (int): Number of heads in the Transformer encoder.
        n_layers (int): Number of Transformer encoder layers.
        mlp_size (int): Number of neurons of the hidden layer of the feed-forward network within the Transformer encoder.
        dropout_p (float): Dropout probability.
    """

    def __init__(
        self,
        n_classes: int,
        img_size: int = 224,
        img_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        mlp_size: int = 3072,
        dropout_p: float = 0.1,
    ):
        """
        Initializes the ViT class.

        Args:
            n_classes (int): Number of output classes for classification.
            img_size (int, optional): The size of the image. Default: 224.
            img_channels (int, optional): Number of channels in the input image. Default: 3
            patch_size (int, optional): Size of each patch. Default: 16
            embed_dim (int, optional): Dimensionality of the patch embeddings. Default: 768
            n_heads (int, optional): Number of heads in the Transformer encoder. Default: 12
            n_layers (int, optional): Number of Transformer encoder layers. Default: 12
            mlp_size (int, optional): Number of neurons of the hidden layer of the feed-forward network within the Transformer encoder. Default: 3072
            dropout_p (float, optional): Dropout probability. Default: 0.1.

        Precondition:
            - 'img_size' must be divisible by 'patch_size'
        """
        super().__init__()
        self.n_classes = n_classes
        self.img_channels = img_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_size = mlp_size
        self.dropout_p = dropout_p
        self.n_patches = img_size * img_size // patch_size**2

        self._embed = PatchEmbed(
            img_channels=img_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            method="conv",
        )
        self._add_class_token = AddClassToken(embed_dim=embed_dim)
        self._pos_embed = PosEmbed(n_embed=self.n_patches + 1, embed_dim=embed_dim)
        self._embed_do = Dropout(p=dropout_p)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_size,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self._mlp_head = Sequential(
            LayerNorm(normalized_shape=embed_dim),
            Linear(in_features=embed_dim, out_features=n_classes),
        )

    def forward(self, x):
        """
        Forward pass of the ViT class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `img_channels`, `img_size`, `img_size`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `n_classes`).
        """
        x = self._embed(x)
        x = self._add_class_token(x)
        x = self._pos_embed(x)
        x = self._embed_do(x)
        x = self._encoder(x)
        x = x[:, 0, :]
        x = self._mlp_head(x)
        return x


# trainable parameters: 85,805,578
class ViT_B_16(ViT):
    def __init__(self, n_classes: int):
        super().__init__(n_classes=n_classes)


# trainable parameters: 87,462,154
class ViT_B_32(ViT):
    def __init__(self, n_classes: int):
        super().__init__(n_classes=n_classes, patch_size=32)


# trainable parameters: 303,310,858
class ViT_L_16(ViT):
    def __init__(self, n_classes: int):
        super().__init__(
            n_classes=n_classes, embed_dim=1024, n_heads=16, n_layers=24, mlp_size=4096
        )


# trainable parameters: 305,519,626
class ViT_L_32(ViT):
    def __init__(self, n_classes: int):
        super().__init__(
            n_classes=n_classes,
            patch_size=32,
            embed_dim=1024,
            n_heads=16,
            n_layers=24,
            mlp_size=4096,
        )


# trainable parameters: 630,776,330
class ViT_H_14(ViT):
    def __init__(self, n_classes: int):
        super().__init__(
            n_classes=n_classes,
            patch_size=14,
            embed_dim=1280,
            n_heads=16,
            n_layers=32,
            mlp_size=5120,
        )


if __name__ == "__main__":
    model_names = {
        "vit_b_16": ViT_B_16,
        "vit_b_32": ViT_B_32,
        "vit_l_16": ViT_L_16,
        "vit_l_32": ViT_L_32,
        "vit_h_14": ViT_H_14,
    }

    parser = ArgumentParser()
    parser.add_argument("model_name", choices=list(model_names.keys()))
    args = parser.parse_args()

    model = model_names[args.model_name]

    summary(
        model(n_classes=10),
        input_size=(
            2,
            3,
            224,
            224,
        ),
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "trainable",
        ),
    )
