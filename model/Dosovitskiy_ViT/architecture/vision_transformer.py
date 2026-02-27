from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type, cast

from torch import Tensor
from torch.nn import (
    GELU,
    Dropout,
    LayerNorm,
    Linear,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator
from model.Dosovitskiy_ViT.architecture.add_class_token import AddClassToken
from model.Dosovitskiy_ViT.architecture.patch_embed import PatchEmbed
from model.Dosovitskiy_ViT.architecture.pos_embed import PosEmbed


class VisionTransformer(Model_):
    """
    Deep learning model for the classification of images.

    This model divides an input image into fixed-size patches, encodes each patch into a learnable embedding, and
    processes the sequence of embeddings through a stack of Transformer encoders. By utilizing self-attention
    mechanisms, the Vision Transformer (ViT) effectively captures both local and global dependencies between patches,
    enabling accurate and robust image classification. This implementation follows the methodology introduced in the
    paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020).

    Attributes:
        config (VisionTransformerConfig): Configuration of the model.
        n_patches (int): Number of patches in the input image.
    """

    config: VisionTransformerConfig

    def __init__(
        self,
        config: VisionTransformerConfig,
    ):
        """
        Initializes the VisionTransformer class.

        Args:
            config (VisionTransformerConfig): Configuration of the model.
        """
        super().__init__(config)
        self.n_patches = config.input_shape[1] * config.input_shape[2] // (config.patch_size[0] * config.patch_size[1])

        self._embed = PatchEmbed(
            img_channels=config.input_shape[0], patch_size=config.patch_size, d_model=config.d_model
        )
        self._add_class_token = AddClassToken(config.d_model)
        self._pos_embed = PosEmbed(n_embed=self.n_patches + 1, d_model=config.d_model)
        self._embed_do = Dropout(p=config.p_drop)
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dff,
            dropout=config.p_drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self._mlp_head = Sequential(
            LayerNorm(config.d_model),
            Linear(in_features=config.d_model, out_features=config.dff),
            GELU(),
            Linear(in_features=config.dff, out_features=config.n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the VisionTransformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *`input_shape`).

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


@dataclass(init=False)
class VisionTransformerConfig(ModelConfig_):
    """
    Configuration class for the Vision Transformer class.

    Attributes:
        description: (str): Description of the model.
        input_shape: (tuple[int, int, int]): Shape of the input image as (channels, width, height).
        n_classes: (int): Number of output classes.
        patch_size: (tuple[int, int]): Size of each patch.
        d_model: (int): Dimensionality of the patch embeddings.
        n_heads: (int): Number of heads in the Transformer encoder.
        n_layers: (int): Number of Transformer encoder layers.
        dff: (int): Dimensionality of the feed-forward network.
        p_drop: (float): Dropout probability.
        class_: (Type[VisionTransformer]): Class of the model.
    """

    class_: ClassVar[Type[VisionTransformer]] = VisionTransformer

    input_shape: tuple[int, int, int]
    n_classes: int
    patch_size: tuple[int, int]
    d_model: int
    n_heads: int
    n_layers: int
    dff: int
    p_drop: float

    @validator.constraints("n_classes", "x > 1")
    @validator.constraints("patch_size", "x > 0")
    @validator.constraints("d_model", "x > 0")
    @validator.constraints("n_heads", "x > 0")
    @validator.constraints("n_layers", "x > 0")
    @validator.constraints("dff", "x > 0")
    @validator.constraints("p_drop", "x >= 0 and x <= 1")
    def __init__(
        self,
        description: str,
        input_shape: tuple[int, int, int] | list[int],
        n_classes: int,
        patch_size: tuple[int, int] | list[int] = (16, 16),
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        dff: int = 3072,
        p_drop: float = 0.1,
    ):
        """
        Initializes the VisionTransformerConfig class.

        Args:
            description: (str): Description of the model.
            input_shape (tuple[int, int, int] | list[int]):
                Shape of the input image as (channels, width, height). Must be a tuple or list of 3 positive integers.
            n_classes (int): Number of output classes. Must be greater than 1.
            patch_size (tuple[int, int] | list[int]):
              Size of each patch. Must be a tuple or list of 2 positive integers. Default: (16, 16).
            d_model (int): Dimensionality of the patch embeddings. Must be a positive integer. Default: 768.
            n_heads (int): Number of heads in the Transformer encoder. Must be a positive integer. Default: 12.
            n_layers (int): Number of Transformer encoder layers. Must be a positive integer. Default: 12.
            dff (int): Dimensionality of the feed-forward network. Must be a positive integer. Default: 3072.
            p_drop (float): Dropout probability. Must be a float in range [0, 1]. Default: 0.1.

        Precondition:
            - width must be divisible by `patch_size[0]`
            - height must be divisible by `patch_size[1]`
        """
        if isinstance(patch_size, list) and len(patch_size) != 2:
            raise ValueError(f"If `patch_size` is a list, it must be of length 2, got {len(patch_size)}.")

        super().__init__(description=description, input_shape=input_shape)

        patch_size_ = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        if not input_shape[1] % patch_size_[0] == 0:
            raise ValueError(
                f"The image width ({input_shape[1]}) must be divisible by the patch size ({patch_size_[0]})."
            )
        if not input_shape[2] % patch_size_[1] == 0:
            raise ValueError(
                f"The image height ({input_shape[2]}) must be divisible by the patch size ({patch_size_[1]})."
            )

        super().__init__(description=description, input_shape=tuple(input_shape))
        self.n_classes = n_classes
        self.patch_size = cast(tuple[int, int], patch_size_)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dff = dff
        self.p_drop = p_drop
