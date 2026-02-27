from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Type, cast

import torch
import torch.nn as nn

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator
from model import model_helper
from model.BrT.architecture.patch_embed import (
    PatchEmbed4d_,
    PatchEmbed4dConfig_,
)
from model.Dosovitskiy_ViT.architecture.add_class_token import AddClassToken
from model.Dosovitskiy_ViT.architecture.pos_embed import PosEmbed


class BrainTransformer(Model_):
    """
    Deep learning model for the classification of fMRI data.

    This is a transformer-based model inspired by the Vision Transformer (ViT) introduced in the paper "An
    Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020).
    It divides the input fMRI data into fixed-size patches, encodes each patch into a learnable embedding, prepends a
    learnable class token to the sequence, and processes the resulting sequence through a stack of Transformer encoders.
    The output corresponding to the class token from the final encoder is then passed through a multi-layer perceptron
    (MLP) to produce the final output.

    Attributes:
        config (BrainTransformerConfig): Configuration of the model.
    """

    config: BrainTransformerConfig

    def __init__(
        self,
        config: BrainTransformerConfig,
    ):
        """
        Initializes the BrainTransformer class.

        Args:
            config (BrainTransformerConfig): Configuration of the model.
        """
        super().__init__(config)

        self._dropout = nn.Dropout(p=config.p_drop)

        self._embed = PatchEmbed4d_.create(config.patch_embed)
        self._add_class_token = AddClassToken(d_model=config.d_model)

        shape = model_helper.get_output_shape(
            input_shape=config.input_shape, module=[self._embed, self._add_class_token]
        )

        self._pos_embed = PosEmbed(n_embed=shape[0], d_model=config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dff,
            dropout=config.p_drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self._mlp_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(in_features=config.d_model, out_features=config.dff),
            nn.GELU(),
            nn.Linear(in_features=config.dff, out_features=config.n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        def init_linear(m: nn.Linear | nn.Conv3d) -> None:
            nn.init.trunc_normal_(m.weight, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if getattr(m, "is_patch_embed", False):
                    init_linear(m)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                init_linear(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BrainTransformer class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *`input_shape`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `n_classes`).
        """
        x = self._embed(x)
        x = self._add_class_token(x)
        x = self._pos_embed(x)
        x = self._dropout(x)
        x = self._encoder(x)
        x = x[:, 0, :]
        x = self._mlp_head(x)
        return x


@dataclass(init=False)
class BrainTransformerConfig(ModelConfig_):
    """
    Configuration class for the BrainTransformer class.

    Attributes:
        description (str): Description of the model.
        input_shape (tuple[int, int, int, int]): Shape of the input fMRI data as (volumes, x, y, z).
        n_classes (int): Number of output classes.
        patch_embed (PatchEmbedConfig_): Configuration of the patch embedding module.
        d_model (int): Dimensionality of the patch embeddings.
        n_heads (int): Number of heads in the Transformer encoder.
        n_layers (int): Number of Transformer encoder layers.
        dff (int): Dimensionality of the feed-forward network.
        p_drop (float): Dropout probability.
        class_ (Type[BrainTransformer]): Class of the model.
    """

    class_: ClassVar[Type[BrainTransformer]] = BrainTransformer

    input_shape: tuple[int, int, int, int]
    n_classes: int
    patch_embed: PatchEmbed4dConfig_
    d_model: int
    n_heads: int
    n_layers: int
    dff: int
    p_drop: float

    @validator.constraints("n_classes", "x > 1")
    @validator.constraints("d_model", "x > 0")
    @validator.constraints("n_heads", "x > 0")
    @validator.constraints("n_layers", "x > 0")
    @validator.constraints("dff", "x > 0")
    @validator.constraints("p_drop", "x >= 0 and x <= 1")
    def __init__(
        self,
        description: str,
        input_shape: tuple[int, int, int, int] | list[int],
        n_classes: int,
        patch_embed: PatchEmbed4dConfig_,
        d_model: int = 768,
        n_heads: int = 6,
        n_layers: int = 6,
        dff: int = 1536,
        p_drop: float = 0.1,
    ):
        """
        Initializes the BrainTransformerConfig class.

        Args:
            description (str): Description of the model.
            input_shape (tuple[int, int, int, int] | list[int]):
              Shape of the input fMRI data as (volumes, x, y, z). Must be a tuple or list of 4 positive integers.
            n_classes (int): Number of output classes. Must be greater than 1.
            patch_embed (PatchEmbedConfig_): Configuration of the patch embedding module.
            d_model (int): Dimensionality of the patch embeddings. Must be a positive integer. Default: 768.
            n_heads (int): Number of heads in the Transformer encoder. Must be a positive integer. Default: 6.
            n_layers (int): Number of Transformer encoder layers. Must be a positive integer. Default: 6.
            dff (int): Dimensionality of the feed-forward network. Must be a positive integer. Default: 1536.
            p_drop (float): Dropout probability. Must be a float in range [0, 1]. Default: 0.1.
        """
        if patch_embed.attr_is_initialized("input_shape"):
            raise ValueError("`patch_embed.input_shape` must not be initialized.")

        if patch_embed.attr_is_initialized("d_model"):
            raise ValueError("`patch_embed.d_model` must not be initialized.")

        input_shape_ = cast(tuple[int, int, int, int], tuple(input_shape))

        patch_embed.input_shape = input_shape_
        patch_embed.d_model = d_model

        super().__init__(description=description, input_shape=input_shape_)

        self.n_classes = n_classes
        self.patch_embed = patch_embed
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dff = dff
        self.p_drop = p_drop

    @classmethod
    def _from_dict(cls: Type[BrainTransformerConfig], dict_: dict[str, Any]) -> BrainTransformerConfig:
        patch_embed_dict = dict_.pop("patch_embed")
        patch_embed_dict.pop("input_shape")
        patch_embed_dict.pop("d_model")

        return cls(
            description=dict_["description"],
            input_shape=dict_["input_shape"],
            n_classes=dict_["n_classes"],
            patch_embed=PatchEmbed4dConfig_.from_dict(patch_embed_dict),
            d_model=dict_["d_model"],
            n_heads=dict_["n_heads"],
            n_layers=dict_["n_layers"],
            dff=dict_["dff"],
            p_drop=dict_["p_drop"],
        )

    def _as_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "input_shape": self.input_shape,
            "n_classes": self.n_classes,
            "patch_embed": self.patch_embed.as_dict(),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dff": self.dff,
            "p_drop": self.p_drop,
        }
