from typing import Literal

import einops.layers.torch as einops
from torch import Tensor
from torch.nn import Conv2d, Linear, Module, Sequential

from helper.validator import validator


class PatchEmbed(Module):
    """
    Learned patch embedding that maps an image into a sequence of patch embeddings.

    This module divides an image into non-overlapping patches and projects them into a specified embedding space using
    either a convolutional or linear layer. It is commonly used in Vision Transformer (ViT) architectures.

    Note:
        The embedding method is determined by `method`:
        - "conv": Embedding is performed using a convolutional layer, which is computationally efficient and commonly
            used in practice. This approach  produces the same result as flattening the patches and projecting them into
            the embedding space using a linear layer but does so in a more optimized manner.
        - "linear": Embedding is performed using a linear layer, as originally described in the paper "An Image is
            Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).

    Attributes:
        img_channels (int): Number of channels in the input image.
        patch_size (tuple[int, int]): Size of each patch.
        d_model (int): Dimensionality of the patch embeddings.
        method (str): Embedding method to use.
    """

    @validator.constraints("img_channels", "x > 0")
    @validator.constraints("patch_size", "x > 0")
    @validator.constraints("d_model", "x > 0")
    def __init__(
        self,
        img_channels: int = 3,
        patch_size: tuple[int, int] = (16, 16),
        d_model: int = 768,
        method: Literal["conv", "linear"] = "conv",
    ):
        """
        Initializes the PatchEmbed class.

        Args:
            img_channels (int): Number of channels in the input image. Must be a positive integer. Default: 3.
            patch_size (tuple[int, int]):
                Size of each patch. Must be a tuple of two positive integers. Default: (16, 16).
            d_model (int): Dimensionality of the patch embeddings. Must be a positive integer. Default: 768.
            method (str): Embedding method to use, either "conv" or "linear". Default: "conv".
            bias (bool): Whether to include a bias term in the embedding layer. Default: False.
        """
        super().__init__()

        self.img_channels = img_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.method = method

        if method == "conv":
            self._embed = Sequential(
                Conv2d(
                    in_channels=self.img_channels,
                    out_channels=self.d_model,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                    bias=False,
                ),
                einops.Rearrange("b e h w -> b (h w) e"),
            )
        elif method == "linear":
            self._embed = Sequential(
                einops.Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=self.patch_size[0],
                    p2=self.patch_size[1],
                ),
                Linear(
                    in_features=self.patch_size[0] * self.patch_size[1] * self.img_channels,
                    out_features=self.d_model,
                    bias=False,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PatchEmbed class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `img_channels`, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches, `d_model`).

        Precondition:
            - width and height must be divisible by `patch_size`.
        """
        assert x.shape[2] % self.patch_size[0] == 0 or x.shape[3] % self.patch_size[1] == 0, (
            f"Input dimensions (width={x.shape[2]}, height={x.shape[3]}) must be divisible by `patch_size`."
            f"({self.patch_size})."
        )

        return self._embed(x)
