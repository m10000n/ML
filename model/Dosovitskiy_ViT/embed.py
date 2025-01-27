from einops.layers.torch import Rearrange
from torch import Tensor, cat, randn
from torch.nn import Conv2d, Linear, Module, Parameter, Sequential


class PatchEmbed(Module):
    """
    A module that divides an image into non-overlapping patches and embeds them into a specified embedding space using either a convolutional or linear layer.

    This class is commonly used in Vision Transformers (ViT) to transform images into sequences of patch embeddings. It takes an input of shape (batch_size,
    img_channels, img_width, img_height), and projects each patch into a vector of dimension `embed_dim`.

    Note:
        The embedding method is determined by the `method` argument:
        - `"conv"`: Embedding is performed using a convolutional layer, which is computationally efficient and commonly used in practice. This approach  produces
        the same result as flattening the patches and projecting them into the embedding space using a linear layer but does so in a more optimized manner.
        - `"linear"`: Embedding is performed using a linear layer, as originally described in the Vision Transformer (ViT) paper, "An Image is Worth 16x16 Words:
        Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).

    Attributes:
        img_channels (int): Number of channels in the input image.
        patch_size (int): Size of each patch.
        embed_dim (int): Dimensionality of the patch embeddings.
        method (str): Embedding method to use, either "conv" or "linear".
        bias (bool): Whether to include a learnable bias term in the embedding layer.
    """

    def __init__(
        self,
        img_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        method: str = "conv",
        bias: bool = False,
    ):
        """
        Initializes the PatchEmbed class.

        Args:
            img_channels (int, optional): Number of channels in the input image. Default: 3.
            patch_size (int, optional): Size of each patch. Default: 16.
            embed_dim (int, optional): Dimensionality of the patch embeddings. Default: 768.
            method (str, optional): Embedding method to use, either "conv" or "linear". Default: "conv".
            bias (bool, optional): Whether to include a bias term in the embedding layer. Default: False.

        Raises:
            ValueError: If an invalid method is provided (not "conv" or "linear").
        """
        super().__init__()
        self.img_channels = img_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.method = method
        self.bias = bias

        if method not in ["conv", "linear"]:
            raise ValueError(
                f"Invalid argument for 'method': {method}. Expected 'conv' or 'linear'."
            )

        if method == "conv":
            self._embed = Sequential(
                Conv2d(
                    in_channels=img_channels,
                    out_channels=embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=bias,
                ),
                Rearrange("b e h w -> b (h w) e"),
            )
        elif method == "linear":
            self._embed = Sequential(
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size,
                    p2=patch_size,
                ),
                Linear(
                    in_features=patch_size * patch_size * img_channels,
                    out_features=embed_dim,
                    bias=bias,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PatchEmbed class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `img_channels`, img_width, img_height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches, `embed_dim`).

        Raises:
            ValueError: img_width and img_height must be divisible by `patch_size`.
        """
        img_width, img_height = x.shape[2], x.shape[3]

        if img_width % self.patch_size != 0 or img_height % self.patch_size:
            raise ValueError(
                f"Input dimensions (width={img_width}, height={img_height}) must be divisible by patch_size={self.patch_size}."
            )

        return self._embed(x)


class AddClassToken(Module):
    """
    A module that prepends a learnable class token to a sequence of patch embeddings.

    This class token is commonly used in Vision Transformers (ViT) to aggregate global information from all patches for tasks such as classification. This class
    token is a learnable parameter that is appended to the sequence of patch embeddings.

    Attributes:
        embed_dim (int): Dimensionality of the patch embeddings.
    """

    def __init__(self, embed_dim: int = 768):
        """
        Initializes the AddClassToken class.

        Args:
            embed_dim (int, optional): The dimensionality of the patch embeddings. Default: 768.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self._class_token = Parameter(randn(1, 1, embed_dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ClassToken class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_patches, `embed_dim`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches + 1, `embed_dim`), where the class token is prepended to the sequence.
        """
        batch_size = x.shape[0]
        class_token = self._class_token.expand(batch_size, 1, self.embed_dim)
        return cat((class_token, x), dim=1)


class PosEmbed(Module):
    """
    A module that adds learnable positional embeddings to a sequence of embeddings.

    Positional embeddings help the model incorporate spatial order information into the input data. By adding these embeddings to both patch embeddings and the class
    token, the model is able to infer positional relationships between patches and better understand spatial context in Vision Transformer (ViT) architectures.

    Attributes:
        n_patches (int): Number of patches in the input sequence. This includes all patches from the image.
        embed_dim (int): Dimensionality of the patch embeddings.
    """

    def __init__(self, n_embed: int = 197, embed_dim: int = 768):
        """
        Initializes the PosEmbed class.

        Args:
            n_embed (int, optional): Number of embeddings in the input sequence. Default: 197.
            embed_dim (int, optional): Dimensionality of the patch embeddings. Default: 768.
        """
        super().__init__()
        self.n_patches = n_embed
        self.embed_dim = embed_dim
        self._pos_embed = Parameter(randn(1, n_embed, embed_dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PosEmbed class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `n_embed`, `embed_dim`).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, `n_embed`, `embed_dim`), where the positional embeddings are added to the input tensor.
        """
        return x + self._pos_embed
