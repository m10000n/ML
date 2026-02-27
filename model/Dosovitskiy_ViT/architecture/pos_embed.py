from torch import Tensor, randn
from torch.nn import Module, Parameter

from helper.validator import validator


class PosEmbed(Module):
    """
    Adds a learnable positional embeddings to a sequence of embeddings.

    This module adds a learnable positional embedding to a sequence of embeddings, allowing the model to incorporate
    spatial order information into the input. By adding these embeddings to both the patch embeddings and the class
    token, the model can infer positional relationships between patches and better capture spatial context, which is
    crucial in Vision Transformer (ViT) architectures.

    Attributes:
        n_embed (int): Number of embeddings in the input sequence
        d_model (int): Dimensionality of the patch embeddings.
    """

    @validator.constraints("n_embed", "x > 0")
    @validator.constraints("d_model", "x > 0")
    def __init__(self, n_embed: int = 197, d_model: int = 768):
        """
        Initializes the PosEmbed class.

        Args:
            n_embed (int): Number of embeddings in the input sequence. Must be a positive integer. Default: 197.
            d_model (int): Dimensionality of the patch embeddings. Must be a positive integer. Default: 768.
        """
        super().__init__()

        self.n_embed = n_embed
        self.d_model = d_model

        self._pos_embed = Parameter(randn(1, n_embed, d_model), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PosEmbed class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, `n_embed`, `d_model`).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, `n_embed`, `d_model`), where the positional
                embeddings are added to the input tensor.
        """
        return x + self._pos_embed
