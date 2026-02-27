from torch import Tensor, cat, randn
from torch.nn import Module, Parameter

from helper.validator import validator


class AddClassToken(Module):
    """
    Prepends a learnable class token to a sequence of patch embeddings.

    This module prepends a learnable class token to a sequence of patch embeddings. This class token is commonly used in
    Vision Transformer (ViT) architectures to aggregate global information from all patches for tasks such as
    classification.

    Attributes:
        d_model (int): Dimensionality of the patch embeddings.
    """

    @validator.constraints("d_model", "x > 0")
    def __init__(self, d_model: int = 768):
        """
        Initializes the AddClassToken class.

        Args:
            d_model (int): The dimensionality of the patch embeddings. Must be a positive integer. Default: 768.
        """
        super().__init__()

        self.d_model = d_model

        self._class_token = Parameter(randn(1, 1, d_model), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the AddClassToken class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_patches, `d_model`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches + 1, `d_model`), where the class token is
                prepended to the sequence.
        """
        batch_size = x.shape[0]
        class_token = self._class_token.expand(batch_size, 1, self.d_model)
        return cat((class_token, x), dim=1)
