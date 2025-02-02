from torch import Tensor
from torch.nn import Embedding, Module

# done


class Embed(Module):
    """
    Implements a learned embedding layer for mapping discrete token indices to dense vector representations.

    This module converts input token indices into continuous vector embeddings using a learned lookup table. The embeddings are scaled by the square root of the
    embedding dimension (`d_model`) to maintain stable gradients during training. It is commonly used in models for natural language processing, sequence modeling,
    and other applications that require efficient representation of categorical inputs.

    Attributes:
        vocab_size (int): Number of unique tokens in the vocabulary.
        d_model (int): Dimensionality of the token embeddings.
        weight (torch.Tensor): Learnable embedding weight matrix of shape (`vocab_size`, `d_model`).
    """

    def __init__(self, vocab_size: int, d_model: int = 448):
        """
        Initializes the Embed class.

        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            d_model (int, optional): Dimensionality of the token embeddings. Default: 512.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self._embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.weight = self._embedding.weight

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Embed class.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, `d_model`), representing the embedded token representations.
        """
        x = self._embedding(x)
        return x * self.d_model**0.5
