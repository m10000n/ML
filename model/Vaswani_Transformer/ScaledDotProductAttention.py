from torch import Tensor, ones_like, triu
from torch.nn import Module, Softmax
from math import sqrt

# done
class ScaledDotProductAttention(Module):
    """
    Implements the scaled dot-product attention mechanism.

    This module computes attention by taking the dot product of the query and key matrices, scaling the result by the square root of the key dimension, and
    applying a softmax function to obtain attention weights. These weights are then used to weight the value matrix, producing the final output. This mechanism
    enables the model to dynamically focus on relevant parts of the input sequence and is a core component of Transformer-based architectures. This implementation
    follows the methodology introduced in the paper "Attention Is All You Need" by Vaswani et al. (2017).
    """

    def __init__(self):
        """
        Initializes the ScaledDotProductAttention class.
        """
        super().__init__()

        self._softmax = Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass of the ScaledDotProductAttention class.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_k).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_k).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_v).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len, seq_len), applied to the attention scores. Default: None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_v), representing the attention-weighted values.
        """
        d_k = query.shape[-1]

        # MatMul
        x = query @ key.mT  # (batch_size, seq_len, seq_len)
        # Scale
        x = x / sqrt(d_k)

        # Mask
        if mask:
            x = x.masked_fill(mask == 0, float("-inf"))

        # Softmax
        x = self._softmax(x)
        # MatMul
        x = x @ value  # (batch_size, seq_len, d_v)
        return x
