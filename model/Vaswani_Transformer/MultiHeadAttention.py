from torch import Tensor, randn
from torch.nn import Linear, Module

from model.Vaswani_Transformer.ScaledDotProductAttention import (
    ScaledDotProductAttention,
)

# done

class MultiHeadAttention(Module):
    """
    Implements the multi-head attention mechanism.

    This module extends scaled dot-product attention by applying multiple attention heads in parallel, allowing the model to attend to different representation
    subspaces simultaneously. Each head independently projects the query, key, and value matrices, computes attention scores using scaled dot-product attention,
    and outputs weighted values. These outputs are then concatenated and linearly projected to form the final representation. By performing attention across multiple
    heads in parallel, the model captures richer dependencies and learns diverse contextual representations. Multi-head attention is a fundamental component of
    Transformer-based architectures and follows the methodology introduced in the paper "Attention Is All You Need" by Vaswani et al. (2017).

    Attributes:
        d_model (int): Dimensionality of the input and output embeddings.
        n_heads (int): Number of attention heads.
        d_k (int): Dimensionality of each attention head.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8):
        """
        Initializes the MultiHeadAttention class.

        Args:
            d_model (int, optional): Dimensionality of the input and output embeddings. Default: 512
            n_heads (int, optional): Number of attention heads. Default: 8

        Precondition:
            - `d_model` must be divisible by 'n_heads'
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self._W_q = Linear(
            in_features=self.d_model, out_features=self.d_model, bias=False
        )
        self._W_k = Linear(
            in_features=self.d_model, out_features=self.d_model, bias=False
        )
        self._W_v = Linear(
            in_features=self.d_model, out_features=self.d_model, bias=False
        )

        self._attention = ScaledDotProductAttention()

        self._W_o = Linear(in_features=d_model, out_features=d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass of the MultiHeadAttention class.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, `d_model`).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, `d_model`).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, `d_model`).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len, seq_len), applied to the attention scores. Default: None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, `d_model`, `d_model`), representing the attention-weighted values.
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        q = self._W_q(query)  # (batch_size, seq_len, `d_model`)
        q = q.view(
            batch_size, seq_len, self.n_heads, self.d_k
        )
        q = q.transpose(1, 2)  # (batch_size, `n_heads`, seq_len, `d_k`)
        k = (
            self._W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        )  # (batch_size, `n_heads`, seq_len, `d_k`)
        v = (
            self._W_v(value)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # (batch_size, `n_heads`, seq_len, `d_k`)

        multi_head = self._attention(
            query=q, key=k, value=v, mask=mask)  # (batch_size, `n_heads`, seq_len, `d_k`)
        multi_head = multi_head.transpose(
            1, 2
        )  # (batch_size, seq_len, `n_heads`, `d_k`)
        multi_head = multi_head.reshape(
            batch_size, seq_len, self.d_model
        )  # (batch_size, `seq_len`, `d_model`)

        x = self._W_o(multi_head)  # (batch_size, `d_model`, `d_model`)
        return x
