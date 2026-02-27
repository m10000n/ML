from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ReLU

from helper.validator import validator
from model.model_helper import Add
from model.Vaswani_Transformer.architecture.multi_head_attention import (
    MultiHeadAttention,
)


class EncoderBlock(Module):
    """
    Implements a single encoder block for a Transformer model.

    This module consists of a multi-head self-attention mechanism followed by a position-wise feed-forward network. Each
    sublayer includes dropout, residual connections and layer normalization for stability. The multi-head attention
    allows the model to attend to different positions in the input sequence simultaneously, while the feed-forward
    network enhances feature transformation. This implementation follows the original Transformer encoder block
    described in "Attention Is All You Need" (Vaswani et al., 2017).

    Attributes:
        d_model (int): Dimensionality of the token embeddings.
        n_heads (int): Number of attention heads.
        dff (int): Dimensionality of the feed-forward network.
        p_drop (float): Dropout probability.
    """

    @validator.constraints("d_model", "x > 0")
    @validator.constraints("n_heads", "x > 0")
    @validator.constraints("dff", "x > 0")
    @validator.constraints("p_drop", "x >= 0 and x <= 1")
    def __init__(self, d_model: int = 512, n_heads: int = 8, dff: int = 2048, p_drop: float = 0.1):
        """
        Initializes the EncoderBlock class.

        Args:
            d_model (int): Dimensionality of the token embeddings. Must be a positive integer. Default: 512.
            n_heads (int): Number of attention heads. Must be a positive integer. Default: 8.
            dff (int): Dimensionality of the feed-forward network. Must be a positive integer. Default: 2048.
            p_drop (float): Dropout probability. Must be in the range [0, 1]. Default: 0.1.
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.dff = dff
        self.p_drop = p_drop

        self._relu = ReLU()
        self._add_residual = Add()
        self._drop_out = Dropout(p=p_drop)

        self._multi_head_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self._layer_norm1 = LayerNorm(normalized_shape=d_model, eps=1e-6)
        self._feed_forward1 = Linear(in_features=d_model, out_features=dff, bias=True)
        self._feed_forward2 = Linear(in_features=dff, out_features=d_model, bias=True)
        self._layer_norm2 = LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass of the EncoderBlock class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len, seq_len), applied to the attention
                scores. Default: None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model), representing the transformed output
                after self-attention, feed-forward, and normalization.
        """
        residual1 = x
        x = self._multi_head_attention(query=x, key=x, value=x, mask=mask)
        x = self._drop_out(x)
        x = self._add_residual(x=x, residual=residual1)
        x = self._layer_norm1(x)
        residual2 = x
        x = self._feed_forward1(x)
        x = self._relu(x)
        x = self._feed_forward2(x)
        x = self._drop_out(x)
        x = self._add_residual(x=x, residual=residual2)
        return self._layer_norm2(x)
