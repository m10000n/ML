from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ReLU

from helper.validator import validator
from model.model_helper import Add
from model.Vaswani_Transformer.architecture.multi_head_attention import (
    MultiHeadAttention,
)


class DecoderBlock(Module):
    """
    Implements a single decoder block for a Transformer model.

    This module consists of a masked multi-head self-attention mechanism, a encoder-decoder attention mechanism that
    attends to the encoder output, and a position-wise feed-forward network. Each sublayer integrates residual
    connections, dropout, and layer normalization to enhance stability. The masked self-attention ensures that each
    token attends only to previous positions, enforcing autoregressive decoding, while the encoder-decoder attention
    enables the decoder to incorporate contextual information from the encoder output. This implementation follows the
    original Transformer decoder block as described in “Attention Is All You Need” (Vaswani et al., 2017).

    Args:
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
        Initializes the DecoderBlock class.

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

        self._m_h_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self._layer_norm1 = LayerNorm(normalized_shape=d_model, eps=1e-6)
        self._e_d_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self._layer_norm2 = LayerNorm(normalized_shape=d_model, eps=1e-6)
        self._feed_forward1 = Linear(in_features=d_model, out_features=dff)
        self._feed_forward2 = Linear(in_features=dff, out_features=d_model)

    def forward(self, x: Tensor, encoder_out: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Forward pass of the DecoderBlock class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, `d_model`).
            encoder_out (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, `d_model`).
            src_mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len, seq_len), applied to the encoder-decoder
                attention scores.
            tgt_mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len, seq_len), applied to the self-attention
                scores.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, `d_model`), representing the transformed decoder
                state after masked self-attention, encoder-decoder attention, and feed-forward transformation.
        """
        residual1 = x
        x = self._m_h_attention(query=x, key=x, value=x, mask=tgt_mask)
        x = self._drop_out(x)
        x = self._add_residual(x=x, residual=residual1)
        x = self._layer_norm1(x)
        residual2 = x
        x = self._e_d_attention(query=x, key=encoder_out, value=encoder_out, mask=src_mask)
        x = self._drop_out(x)
        x = self._add_residual(x, residual2)
        x = self._layer_norm2(x)
        residual3 = x
        x = self._feed_forward1(x)
        x = self._relu(x)
        x = self._feed_forward2(x)
        x = self._drop_out(x)
        return self._add_residual(x=x, residual=residual3)
