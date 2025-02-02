from torch import Tensor
from torch.nn import Dropout, Module, ReLU, LayerNorm, Linear

from model.helper import AddResidual
from model.Vaswani_Transformer.MultiHeadAttention import MultiHeadAttention

# done

class DecoderBlock(Module):
    """
    Implements a single decoder block for a Transformer model.

    This module consists of a masked multi-head self-attention mechanism, a cross-attention mechanism that attends to the encoder output, and a position-wise
    feed-forward network. Each sublayer integrates residual connections, dropout, and layer normalization to enhance stability. The masked self-attention ensures
    that each token attends only to previous positions, enforcing autoregressive decoding, while the cross-attention enables the decoder to incorporate contextual
    information from the encoder output. This implementation follows the original Transformer decoder block as described in “Attention Is All You Need”
    (Vaswani et al., 2017).
    """
    def __init__(
        self, d_model: int = 512, n_heads: int = 8, dff: int = 2048, p_drop: float = 0.1
    ):
        """
        Initializes the DecoderBlock class.

        Args:
            d_model (int, optional): Dimensionality of the token embeddings. Default: 512.
            n_heads (int, optional): Number of attention heads. Default: 8.
            dff (int, optional): Dimensionality of the feed-forward network. Default: 2048.
            p_drop (float, optional): Dropout probability. Default: 0.1.
        """
        super().__init__()
        self._relu = ReLU()
        self._add_residual = AddResidual()
        self._drop_out = Dropout(p=p_drop)

        self._multi_head_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads
        )
        self._layer_norm1 = LayerNorm(normalized_shape=d_model, eps=1e-6)
        self._cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self._layer_norm2 = LayerNorm(normalized_shape=d_model, eps=1e-6)
        self._feed_forward1 = Linear(in_features=d_model, out_features=dff)
        self._feed_forward2 = Linear(in_features=dff, out_features=d_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor
    ) -> Tensor:
        """
        Forward pass of the DecoderBlock class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            encoder_out (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len, seq_len), applied to the encoder-decoder attention scores.
            tgt_mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len, seq_len), applied to the self-attention scores.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model), representing the transformed decoder state after masked self-attention,
            encoder-decoder attention, and feed-forward transformation.
        """
        residual1 = x
        x = self._multi_head_attention(query=x, key=x, value=x, mask=tgt_mask)
        x = self._drop_out(x)
        x = self._add_residual(x=x, residual=residual1)
        x = self._layer_norm1(x)
        residual2 = x
        x = self._cross_attention(query=x, key=encoder_out, value=encoder_out, mask=src_mask)
        x = self._drop_out(x)
        x = self._add_residual(x, residual2)
        x = self._layer_norm2(x)
        residual3 = x
        x = self._feed_forward1(x)
        x = self._relu(x)
        x = self._feed_forward2(x)
        x = self._drop_out(x)
        return self._add_residual(x=x, residual=residual3)
